from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # Field không được sử dụng, có thể bỏ
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import requests
import polyline  # polyline được dùng trong get_route_geometry_impl
import io
import urllib.parse  # Thêm import này

from src.models.geneticAlgorithm import genetic_algorithm_vrptw_solver, LocationModel, VRPTWSolutionOutput, VRPTWDataInput as GA_VRPTWDataInput


class VRPTWDataInputAPI(BaseModel):
    locations: List[LocationModel]
    depot_idx: int
    distance_matrix: Optional[List[List[float]]] = None
    time_matrix: Optional[List[List[float]]] = None
    vehicle_capacity: int = 100


class SearchResultItem(BaseModel):
    name: str
    latitude: float
    longitude: float


class MatrixInputLocation(BaseModel):
    latitude: float
    longitude: float


class MatricesResponse(BaseModel):
    distances: Optional[List[List[float]]] = None
    durations: Optional[List[List[float]]] = None
    error: Optional[str] = None


class RouteGeometryRequest(BaseModel):
    start_coord: List[float]  # [lat, lon]
    end_coord: List[float]   # [lat, lon]


class ParsedExcelResponse(BaseModel):
    locations: Optional[List[LocationModel]] = None
    error: Optional[str] = None
    message: Optional[str] = None


app = FastAPI(title="VRPTW Solver Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def search_location_impl(query: str) -> List[SearchResultItem]:

    params = {'q': query, 'format': 'json', 'limit': 5}
    # Thêm thông tin liên hệ nếu có thể
    headers = {
        'User-Agent': 'VRPTW_Solver_Backend/1.0 (github.com/yourusername/yourrepo)'}

    print(
        f"Nominatim search_location_impl: Sending query '{query}' with params: {params}")

    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=60
        )
        print(
            f"Nominatim response status: {response.status_code} for query: '{query}'")
        # print(f"Nominatim response headers: {response.headers}") # Debug
        # print(f"Nominatim response text: {response.text[:500]}...") # In một phần response text để debug

        response.raise_for_status()  # Sẽ raise HTTPError cho mã lỗi 4xx/5xx

        # Điều này có thể raise JSONDecodeError nếu response không phải JSON hợp lệ
        data = response.json()

        results = []
        if isinstance(data, list):  # Nominatim thường trả về một list các kết quả
            for item in data:
                if isinstance(item, dict) and 'display_name' in item and 'lat' in item and 'lon' in item:
                    try:
                        results.append(SearchResultItem(
                            name=item['display_name'],
                            latitude=float(item['lat']),
                            longitude=float(item['lon'])
                        ))
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing item: {item}. Error: {e}")
                else:
                    print(f"Skipping malformed item: {item}")
        else:
            print(
                f"Unexpected data format from Nominatim (expected list): {type(data)}")

        return results

    except requests.exceptions.HTTPError as http_err:
        print(
            f"HTTP error occurred with Nominatim: {http_err} - Response: {response.text[:500] if response else 'No response'}")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error with Nominatim: {conn_err}")
        return []
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error with Nominatim: {timeout_err}")
        return []
    except requests.exceptions.RequestException as req_err:
        print(f"An ambiguous request error occurred with Nominatim: {req_err}")
        return []
    except ValueError as json_decode_err:  # Cụ thể cho lỗi parse JSON
        print(f"JSON decode error from Nominatim response: {json_decode_err}")
        print(
            f"Response text that failed to parse: {response.text[:500] if response else 'No response text'}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred in search_location_impl: {e}")
        import traceback
        traceback.print_exc()  # In đầy đủ traceback để debug
        return []


def calculate_distance_matrix_api_impl(locations: List[MatrixInputLocation]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if len(locations) < 2:
        return None, None, "Need at least 2 points"
    try:
        coords = ";".join(
            f"{loc.longitude},{loc.latitude}" for loc in locations)
        response = requests.get(
            f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration", timeout=30)
        response.raise_for_status()
        data = response.json()
        return np.array(data['distances'])/1000, np.array(data['durations'])/60, None
    except requests.exceptions.HTTPError as http_err:
        return None, None, f"OSRM API HTTP Error: {http_err} - {response.text if response else ''}"
    except Exception as e:
        return None, None, f"OSRM API Error: {str(e)}"


def get_route_geometry_impl(start_coord: List[float], end_coord: List[float]) -> Optional[List[List[float]]]:
    try:
        response = requests.get(f"https://router.project-osrm.org/route/v1/driving/{start_coord[1]},{start_coord[0]};{end_coord[1]},{end_coord[0]}",
                                params={'overview': 'full', 'geometries': 'polyline'}, timeout=30)
        response.raise_for_status()
        data = response.json()
        return polyline.decode(data['routes'][0]['geometry']) if data.get('routes') else None
    except Exception:  # Giữ im lặng cho lỗi này vì nó không quá quan trọng
        return None


def solve_vrptw_with_ga(data_api: VRPTWDataInputAPI) -> VRPTWSolutionOutput:
    ga_input_data = GA_VRPTWDataInput(
        locations=data_api.locations,
        depot_idx=data_api.depot_idx,
        distance_matrix=data_api.distance_matrix,
        time_matrix=data_api.time_matrix,
        vehicle_capacity=data_api.vehicle_capacity
    )
    solution = genetic_algorithm_vrptw_solver(ga_input_data)
    return solution

# --- FastAPI Endpoints ---


@app.get("/search_location/", response_model=List[SearchResultItem])
async def api_search_location(query: str):
    if not query:
        raise HTTPException(
            status_code=400, detail="Query parameter 'q' is required.")
    

    results = search_location_impl(query)

    return results


@app.post("/calculate_matrices/", response_model=MatricesResponse)
async def api_calculate_matrices(locations_input: List[MatrixInputLocation]):
    if not locations_input or len(locations_input) < 2:
        return MatricesResponse(error="At least two locations are required.")
    distances, durations, error = calculate_distance_matrix_api_impl(
        locations_input)
    if error:
        return MatricesResponse(error=error)  # Trả về lỗi nếu có
    if distances is None or durations is None:  # Kiểm tra kỹ hơn
        return MatricesResponse(error="Failed to calculate matrices from OSRM.")
    return MatricesResponse(distances=distances.tolist(), durations=durations.tolist())


@app.post("/route_geometry/", response_model=Optional[List[List[float]]])
async def api_get_route_geometry(request: RouteGeometryRequest):
    return get_route_geometry_impl(request.start_coord, request.end_coord)


@app.post("/solve_vrptw/", response_model=VRPTWSolutionOutput)
async def api_solve_vrptw(data: VRPTWDataInputAPI):
    if not data.locations:
        raise HTTPException(status_code=400, detail="No locations provided.")
    if data.depot_idx < 0 or data.depot_idx >= len(data.locations):
        raise HTTPException(status_code=400, detail="Invalid depot index.")

    actual_depot_found = False
    if 0 <= data.depot_idx < len(data.locations) and data.locations[data.depot_idx].is_depot:
        actual_depot_found = True
    else:
        for idx, loc in enumerate(data.locations):
            if loc.is_depot:
                data.depot_idx = idx
                actual_depot_found = True
                print(
                    f"Warning: Original depot_idx was problematic or pointed to a non-depot. Using first found depot at index {idx}.")
                break
    if not actual_depot_found:
        raise HTTPException(
            status_code=400, detail="No valid depot found in the locations list.")

    if data.distance_matrix is None or data.time_matrix is None:
        raise HTTPException(
            status_code=400, detail="Distance and Time matrices are required for solving.")
    return solve_vrptw_with_ga(data)


@app.post("/parse_excel/", response_model=ParsedExcelResponse)
async def api_parse_excel(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        required_cols = ['Tên địa điểm', 'Vĩ độ', 'Kinh độ', 'Loại']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return ParsedExcelResponse(error=f"Thiếu cột: {', '.join(missing_cols)}")

        def time_to_minutes(time_str):
            try:
                if pd.isna(time_str) or time_str == '':
                    return 0
                if ':' in str(time_str):
                    h, m = map(int, str(time_str).split(':'))
                    return h * 60 + m
                return int(time_str)
            except:
                return 0

        new_locations_data = []
        for idx, row in df.iterrows():
            new_locations_data.append(LocationModel(
                id=idx + 1,
                name=row['Tên địa điểm'],
                latitude=row['Vĩ độ'],
                longitude=row['Kinh độ'],
                demand=int(row.get('Nhu cầu', 10) if pd.notna(
                    row.get('Nhu cầu', 10)) else 10),
                earliest_time=time_to_minutes(row.get('Giờ mở', '08:00')),
                latest_time=time_to_minutes(row.get('Giờ đóng', '17:00')),
                service_time=int(row.get('Thời gian phục vụ', 15) if pd.notna(
                    row.get('Thời gian phục vụ', 15)) else 15),
                description=str(row.get('Mô tả', '')),
                is_depot='kho' in str(row['Loại']).lower(
                ) or 'depot' in str(row['Loại']).lower()
            ))

        if not any(loc.is_depot for loc in new_locations_data):
            return ParsedExcelResponse(error="File phải có ít nhất 1 kho (Loại = 'Kho' hoặc 'Depot')")
        return ParsedExcelResponse(locations=new_locations_data, message=f"Đã import {len(new_locations_data)} địa điểm!")
    except Exception as e:
        print(f"Error parsing Excel: {e}")  # In lỗi ra console backend
        import traceback
        traceback.print_exc()
        return ParsedExcelResponse(error=f"Lỗi đọc file Excel: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Đảm bảo tên file khớp, ví dụ nếu file này là main.py thì "main:app"
    uvicorn.run("fastAPI:app", host="0.0.0.0", port=8000, reload=True)
