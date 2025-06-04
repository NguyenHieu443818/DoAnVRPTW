from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field # Field không được sử dụng, có thể bỏ
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
import requests
import polyline
import io
# Giả sử file geneticAlgorithm.py nằm trong thư mục src/models
# và chứa hàm genetic_algorithm_vrptw_solver đã được cập nhật
# để không yêu cầu max_time.
from src.models.geneticAlgorithm import genetic_algorithm_vrptw_solver, LocationModel, VRPTWSolutionOutput, VRPTWDataInput as GA_VRPTWDataInput

# --- Data classes (Pydantic for FastAPI) ---
# Sử dụng lại LocationModel và VRPTWSolutionOutput từ geneticAlgorithm.py nếu chúng giống hệt
# Nếu có sự khác biệt nhỏ, bạn có thể định nghĩa lại ở đây.
# Để đơn giản, tôi sẽ giả định chúng ta có thể import và dùng trực tiếp.

# class LocationModel(BaseModel): # Đã import từ geneticAlgorithm
#     id: int
#     name: str
#     latitude: float
#     longitude: float
#     demand: int
#     earliest_time: int
#     latest_time: int
#     service_time: int
#     description: str
#     is_depot: bool

class VRPTWDataInputAPI(BaseModel): # Đổi tên để tránh xung đột nếu GA_VRPTWDataInput khác
    locations: List[LocationModel]
    depot_idx: int
    distance_matrix: Optional[List[List[float]]] = None
    time_matrix: Optional[List[List[float]]] = None
    vehicle_capacity: int = 100
    # max_time không còn ở đây

# class VRPTWSolutionOutput(BaseModel): # Đã import từ geneticAlgorithm
#     routes: List[List[int]]
#     total_distance: float
#     total_time: float
#     num_vehicles_used: int
#     is_feasible: bool

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
    start_coord: List[float] # [lat, lon]
    end_coord: List[float]   # [lat, lon]

class ParsedExcelResponse(BaseModel):
    locations: Optional[List[LocationModel]] = None # Sử dụng LocationModel đã import
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
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search",
                                params={'q': query, 'format': 'json', 'limit': 5},
                                headers={'User-Agent': 'VRPTW_Solver_Backend/1.0'}, timeout=10)
        if response.status_code == 200:
            return [SearchResultItem(name=item['display_name'], latitude=float(item['lat']), longitude=float(item['lon']))
                    for item in response.json()]
        return []
    except Exception:
        return []

def calculate_distance_matrix_api_impl(locations: List[MatrixInputLocation]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if len(locations) < 2:
        return None, None, "Need at least 2 points"
    try:
        coords = ";".join(f"{loc.longitude},{loc.latitude}" for loc in locations)
        response = requests.get(
            f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return np.array(data['distances'])/1000, np.array(data['durations'])/60, None
        return None, None, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return None, None, str(e)

def get_route_geometry_impl(start_coord: List[float], end_coord: List[float]) -> Optional[List[List[float]]]:
    try:
        response = requests.get(f"https://router.project-osrm.org/route/v1/driving/{start_coord[1]},{start_coord[0]};{end_coord[1]},{end_coord[0]}",
                                params={'overview': 'full', 'geometries': 'polyline'}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return polyline.decode(data['routes'][0]['geometry']) if data.get('routes') else None
    except Exception:
        return None

# Hàm này bây giờ là wrapper cho GA solver
def solve_vrptw_with_ga(data_api: VRPTWDataInputAPI) -> VRPTWSolutionOutput:
    # Chuyển đổi từ VRPTWDataInputAPI (không có max_time)
    # sang GA_VRPTWDataInput (cũng không có max_time theo yêu cầu)
    
    # Tạo đối tượng GA_VRPTWDataInput từ data_api
    # Đảm bảo các trường khớp với định nghĩa của GA_VRPTWDataInput trong file geneticAlgorithm.py
    ga_input_data = GA_VRPTWDataInput(
        locations=data_api.locations,
        depot_idx=data_api.depot_idx,
        distance_matrix=data_api.distance_matrix,
        time_matrix=data_api.time_matrix,
        vehicle_capacity=data_api.vehicle_capacity
        # max_time không được truyền nữa
    )
    
    solution = genetic_algorithm_vrptw_solver(ga_input_data)
    return solution

# --- FastAPI Endpoints ---
@app.get("/search_location/", response_model=List[SearchResultItem])
async def api_search_location(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
    return search_location_impl(query)

@app.post("/calculate_matrices/", response_model=MatricesResponse)
async def api_calculate_matrices(locations_input: List[MatrixInputLocation]):
    if not locations_input or len(locations_input) < 2:
        return MatricesResponse(error="At least two locations are required.")
    distances, durations, error = calculate_distance_matrix_api_impl(locations_input)
    if error:
        return MatricesResponse(error=error)
    if distances is None or durations is None:
        return MatricesResponse(error="Failed to calculate matrices for an unknown reason.")
    return MatricesResponse(distances=distances.tolist(), durations=durations.tolist())

@app.post("/route_geometry/", response_model=Optional[List[List[float]]])
async def api_get_route_geometry(request: RouteGeometryRequest):
    return get_route_geometry_impl(request.start_coord, request.end_coord)

@app.post("/solve_vrptw/", response_model=VRPTWSolutionOutput) # response_model là VRPTWSolutionOutput từ GA
async def api_solve_vrptw(data: VRPTWDataInputAPI): # data là VRPTWDataInputAPI (không có max_time)
    if not data.locations:
        raise HTTPException(status_code=400, detail="No locations provided.")
    if data.depot_idx < 0 or data.depot_idx >= len(data.locations):
        raise HTTPException(status_code=400, detail="Invalid depot index.")
    
    actual_depot_found = False
    # Kiểm tra và có thể tự động sửa depot_idx nếu cần
    if 0 <= data.depot_idx < len(data.locations) and data.locations[data.depot_idx].is_depot:
        actual_depot_found = True
    else:
        # Nếu depot_idx ban đầu không hợp lệ, thử tìm depot đầu tiên
        for idx, loc in enumerate(data.locations):
            if loc.is_depot:
                data.depot_idx = idx # Cập nhật lại depot_idx
                actual_depot_found = True
                print(f"Warning: Original depot_idx was problematic or pointed to a non-depot. Using first found depot at index {idx}.")
                break
    
    if not actual_depot_found:
        raise HTTPException(status_code=400, detail="No valid depot found in the locations list.")

    if data.distance_matrix is None or data.time_matrix is None:
         raise HTTPException(status_code=400, detail="Distance and Time matrices are required for solving.")

    # Gọi hàm giải GA (đã được cập nhật để không dùng max_time)
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
                if pd.isna(time_str) or time_str == '': return 0
                if ':' in str(time_str):
                    h, m = map(int, str(time_str).split(':'))
                    return h * 60 + m
                return int(time_str)
            except: return 0

        new_locations_data = []
        for idx, row in df.iterrows():
            new_locations_data.append(LocationModel(
                id=idx + 1,
                name=row['Tên địa điểm'],
                latitude=row['Vĩ độ'],
                longitude=row['Kinh độ'],
                demand=int(row.get('Nhu cầu', 10) if pd.notna(row.get('Nhu cầu', 10)) else 10),
                earliest_time=time_to_minutes(row.get('Giờ mở', '08:00')),
                latest_time=time_to_minutes(row.get('Giờ đóng', '17:00')),
                service_time=int(row.get('Thời gian phục vụ', 15) if pd.notna(row.get('Thời gian phục vụ', 15)) else 15),
                description=str(row.get('Mô tả', '')),
                is_depot= 'kho' in str(row['Loại']).lower() or 'depot' in str(row['Loại']).lower()
            ))
        
        if not any(loc.is_depot for loc in new_locations_data):
            return ParsedExcelResponse(error="File phải có ít nhất 1 kho (Loại = 'Kho' hoặc 'Depot')")
        return ParsedExcelResponse(locations=new_locations_data, message=f"Đã import {len(new_locations_data)} địa điểm!")
    except Exception as e:
        return ParsedExcelResponse(error=f"Lỗi đọc file: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastAPI:app", host="0.0.0.0", port=8000, reload=True)
