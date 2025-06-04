import streamlit as st
import pandas as pd
import folium
import requests
from streamlit_folium import st_folium
import numpy as np
import polyline
from typing import List, Dict, Optional
import random
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
import io


# Config
st.set_page_config(page_title="VRPTW Solver", page_icon="🚛", layout="wide")
st.title("🚛 VRPTW Solver - Vehicle Routing Problem with Time Windows")

# Session state init
defaults = {
    'locations': pd.DataFrame(columns=['id', 'name', 'latitude', 'longitude', 'demand', 'earliest_time', 'latest_time', 'service_time', 'description', 'is_depot']),
    'distance_matrix': None, 'time_matrix': None, 'vrptw_solution': None, 'next_id': 1, 'search_results': []
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Data classes


class VRPTWData:
    def __init__(self):
        self.locations, self.depot_idx, self.distance_matrix, self.time_matrix = [], 0, None, None
        self.vehicle_capacity, self.max_time = 100, 480


class VRPTWSolution:
    def __init__(self):
        self.routes, self.total_distance, self.total_time, self.num_vehicles_used, self.is_feasible = [], 0, 0, 0, True

# Utility functions


def search_location(query):
    try:
        response = requests.get("https://nominatim.openstreetmap.org/search",
                                params={'q': query,
                                        'format': 'json', 'limit': 5},
                                headers={'User-Agent': 'VRPTW_Solver/1.0'}, timeout=10)
        return [{'name': item['display_name'], 'latitude': float(item['lat']), 'longitude': float(item['lon'])}
                for item in response.json()] if response.status_code == 200 else []
    except:
        return []


def calculate_distance_matrix_api(locations):
    if len(locations) < 2:
        return None, None, "Need at least 2 points"
    try:
        coords = ";".join(
            f"{loc['longitude']},{loc['latitude']}" for loc in locations)
        response = requests.get(
            f"https://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance,duration", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return np.array(data['distances'])/1000, np.array(data['durations'])/60, None
        return None, None, f"API Error: {response.status_code}"
    except Exception as e:
        return None, None, str(e)


def get_route_geometry(start_coord, end_coord):
    try:
        response = requests.get(f"https://router.project-osrm.org/route/v1/driving/{start_coord[1]},{start_coord[0]};{end_coord[1]},{end_coord[0]}",
                                params={'overview': 'full', 'geometries': 'polyline'}, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return polyline.decode(data['routes'][0]['geometry']) if data['routes'] else None
    except:
        return None


def simple_vrptw_solver(data: VRPTWData) -> VRPTWSolution:
    solution = VRPTWSolution()
    if data.distance_matrix is None or len(data.locations) <= 1:
        solution.is_feasible = False
        return solution

    customers = [i for i, loc in enumerate(
        data.locations) if not loc.get('is_depot', False)]
    if not customers:
        solution.is_feasible = False
        return solution

    unvisited, routes, depot_idx = set(customers), [], data.depot_idx

    while unvisited:
        route, current_load, current_time, current_location = [], 0, 0, depot_idx

        while unvisited:
            best_customer, best_distance = None, float('inf')

            for customer_idx in unvisited:
                customer = data.locations[customer_idx]
                if current_load + customer.get('demand', 0) > data.vehicle_capacity:
                    continue

                travel_time = data.time_matrix[current_location][customer_idx]
                arrival_time = current_time + travel_time
                if arrival_time > customer.get('latest_time', 1440):
                    continue

                distance = data.distance_matrix[current_location][customer_idx]
                if distance < best_distance:
                    best_distance, best_customer = distance, customer_idx

            if best_customer is None:
                break

            route.append(best_customer)
            unvisited.remove(best_customer)
            customer = data.locations[best_customer]
            current_load += customer.get('demand', 0)

            travel_time = data.time_matrix[current_location][best_customer]
            arrival_time = current_time + travel_time
            start_service_time = max(
                arrival_time, customer.get('earliest_time', 0))
            current_time = start_service_time + customer.get('service_time', 0)
            current_location = best_customer

        if route:
            routes.append(route)

    solution.routes, solution.num_vehicles_used, solution.is_feasible = routes, len(
        routes), len(unvisited) == 0

    # Calculate totals
    total_distance = total_time = 0
    for route in routes:
        if route:
            route_distance = data.distance_matrix[depot_idx][route[0]
                                                             ] + data.distance_matrix[route[-1]][depot_idx]
            route_time = data.time_matrix[depot_idx][route[0]
                                                     ] + data.time_matrix[route[-1]][depot_idx]
            for i in range(len(route) - 1):
                route_distance += data.distance_matrix[route[i]][route[i + 1]]
                route_time += data.time_matrix[route[i]][route[i + 1]]
            total_distance += route_distance
            total_time += route_time

    solution.total_distance, solution.total_time = total_distance, total_time
    return solution


def get_map_center():
    return [st.session_state.locations['latitude'].mean(), st.session_state.locations['longitude'].mean()] if len(st.session_state.locations) > 0 else [21.0285, 105.8542]


def add_location(result=None, lat=None, lng=None, is_depot=False):
    # Fixed: Use proper boolean check for depot existence
    has_depot = (not st.session_state.locations.empty and
                 st.session_state.locations['is_depot'].any())

    if lat is not None and lng is not None:  # From map click
        is_depot = not has_depot
        name = f"{'Kho' if is_depot else 'Điểm'} {st.session_state.next_id}"
        latitude, longitude = lat, lng
    else:  # From search
        name = result['name'].split(',')[0][:50]
        latitude, longitude = result['latitude'], result['longitude']

    new_location = pd.DataFrame([{
        'id': st.session_state.next_id, 'name': name, 'latitude': latitude, 'longitude': longitude,
        'demand': 0 if is_depot else 10, 'earliest_time': 0 if is_depot else 480,
        'latest_time': 1440 if is_depot else 1020, 'service_time': 0 if is_depot else 15,
        'description': "Kho" if is_depot else "", 'is_depot': is_depot
    }])

    st.session_state.locations = pd.concat(
        [st.session_state.locations, new_location], ignore_index=True)
    st.session_state.next_id += 1

    # Reset ma trận khi thêm địa điểm mới
    st.session_state.distance_matrix = None
    st.session_state.time_matrix = None
    st.session_state.vrptw_solution = None


# Main layout
col1, col2 = st.columns([1, 1])

with col1:

    st.subheader("📁 Import/Export Excel")

    # Export Excel - Tạo file mẫu
    if st.button("📤 Tạo file mẫu Excel", use_container_width=True):
        # Tạo dữ liệu mẫu
        sample_data = pd.DataFrame({
            'Tên địa điểm': ['Kho chính', 'Siêu thị A', 'Cửa hàng B', 'Khách hàng C'],
            'Vĩ độ': [21.0285, 21.0345, 21.0195, 21.0405],
            'Kinh độ': [105.8542, 105.8602, 105.8482, 105.8662],
            'Nhu cầu': [0, 15, 20, 10],
            'Giờ mở': ['00:00', '08:00', '09:00', '08:30'],
            'Giờ đóng': ['23:59', '17:00', '18:00', '16:30'],
            'Thời gian phục vụ': [0, 15, 20, 10],
            'Loại': ['Kho', 'Khách hàng', 'Khách hàng', 'Khách hàng']
        })

    # Export dữ liệu hiện tại
    if not st.session_state.locations.empty:
        if st.button("📤 Xuất dữ liệu hiện tại", use_container_width=True):
            # Chuẩn bị dữ liệu xuất
            export_data = st.session_state.locations.copy()
            export_data['Giờ mở'] = export_data['earliest_time'].apply(
                lambda x: f"{x//60:02d}:{x % 60:02d}")
            export_data['Giờ đóng'] = export_data['latest_time'].apply(
                lambda x: f"{x//60:02d}:{x % 60:02d}")
            export_data['Loại'] = export_data['is_depot'].apply(
                lambda x: 'Kho' if x else 'Khách hàng')

            # Chọn và đổi tên cột
            export_columns = {
                'name': 'Tên địa điểm',
                'latitude': 'Vĩ độ',
                'longitude': 'Kinh độ',
                'demand': 'Nhu cầu',
                'Giờ mở': 'Giờ mở',
                'Giờ đóng': 'Giờ đóng',
                'service_time': 'Thời gian phục vụ',
                'Loại': 'Loại'
            }

            final_data = export_data[list(export_columns.keys())].rename(
                columns=export_columns)

            buffer = io.BytesIO()
            final_data.to_excel(buffer, sheet_name='Khách hàng', index=False)
            buffer.seek(0)

            st.download_button(
                label="💾 Tải xuống Excel",
                data=buffer.getvalue(),
                file_name=f"khachhang_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    
    uploaded_file = st.file_uploader(
        "📥 Tải lên file Excel", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        # Kiểm tra xem file đã được xử lý chưa
        file_key = f"processed_file_{uploaded_file.name}_{uploaded_file.size}"
        
        if file_key not in st.session_state:
            try:
                df = pd.read_excel(uploaded_file)

                # Kiểm tra cột bắt buộc
                required_cols = ['Tên địa điểm', 'Vĩ độ', 'Kinh độ', 'Loại']
                missing_cols = [
                    col for col in required_cols if col not in df.columns]

                if missing_cols:
                    st.error(f"❌ Thiếu cột: {', '.join(missing_cols)}")
                    st.session_state[file_key] = "error"
                else:
                    # Xử lý dữ liệu
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

                    # Tạo DataFrame mới
                    new_locations = pd.DataFrame({
                        'id': range(1, len(df) + 1),
                        'name': df['Tên địa điểm'],
                        'latitude': df['Vĩ độ'],
                        'longitude': df['Kinh độ'],
                        'demand': df.get('Nhu cầu', 10).fillna(10),
                        'earliest_time': df.get('Giờ mở', '08:00').apply(time_to_minutes),
                        'latest_time': df.get('Giờ đóng', '17:00').apply(time_to_minutes),
                        'service_time': df.get('Thời gian phục vụ', 15).fillna(15),
                        'description': df.get('Mô tả', ''),
                        'is_depot': df['Loại'].str.lower().str.contains('kho|depot', na=False)
                    })

                    # Kiểm tra có depot không
                    if not new_locations['is_depot'].any():
                        st.error("❌ File phải có ít nhất 1 kho (Loại = 'Kho')")
                        st.session_state[file_key] = "error"
                    else:
                        # Reset index để đảm bảo liên tục từ 0
                        new_locations = new_locations.reset_index(drop=True)

                        # Cập nhật session state
                        st.session_state.locations = new_locations
                        st.session_state.next_id = len(new_locations) + 1

                        # Reset các ma trận
                        st.session_state.distance_matrix = None
                        st.session_state.time_matrix = None
                        st.session_state.vrptw_solution = None

                        # Đánh dấu file đã được xử lý thành công
                        st.session_state[file_key] = "success"
                        
                        st.success(f"✅ Đã import {len(new_locations)} địa điểm!")

            except Exception as e:
                st.error(f"❌ Lỗi đọc file: {str(e)}")
                st.info("💡 Hãy tải file mẫu để xem định dạng đúng")
                st.session_state[file_key] = "error"
        
        else:
            # File đã được xử lý trước đó
            if st.session_state[file_key] == "success":
                st.success(f"✅ File đã được import thành công!")
            elif st.session_state[file_key] == "error":
                st.error("❌ File này đã có lỗi khi xử lý")
        
        # Thêm nút để reset file upload
        if st.button("🔄 Tải file mới", help="Click để có thể tải file khác"):
            # Xóa tất cả các key liên quan đến file đã xử lý
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("processed_file_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.rerun()

    # Search
    st.subheader("🔍 Thêm địa điểm")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input(
            "Tìm kiếm:", placeholder="VD: Siêu thị Big C Hà Nội")
    with search_col2:
        if st.button("Tìm kiếm", use_container_width=True):
            if search_query:
                with st.spinner("Đang tìm kiếm..."):
                    st.session_state.search_results = search_location(
                        search_query)
                    if st.session_state.search_results:
                        st.success(
                            f"Tìm thấy {len(st.session_state.search_results)} kết quả")
                    else:
                        st.warning("Không tìm thấy kết quả")

    # Display search results
    if st.session_state.search_results:
        st.write("**Kết quả tìm kiếm:**")
        for i, result in enumerate(st.session_state.search_results):
            col_info, col_depot, col_customer = st.columns([3, 1, 1])
            with col_info:
                st.write(f"📍 {result['name'][:60]}...")
            with col_depot:
                if st.button("Kho", key=f"depot_{i}", use_container_width=True):
                    add_location(result, is_depot=True)
                    st.session_state.search_results = []
                    st.success("Đã thêm kho!")
                    st.rerun()
            with col_customer:
                if st.button("KH", key=f"customer_{i}", use_container_width=True):
                    add_location(result, is_depot=False)
                    st.session_state.search_results = []
                    st.success("Đã thêm khách hàng!")
                    st.rerun()

        if st.button("❌ Xóa kết quả"):
            st.session_state.search_results = []
            st.rerun()

    # Vehicle config
    st.subheader("🚛 Cấu hình xe")
    vehicle_capacity = st.number_input(
        "Tải trọng xe:", min_value=1, max_value=1000, value=100)
    max_time = st.number_input(
        "Thời gian làm việc tối đa (phút):", min_value=60, max_value=720, value=480)

    # Location management
    st.subheader("📍 Danh sách địa điểm")

    if not st.session_state.locations.empty:
        display_locations = st.session_state.locations.copy()
        display_locations['type'] = display_locations['is_depot'].apply(
            lambda x: "🏭 Kho" if x else "👥 Khách hàng")
        display_locations['earliest_time_str'] = display_locations['earliest_time'].apply(
            lambda x: f"{x//60:02d}:{x % 60:02d}")
        display_locations['latest_time_str'] = display_locations['latest_time'].apply(
            lambda x: f"{x//60:02d}:{x % 60:02d}")

        edited_locations = st.data_editor(
            display_locations[['type', 'name', 'demand', 'earliest_time_str',
                               'latest_time_str', 'service_time', 'description']],
            column_config={
                "type": st.column_config.TextColumn("Loại", disabled=True, width="small"),
                "name": st.column_config.TextColumn("Tên địa điểm", width="medium"),
                "demand": st.column_config.NumberColumn("Nhu cầu", min_value=0, max_value=100, width="small"),
                "earliest_time_str": st.column_config.TextColumn("Thời gian sớm nhất", width="small"),
                "latest_time_str": st.column_config.TextColumn("Thời gian muộn nhất", width="small"),
                "service_time": st.column_config.NumberColumn("Thời gian phục vụ (phút)", min_value=0, width="small"),
                "description": st.column_config.TextColumn("Mô tả")
            },
            hide_index=True, use_container_width=True, num_rows="dynamic"
        )

        # Update from editor
        for i, row in edited_locations.iterrows():
            if i < len(st.session_state.locations):
                try:
                    earliest_parts = row['earliest_time_str'].split(':')
                    latest_parts = row['latest_time_str'].split(':')
                    earliest_minutes = int(
                        earliest_parts[0]) * 60 + int(earliest_parts[1])
                    latest_minutes = int(
                        latest_parts[0]) * 60 + int(latest_parts[1])

                    for col, val in [('name', row['name']), ('demand', row['demand']),
                                     ('earliest_time',
                                      earliest_minutes), ('latest_time', latest_minutes),
                                     ('service_time', row['service_time']), ('description', row['description'])]:
                        st.session_state.locations.at[i, col] = val
                except:
                    pass

        if st.button("🗑️ Xóa tất cả địa điểm"):
            for key in ['locations', 'distance_matrix', 'time_matrix', 'vrptw_solution']:
                st.session_state[key] = defaults[key] if key == 'locations' else None
            st.rerun()
    else:
        st.info("Chưa có địa điểm nào. Hãy tìm kiếm hoặc click vào bản đồ để thêm.")

    # Algorithm
    st.subheader("🧮 Thuật toán VRPTW")
    depot_count = st.session_state.locations['is_depot'].sum(
    ) if not st.session_state.locations.empty else 0
    customer_count = len(st.session_state.locations) - \
        depot_count if not st.session_state.locations.empty else 0

    if depot_count > 0 and customer_count > 0:
        if st.button("📊 Tính ma trận khoảng cách", type="secondary", use_container_width=True):
            # Kiểm tra có depot không
            if not st.session_state.locations['is_depot'].any():
                st.error("❌ Không tìm thấy kho trong dữ liệu!")
            else:
                with st.spinner("Đang tính toán..."):
                    locations = [{'latitude': row['latitude'], 'longitude': row['longitude']}
                                 for _, row in st.session_state.locations.iterrows()]
                    distance_matrix, time_matrix, error = calculate_distance_matrix_api(
                        locations)
                    

                    if error:
                        st.error(f"Lỗi: {error}")
                    else:
                        st.session_state.distance_matrix, st.session_state.time_matrix = distance_matrix, time_matrix
                        st.success("✅ Đã tính xong ma trận!")
                        # Debug info
                        st.write(
                            f"Kích thước ma trận: {distance_matrix.shape}")

        if st.session_state.distance_matrix is not None:
            print(st.session_state.distance_matrix)
            st.success("✅ Ma trận đã sẵn sàng")

            # Thêm thông tin debug
            st.write(
                f"🔍 Debug: Số địa điểm = {len(st.session_state.locations)}, Kích thước ma trận = {st.session_state.distance_matrix.shape}")
            if st.button("🚀 Giải VRPTW", type="primary", use_container_width=True):
      
                # Kiểm tra depot tồn tại
                depot_mask = st.session_state.locations['is_depot']
                if not depot_mask.any():
                    st.error("❌ Không tìm thấy kho!")
                else:
                    with st.spinner("Đang giải..."):
                        vrptw_data = VRPTWData()
                        vrptw_data.locations = st.session_state.locations.to_dict(
                            'records')

                        # Tìm vị trí đầu tiên có is_depot = True trong DataFrame
                        depot_row_index = st.session_state.locations[depot_mask].index[0]
                        # Đây là index trong DataFrame (0-based)
                        vrptw_data.depot_idx = depot_row_index

                        # Debug info
                        st.write(
                            f"🔍 Debug: Depot index = {vrptw_data.depot_idx}")
                        st.write(
                            f"🔍 Debug: Depot name = {st.session_state.locations.iloc[vrptw_data.depot_idx]['name']}")

                        vrptw_data.distance_matrix, vrptw_data.time_matrix = st.session_state.distance_matrix, st.session_state.time_matrix
                        vrptw_data.vehicle_capacity, vrptw_data.max_time = vehicle_capacity, max_time

                        st.session_state.vrptw_solution = simple_vrptw_solver(
                            vrptw_data)

                        if st.session_state.vrptw_solution.is_feasible:
                            st.success(
                                f"✅ Thành công! Sử dụng {st.session_state.vrptw_solution.num_vehicles_used} xe")
                        else:
                            st.error("❌ Không tìm được giải pháp")
                            # Thêm debug info
                            st.write(
                                f"🔍 Debug: Số route = {len(st.session_state.vrptw_solution.routes)}")
    else:
        if depot_count == 0:
            st.warning("⚠️ Cần có ít nhất 1 kho")
        if customer_count == 0:
            st.warning("⚠️ Cần có ít nhất 1 khách hàng")

    # Display solution
    if st.session_state.vrptw_solution:
        solution = st.session_state.vrptw_solution
        print(solution.routes)

        st.subheader("📊 Kết quả VRPTW")

        col1_result, col2_result, col3_result = st.columns(3)
        with col1_result:
            st.metric("Số xe", solution.num_vehicles_used)
        with col2_result:
            st.metric("Tổng khoảng cách", f"{solution.total_distance:.2f} km")
        with col3_result:
            st.metric("Tổng thời gian", f"{solution.total_time:.1f} phút")

        if solution.routes:
            st.write("**Chi tiết lộ trình:**")
            depot_name = st.session_state.locations[st.session_state.locations['is_depot']]['name'].iloc[0]

            for i, route in enumerate(solution.routes):
                if route:
                    st.write(f"🚛 **Xe {i+1}:**")
                    route_text = f"{depot_name} → " + " → ".join(
                        [st.session_state.locations.iloc[idx]['name'] for idx in route]) + f" → {depot_name}"
                    total_demand = sum(
                        st.session_state.locations.iloc[idx]['demand'] for idx in route)
                    st.write(f"   {route_text}")
                    st.write(
                        f"   📦 Tổng tải: {total_demand}/{vehicle_capacity}")

with col2:
    st.subheader("🗺️ Bản đồ")
    st.info("💡 Click vào bản đồ để thêm địa điểm")

    # Create map
    m = folium.Map(location=get_map_center(), zoom_start=12)

    # Add markers
    if not st.session_state.locations.empty:
        for idx, location in st.session_state.locations.iterrows():
            earliest_str = f"{location['earliest_time']//60:02d}:{location['earliest_time'] % 60:02d}"
            latest_str = f"{location['latest_time']//60:02d}:{location['latest_time'] % 60:02d}"

            folium.Marker(
                [location['latitude'], location['longitude']],
                tooltip=f"{location['name']} ({'Kho' if location['is_depot'] else f'Demand: {location['demand']}'})",
                popup=f"<b>{location['name']}</b><br>Loại: {'Kho' if location['is_depot'] else 'Khách hàng'}<br>Nhu cầu: {location['demand']}<br>Thời gian: {earliest_str} - {latest_str}<br>Phục vụ: {location['service_time']} phút",
                icon=folium.Icon(color='red' if location['is_depot'] else 'blue',
                                 icon='home' if location['is_depot'] else 'user', prefix='fa')
            ).add_to(m)

    # Draw routes
    if st.session_state.vrptw_solution and st.session_state.vrptw_solution.routes:
        colors = ['red', 'green', 'blue', 'purple', 'orange',
                  'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
        depot_idx = st.session_state.locations[st.session_state.locations['is_depot']].index[0]
        depot_location = st.session_state.locations.iloc[depot_idx]

        for i, route in enumerate(st.session_state.vrptw_solution.routes):
            if route:
                color = colors[i % len(colors)]
                route_coords = [
                    {'latitude': depot_location['latitude'], 'longitude': depot_location['longitude']}]

                for location_idx in route:
                    location = st.session_state.locations.iloc[location_idx]
                    route_coords.append(
                        {'latitude': location['latitude'], 'longitude': location['longitude']})
                route_coords.append(
                    {'latitude': depot_location['latitude'], 'longitude': depot_location['longitude']})

                # Draw route segments
                for j in range(len(route_coords) - 1):
                    start_coord = [route_coords[j]['latitude'],
                                   route_coords[j]['longitude']]
                    end_coord = [route_coords[j+1]['latitude'],
                                 route_coords[j+1]['longitude']]

                    geometry = get_route_geometry(start_coord, end_coord)
                    
                    
                    folium.PolyLine(
                        locations=geometry if geometry else [
                            start_coord, end_coord],
                        weight=4, color=color, opacity=0.8, popup=f"Xe {i+1} - Đoạn {j+1}"
                    ).add_to(m)

                # Add route numbers
                for j, location_idx in enumerate(route):
                    location = st.session_state.locations.iloc[location_idx]
                    folium.Marker(
                        [location['latitude'], location['longitude']],
                        icon=folium.DivIcon(
                            html=f'<div style="background-color: {color}; border: 2px solid white; border-radius: 50%; width: 25px; height: 25px; text-align: center; line-height: 21px; font-weight: bold; font-size: 12px; color: white;">{j+1}</div>',
                            icon_size=(25, 25), icon_anchor=(12, 12)
                        )
                    ).add_to(m)

    # Handle map clicks
    map_result = st_folium(m, width=700, height=500,
                           returned_objects=["last_clicked"])

    if map_result.get('last_clicked'):
        clicked_lat, clicked_lng = map_result['last_clicked']['lat'], map_result['last_clicked']['lng']
        add_location(lat=clicked_lat, lng=clicked_lng)
        st.success("✅ Đã thêm địa điểm mới!")
        st.rerun()
