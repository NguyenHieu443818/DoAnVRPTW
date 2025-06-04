import streamlit as st
import pandas as pd
import folium
import requests
from streamlit_folium import st_folium
import numpy as np
import io

# --- Config ---
st.set_page_config(page_title="VRPTW Solver", page_icon="🚛", layout="wide")
st.title("VRPTW Solver - Vehicle Routing Problem with Time Windows")

BACKEND_URL = "http://127.0.0.1:8000"

# --- Session state init ---
defaults = {
    'locations': pd.DataFrame(columns=['id', 'name', 'latitude', 'longitude', 'demand', 'earliest_time', 'latest_time', 'service_time', 'description', 'is_depot']),
    'distance_matrix': None,
    'time_matrix': None,
    'vrptw_solution': None,
    'next_id': 1,
    'search_results': [],
    'selected_row_index': None, 
    'last_map_click_for_update': None 
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

class VRPTWSolution:
    def __init__(self, routes=None, total_distance=0, total_time=0, num_vehicles_used=0, is_feasible=True):
        self.routes = routes if routes is not None else []
        self.total_distance = total_distance
        self.total_time = total_time
        self.num_vehicles_used = num_vehicles_used
        self.is_feasible = is_feasible

def search_location_api(query):
    try:
        response = requests.get(f"{BACKEND_URL}/search_location/", params={'query': query}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối khi tìm kiếm: {e}")
        return []
    except Exception as e:
        st.error(f"Lỗi khi tìm kiếm: {e}")
        return []

def calculate_matrices_api(locations_df):
    if len(locations_df) < 2:
        return None, None, "Cần ít nhất 2 địa điểm"
    payload = locations_df[['latitude', 'longitude']].to_dict('records')
    try:
        response = requests.post(f"{BACKEND_URL}/calculate_matrices/", json=payload, timeout=45)
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            return None, None, data["error"]
        dist_matrix = np.array(data.get("distances", [])) if data.get("distances") else None
        time_matrix = np.array(data.get("durations", [])) if data.get("durations") else None
        if dist_matrix is None or time_matrix is None or dist_matrix.size == 0 or time_matrix.size == 0:
             return None, None, "Không nhận được ma trận hợp lệ từ backend."
        return dist_matrix, time_matrix, None
    except requests.exceptions.RequestException as e:
        return None, None, f"Lỗi kết nối khi tính ma trận: {e}"
    except Exception as e:
        return None, None, f"Lỗi khi tính ma trận: {e}"

def get_route_geometry_api(start_coord, end_coord):
    payload = {"start_coord": start_coord, "end_coord": end_coord}
    try:
        response = requests.post(f"{BACKEND_URL}/route_geometry/", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None
    except Exception:
        return None

def solve_vrptw_api(locations_df, depot_idx, dist_matrix, time_matrix, vehicle_capacity):
    payload = {
        "locations": locations_df.to_dict('records'),
        "depot_idx": depot_idx,
        "distance_matrix": dist_matrix.tolist() if dist_matrix is not None else None,
        "time_matrix": time_matrix.tolist() if time_matrix is not None else None,
        "vehicle_capacity": vehicle_capacity
    }
    try:
        response = requests.post(f"{BACKEND_URL}/solve_vrptw/", json=payload, timeout=60)
        response.raise_for_status()
        solution_data = response.json()
        return VRPTWSolution(**solution_data)
    except requests.exceptions.HTTPError as e:
        st.error(f"Lỗi từ backend khi giải VRPTW: {e.response.status_code} - {e.response.text}")
        return VRPTWSolution(is_feasible=False)
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối khi giải VRPTW: {e}")
        return VRPTWSolution(is_feasible=False)
    except Exception as e:
        st.error(f"Lỗi khi giải VRPTW: {e}")
        return VRPTWSolution(is_feasible=False)

def parse_excel_api(uploaded_file_obj):
    files = {'file': (uploaded_file_obj.name, uploaded_file_obj.getvalue(), uploaded_file_obj.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/parse_excel/", files=files, timeout=20)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"Lỗi từ backend khi xử lý Excel: {e.response.status_code} - {e.response.text}")
        return {"error": f"Lỗi backend: {e.response.status_code}"}
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi kết nối khi xử lý Excel: {e}")
        return {"error": f"Lỗi kết nối: {e}"}
    except Exception as e:
        st.error(f"Lỗi khi xử lý Excel: {e}")
        return {"error": f"Lỗi không xác định: {e}"}

def get_map_center():
    if not st.session_state.locations.empty:
        return [st.session_state.locations['latitude'].mean(), st.session_state.locations['longitude'].mean()]
    return [21.0285, 105.8542]

def add_location(result=None, lat=None, lng=None, is_depot_param=False, update_existing_idx=None): # <<< THÊM update_existing_idx
    if update_existing_idx is not None and lat is not None and lng is not None:
        # --- CẬP NHẬT ĐIỂM HIỆN TẠI ---
        if 0 <= update_existing_idx < len(st.session_state.locations):
            st.session_state.locations.at[update_existing_idx, 'latitude'] = lat
            st.session_state.locations.at[update_existing_idx, 'longitude'] = lng
            
            # Reset ma trận và giải pháp
            st.session_state.distance_matrix = None
            st.session_state.time_matrix = None
            st.session_state.vrptw_solution = None
            st.success(f"Đã cập nhật vị trí cho điểm: {st.session_state.locations.at[update_existing_idx, 'name']}")
            st.session_state.selected_row_index = None # Bỏ chọn sau khi cập nhật
            st.session_state.last_map_click_for_update = None # Reset cờ
            st.rerun() # Quan trọng: rerun để selectbox và các thành phần khác cập nhật
        else:
            st.error("Lỗi: Index điểm cần cập nhật không hợp lệ.")
        return # Kết thúc sớm nếu là cập nhật

    # --- THÊM ĐIỂM MỚI (logic cũ) ---
    has_depot = (not st.session_state.locations.empty and
                 st.session_state.locations['is_depot'].any())

    if lat is not None and lng is not None:
        is_depot_val = not has_depot if not is_depot_param else is_depot_param
        name = f"{'Kho' if is_depot_val else 'Điểm'} {st.session_state.next_id}"
        latitude, longitude = lat, lng
    elif result:
        is_depot_val = is_depot_param
        name = result['name'].split(',')[0][:50]
        latitude, longitude = result['latitude'], result['longitude']
    else:
        st.error("Không có thông tin để thêm địa điểm.")
        return

    new_location_data = {
        'id': st.session_state.next_id, 'name': name, 'latitude': latitude, 'longitude': longitude,
        'demand': 0 if is_depot_val else 10, 'earliest_time': 0 if is_depot_val else 480,
        'latest_time': 1439 if is_depot_val else 1020,
        'service_time': 0 if is_depot_val else 15,
        'description': "Kho" if is_depot_val else "", 'is_depot': is_depot_val
    }
    new_location_df = pd.DataFrame([new_location_data])

    st.session_state.locations = pd.concat(
        [st.session_state.locations, new_location_df], ignore_index=True).reset_index(drop=True) # reset_index quan trọng
    st.session_state.next_id += 1

    st.session_state.distance_matrix = None
    st.session_state.time_matrix = None
    st.session_state.vrptw_solution = None
    st.success(f"Đã thêm địa điểm mới: {name}")
    st.rerun()


# --- Main layout ---
col1, col2 = st.columns([1, 1])

with col1:
    # ... (Phần Import/Export Excel giữ nguyên) ...
    st.subheader("Import/Export Excel")

    if not st.session_state.locations.empty:
        if st.button("Xuất dữ liệu hiện tại", use_container_width=True):
            export_data = st.session_state.locations.copy()
            export_data['Giờ mở'] = export_data['earliest_time'].apply(lambda x: f"{int(x)//60:02d}:{int(x) % 60:02d}")
            export_data['Giờ đóng'] = export_data['latest_time'].apply(lambda x: f"{int(x)//60:02d}:{int(x) % 60:02d}")
            export_data['Loại'] = export_data['is_depot'].apply(lambda x: 'Kho' if x else 'Khách hàng')
            export_columns_map = {
                'name': 'Tên địa điểm', 'latitude': 'Vĩ độ', 'longitude': 'Kinh độ',
                'demand': 'Nhu cầu', 'Giờ mở': 'Giờ mở', 'Giờ đóng': 'Giờ đóng',
                'service_time': 'Thời gian phục vụ', 'description': 'Mô tả', 'Loại': 'Loại'
            }
            final_export_cols = [col for col in export_columns_map.keys() if col in export_data.columns]
            final_data = export_data[final_export_cols].rename(columns=export_columns_map)
            buffer = io.BytesIO()
            final_data.to_excel(buffer, sheet_name='Khách hàng', index=False)
            buffer.seek(0)
            st.download_button(
                label="Tải xuống Excel hiện tại",
                data=buffer.getvalue(),
                file_name=f"VRPTW_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    uploaded_file = st.file_uploader("Tải lên file Excel", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        file_key = f"processed_file_{uploaded_file.name}_{uploaded_file.size}"
        if file_key not in st.session_state:
            with st.spinner("Đang xử lý file Excel..."):
                parsed_result = parse_excel_api(uploaded_file)
            if parsed_result.get("error"):
                st.error(f"Lỗi xử lý file: {parsed_result['error']}")
                st.info("Hãy tải file mẫu để xem định dạng đúng.")
                st.session_state[file_key] = "error"
            elif parsed_result.get("locations"):
                imported_locations = pd.DataFrame(parsed_result["locations"])
                for col in defaults['locations'].columns:
                    if col not in imported_locations.columns:
                        if col in ['demand', 'earliest_time', 'latest_time', 'service_time', 'id']:
                             imported_locations[col] = 0
                        elif col == 'is_depot':
                             imported_locations[col] = False
                        else:
                             imported_locations[col] = ''
                st.session_state.locations = imported_locations.reset_index(drop=True)
                st.session_state.next_id = (st.session_state.locations['id'].max() + 1) if not st.session_state.locations.empty else 1
                st.session_state.distance_matrix = None
                st.session_state.time_matrix = None
                st.session_state.vrptw_solution = None
                st.session_state[file_key] = "success"
                st.success(parsed_result.get("message", f"Đã import {len(imported_locations)} địa điểm!"))
            else:
                st.error(" Không nhận được dữ liệu địa điểm từ backend.")
                st.session_state[file_key] = "error"
        else:
            if st.session_state[file_key] == "success":
                st.success(f"File đã được import thành công trước đó!")
            elif st.session_state[file_key] == "error":
                st.error(" File này đã có lỗi khi xử lý trước đó.")
        if st.button("🔄 Tải file mới", help="Click để có thể tải file khác"):
            keys_to_remove = [key for key in st.session_state.keys() if key.startswith("processed_file_")]
            for key in keys_to_remove:
                del st.session_state[key]
            st.rerun()

    st.subheader("Thêm địa điểm")
    search_col1, search_col2 = st.columns([3, 1])
    with search_col1:
        search_query = st.text_input("Tìm kiếm:", placeholder="VD: Siêu thị Big C Hà Nội")
    with search_col2:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        if st.button("Tìm", use_container_width=True):
            if search_query:
                with st.spinner("Đang tìm kiếm..."):
                    st.session_state.search_results = search_location_api(search_query)
                if st.session_state.search_results:
                    st.success(f"Tìm thấy {len(st.session_state.search_results)} kết quả.")
                else:
                    st.warning("Không tìm thấy kết quả.")
            else:
                st.warning("Vui lòng nhập địa điểm cần tìm.")

    if st.session_state.search_results:
        st.write("**Kết quả tìm kiếm:**")
        for i, result in enumerate(st.session_state.search_results):
            col_info, col_depot, col_customer = st.columns([3, 1, 1])
            with col_info:
                st.write(f"{result['name'][:60]}...")
            with col_depot:
                if st.button("Kho", key=f"depot_{i}", use_container_width=True):
                    add_location(result=result, is_depot_param=True) # Không cập nhật, chỉ thêm mới
                    st.session_state.search_results = [] # Xóa sau khi thêm
                    st.rerun() # add_location đã có rerun
            with col_customer:
                if st.button("KH", key=f"customer_{i}", use_container_width=True):
                    add_location(result=result, is_depot_param=False) # Không cập nhật, chỉ thêm mới
                    st.session_state.search_results = [] # Xóa sau khi thêm
                    st.rerun() # add_location đã có rerun
        if st.button(" Xóa kết quả tìm kiếm"):
            st.session_state.search_results = []
            st.rerun()


    st.subheader("Cấu hình xe")
    vehicle_capacity = st.number_input("Tải trọng xe:", min_value=1, value=100, step=10)

    st.subheader("Danh sách địa điểm")

    # --- BẮT ĐẦU TÍCH HỢP LOGIC CHỌN VÀ SỬA ---
    if not st.session_state.locations.empty:
        row_options = ["-- Chọn điểm để sửa/xóa --"] + [
            f"{idx}:{row['id']}:{row['name']}" for idx, row in st.session_state.locations.iterrows()
        ] # Thêm DataFrame index (idx) vào option để dễ lấy selected_row_index

        # Lấy index của selectbox dựa trên st.session_state.selected_row_index
        current_selection_str = None
        if st.session_state.selected_row_index is not None:
             # Cần tìm lại option string dựa trên selected_row_index
            try:
                selected_loc_data = st.session_state.locations.iloc[st.session_state.selected_row_index]
                current_selection_str = f"{st.session_state.selected_row_index}:{selected_loc_data['id']}:{selected_loc_data['name']}"
            except IndexError: # Nếu index không còn hợp lệ (ví dụ sau khi xóa)
                st.session_state.selected_row_index = None


        selected_option_value = st.selectbox(
            "Chọn điểm để cập nhật vị trí hoặc xóa:",
            options=row_options,
            index=row_options.index(current_selection_str) if current_selection_str and current_selection_str in row_options else 0,
            key="selectbox_location_edit"
        )

        if selected_option_value != "-- Chọn điểm để sửa/xóa --":
            try:
                # Option format: "df_index:id:name"
                selected_df_idx = int(selected_option_value.split(":")[0])
                if 0 <= selected_df_idx < len(st.session_state.locations):
                     st.session_state.selected_row_index = selected_df_idx
                else: # Lựa chọn không còn hợp lệ (ví dụ sau khi xóa hàng và selectbox chưa kịp cập nhật)
                    st.session_state.selected_row_index = None
                    # st.warning("Lựa chọn không còn hợp lệ, vui lòng chọn lại.") # Có thể gây rerun loop
            except (ValueError, IndexError):
                st.session_state.selected_row_index = None # Lỗi parse thì bỏ chọn
        else:
            st.session_state.selected_row_index = None

        # Các nút hành động khi một điểm được chọn
        if st.session_state.selected_row_index is not None:
            idx_to_edit = st.session_state.selected_row_index
            try:
                selected_point_name = st.session_state.locations.iloc[idx_to_edit]['name']
                st.markdown(f"Đang chọn: **{selected_point_name}** (ID: {st.session_state.locations.iloc[idx_to_edit]['id']})")

                col_update_map, col_delete_point = st.columns(2)
                with col_update_map:
                    st.info("Click vào bản đồ để cập nhật Vĩ độ/Kinh độ cho điểm đã chọn.")
                with col_delete_point:
                    if st.button(f"🗑️ Xóa điểm '{selected_point_name}'", type="primary", use_container_width=True, key=f"delete_loc_{idx_to_edit}"):
                        st.session_state.locations = st.session_state.locations.drop(idx_to_edit).reset_index(drop=True)
                        st.session_state.selected_row_index = None # Bỏ chọn sau khi xóa
                        # Reset ma trận và giải pháp
                        st.session_state.distance_matrix = None
                        st.session_state.time_matrix = None
                        st.session_state.vrptw_solution = None
                        st.success(f"Đã xóa điểm: {selected_point_name}")
                        st.rerun()
            except IndexError:
                st.warning("Điểm đã chọn không còn tồn tại. Vui lòng chọn lại.")
                st.session_state.selected_row_index = None # Reset nếu có lỗi
                st.rerun() # Rerun để selectbox cập nhật
        else:
            st.info("Click vào bản đồ để thêm điểm mới. Chọn một điểm từ danh sách trên để sửa vị trí hoặc xóa.")
        
        st.markdown("---") # Phân cách

    # --- KẾT THÚC TÍCH HỢP LOGIC CHỌN VÀ SỬA ---


    # Hiển thị data_editor (logic cũ vẫn giữ nguyên để sửa các trường khác)
    if not st.session_state.locations.empty:
        display_locations = st.session_state.locations.copy()
        display_locations['type'] = display_locations['is_depot'].apply(lambda x: "🏭 Kho" if x else "👥 Khách hàng")
        display_locations['earliest_time_str'] = display_locations['earliest_time'].apply(lambda x: f"{int(x)//60:02d}:{int(x) % 60:02d}")
        display_locations['latest_time_str'] = display_locations['latest_time'].apply(lambda x: f"{int(x)//60:02d}:{int(x) % 60:02d}")

        # Đánh dấu hàng được chọn trong data_editor (tùy chọn, có thể phức tạp hóa)
        # Hiện tại data_editor không hỗ trợ trực tiếp việc highlight hàng dễ dàng

        edited_df_from_editor = st.data_editor(
            display_locations[['id','type', 'name', 'demand', 'earliest_time_str', 'latest_time_str', 'service_time', 'description']],
            column_config={
                "id": st.column_config.NumberColumn("ID", disabled=True, width="small"),
                "type": st.column_config.TextColumn("Loại", disabled=True, width="small"),
                "name": st.column_config.TextColumn("Tên địa điểm", width="medium"),
                "demand": st.column_config.NumberColumn("Nhu cầu", min_value=0, width="small"),
                "earliest_time_str": st.column_config.TextColumn("Sớm nhất", width="small"),
                "latest_time_str": st.column_config.TextColumn("Muộn nhất", width="small"),
                "service_time": st.column_config.NumberColumn("TG Phục vụ (phút)", min_value=0, width="small"),
                "description": st.column_config.TextColumn("Mô tả")
            },
            hide_index=True, use_container_width=True, num_rows="fixed", # "fixed" để tránh thêm/xóa hàng qua data_editor nếu dùng selectbox
            key="location_editor_main"
        )

        if edited_df_from_editor is not None:
            edited_rows_map = {row['id']: row for _, row in edited_df_from_editor.iterrows()}
            changed_by_editor = False
            for i, original_loc in st.session_state.locations.iterrows(): # Lặp qua st.session_state.locations gốc
                loc_id = original_loc['id']
                if loc_id in edited_rows_map:
                    edited_row = edited_rows_map[loc_id]
                    try:
                        if st.session_state.locations.at[i, 'name'] != edited_row['name']:
                            st.session_state.locations.at[i, 'name'] = edited_row['name']
                            changed_by_editor = True
                        if st.session_state.locations.at[i, 'demand'] != edited_row['demand']:
                            st.session_state.locations.at[i, 'demand'] = edited_row['demand']
                            changed_by_editor = True
                        if st.session_state.locations.at[i, 'service_time'] != edited_row['service_time']:
                            st.session_state.locations.at[i, 'service_time'] = edited_row['service_time']
                            changed_by_editor = True
                        if st.session_state.locations.at[i, 'description'] != edited_row['description']:
                            st.session_state.locations.at[i, 'description'] = edited_row['description']
                            changed_by_editor = True
                        
                        early_h, early_m = map(int, edited_row['earliest_time_str'].split(':'))
                        earliest_minutes = early_h * 60 + early_m
                        if st.session_state.locations.at[i, 'earliest_time'] != earliest_minutes:
                            st.session_state.locations.at[i, 'earliest_time'] = earliest_minutes
                            changed_by_editor = True

                        late_h, late_m = map(int, edited_row['latest_time_str'].split(':'))
                        latest_minutes = late_h * 60 + late_m
                        if st.session_state.locations.at[i, 'latest_time'] != latest_minutes:
                            st.session_state.locations.at[i, 'latest_time'] = latest_minutes
                            changed_by_editor = True
                            
                    except ValueError:
                        st.warning(f"Lỗi định dạng thời gian cho địa điểm ID {loc_id} khi sửa bằng bảng. Sử dụng HH:MM.")
                    except Exception:
                        pass
            
            if changed_by_editor:
                st.session_state.distance_matrix = None
                st.session_state.time_matrix = None
                st.session_state.vrptw_solution = None
                st.success("Đã cập nhật thông tin từ bảng chỉnh sửa.")
                st.rerun()

        if st.button("🗑️ Xóa tất cả địa điểm", use_container_width=True, key="delete_all_locations"):
            st.session_state.locations = pd.DataFrame(columns=defaults['locations'].columns)
            st.session_state.next_id = 1
            st.session_state.selected_row_index = None # Reset lựa chọn
            st.session_state.distance_matrix = None
            st.session_state.time_matrix = None
            st.session_state.vrptw_solution = None
            st.session_state.search_results = []
            st.rerun()
    else:
        st.info("Chưa có địa điểm nào. Hãy tìm kiếm, tải file Excel hoặc click vào bản đồ để thêm.")

    st.subheader("Thuật toán VRPTW")
    depot_count = st.session_state.locations['is_depot'].sum() if not st.session_state.locations.empty else 0
    customer_count = len(st.session_state.locations) - depot_count

    if depot_count > 0 and customer_count > 0:
        if st.button("Tính ma trận khoảng cách & thời gian", type="secondary", use_container_width=True):
            if st.session_state.locations['is_depot'].any():
                with st.spinner("Đang tính toán ma trận... (có thể mất vài giây)"):
                    dist_m, time_m, error = calculate_matrices_api(st.session_state.locations)
                if error:
                    st.error(f"Lỗi khi tính ma trận: {error}")
                elif dist_m is not None and time_m is not None:
                    st.session_state.distance_matrix = dist_m
                    st.session_state.time_matrix = time_m
                    st.success("Đã tính xong ma trận!")
                else:
                    st.error(" Không thể tính ma trận. Kiểm tra lại địa điểm hoặc kết nối backend.")
            else:
                st.error(" Cần có ít nhất 1 kho để tính ma trận.")

        if st.session_state.distance_matrix is not None and st.session_state.time_matrix is not None:
            st.success(f"Ma trận đã sẵn sàng (Kích thước: {st.session_state.distance_matrix.shape})")
            if st.button("Giải VRPTW", type="primary", use_container_width=True):
                depot_mask = st.session_state.locations['is_depot']
                if not depot_mask.any():
                    st.error(" Không tìm thấy kho!")
                else:
                    depot_df_idx = st.session_state.locations[depot_mask].index[0]
                    with st.spinner("Đang giải VRPTW..."):
                        solution = solve_vrptw_api(
                            st.session_state.locations,
                            int(depot_df_idx),
                            st.session_state.distance_matrix,
                            st.session_state.time_matrix,
                            vehicle_capacity
                        )
                    st.session_state.vrptw_solution = solution
                    if solution and solution.is_feasible:
                        st.success(f"Giải thành công! Sử dụng {solution.num_vehicles_used} xe.")
                    else:
                        st.error(" Không tìm được giải pháp khả thi.")
                        if solution:
                             st.write(f"Số route tìm được (có thể không khả thi): {len(solution.routes)}")
    else:
        if depot_count == 0: st.warning("Cần có ít nhất 1 kho.")
        if customer_count == 0: st.warning("Cần có ít nhất 1 khách hàng.")

    if st.session_state.vrptw_solution:
        sol = st.session_state.vrptw_solution
        st.subheader("Kết quả VRPTW")
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Số xe", sol.num_vehicles_used if sol.is_feasible else "N/A")
        res_col2.metric("Tổng khoảng cách", f"{sol.total_distance:.2f} km" if sol.is_feasible else "N/A")
        res_col3.metric("Tổng thời gian", f"{sol.total_time:.1f} phút" if sol.is_feasible else "N/A")

        if sol.routes and sol.is_feasible:
            st.write("**Chi tiết lộ trình:**")
            depot_series = st.session_state.locations[st.session_state.locations['is_depot']]
            depot_name = depot_series['name'].iloc[0] if not depot_series.empty else "Kho"
            for i, route_indices in enumerate(sol.routes):
                if route_indices:
                    st.write(f"**Xe {i+1}:**")
                    valid_route_names = []
                    total_demand_route = 0
                    for idx in route_indices:
                        if 0 <= idx < len(st.session_state.locations):
                            loc_data = st.session_state.locations.iloc[idx]
                            valid_route_names.append(loc_data['name'])
                            total_demand_route += loc_data['demand']
                        else:
                            valid_route_names.append(f"Lỗi_Index_{idx}")
                    route_text = f"{depot_name} → " + " → ".join(valid_route_names) + f" → {depot_name}"
                    st.write(f"   {route_text}")
                    st.write(f"   📦 Tổng tải: {total_demand_route}/{vehicle_capacity}")
        elif not sol.is_feasible:
            st.warning("Không có lộ trình khả thi để hiển thị.")


with col2: # Phần bản đồ
    st.subheader("Bản đồ")
    if st.session_state.selected_row_index is not None:
        try:
            selected_name_map = st.session_state.locations.iloc[st.session_state.selected_row_index]['name']
            st.info(f"Click vào bản đồ để cập nhật vị trí cho: **{selected_name_map}**.")
        except IndexError:
             st.info("Click vào bản đồ để thêm điểm mới.") # Fallback
    else:
        st.info("Click vào bản đồ để thêm điểm mới.")


    m = folium.Map(location=get_map_center(), zoom_start=12, tiles="OpenStreetMap")

    if not st.session_state.locations.empty:
        for idx, loc in st.session_state.locations.iterrows(): # Dùng idx ở đây
            earliest_str = f"{int(loc['earliest_time'])//60:02d}:{int(loc['earliest_time']) % 60:02d}"
            latest_str = f"{int(loc['latest_time'])//60:02d}:{int(loc['latest_time']) % 60:02d}"
            popup_html = (f"<b>{loc['name']} (ID: {loc['id']})</b><br>"
                          f"Loại: {'Kho' if loc['is_depot'] else 'Khách hàng'}<br>"
                          f"Nhu cầu: {loc['demand']}<br>"
                          f"Thời gian: {earliest_str} - {latest_str}<br>"
                          f"Phục vụ: {loc['service_time']} phút")
            
            marker_color = 'orange' if idx == st.session_state.selected_row_index else ('red' if loc['is_depot'] else 'blue')
            
            folium.Marker(
                [loc['latitude'], loc['longitude']],
                tooltip=f"{loc['name']} ({'Kho' if loc['is_depot'] else f'Cần: {loc['demand']}'})",
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color=marker_color,
                                 icon='home' if loc['is_depot'] else ('star' if idx == st.session_state.selected_row_index else 'user'), 
                                 prefix='fa')
            ).add_to(m)

    # ... (Phần vẽ route trên bản đồ giữ nguyên) ...
    if st.session_state.vrptw_solution and st.session_state.vrptw_solution.is_feasible and st.session_state.vrptw_solution.routes:
        sol = st.session_state.vrptw_solution
        colors = ['#FF0000', '#008000', '#0000FF', '#800080', '#FFA500', 
                  '#A52A2A', '#FFC0CB', '#00FFFF', '#00008B', '#006400']
        depot_df_mask = st.session_state.locations['is_depot']
        if depot_df_mask.any():
            depot_loc_series = st.session_state.locations[depot_df_mask].iloc[0]
            depot_coords = [depot_loc_series['latitude'], depot_loc_series['longitude']]
            for i, route_indices in enumerate(sol.routes):
                if route_indices:
                    color = colors[i % len(colors)]
                    full_path_coords_for_geom = [depot_coords]
                    for loc_idx in route_indices:
                         if 0 <= loc_idx < len(st.session_state.locations):
                            customer_loc = st.session_state.locations.iloc[loc_idx]
                            full_path_coords_for_geom.append([customer_loc['latitude'], customer_loc['longitude']])
                    full_path_coords_for_geom.append(depot_coords)
                    for j in range(len(full_path_coords_for_geom) - 1):
                        start_c = full_path_coords_for_geom[j]
                        end_c = full_path_coords_for_geom[j+1]
                        geometry = get_route_geometry_api(start_c, end_c)
                        if geometry:
                            folium.PolyLine(
                                locations=geometry, weight=4, color=color, opacity=0.8,
                                popup=f"Xe {i+1} - Đoạn {j+1}"
                            ).add_to(m)
                        else:
                            folium.PolyLine(
                                locations=[start_c, end_c], weight=3, color=color, opacity=0.6, dash_array='5, 5',
                                popup=f"Xe {i+1} - Đoạn {j+1} (Fallback)"
                            ).add_to(m)
                    for k, loc_idx in enumerate(route_indices):
                        if 0 <= loc_idx < len(st.session_state.locations):
                            customer_loc = st.session_state.locations.iloc[loc_idx]
                            folium.Marker(
                                [customer_loc['latitude'], customer_loc['longitude']],
                                icon=folium.DivIcon(
                                    html=f'<div style="font-size: 10pt; font-weight: bold; color: white; background-color:{color}; border-radius: 50%; width: 20px; height: 20px; text-align:center; line-height:20px;">{k+1}</div>',
                                    icon_size=(20,20), icon_anchor=(10,10)
                                )
                            ).add_to(m)
    
    map_output = st_folium(m, width="100%", height=500, key="folium_map_main_vrptw", returned_objects=["last_clicked"])

    if map_output and map_output.get("last_clicked"):
        lat = map_output["last_clicked"]["lat"]
        lng = map_output["last_clicked"]["lng"]
        
        # Kiểm tra xem có đang chọn điểm để cập nhật không
        if st.session_state.selected_row_index is not None:
            # Đánh dấu rằng click này là để cập nhật, và gọi add_location
            # add_location sẽ kiểm tra selected_row_index
            add_location(lat=lat, lng=lng, update_existing_idx=st.session_state.selected_row_index)
            # st.session_state.selected_row_index = None # Bỏ chọn sau khi click để cập nhật
            # Không rerun ở đây, add_location sẽ rerun
        else:
            # Thêm điểm mới như bình thường
            add_location(lat=lat, lng=lng) # is_depot sẽ được quyết định trong add_location
            # Không rerun ở đây, add_location sẽ rerun