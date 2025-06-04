# demo các hàm trong phương thức
import numpy as np
import array
from numpy import ndarray
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# Kiểm tra kiểu dữ liệu


def check_data_type(obj):
    if isinstance(obj, list):
        return "List"
    elif isinstance(obj, tuple):
        return "Tuple"
    elif isinstance(obj, dict):
        return "Dictionary"
    elif isinstance(obj, array.array):
        return "Array"
    elif isinstance(obj, np.ndarray):
        return "ndarray"
    else:
        return "Không phải kiểu dữ liệu được kiểm tra"
# Tính khoảng cách từ 1 điểm đến tất cả các điểm còn lại bằng euclidean


def distance_cdist(X: np.ndarray, Y: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    # return distance_euclidean(X,Y) if metric=='euclidean' else distance_chebyshev(X,Y)
    from scipy.spatial.distance import cdist
    return cdist(X, Y, metric=metric)

# Ma trận độ thuộc ra nhãn (giải mờ)


def extract_labels(membership: np.ndarray) -> np.ndarray:
    return np.argmax(membership, axis=1)

# # Chia các điểm vào các cụm
# def extract_clusters(data: np.ndarray, labels: np.ndarray, n_cluster: int = 0) -> list:
#     if n_cluster == 0:
#         n_cluster = np.unique(labels)
#     return [data[labels == i] for i in range(n_cluster)]
# Chia các điểm vào các cụm


def extract_clusters(labels: np.ndarray, n_cluster: int = 0) -> list:
    if n_cluster == 0:
        n_cluster = np.unique(labels)
    return [np.argwhere([labels == i]).T[1,] + 1 for i in range(n_cluster)]

# Làm tròn số


def round_float(number: float, n: int = 2) -> float:
    if n == 0:
        return int(number)
    return round(number, n)


def create_graph(coords):
    """
    Tạo ma trận khoảng cách từ danh sách tọa độ.
    :param coords: Danh sách tọa độ (danh sách các tuple dạng (x, y))
    :return: Ma trận khoảng cách
    """
    # num_nodes = len(coords)
    # distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    # for i in range(num_nodes):
    #     for j in range(i + 1, num_nodes):
    #         dist = np.linalg.norm(coords[i] - coords[j])
    #         distance_matrix[i][j] = dist
    #         distance_matrix[j][i] = dist  # Ma trận đối xứng

    return distance_cdist(coords, coords)

# Biểu diễn các điểm lên


def visualize_clusters(data, labels, centers):
    plt.figure(figsize=(10, 10))
    plt.scatter(data[0, 0], data[0, 1], c='black',
                marker='o', s=200, label='Starter')
    # Xóa hàng đầu tiên
    data = np.delete(data, 0, 0)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = data[labels == label]
        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1])
    # Convert centers to a NumPy array for proper indexing
    centers_array = np.array(centers)
    print(np.shape(centers_array))

    # plt.scatter(centers_array[:, 0], centers_array[:, 1],
    #             c='red', marker='x', s=200, label='Centers')
    plt.legend()
    plt.title('C')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    


def visualize_vrptw_routes(coordinates, routes, depot_index=0, save_path=None):
    """
    Vẽ các tuyến đường trong bài toán VRPTW với mỗi tuyến (xe) có màu khác nhau.

    Parameters:
        coordinates (np.ndarray): Mảng numpy với shape (n, 2) chứa tọa độ [x, y] của depot và khách hàng.
        routes (list of list of int): Danh sách các tuyến, mỗi tuyến là list các chỉ số (int).
        depot_index (int): Chỉ số depot trong mảng tọa độ.
        save_path (str): Nếu muốn lưu ảnh, truyền vào đường dẫn file.
    """
    # Kiểm tra coordinates hợp lệ
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("'coordinates' phải là numpy array.")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("'coordinates' phải có shape (n, 2) — mỗi hàng là [x, y].")

    n_points = coordinates.shape[0]

    fig, ax = plt.subplots(figsize=(10, 8))

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    ax.scatter(x, y, c='black', label='Customer', zorder=3)
    ax.scatter(x[depot_index], y[depot_index], c='red', s=150, marker='*', label='Depot', zorder=4)

    # for i, (xi, yi) in enumerate(zip(x, y)):
    #     ax.annotate(str(i), (xi + 0.5, yi + 0.5), fontsize=8)

    color_map = plt.cm.get_cmap('tab20', len(routes))

    for i, route in enumerate(routes):
        if not isinstance(route, list):
            raise TypeError(f"Route {i+1} phải là list các chỉ số (int), nhưng nhận được: {type(route)}")

        for idx in route:
            if not isinstance(idx, int):
                raise TypeError(f"Route {i+1} chứa phần tử không phải int: {idx}")
            if not (0 <= idx < n_points):
                raise IndexError(f"Route {i+1} chứa chỉ số không hợp lệ: {idx}")

        route_coords = coordinates[route]
        ax.plot(route_coords[:, 0], route_coords[:, 1],
                label=f'Vehicle {i+1}', linewidth=2, color=color_map(i), marker='o')

    ax.set_title('VRPTW - Vehicle Routes Visualization')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def write_excel_file(data_excels, data_files, data_name, run_time, algorithms, title_names, fileio):
    from xlsxwriter import Workbook
    import numpy as np

    workbook = Workbook(fileio)
    for idx_a, algorithm in enumerate(algorithms):
        worksheet = workbook.add_worksheet(data_name+algorithm+str(idx_a))
        titformat = workbook.add_format(
            {'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 14})
        char_data_end = (len(data_files)*len(title_names) // 26) * \
            "A" + chr(ord('A')+len(data_files)*len(title_names) % 26)
        worksheet.merge_range(
            f'A1:{char_data_end}1', f'Kết quả chạy thử bộ dữ liệu bằng {algorithm}', titformat)
        hedformat = workbook.add_format(
            {'bold': 1, 'border': 1, 'align': 'center'})
        # In các bộ dữ liệu
        for idx_d, dat in enumerate(data_files):
            # Tính số lượng chữ cái A được lặp và chữ cái cuối cùng của chuỗi trong excel
            char_start = ((idx_d*len(title_names)+1) // 26)*"A" + \
                chr(ord('A')+((idx_d*len(title_names)+1) % 26))
            char_end = ((idx_d+1)*len(title_names) // 26)*"A" + \
                chr(ord('A')+((idx_d+1)*len(title_names) % 26))
            worksheet.merge_range(
                f'{char_start}2:{char_end}2', dat[:-4], titformat)

        # Tạo một list titles chứa các tiêu đề cho các cột
        titles = ['Lần chạy'] + (title_names*len(data_files))

        # Duyệt qua danh sách titles và ghi từng tiêu đề vào hàng thứ ba (chỉ số 2) của worksheet, áp dụng hedformat
        for idx_t, title in enumerate(titles):
            worksheet.write(2, idx_t, title, hedformat)

        # Tạo một đối tượng định dạng colformat cho viền của các ô
        colformat = workbook.add_format({'border': 1})

        # Định dạng lại dữ liệu đầu vào
        data = np.array(data_excels[idx_a])
        # số bộ dữ liệu x số lần chạy  x số thuộc tính
        data = np.reshape(
            data, (len(data_files), run_time, len(title_names)))

        # số lần chạy + 1 x (số bộ dữ liệu x số thuộc tính)
        data = data.transpose(1, 0, 2).reshape(
            run_time, len(data_files)*len(title_names))

        # Tính giá trị trung bình cho các lần tính
        data_mean = np.mean(data, axis=0, keepdims=True)
        data = np.append(data, data_mean, axis=0)

        for row, dr in enumerate(data):
            dat = [row + 1]  # Tính cho hàng giá trị trung bình
            dat = dat + list(dr)
            for i, item in enumerate(dat):
                worksheet.write(row + 3, i, item, colformat)
        worksheet.write(row+3, 0, "TB", colformat)
    workbook.close()
