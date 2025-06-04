import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.GeneticAlgorithm import GA, Individual
from algorithm.VariableNeighborhoodSearchAlgorithm import VNS
from algorithm.kmeans import Kmeans
import math
import random
from ultility.utilities import round_float, write_excel_file, distance_cdist, create_graph
import os
import time


class GA_VNS(GA, VNS):
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.15, vehcicle_capacity: float = 200, conserve_rate: float = 0.1, M: float = 50, customers: list = None, graph_data: np.ndarray = None, list_n_l: list = None, beta_0: int = 0, beta_1: int = 0):
        GA.__init__(self, individual=individual, generation=generation, crossover_rate=crossover_rate, mutation_rate=mutation_rate,
                    vehcicle_capacity=vehcicle_capacity, conserve_rate=conserve_rate, M=M, customers=customers, graph_data=graph_data)
        VNS.__init__(self, list_n_l=list_n_l, beta_0=beta_0,
                     beta_1=beta_1, generation=generation)

    # Hàm kiểm tra xem chuỗi gen mới có tốt hơn chuỗi gen cũ
    def is_better(self, new_gene: list, gene: list):
        _, fitness_gene, _ = GA.cal_fitness_individualV2(self, individual=gene)

        _, fitness_new_gene, _ = GA.cal_fitness_individualV2(self,
                                                             individual=new_gene, fitness_to_branch_bound=fitness_gene)

        return True if fitness_new_gene > fitness_gene else False

    # Tạm thời chưa tối ưu code -> đề xuất sửa thành thuật toán nhánh cận tăng tốc độ tính toán
    def generate_first_individual(self, cluster):
        # # Chuyển np.ndarray sang list để sort không bị lỗi
        # cluster = list(cluster)
        copy_cluster = cluster.copy()
        # Sắp xếp tăng dần theo dueTime, nếu bằng nhau thì theo readyTime
        copy_cluster.sort(key=lambda x: (
            self.customers[x].dueTime, self.customers[x].readyTime))

        new_individual = [copy_cluster[0]]
        del copy_cluster[0]

        while copy_cluster:
            cus_min = 0
            idx_min = 0
            fitness_min = float('inf')  # sửa thành 'inf' vì đang tìm min

            for idx, cus in enumerate(copy_cluster):
                test = new_individual + [cus]  # tạo bản copy tạm để đánh giá
                _, fitness_part, _ = self.cal_fitness_individualV2(
                    individual=test, fitness_to_branch_bound=fitness_min)
                if fitness_part < fitness_min:  # cập nhật nếu tốt hơn
                    fitness_min = fitness_part
                    cus_min = cus
                    idx_min = idx

            # Chèn khách hàng tốt nhất
            new_individual.append(cus_min)
            del copy_cluster[idx_min]

        return new_individual

    def initialPopulation(self, cluster) -> list:
        # route = GA.individualToRoute(self,individual=cluster)
        # self.population = [Individual(customerList=[x for sub in route for x in (
        #     random.sample(sub, len(sub)))]) for _ in range(self._individual)]
        self.population = [Individual(customerList=random.sample(
            cluster, len(cluster))) for _ in range(self._individual - 1)]
        self.population.append(Individual(customerList=cluster))

    def TwoPointCrossover(self, dad, mom):
         # Chuyển đổi cá thể thành các sub_route
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        # Chọn điểm cắt cho mẹ
        pos1, pos2 = sorted(random.choices(range(len(sub_route_mom)), k=2))
            
        sub_mom = [item for sublist in sub_route_mom[pos1:pos2+1]
                       for item in sublist]

        
        pos1, pos2 = sorted(random.choices(range(len(sub_route_dad)), k=2))
            
        sub_dad = [item for sublist in sub_route_dad[pos1:pos2+1]
                       for item in sublist]
        

        # Tạo filter_dad bằng cách loại bỏ phần tử của sub_mom
        filter_dad = [item for item in dad if item not in sub_mom]
        gene_child_1 = filter_dad[:pos1] + sub_mom + filter_dad[pos1:]

        # Tạo filter_mom bằng cách loại bỏ phần tử của sub_dad
        filter_mom = [item for item in mom if item not in sub_dad]
        gene_child_2 = filter_mom[:pos1] + sub_dad + filter_mom[pos1:]

        return gene_child_1, gene_child_2

    def hybird(self):
        # Lấy vị trí bắt đầu của cá thể được phép lai ghép trong quần thể
        index = math.floor(self._conserve_rate*self._individual)

        while (len(self.population) < self._individual):
            # Lấy ra tỉ lệ sinh của quần thể lần này
            hybird_rate = random.random()

            # Lấy ngẫu nhiên ra 2 cá thể đem trao đổi chéo
            dad, mom = random.sample(self.population[index:], 2)

            # Kiểm tra tỉ lệ sinh so với tủi lệ trao đổi chéo
            if (hybird_rate > self._crossover_rate):
                continue

            # Tiến hành trao đổi chéo
            gene_child_1, gene_child_2 = self.TwoPointCrossover(
                dad.customerList, mom.customerList)

            gene_child_3, gene_child_4 = self.TwoPointCrossover(
                dad.customerList, self.population[0].customerList)
            # Kiếm tra tỉ lệ sinh với tỉ lệ đột biến gen
            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)
                gene_child_3 = self.mutation(gene_child_3)
                gene_child_4 = self.mutation(gene_child_4)

            self.population.append(Individual(customerList=gene_child_1))
            self.population.append(Individual(customerList=gene_child_2))
            self.population.append(Individual(customerList=gene_child_3))
            self.population.append(Individual(customerList=gene_child_4))

            if (len(self.population) > self._individual):
                del self.population[self._individual:]
                break

    def fit(self, cluster):
        _start_time = time.time()
        cluster = self.generate_first_individual(cluster)
        self.initialPopulation(cluster)
        self.cal_fitness_population()

        for _ in range(self._generation):
            self.selection()
            self.hybird()

            self.cal_fitness_population()
            self.population[0].customerList = VNS.fit(self,
                                                      gene=self.population[0].customerList)

        self.process_time += round_float(time.time() - _start_time)

        self.cal_fitness_population()
        self.selection()

        self.best_fitness_global += self.population[0].fitness
        self.best_distance_global += self.population[0].distance
        best_route = self.individualToRoute(self.population[0].customerList)
        self.best_route_global.append(best_route)
        self.route_count_global += len(best_route)
        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

    def fit_allClusters(self, clusters):
        # clusters = [[12, 13, 14, 15, 16, 17, 18, 19, 77, 82, 83, 84, 86, 87, 88, 90, 91, 96]]

        clusters = self.re_cluster_by_timewindow(clusters)
        # print('Số cụm sau đi phân cụm lại',len(clusters),clusters)
        # exit()

        for i in range(len(clusters)):
            self.fit(cluster=clusters[i])

        return self.best_fitness_global, self.best_route_global, self.best_distance_global, self.route_count_global, self.process_time


if __name__ == "__main__":
    # Thông số K-means
    N_CLUSTER = 1
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100
    # Thông số GA
    INDIVIDUAL = 100
    GENERATION = 100
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    VEHCICLE_CAPACITY = 700
    CONSERVE_RATE = 0.1
    M = 0
    # Thông số VNS
    LIST_N_L = [2] * 6
    BETA_0 = 5
    BETA_1 = 2
    # Thông số ngoài
    DATA_ID = "C101"  # File dữ liệu cụ thể
    DATA_NAME = "C1"  # Bộ dữ liệu
    DATA_NUMBER_CUS = "100"  # Số lượng khách hàng
    RUN_TIMES = 10  # Số lượng chạy
    EXCEL_FILE = None  # File excel xuất ra kết quả bộ dữ liệu
    FILE_EXCEL_PATH = "result/"
    FILE_NAME = "_TestKmeans"
    # ================================================================================================================
    if (DATA_ID != None):
        url_data = "data/txt/"+DATA_NUMBER_CUS+"/"+DATA_ID[:-2]+"/"
        data_files = [DATA_ID+".txt"]
    else:
        url_data = "data/txt/"+DATA_NUMBER_CUS+"/"+DATA_NAME+"/"
        data_files = sorted(
            [f for f in os.listdir(url_data) if f.endswith(('.txt'))])
        EXCEL_FILE = FILE_EXCEL_PATH + DATA_NAME + FILE_NAME + ".xlsx"

    len_data = len(data_files)
    print(f"Bộ dữ liệu {DATA_NAME}: {data_files}")
    run_time_data = 0
    route_count_data = 0
    distance_data = 0
    fitness_data = 0

    C_scope = range(1, 10)
    data_excel = []
    for data_file in data_files:
        run_time_mean = 0
        route_count_mean = 0
        distance_mean = 0
        fitness_mean = 0

        _start_time = time.time()
        # Khởi tạo dữ liệu
        capacity, data, customers = load_txt_dataset(
            url=url_data, name_of_id=data_file)
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))

        print("#K-means =============================")
        # chạy kmeans
        _start_time = time.time()
        kmeans = Kmeans(epsilon=EPSILON, maxiter=MAX_ITER, n_cluster=N_CLUSTER)
        warehouse = data[0]
        data_kmeans = np.delete(data, 0, 0)
        # U1, V1, step = kmeans.k_means(data_kmeans)
        # print("Thời gian chạy K-means:", round_float(time.time() - _start_time))
        graph_data = create_graph(coords=data)
        # cluster = kmeans.data_to_cluster(U1)

        U1, V1, step = kmeans.k_means_lib_sorted(data_kmeans, warehouse)
        cluster = kmeans.data_to_cluster(U1)
        print(cluster)
        # kmeans.elbow_k_means_lib(data_kmeans,C_scope)
        # kmeans.elbow_k_means(data_kmeans,C_scope)
        # -------------------------------------------------------------------------------------------------------------------------------------

        print("#GA =============================")

        for j in range(RUN_TIMES):

            ga = GA_VNS(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                        vehcicle_capacity=capacity, conserve_rate=CONSERVE_RATE, M=M, customers=customers, graph_data=graph_data, list_n_l=LIST_N_L, beta_0=BETA_0, beta_1=BETA_1)

            best_fitness_global, best_route_global, best_distance_global, route_count_global, process_time = ga.fit_allClusters(
                clusters=cluster)

            run_time_mean += process_time

            print(
                f"Thời gian chạy {data_file[:-4]} lần", j+1, ":", process_time)
            print("Fitness: ", round_float(best_fitness_global))
            print("Distance: ", round_float(best_distance_global))
            print("Số lượng route: ", route_count_global)
            print(best_route_global)
            route_count_mean += route_count_global
            distance_mean += best_distance_global
            fitness_mean += best_fitness_global
            data_excel += [[route_count_global,
                            round_float(best_distance_global), process_time]]
            print("===================================")
        # Thống kê file dữ liệu

        print(f"#Thống kê {data_file[:-4]} =============================")
        print("Số lượt chạy mỗi bộ dữ liệu ", RUN_TIMES)
        print("Fitness trung bình: ", round_float(fitness_mean/RUN_TIMES))
        print("Số lượng route trung bình: ",
              round_float(route_count_mean/RUN_TIMES))
        print("Thời gian di chuyển trung bình: ",
              round_float(distance_mean/RUN_TIMES))
        print("Thời gian chạy trung bình: ",
              round_float(run_time_mean/RUN_TIMES))
        run_time_data += round_float(run_time_mean/RUN_TIMES)
        route_count_data += round_float(route_count_mean/RUN_TIMES)
        distance_data += round_float(distance_mean/RUN_TIMES)
        fitness_data += round_float(fitness_mean/RUN_TIMES)
        data_excel += [[round_float(route_count_mean/RUN_TIMES),
                        round_float(distance_mean/RUN_TIMES)]]
        print("====================================================================================================================")

    # Thống kê data
    print("=====================================================================================================================================")
    print(f"#Thống kê {DATA_NAME} =============================")
    print("Số lượt chạy mỗi bộ dữ liệu ", RUN_TIMES)
    print("Fitness trung bình: ", round_float(fitness_data/len_data))
    print("Số lượng route trung bình: ", round_float(route_count_data/len_data))
    print("Thời gian di chuyển trung bình: ",
          round_float(distance_data/len_data))
    print("Thời gian chạy trung bình: ", round_float(run_time_data/len_data))
