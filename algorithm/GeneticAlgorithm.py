import numpy as np
from ultility.readDataFile import load_txt_dataset
from algorithm.kmeans import Kmeans
import math
import random
from ultility.utilities import round_float, write_excel_file, distance_cdist, create_graph
import os
import time

# Khách hàng


class Individual():
    def __init__(self, customerList: list = None, fitness: float = 0, distance: float = 0):
        self.customerList = customerList
        self.fitness = fitness
        self.distance = distance

    def print(self):
        print(self.customerList, ' ', self.fitness, ' ', self.distance)

# Thuật toán di truyền


class GA:
    def __init__(self, individual: int = 4500, generation: int = 100, crossover_rate: float = 0.8, mutation_rate: float = 0.15, vehcicle_capacity: float = 200, conserve_rate: float = 0.1, M: float = 50, customers: list = None, graph_data: np.ndarray = None):
        self._individual = individual  # số cá thể
        self._generation = generation  # số thế hệ
        self._crossover_rate = crossover_rate  # tỉ lệ trao đổi chéo
        self._mutation_rate = mutation_rate  # tỉ lệ đột biến
        self._conserve_rate = conserve_rate  # tỉ lệ bảo tồn

        self._vehcicle_capacity = vehcicle_capacity  # trọng tải của xe
        self._M = M  # sai số thời gian
        self.customers = customers  # Dữ liệu khách hàng
        self.graph_data = graph_data  # Dữ liệu đồ thị

        self.best_distance_global = 0
        self.route_count_global = 0
        self.best_fitness_global = 0
        self.best_route_global = []

        self.best_fitness_pM = -1
        self.best_fitness_pD = -1

        self.process_time = 0

    # Tối ưu cục bộ(không kiểm tra điều kiện)

    def initialPopulation(self, cluster) -> list:
        self.population = [Individual(customerList=random.sample(
            cluster, len(cluster))) for _ in range(self._individual)]

        # [a.print() for a in self.population]
        # print(isinstance(self.population[0].customerList,np.ndarray))
        # exit()
        # print(self.population[-1].fitness)

    # Tách thành các lộ trình con
    def individualToRoute(self, individual: list):
        route = []  # Lộ trình tổng
        vehicle_load = 0  # trọng tải xe
        sub_route = []  # Lộ trình con
        elapsed_time = 0  # Mốc thời gian hiện tại của xe
        last_customer_id = 0  # Vị khách đã xét trước đó
        for customer_id in individual:
            customer = self.customers[customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand

            # Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            moving_time = np.linalg.norm(
                customer.xy_coord-self.customers[last_customer_id].xy_coord)

            # Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M -
                               elapsed_time - moving_time, 0)
            # Thời gian di chuyển từ điểm đang xét về kho
            return_time = np.linalg.norm(
                customer.xy_coord-self.customers[0].xy_coord)

            update_elapsed_time = elapsed_time + service_time + \
                moving_time + waiting_time + return_time

            if (update_vehicle_load <= self._vehcicle_capacity) and (update_elapsed_time <= self.customers[0].dueTime + self._M):
                sub_route.append(customer_id)
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
            else:
                route.append(sub_route)
                sub_route = [customer_id]
                vehicle_load = demand
                elapsed_time = service_time + return_time + \
                    max(customer.readyTime - self._M - return_time, 0)
            last_customer_id = customer_id
        if sub_route:
            # Lưu lộ trình con còn lại sau khi xét hết các điểm
            route.append(sub_route)
        route = [[int(node) for node in arr] for arr in route if len(arr) > 0]

        return route  # 1 list các array

    # Tính mức độ thích nghi trên toàn bộ quần thể
    def cal_fitness_population(self):
        for individual in self.population:
            _, individual.fitness, individual.distance = self.cal_fitness_individualV2(
                individual.customerList)
        # [a.print() for a in self.population]
        # exit()
        # print(self.population)

    def cal_fitness_individual(self, individual):
        route = self.individualToRoute(individual)
        fitness = 0
        distance = 0
        for sub_route in route:
            sub_route_time_cost = 0  # Thời gian đợi và phạt của 1 route con
            sub_route_distance = 0  # Thời gian di chuyển của 1 route con
            elapsed_time = 0
            last_customer_id = 0
            for customer_id in sub_route:
                customer = self.customers[customer_id]

                # Thời gian di chuyển giữa 2 điểm
                moving_time = np.linalg.norm(
                    customer.xy_coord-self.customers[last_customer_id].xy_coord)

                # Cập nhật thời gian di chuyển
                sub_route_distance += moving_time

                # Mốc thời gian đến khách hàng thứ customer_id
                arrive_time = moving_time + elapsed_time

                # Thời gian chờ đợi nếu xe đang ở thời gian chưa đến thời gian bắt đầu phục vụ
                waiting_time = max(customer.readyTime -
                                   self._M - arrive_time, 0)

                # Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
                delay_time = max(arrive_time - customer.dueTime - self._M, 0)

                # Cập nhật thời gian đợi và phạt
                sub_route_time_cost += waiting_time + delay_time
                # Cập nhật mốc thời gian mới của xe
                elapsed_time = arrive_time + customer.serviceTime + waiting_time
                # Cập nhật khách hàng đang xét
                last_customer_id = customer_id
            # Thời gian di chuyển từ điểm cuối cùng về kho
            return_time = np.linalg.norm(
                self.customers[last_customer_id].xy_coord-self.customers[0].xy_coord)
            sub_route_distance += return_time
            fitness += sub_route_distance + sub_route_time_cost
            distance += sub_route_distance

        return fitness, distance

    def cal_fitness_individualV2(self, individual: list, fitness_to_branch_bound: float = None):
        route = []  # Lộ trình tổng
        sub_route = []  # Lộ trình con
        vehicle_load = 0  # trọng tải xe
        elapsed_time = 0  # Mốc thời gian hiện tại của xe
        last_customer_id = 0  # Vị khách đã xét trước đó
        fitness = 0
        distance = 0
        depot = self.customers[0]  # Lấy dữ liệu kho
        for customer_id in individual:
            customer = self.customers[customer_id]
            # last_customer = self.customers[last_customer_id]
            demand = customer.demand
            update_vehicle_load = vehicle_load + demand

            # Thời gian phục vụ
            service_time = customer.serviceTime

            # Thời gian di chuyển giữa 2 điểm
            # moving_time = np.linalg.norm(
            #     customer.xy_coord-last_customer.xy_coord)
            moving_time = self.graph_data[customer_id][last_customer_id]

            # Mốc thời gian đến khách hàng thứ customer_id
            arrive_time = moving_time + elapsed_time

            # Thời gian chờ đợi khi xe đã di chuyển đến điểm hiện tại
            waiting_time = max(customer.readyTime - self._M - arrive_time, 0)

            # Thời gian phạt của mốc thời gian xe với mốc thời gian muộn nhất có thể giao
            delay_time = max(arrive_time - customer.dueTime - self._M, 0)

            # Thời gian di chuyển từ điểm đang xét về kho
            # return_time = np.linalg.norm(customer.xy_coord-depot.xy_coord)
            return_time = self.graph_data[customer_id][0]

            update_elapsed_time = arrive_time + service_time + waiting_time + return_time

            if (update_vehicle_load <= self._vehcicle_capacity) and (update_elapsed_time <= depot.dueTime + self._M):
                vehicle_load = update_vehicle_load
                elapsed_time = update_elapsed_time - return_time
                sub_route.append(customer_id)
                # Cập nhật thời gian di chuyển
                distance += moving_time
                # Cập nhật thời gian đợi và phạt
                fitness += waiting_time + delay_time
            else:
                route.append(sub_route)
                distance += return_time + self.graph_data[last_customer_id][0]
                sub_route = [customer_id]
                # Cập nhật khoảng cách di chuyển từ điểm kết thúc về kho và bắt đầu từ kho đến điểm (Không cần phải tính phạt vì sẽ là bị dữ liệu sai)
                waiting_time = max(customer.readyTime -
                                   self._M - return_time, 0)
                fitness += waiting_time
                vehicle_load = demand
                elapsed_time = service_time + return_time + waiting_time

            # print(fitness)
            last_customer_id = customer_id

        if (fitness_to_branch_bound is not None) and (fitness > fitness_to_branch_bound):
            return route, fitness, distance

        if sub_route != []:
            route.append(sub_route)
            distance += self.graph_data[last_customer_id][0]
            fitness += distance

        route = [arr for arr in route if len(arr) > 0]

        return route, fitness, distance

    def cal_fitness_sub_route(self, route):
        sub_route_result = []
        for sub_route in route:
            _, fitness, _ = self.cal_fitness_individualV2(sub_route)
            sub_route_result.append(fitness)
        return sub_route_result

    def selection(self):
        # Sắp xếp quần thể theo chiều tăng dần
        self.population.sort(key=lambda x: x.fitness)
        # [a.print() for a in self.population]
        # vị trí xóa = (1-tỉ lệ bảo tồn)*số cá thể/2 + tỉ lệ bảo tồn *số cá thể
        positionToDel = math.floor(self._individual*(1+self._conserve_rate)/2)
        del self.population[positionToDel:]

    def SinglePointCrossover(self, dad, mom):
        assert len(dad) == len(mom), "Dad and Mom must have the same length."

        pos1 = random.randrange(len(mom))

        # Lấy phần tử ở dad không xuất hiện trong mom[pos1:]
        mom_tail_set = set(mom[pos1:])
        filter_dad = [gene for gene in dad if gene not in mom_tail_set]

        # Lấy phần tử ở mom không xuất hiện trong dad[pos1:]
        dad_tail_set = set(dad[pos1:])
        filter_mom = [gene for gene in mom if gene not in dad_tail_set]

        # Ghép gene
        gene_child_1 = filter_dad[:pos1] + mom[pos1:]
        gene_child_2 = filter_mom[:pos1] + dad[pos1:]

        return gene_child_1, gene_child_2

    def heuristic_SinglePointCrossover(self, dad, mom):
        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        fitness_sub_route_mom = self.cal_fitness_sub_route(sub_route_mom)
        fitness_sub_route_dad = self.cal_fitness_sub_route(sub_route_dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if self.best_fitness_pM <= best_fitness_sub_route_mom:
            pos1 = random.randrange(len(mom))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(
                best_fitness_sub_route_mom)]
            # tìm vị trí phần tử trong list
            pos1 = mom.index(best_sub_route_mom[0])

        # Tạo con 1
        mom_tail_set = set(mom[pos1:])
        filter_dad = [gene for gene in dad if gene not in mom_tail_set]
        gene_child_1 = filter_dad[:pos1] + mom[pos1:]

        # ==========================================

        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if self.best_fitness_pD <= best_fitness_sub_route_dad:
            pos1 = random.randrange(len(dad))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(
                best_fitness_sub_route_dad)]
            pos1 = dad.index(best_sub_route_dad[0])

        # Tạo con 2
        dad_tail_set = set(dad[pos1:])
        filter_mom = [gene for gene in mom if gene not in dad_tail_set]
        gene_child_2 = filter_mom[:pos1] + dad[pos1:]

        # Cập nhật best fitness
        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    # def TwoPointCrossover(self, dad, mom):
    #     size = len(mom)
    #     pos1, pos2 = sorted(random.sample(range(size), 2))

    #     # Các gene sẽ lấy từ mom[pos1:pos2] và dad[pos1:pos2]
    #     mid_mom = mom[pos1:pos2]
    #     mid_dad = dad[pos1:pos2]

    #     # Tạo filter: lấy những gene trong dad không xuất hiện trong mid_mom
    #     mid_mom_set = set(mid_mom)
    #     filter_dad = [gene for gene in dad if gene not in mid_mom_set]

    #     # Tạo filter: lấy những gene trong mom không xuất hiện trong mid_dad
    #     mid_dad_set = set(mid_dad)
    #     filter_mom = [gene for gene in mom if gene not in mid_dad_set]

    #     # Ghép các phần tử thành con
    #     gene_child_1 = filter_dad[:pos1] + mid_mom + filter_dad[pos1:]
    #     gene_child_2 = filter_mom[:pos1] + mid_dad + filter_mom[pos1:]

    #     return gene_child_1, gene_child_2

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

    def heuristic_TwoPointCrossoverV1(self, dad, mom):
        size = len(mom)

        sub_route_mom = self.individualToRoute(mom)
        sub_route_dad = self.individualToRoute(dad)

        fitness_sub_route_mom = self.cal_fitness_sub_route(sub_route_mom)
        fitness_sub_route_dad = self.cal_fitness_sub_route(sub_route_dad)

        best_fitness_sub_route_mom = min(fitness_sub_route_mom)
        if self.best_fitness_pM <= best_fitness_sub_route_mom:
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_mom = sub_route_mom[fitness_sub_route_mom.index(
                best_fitness_sub_route_mom)]
            pos1 = mom.index(best_sub_route_mom[0])
            # chú ý +1 vì slicing bên phải mở (khác np)
            pos2 = mom.index(best_sub_route_mom[-1]) + 1

        # Tạo gene_child_1
        mid_mom = mom[pos1:pos2]
        mid_mom_set = set(mid_mom)
        filter_dad = [gene for gene in dad if gene not in mid_mom_set]
        gene_child_1 = filter_dad[:pos1] + mid_mom + filter_dad[pos1:]

        # ========================

        best_fitness_sub_route_dad = min(fitness_sub_route_dad)
        if self.best_fitness_pD <= best_fitness_sub_route_dad:
            pos1, pos2 = sorted(random.sample(range(size), 2))
        else:
            best_sub_route_dad = sub_route_dad[fitness_sub_route_dad.index(
                best_fitness_sub_route_dad)]
            pos1 = dad.index(best_sub_route_dad[0])
            pos2 = dad.index(best_sub_route_dad[-1]) + 1  # chú ý +1

        # Tạo gene_child_2
        mid_dad = dad[pos1:pos2]
        mid_dad_set = set(mid_dad)
        filter_mom = [gene for gene in mom if gene not in mid_dad_set]
        gene_child_2 = filter_mom[:pos1] + mid_dad + filter_mom[pos1:]

        # Cập nhật best fitness
        self.best_fitness_pM = best_fitness_sub_route_mom
        self.best_fitness_pD = best_fitness_sub_route_dad

        return gene_child_1, gene_child_2

    def PMXCrossover(self, dad, mom):
        size = len(mom)
        pos1, pos2 = sorted(random.sample(range(size), 2))

        # Tạo gene con, ban đầu toàn None
        gene_child_1 = [None] * size
        gene_child_2 = [None] * size

        # Tạo ánh xạ
        mapping_gene = {dad[i]: mom[i] for i in range(pos1, pos2)}
        reverse_mapping_gene = {mom[i]: dad[i] for i in range(pos1, pos2)}

        # Ánh xạ cho mỗi vị trí
        for idx_p in range(size):
            d = dad[idx_p]
            m = mom[idx_p]

            if pos1 <= idx_p < pos2:
                gene_child_1[idx_p] = mom[idx_p]
                gene_child_2[idx_p] = dad[idx_p]
                continue

            # Ánh xạ phần gen của bố
            while d in reverse_mapping_gene:
                d = reverse_mapping_gene[d]

            # Ánh xạ phần gen của mẹ
            while m in mapping_gene:
                m = mapping_gene[m]

            gene_child_1[idx_p] = d
            gene_child_2[idx_p] = m

        return gene_child_1, gene_child_2

    def BestCostRouteCrossover(self, dad, mom):
        # Tách các route con từ các lộ trình cha và mẹ
        route_dad = self.individualToRoute(dad)
        route_mom = self.individualToRoute(mom)

        # Chọn ngẫu nhiên sub_route từ cha và mẹ
        sub_route_mom = route_mom[random.randint(0, len(route_mom) - 1)]
        sub_route_dad = route_dad[random.randint(0, len(route_dad) - 1)]

        # Tráo đổi cho nhau và tiến hành xóa những phần tử đấy
        gene_child_1 = [gene for gene in dad if gene not in sub_route_mom]
        gene_child_2 = [gene for gene in mom if gene not in sub_route_dad]

        # Sau đó lấp lại bằng cách dùng phương pháp tham lam
        def greedySearch(intersect_gene, diff_gene):
            common_part = list(intersect_gene)
            # Thực hiện thuật toán nhánh cận
            # diff_gene: đoạn gen lắp vào
            # common_part: đoạn gen trả về
            for gen in diff_gene:
                idx_cus_min = 0
                fitness_min = float('-inf')
                # Tìm vị trí chèn tốt nhất cho gen này
                for idx in range(len(common_part) + 1):
                    _, fitness_part, _ = self.cal_fitness_individualV2(
                        individual=common_part[:idx] + [gen] + common_part[idx:], fitness_to_branch_bound=fitness_min)
                    if fitness_min > fitness_part:
                        fitness_min = fitness_part
                        idx_cus_min = idx

                # Chèn khách hàng
                common_part.insert(idx_cus_min, gen)
            return common_part

        gene_child_1 = greedySearch(gene_child_1, sub_route_mom)
        gene_child_2 = greedySearch(gene_child_2, sub_route_dad)

        return gene_child_1, gene_child_2

    def STPBCrossover(self, dad, mom):  # 3411.464
        # probabilities = [0.1,0.3,0.3,0.1,0.2]
        probabilities = [0.25, 0.25, 0.25, 0.25]

        choice = np.random.choice(
            [i for i in range(len(probabilities))], p=probabilities)
        match choice:
            case 0:
                gene_child_1, gene_child_2 = self.heuristic_SinglePointCrossover(
                    dad, mom)
            case 1:
                gene_child_1, gene_child_2 = self.heuristic_TwoPointCrossoverV1(
                    dad, mom)  # Order
            case 2:
                gene_child_1, gene_child_2 = self.PMXCrossover(dad, mom)
            case 3:
                gene_child_1, gene_child_2 = self.BestCostRouteCrossover(
                    dad, mom)

        return gene_child_1, gene_child_2

    def re_cluster_by_timewindow(self, clusters):

        def check_concatenate(cluster1, cluster2):
            # Nối hai cụm
            total_cluster = cluster1 + cluster2

            # Tính thời gian phục vụ cho toàn bộ khách hàng
            check = sum([self.customers[i].serviceTime for i in total_cluster])

            # Tính trọng tải của toàn bộ khách hàng
            check_capacity = sum(
                [self.customers[i].demand for i in total_cluster])

            # Kiểm tra điều kiện ràng buộc trọng tải
            if check_capacity > self._vehcicle_capacity:
                return False

            # Lấy tọa độ của khách hàng từ các cụm
            cluster1_coords = [self.customers[i].xy_coord for i in cluster1]
            cluster2_coords = [self.customers[i].xy_coord for i in cluster2]

            # Kết hợp tọa độ của hai cụm
            total_cluster_coords = cluster1_coords + cluster2_coords

            # Tính khoảng cách giữa các điểm trong cluster1 và cluster2
            distance = distance_cdist(
                total_cluster_coords, total_cluster_coords, metric='euclidean')

            # Tính khoảng cách trung bình giữa 2 điểm khách hàng trong cụm ghép
            aver_dist = np.mean(np.nonzero(distance))

            # Tính khoảng cách từ depot đến các điểm trong cụm ghép
            distance_to_depot = distance_cdist(
                total_cluster_coords, [self.customers[0].xy_coord], metric='euclidean')

            # Khoảng cách trung bình từ kho đến cụm ghép
            avg_distance_to_depot = np.min(distance_to_depot)

            # Thời gian di chuyển ước lượng
            check += 2 * avg_distance_to_depot + \
                (len(total_cluster_coords) - 1) * aver_dist

            return check <= self.customers[0].dueTime

        def concatenate_arrays(array, index1, index2):
            # Nối hai mảng theo chỉ số đã cho
            return [array[index1] + array[index2]] + [array[i] for i in range(len(array)) if i != index1 and i != index2]

        i = 0
        while i < len(clusters) - 1:
            j = i + 1
            while j < len(clusters):
                if check_concatenate(clusters[i], clusters[j]):
                    clusters = concatenate_arrays(clusters, i, j)
                    # Sau khi nối, không cần kiểm tra lại j
                    j = i + 1  # Reset j để kiểm tra lại từ i
                else:
                    j += 1  # Chỉ tăng j nếu không nối
            i += 1

        return clusters

    def mutation(self, child):
        child_new = child.copy()  # Tạo bản sao của child (dùng list.copy() thay vì np.copy)

        pos1, pos2, pos3, pos4 = sorted(random.sample(range(len(child)), 4))

        # Thực hiện thao tác hoán vị các phần tử trong child
        child_new = child[:pos1] + child[pos3:pos4+1] + \
            child[pos2+1:pos3] + child[pos1:pos2+1] + child[pos4+1:]

        return child_new

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
            gene_child_1, gene_child_2 = self.STPBCrossover(
                dad.customerList, mom.customerList)

            # Kiếm tra tỉ lệ sinh với tỉ lệ đột biến gen
            if hybird_rate <= self._mutation_rate:
                gene_child_1 = self.mutation(gene_child_1)
                gene_child_2 = self.mutation(gene_child_2)

            child1 = Individual(customerList=gene_child_1)
            # child1.fitness,child1.distance = self.cal_fitness_individualV2(gene_child_1)
            self.population.append(child1)

            if (len(self.population) < self._individual):
                child2 = Individual(customerList=gene_child_2)
                # child2.fitness,child2.distance = self.cal_fitness_individualV2(gene_child_2)
                self.population.append(child2)

    def fit(self, cluster):
        _start_time = time.time()
        self.initialPopulation(cluster)
        for _ in range(self._generation):
            self.cal_fitness_population()

            self.selection()
            self.hybird()
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
    N_CLUSTER = 20
    EPSILON = 1e-5
    MAX_ITER = 1000
    NUMBER_OF_CUSTOMER = 100
    # Thông số GA
    INDIVIDUAL = 10
    GENERATION = 10
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    VEHCICLE_CAPACITY = 700
    CONSERVE_RATE = 0.1
    M = 0
    # Thông số ngoài
    DATA_ID = "R101"  # File dữ liệu cụ thể
    DATA_NAME = "R1"  # Bộ dữ liệu
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
        data, customers = load_txt_dataset(
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

            ga = GA(individual=INDIVIDUAL, generation=GENERATION, crossover_rate=CROSSOVER_RATE, mutation_rate=MUTATION_RATE,
                    vehcicle_capacity=VEHCICLE_CAPACITY, conserve_rate=CONSERVE_RATE, M=M, customers=customers, graph_data=graph_data)

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

    title_name = ['Route', 'Distance', 'RunTime']
    # if (DATA_NAME != None):
    #     # write_excel_file(data_excels=data_excel,data_files=data_files,run_time=RUN_TIMES,alogirthm='GA',fileio=EXCEL_FILE)
    #     write_excel_file(data_excels=data_excel,data_files=data_files,run_time=RUN_TIMES,alogirthms='GA',title_names=title_name,fileio=EXCEL_FILE)

    # count = 1
    # for i in range(len(best_route)):
    #     for j in range(len(best_route[i])):
    #         print('route ',count,best_route[i][j])
    #         count+=1
    # print('best_fitness',best_fitness)
