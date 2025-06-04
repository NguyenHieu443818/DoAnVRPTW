import random
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pydantic import BaseModel
import math
import time

class LocationModel(BaseModel):
    id: int
    name: str
    latitude: float
    longitude: float
    demand: int
    earliest_time: int
    latest_time: int
    service_time: int
    description: str
    is_depot: bool

class VRPTWDataInput(BaseModel):
    locations: List[LocationModel]
    depot_idx: int
    distance_matrix: List[List[float]]
    time_matrix: List[List[float]]
    vehicle_capacity: int = 100
    # max_time: int = 480 # Đã loại bỏ

class VRPTWSolutionOutput(BaseModel):
    routes: List[List[int]]
    total_distance: float
    total_time: float
    num_vehicles_used: int
    is_feasible: bool

POPULATION_SIZE = 50
NUM_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
TOURNAMENT_SIZE = 5

def calculate_route_details(route: List[int], data: VRPTWDataInput, locations_list: List[Dict],
                            dist_matrix: np.ndarray, time_matrix: np.ndarray) -> Tuple[float, float, float, bool, float, float]:
    route_dist = 0.0
    route_time = 0.0
    current_load = 0.0
    current_time_vehicle = 0.0
    current_loc_idx = data.depot_idx
    is_feasible_route = True
    time_window_penalty = 0.0
    capacity_penalty = 0.0

    if not route:
        return 0.0, 0.0, 0.0, True, 0.0, 0.0

    first_customer_idx = route[0]
    travel_to_first = time_matrix[current_loc_idx][first_customer_idx]
    route_dist += dist_matrix[current_loc_idx][first_customer_idx]
    current_time_vehicle += travel_to_first
    
    arrival_at_first = current_time_vehicle
    customer_data = locations_list[first_customer_idx]
    
    service_start_time = max(arrival_at_first, customer_data['earliest_time'])
    if service_start_time > customer_data['latest_time']:
        is_feasible_route = False
        time_window_penalty += (service_start_time - customer_data['latest_time']) * 10

    current_time_vehicle = service_start_time + customer_data['service_time']
    current_load += customer_data['demand']
    current_loc_idx = first_customer_idx

    for i in range(len(route) - 1):
        from_idx = route[i]
        to_idx = route[i+1]
        
        travel_time_segment = time_matrix[from_idx][to_idx]
        route_dist += dist_matrix[from_idx][to_idx]
        current_time_vehicle += travel_time_segment

        arrival_at_next = current_time_vehicle
        customer_data = locations_list[to_idx]

        service_start_time = max(arrival_at_next, customer_data['earliest_time'])
        if service_start_time > customer_data['latest_time']:
            is_feasible_route = False
            time_window_penalty += (service_start_time - customer_data['latest_time']) * 10

        current_time_vehicle = service_start_time + customer_data['service_time']
        current_load += customer_data['demand']
        current_loc_idx = to_idx

    last_customer_idx = route[-1]
    route_dist += dist_matrix[last_customer_idx][data.depot_idx]
    current_time_vehicle += time_matrix[last_customer_idx][data.depot_idx]

    if current_load > data.vehicle_capacity:
        is_feasible_route = False
        capacity_penalty += (current_load - data.vehicle_capacity) * 10
    
    # if current_time_vehicle > data.max_time: # Đã loại bỏ kiểm tra max_time
    #     is_feasible_route = False
    #     time_window_penalty += (current_time_vehicle - data.max_time) * 5 

    return route_dist, current_time_vehicle, current_load, is_feasible_route, time_window_penalty, capacity_penalty

def calculate_fitness(chromosome: List[List[int]], data: VRPTWDataInput,
                      locations_list: List[Dict], dist_matrix: np.ndarray, time_matrix: np.ndarray) -> float:
    total_dist_solution = 0.0
    total_time_solution = 0.0
    total_penalty = 0.0
    all_routes_feasible = True

    if not chromosome:
        return 0.0

    for route in chromosome:
        if not route: continue

        route_dist, route_time, _, is_route_feasible, tw_penalty, cap_penalty = \
            calculate_route_details(route, data, locations_list, dist_matrix, time_matrix)
        
        total_dist_solution += route_dist
        total_time_solution += route_time
        total_penalty += tw_penalty + cap_penalty
        if not is_route_feasible:
            all_routes_feasible = False

    cost = total_dist_solution
    if cost == 0 and total_penalty == 0:
        return 1.0
    
    fitness_value = 1.0 / (cost + total_penalty + 1e-6)
    
    return fitness_value

def initialize_population(data: VRPTWDataInput, locations_list: List[Dict],
                          dist_matrix: np.ndarray, time_matrix: np.ndarray) -> List[List[List[int]]]:
    population = []
    customer_indices = [i for i, loc in enumerate(locations_list) if not loc['is_depot']]

    for _ in range(POPULATION_SIZE):
        shuffled_customers = random.sample(customer_indices, len(customer_indices))
        individual_routes = []
        current_route = []
        current_load = 0
        current_time_vehicle = 0
        current_loc_idx_in_route = data.depot_idx

        for cust_idx in shuffled_customers:
            customer = locations_list[cust_idx]
            demand = customer['demand']
            travel_to_cust = time_matrix[current_loc_idx_in_route][cust_idx]
            arrival_at_cust = current_time_vehicle + travel_to_cust
            service_start = max(arrival_at_cust, customer['earliest_time'])
            service_end = service_start + customer['service_time']
            time_back_to_depot = time_matrix[cust_idx][data.depot_idx]
            
            if (current_load + demand <= data.vehicle_capacity and
                service_start <= customer['latest_time']):
                # service_end + time_back_to_depot <= data.max_time): # Đã loại bỏ kiểm tra max_time
                current_route.append(cust_idx)
                current_load += demand
                current_time_vehicle = service_end
                current_loc_idx_in_route = cust_idx
            else:
                if current_route:
                    individual_routes.append(current_route)
                current_route = [cust_idx]
                current_load = demand
                travel_from_depot = time_matrix[data.depot_idx][cust_idx]
                arrival_at_cust_new_route = travel_from_depot
                service_start_new_route = max(arrival_at_cust_new_route, customer['earliest_time'])
                if service_start_new_route > customer['latest_time']:
                    current_route = []
                    current_load = 0
                    current_time_vehicle = 0
                    current_loc_idx_in_route = data.depot_idx
                    continue
                current_time_vehicle = service_start_new_route + customer['service_time']
                current_loc_idx_in_route = cust_idx
        
        if current_route:
            individual_routes.append(current_route)
        
        visited_in_individual = set()
        for r in individual_routes:
            for c_idx in r:
                visited_in_individual.add(c_idx)
        
        if len(visited_in_individual) == len(customer_indices):
             population.append(individual_routes)

    if not population and customer_indices:
        fallback_individual = []
        for cust_idx in customer_indices:
            customer = locations_list[cust_idx]
            time_depot_cust = time_matrix[data.depot_idx][cust_idx]
            service_start = max(time_depot_cust, customer['earliest_time'])
            time_cust_depot = time_matrix[cust_idx][data.depot_idx]
            total_route_time = service_start + customer['service_time'] + time_cust_depot

            if (customer['demand'] <= data.vehicle_capacity and
                service_start <= customer['latest_time']):
                # total_route_time <= data.max_time): # Đã loại bỏ kiểm tra max_time
                fallback_individual.append([cust_idx])
        if fallback_individual and len(fallback_individual) == len(customer_indices):
            population.append(fallback_individual)

    if not population:
        print("Warning: Population is still empty.")
    
    return population

def selection(population: List[List[List[int]]], fitness_values: List[float]) -> List[List[int]]:
    selected_parents = []
    for _ in range(len(population)):
        tournament_indices = random.sample(range(len(population)), TOURNAMENT_SIZE)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index_in_tournament = np.argmax(tournament_fitness)
        selected_parents.append(population[tournament_indices[winner_index_in_tournament]])
    return selected_parents

def crossover(parent1: List[List[int]], parent2: List[List[int]], data: VRPTWDataInput,
              locations_list: List[Dict], dist_matrix: np.ndarray, time_matrix: np.ndarray) -> Tuple[List[List[int]], List[List[int]]]:
    child1_routes, child2_routes = [], []
    num_routes_from_p1 = random.randint(1, len(parent1)) if parent1 else 0
    p1_selected_routes_indices = random.sample(range(len(parent1)), num_routes_from_p1) if parent1 else []
    child1_routes = [parent1[i] for i in p1_selected_routes_indices]
    customers_in_child1 = set()
    for route in child1_routes:
        for cust in route:
            customers_in_child1.add(cust)
    remaining_customers_p2_ordered = []
    for route_p2 in parent2:
        for cust_p2 in route_p2:
            if cust_p2 not in customers_in_child1:
                remaining_customers_p2_ordered.append(cust_p2)
    if remaining_customers_p2_ordered:
        for cust_idx in remaining_customers_p2_ordered:
            child1_routes.append([cust_idx])
    num_routes_from_p2 = random.randint(1, len(parent2)) if parent2 else 0
    p2_selected_routes_indices = random.sample(range(len(parent2)), num_routes_from_p2) if parent2 else []
    child2_routes = [parent2[i] for i in p2_selected_routes_indices]
    customers_in_child2 = set()
    for route in child2_routes:
        for cust in route:
            customers_in_child2.add(cust)
    remaining_customers_p1_ordered = []
    for route_p1 in parent1:
        for cust_p1 in route_p1:
            if cust_p1 not in customers_in_child2:
                remaining_customers_p1_ordered.append(cust_p1)
    if remaining_customers_p1_ordered:
        for cust_idx in remaining_customers_p1_ordered:
            child2_routes.append([cust_idx])

    def repair_chromosome(chromosome_routes: List[List[int]], all_customer_indices: List[int]):
        repaired_chromosome = []
        served_customers = set()
        for route in chromosome_routes:
            if not route: continue
            unique_cust_in_route = []
            for cust in route:
                if cust not in served_customers:
                    unique_cust_in_route.append(cust)
                    served_customers.add(cust)
            if unique_cust_in_route:
                repaired_chromosome.append(unique_cust_in_route)
        unserved_customers = [c for c in all_customer_indices if c not in served_customers]
        if unserved_customers:
            for cust in unserved_customers:
                repaired_chromosome.append([cust])
        return repaired_chromosome

    all_customers = [i for i, loc in enumerate(locations_list) if not loc['is_depot']]
    child1_routes = repair_chromosome(child1_routes, all_customers)
    child2_routes = repair_chromosome(child2_routes, all_customers)
    return child1_routes, child2_routes

def mutation(chromosome: List[List[int]], data: VRPTWDataInput,
             locations_list: List[Dict], dist_matrix: np.ndarray, time_matrix: np.ndarray) -> List[List[int]]:
    mutated_chromosome = [list(route) for route in chromosome]
    if not mutated_chromosome or sum(len(r) for r in mutated_chromosome) < 2 :
        return mutated_chromosome
    mutation_type = random.choice(["swap_within_route", "swap_between_routes"]) # Removed other types for simplicity now

    if mutation_type == "swap_within_route" and any(len(r) >= 2 for r in mutated_chromosome):
        eligible_routes_indices = [i for i, r in enumerate(mutated_chromosome) if len(r) >= 2]
        if not eligible_routes_indices: return mutated_chromosome
        route_idx_to_mutate = random.choice(eligible_routes_indices)
        route_to_mutate = mutated_chromosome[route_idx_to_mutate]
        idx1, idx2 = random.sample(range(len(route_to_mutate)), 2)
        route_to_mutate[idx1], route_to_mutate[idx2] = route_to_mutate[idx2], route_to_mutate[idx1]
    elif mutation_type == "swap_between_routes" and len(mutated_chromosome) >= 1 and sum(len(r) for r in mutated_chromosome) >=2 : # Ensure at least one route and two customers overall
        all_customer_positions = []
        for r_idx, route in enumerate(mutated_chromosome):
            for c_pos, _ in enumerate(route):
                all_customer_positions.append((r_idx, c_pos))
        if len(all_customer_positions) < 2: return mutated_chromosome
        pos1_info, pos2_info = random.sample(all_customer_positions, 2)
        r1_idx, c1_pos = pos1_info
        r2_idx, c2_pos = pos2_info
        cust1_val = mutated_chromosome[r1_idx][c1_pos]
        cust2_val = mutated_chromosome[r2_idx][c2_pos]
        mutated_chromosome[r1_idx][c1_pos] = cust2_val
        mutated_chromosome[r2_idx][c2_pos] = cust1_val
    return mutated_chromosome

def genetic_algorithm_vrptw_solver(data_input: VRPTWDataInput) -> VRPTWSolutionOutput:
    solution = VRPTWSolutionOutput(routes=[], total_distance=0, total_time=0, num_vehicles_used=0, is_feasible=False)
    if data_input.distance_matrix is None or data_input.time_matrix is None or len(data_input.locations) <= 1:
        return solution
    locations_list_of_dicts = [loc.model_dump() for loc in data_input.locations]
    distance_matrix_np = np.array(data_input.distance_matrix)
    time_matrix_np = np.array(data_input.time_matrix)
    customer_indices = [i for i, loc_dict in enumerate(locations_list_of_dicts) if not loc_dict.get('is_depot', False)]
    if not customer_indices:
        solution.is_feasible = True
        return solution

    population = initialize_population(data_input, locations_list_of_dicts, distance_matrix_np, time_matrix_np)
    if not population:
        print("GA Error: Could not initialize population.")
        return solution

    best_chromosome_overall = None
    best_fitness_overall = -1.0

    for generation in range(NUM_GENERATIONS):
        fitness_values = [calculate_fitness(ind, data_input, locations_list_of_dicts, distance_matrix_np, time_matrix_np) for ind in population]
        current_best_idx = np.argmax(fitness_values)
        if fitness_values[current_best_idx] > best_fitness_overall:
            best_fitness_overall = fitness_values[current_best_idx]
            best_chromosome_overall = [list(r) for r in population[current_best_idx]]
        parents = selection(population, fitness_values)
        next_population = []
        if best_chromosome_overall:
            if isinstance(best_chromosome_overall, list) and all(isinstance(r, list) for r in best_chromosome_overall):
                next_population.append([list(r) for r in best_chromosome_overall])
        while len(next_population) < POPULATION_SIZE:
            if not parents: break # Avoid error if parents list is empty
            p1, p2 = random.sample(parents, 2) if len(parents) >= 2 else (parents[0], parents[0]) # Handle case with <2 parents
            child1, child2 = p1, p2
            if random.random() < CROSSOVER_RATE:
                child1, child2 = crossover(p1, p2, data_input, locations_list_of_dicts, distance_matrix_np, time_matrix_np)
            if random.random() < MUTATION_RATE:
                child1 = mutation(child1, data_input, locations_list_of_dicts, distance_matrix_np, time_matrix_np)
            if random.random() < MUTATION_RATE:
                child2 = mutation(child2, data_input, locations_list_of_dicts, distance_matrix_np, time_matrix_np)
            next_population.append(child1)
            if len(next_population) < POPULATION_SIZE:
                next_population.append(child2)
        if not next_population: # If elitism didn't add anything and no children were made
            print(f"Warning: Next population is empty at generation {generation+1}. Stopping.")
            break
        population = next_population[:POPULATION_SIZE]
        # print(f"Generation {generation+1}/{NUM_GENERATIONS} - Best Fitness: {best_fitness_overall:.6f}")

    if best_chromosome_overall:
        final_routes = best_chromosome_overall
        total_dist = 0
        total_time = 0
        num_vehicles = len(final_routes)
        solution_is_feasible = True
        all_served_customers_final = set()
        for route in final_routes:
            if not route: 
                num_vehicles -=1
                continue
            for cust_idx_in_route in route:
                all_served_customers_final.add(cust_idx_in_route)
            r_dist, r_time, r_load, r_feasible, r_tw_penalty, r_cap_penalty = \
                calculate_route_details(route, data_input, locations_list_of_dicts, distance_matrix_np, time_matrix_np)
            total_dist += r_dist
            total_time += r_time
            if not r_feasible or r_tw_penalty > 0 or r_cap_penalty > 0:
                solution_is_feasible = False
        if len(all_served_customers_final) != len(customer_indices):
            solution_is_feasible = False
        solution.routes = [[int(c) for c in r] for r in final_routes if r]
        solution.total_distance = round(total_dist, 2)
        solution.total_time = round(total_time, 1)
        solution.num_vehicles_used = len(solution.routes)
        solution.is_feasible = solution_is_feasible
    else:
        print("GA: No best chromosome found.")
    return solution