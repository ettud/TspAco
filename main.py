import random
import numpy as np


# solves travelling salesman problem with ant colony system
# graph should be complete though it is not checked by function
# http://staff.washington.edu/paymana/swarm/stutzle99-eaecs.pdf
# arguments:
# adjacency_matrix  - adjacency matrix of graph of cities
# number_of_ants    - number of ants that will be used in algorithm
# alpha             - coef. of influence of pheromons;
# if alpha = 0 then pheromons do not play any role
# beta              - coef. of influence of length between nodes;
# if beta != 0 than ants are more likely to choose closer cities
# p                 - coef. of pheromon evaporation and also coef. of new pheromon trails;
# if p = 0 then new pheromon trails will not appear and initial pheromon trails will stay forever
# iterations        - number of iterations of algorithm
def solve_tsp_with_aco(adjacency_matrix, number_of_ants, alpha, beta, p, iterations):
    if isinstance(adjacency_matrix, np.ndarray):
        shape = adjacency_matrix.shape
        if shape[0] != shape[1]:
            raise ValueError("adjacency_matrix must be not just a 2-D array but also a quadratic matrix")
    else:
        raise ValueError("adjacency_matrix must be a numpy.ndarray")

    if not isinstance(number_of_ants, type(1)):
        raise ValueError("number_of_ants must be a number")

    if number_of_ants <= 0:
        raise ValueError("number_of_ants must be a positive number")

    if not 0 < p <= 1:
        raise ValueError("p must be greater than 0 but not greater than 1")

    def find_next_node(adjacency_matrix, powed_adjacency_matrix, powed_pheromon_table, current_node, previous_nodes, alpha, beta):
        # calculate probability to go to next_node from current_node omitting visited nodes
        def calculate_probability(adjacency_matrix, powed_pheromon_table, current_node, next_node, previous_nodes, alpha, beta):
            if adjacency_matrix[current_node][next_node] is None:
                return 0
            sum = 0
            for i in range(len(adjacency_matrix)):
                if i != current_node and not previous_nodes.__contains__(i):
                    sum += powed_pheromon_table[current_node][i] / powed_adjacency_matrix[current_node][i]
            return powed_pheromon_table[current_node][next_node] / powed_adjacency_matrix[current_node][next_node] / sum

        probabilities = np.zeros(len(adjacency_matrix), float)
        for i in range(len(adjacency_matrix)):
            if i != current_node and not previous_nodes.__contains__(i):
                probabilities[i] = calculate_probability(adjacency_matrix, powed_pheromon_table, current_node, i, previous_nodes, alpha, beta)
        random_number = random.random()
        s = 0
        for i in range(len(probabilities)):
            if probabilities[i] > 0:
                if s <= random_number < (s + probabilities[i]):
                    return i
                s += probabilities[i]
        # in case there are problems caused by calculation errors connected with float math
        # the closest node will be returned:
        cur_closest_node_number = None
        for i in range(len(adjacency_matrix[current_node])):
            if i != current_node and previous_nodes.__contains__(i):
                if cur_closest_node_number is None:
                    cur_closest_node_number = i
                elif adjacency_matrix[current_node][i] != 0 and adjacency_matrix[current_node][i] < adjacency_matrix[current_node][cur_closest_node_number]:
                    cur_closest_node_number = i
        return cur_closest_node_number

    # function is used to know to which graph arc add pheromons
    def is_subarray(subarray, array):
        for i in range(len(array)+1-len(subarray)):
            for j in range(len(subarray)):
                if array[i+j] != subarray[j]:
                    break
            else:
                return True
        return False

    def create_ant_route(adjacency_matrix, powed_adjacency_matrix, powed_pheromon_table, first_node, alpha, beta):
        ant_route = np.array([first_node], np.uint8)
        ant_route_length = 0

        not_visited_nodes = list(range(len(adjacency_matrix)))
        not_visited_nodes.remove(ant_route[0])
        while len(not_visited_nodes) != 0:
            previous_node = ant_route[-2] if len(ant_route) > 1 else None
            next_node = find_next_node(adjacency_matrix, powed_adjacency_matrix, powed_pheromon_table, ant_route[-1], ant_route, alpha, beta)
            ant_route = np.append(ant_route, next_node)
            not_visited_nodes.remove(next_node)
            if previous_node is not None:
                ant_route_length += adjacency_matrix[previous_node][next_node]
        ant_route = np.append(ant_route, ant_route[0])  # visit the first node to loop the route
        return [ant_route, ant_route_length]

    t = 0
    pheromon_table = np.array([[0.5] * len(adjacency_matrix)] * len(adjacency_matrix), float)
    ant_routes = [None] * number_of_ants
    best_ant_route = None
    best_ant_route_length = None

    powed_adjacency_matrix = np.power(adjacency_matrix, beta)

    while t < iterations:
        powed_pheromon_table = np.power(pheromon_table, alpha)
        # put each ant in random node but try to put them in different:
        free_nodes = np.empty(shape=0)
        first_nodes = np.zeros(shape=(len(ant_routes)), dtype=np.uint8)
        for i in range(len(ant_routes)):
            if len(free_nodes) == 0:
                free_nodes = np.arange(len(adjacency_matrix))
            target_node_number = random.randint(0, len(free_nodes)-1)
            first_nodes[i] = free_nodes[target_node_number]
            ant_routes[i] = np.array([free_nodes[target_node_number]], np.uint8)
            free_nodes = np.delete(free_nodes, target_node_number)

        ant_routes_length = np.empty(shape=(number_of_ants), dtype=np.uint16)

        # Tried to advance this cycle by using multiprocessing.process
        # Apparently Python deals with optimization in this case better
        # Maybe multiprocessing would make sense for big numbers of ants
        # However in my opinion big number of ants should be used in big TCP
        # But as far as I know ACO shows bad results in big TCP
        for k in range(number_of_ants):
            ant_routes[k], ant_routes_length[k] = create_ant_route(adjacency_matrix, powed_adjacency_matrix, powed_pheromon_table, first_nodes[k], alpha, beta)

        best_in_cycle_ant_route = None
        best_in_cycle_ant_route_length = None
        for k in range(number_of_ants):
            k_ant_route_length = 0
            for i in range(1, len(ant_routes[k])):
                k_ant_route_length += adjacency_matrix[ant_routes[k][i-1]][ant_routes[k][i]]
            if best_in_cycle_ant_route is None:
                best_in_cycle_ant_route = ant_routes[k]
                best_in_cycle_ant_route_length = k_ant_route_length
            elif k_ant_route_length < best_in_cycle_ant_route_length:
                best_in_cycle_ant_route = ant_routes[k]
                best_in_cycle_ant_route_length = k_ant_route_length

        if best_ant_route_length is None:
            best_ant_route = best_in_cycle_ant_route
            best_ant_route_length = best_in_cycle_ant_route_length
        if best_in_cycle_ant_route_length < best_ant_route_length:
            best_ant_route = best_in_cycle_ant_route
            best_ant_route_length = best_in_cycle_ant_route_length

        pheromon_table *= (1 - p)
        for i in range(len(pheromon_table)):
            for j in range(len(pheromon_table)):
                for k in range(number_of_ants):
                    if is_subarray([i, j], ant_routes[k]):
                        pheromon_table[i][j] += p/ant_routes_length[k]

        t += 1

    return [best_ant_route, best_ant_route_length]


if __name__ == '__main__':
    alpha = 0.3
    beta = 0.75
    p = 0.75
    t = 80
    ants_number = 10
    matrix = np.array([input().split(" ")])
    number_of_rows = len(matrix[0])
    i = 1
    for i in range(number_of_rows - 1):
        matrix = np.append(matrix, [input().split(" ")])
    matrix.shape = [number_of_rows, number_of_rows]
    matrix = matrix.astype(np.int32)
    print(solve_tsp_with_aco(matrix, ants_number, alpha, beta, p, t)[1])
    exit(0)
