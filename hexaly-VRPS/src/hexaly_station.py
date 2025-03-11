import hexaly.optimizer
from load_data_from_chains import get_data as get_data

# from load_data_from_chains_online import get_data_Hexlay as get_data
import folium
from PIL import Image
import argparse
import io
import os
import sys
import math
import os


def pdptw_solver(
    n,
    P,
    D,
    C,
    tau,
    add_time_for_break,
    c,
    add_cost_for_break,
    a,
    b,
    s,
    ell,
    num_vehicles,
    str_time_limit,
    sol_file,
    chain_id,
    delta=900,
):
    #
    # Read instance data
    #
    (
        nb_customers,
        nb_vehicles,
        vehicle_capacity,
        dist_matrix_data,
        dist_depot_data,
        time_matrix_data,
        time_depot_data,
        add_cost_matrix_data,
        add_cost_depot_data,
        add_time_matrix_data,
        add_time_depot_data,
        demands_data,
        service_time_data,
        earliest_start_data,
        latest_end_data,
        pick_up_index,
        delivery_index,
        max_horizon,
    ) = read_input_pdptw(
        n,
        P,
        D,
        C,
        tau,
        add_time_for_break,
        c,
        add_cost_for_break,
        a,
        b,
        s,
        ell,
        num_vehicles,
        str_time_limit,
        sol_file,
        chain_id,
        delta,
    )

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        #
        # Declare the optimization model
        #
        model = optimizer.model

        # Sequence of customers visited by each vehicle
        customers_sequences = [model.list(nb_customers) for k in range(nb_vehicles)]

        # All customers must be visited by exactly one vehicle
        model.constraint(model.partition(customers_sequences))

        demands = model.array(demands_data)  # Loads of the customers
        earliest = model.array(earliest_start_data)  # a[i]
        latest = model.array(latest_end_data)  # b[i]
        service_time = model.array(service_time_data)  # s[i]
        dist_matrix = model.array(dist_matrix_data)  # cost matrix
        dist_depot = model.array(dist_depot_data)  # cost depots
        time_matrix = model.array(time_matrix_data)
        time_depot = model.array(time_depot_data)
        add_cost_matrix = model.array(add_cost_matrix_data)
        add_cost_depot = model.array(add_cost_depot_data)
        add_time_matrix = model.array(add_time_matrix_data)
        add_time_depot = model.array(add_time_depot_data)

        dist_routes = [None] * nb_vehicles  # Distance traveled by each vehicle before stationing
        acc_dist_routes = [None] * nb_vehicles  # Actual distance traveled by each vehicle after stationing
        end_time = [None] * nb_vehicles  # End time of visit at node before considering stationing
        arrival_time = [None] * nb_vehicles  # Arrival time at each node after considering stationing
        station_needed = [None] * nb_vehicles
        home_lateness = [
            None
        ] * nb_vehicles  # Used to compute sum of lateness must be 0 for solution to be feasible (set in constraints already)
        lateness = [None] * nb_vehicles
        route_quantity = [None] * nb_vehicles

        M = 10000
        # A vehicle is used if it visits at least one customer
        vehicles_used = [(model.count(customers_sequences[k]) > 0) for k in range(nb_vehicles)]
        nb_vehicles_used = model.sum(vehicles_used)

        # Pickups and deliveries
        customers_sequences_array = model.array(customers_sequences)
        for i in range(nb_customers):
            if pick_up_index[i] == -1:
                pick_up_list_index = model.find(customers_sequences_array, i)
                delivery_list_index = model.find(customers_sequences_array, delivery_index[i])
                model.constraint(
                    pick_up_list_index == delivery_list_index
                )  # Constraint to ensure that pickup and delivery are in the same route
                pick_up_list = model.at(customers_sequences_array, pick_up_list_index)
                delivery_list = model.at(customers_sequences_array, delivery_list_index)
                model.constraint(
                    model.index(pick_up_list, i) < model.index(delivery_list, delivery_index[i])
                )  # Constraint to ensure that pickup index is before delivery index

        for k in range(nb_vehicles):
            # vehicle sequence
            sequence = customers_sequences[k]
            c = model.count(sequence)  # length of the vehicle's sequence

            # The quantity needed in each route must not exceed the vehicle capacity at any
            # point in the sequence
            # Explanation: i is the index of the current node, prev is demand of the previous node

            # Compute the quantity of each route
            demand_lambda = model.lambda_function(lambda i, prev: prev + demands[sequence[i]])

            route_quantity[k] = model.array(model.range(0, c), demand_lambda, 0)

            # Set constraint for quantity/load to be less than vehicle capacity
            quantity_1_lambda = model.lambda_function(lambda i: route_quantity[k][i] <= vehicle_capacity)
            model.constraint(model.and_(model.range(0, c), quantity_1_lambda))

            # Set constraint for quantity/load to be greater than 0
            quantity_2_lambda = model.lambda_function(lambda i: route_quantity[k][i] >= 0)
            model.constraint(model.and_(model.range(0, c), quantity_2_lambda))

            # Distance traveled by each vehicle before we consider stationing
            dist_lambda = model.lambda_function(lambda i: model.at(dist_matrix, sequence[i - 1], sequence[i]))
            dist_routes[k] = model.sum(model.range(1, c), dist_lambda) + model.iif(
                c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0
            )

            # End of each visit before we consider stationing
            end_lambda = model.lambda_function(
                lambda i, prev: model.max(
                    earliest[sequence[i]],
                    model.iif(
                        i == 0,
                        time_depot[sequence[0]],
                        prev + model.at(time_matrix, sequence[i - 1], sequence[i]) + service_time[sequence[i]],
                    ),
                )
            )

            end_time[k] = model.array(model.range(0, c), end_lambda, 0)

            late_end_lambda = model.lambda_function(lambda i: end_time[k][i] <= latest[sequence[i]])

            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), late_end_lambda))

            # Station Needed
            station_lambda = model.lambda_function(
                lambda i, prev: model.iif(
                    i == (c - 1),
                    0,
                    model.iif(
                        end_time[k][i + 1] - end_time[k][i] - model.at(time_matrix, sequence[i], sequence[i + 1])
                        >= delta,
                        1,
                        0,
                    ),
                )
            )

            station_needed[k] = model.array(model.range(0, c), station_lambda, 0)

            arrival_lambda = model.lambda_function(
                lambda i, prev: model.max(
                    earliest[sequence[i]],
                    model.iif(
                        i == 0,
                        time_depot[sequence[0]],
                        model.iif(
                            station_needed[k][i - 1] == 1,
                            prev
                            + model.at(time_matrix, sequence[i - 1], sequence[i])
                            + model.at(add_time_matrix, sequence[i - 1], sequence[i])
                            + service_time[sequence[i]],
                            prev + model.at(time_matrix, sequence[i - 1], sequence[i]) + service_time[sequence[i]],
                        ),
                    ),
                )
            )

            arrival_time[k] = model.array(model.range(0, c), arrival_lambda, 0)

            late_arrival_lambda = model.lambda_function(lambda i: arrival_time[k][i] <= latest[sequence[i]])

            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), late_arrival_lambda))

            # New Distance after considering stationing
            act_dist_lambda = model.lambda_function(
                lambda i: model.iif(
                    station_needed[k][i] == 1,
                    model.at(dist_matrix, sequence[i - 1], sequence[i])
                    + model.at(add_cost_matrix, sequence[i - 1], sequence[i]),
                    model.at(dist_matrix, sequence[i - 1], sequence[i]),
                )
            )

            acc_dist_routes[k] = model.sum(model.range(1, c), act_dist_lambda) + model.iif(
                c > 0, dist_depot[sequence[0]] + dist_depot[sequence[c - 1]], 0
            )

            # Station Load must be zero if station is needed
            # station_load_lambda = model.lambda_function(
            #     lambda i : route_quantity[k][i] <= M*(1 - station_needed[k][i]))

            station_load_lambda = model.lambda_function(
                lambda i: model.iif(station_needed[k][i] == 1, route_quantity[k][i] == 0, True)
            )

            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), station_load_lambda))

            # Arriving home after max_horizon
            home_lateness[k] = model.iif(
                vehicles_used[k], model.max(0, arrival_time[k][c - 1] + time_depot[sequence[c - 1]] - max_horizon), 0
            )

            # #Arriving home after max_horizon ----- OLD
            # home_lateness[k] = model.iif(
            #     vehicles_used[k],
            #     model.max(
            #         0,
            #         end_time[k][c - 1] + time_depot[sequence[c - 1]] - max_horizon),
            #     0)

            # Completing visit after latest_arrival
            late_selector = model.lambda_function(lambda i: model.max(0, arrival_time[k][i] - latest[sequence[i]]))
            lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_selector)

            # #Completing visit after latest_arrival --- OLD
            # late_selector = model.lambda_function(
            #     lambda i: model.max(0, end_time[k][i] - latest[sequence[i]]))
            # lateness[k] = home_lateness[k] + model.sum(model.range(0, c), late_selector)

        # Total lateness (must be 0 for the solution to be valid)
        total_lateness = model.sum(lateness)

        # Total distance traveled
        total_distance = model.div(model.round(100 * model.sum(dist_routes)), 100)
        act_total_distance = model.div(model.round(100 * model.sum(acc_dist_routes)), 100)

        # Objective: minimize the number of vehicles used, then minimize the distance traveled
        model.minimize(total_lateness)
        model.minimize(nb_vehicles_used)
        model.minimize(act_total_distance)
        model.minimize(total_distance)

        model.close()

        # Parameterize the solver
        optimizer.param.time_limit = int(str_time_limit)

        optimizer.solve()

        # Collect solution data
        solution_data = []
        total_distance_value = total_distance.value
        # act_dist_value = act_total_distance.value

        for k in range(nb_vehicles):
            if vehicles_used[k].value != 1:
                continue

            sequence_cust = [customer + 1 for customer in customers_sequences[k].value]
            print(sequence_cust)
            end_times = [time for time in end_time[k].value]
            arrival_times = [time for time in arrival_time[k].value]
            station_needs = [station_need for station_need in station_needed[k].value]
            quantities = [quantity for quantity in route_quantity[k].value]
            route_data = {
                "vehicle": k,
                "route": sequence_cust,  # Add depot at start and end
                "ends": end_times,
                "arr_times": arrival_times,
                "stations": station_needs,
                "distance": dist_routes[k].value,
                "quantity": quantities,
            }
            solution_data.append(route_data)

        # write_solution_to_file(sol_file, chain_id, solution_data, ell, earliest_start_data, latest_end_data, tau, add_time_for_break, total_distance_value)
        # Write the solution
        if sol_file is not None:
            with open(sol_file, "w") as f:
                f.write(f"Chain Id: {chain_id}\n")
                sol_load = 0
                for route in solution_data:
                    load = 0
                    f.write(f"Route for vehicle {route['vehicle']}:\n")
                    for i in range(len(route["route"])):
                        customer = route["route"][i]
                        load += ell[customer]
                        end_time = route["ends"][i]
                        arrival = route["arr_times"][i]

                        time_window_start = earliest_start_data[customer - 1]
                        time_window_end = latest_end_data[customer - 1]
                        travel_time = (
                            tau[route["route"][i - 1], route["route"][i]] if i > 0 else tau[0, route["route"][i]]
                        )
                        add_time = (
                            add_time_for_break[route["route"][i - 1], route["route"][i]]
                            if i > 0
                            else add_time_for_break[0, route["route"][i]]
                        )
                        arrival_diff = arrival - route["arr_times"][i - 1] if i > 0 else 0
                        arrival_diff = arrival_diff - travel_time if i > 0 else 0
                        quantity = route["quantity"][i]
                        try:
                            station_need = route["stations"][i]
                            f.write(
                                f" {customer}  End Time({end_time}) Arrival({arrival}) Quantity({quantity}) Load({load}) Station Needed ({station_need}) TimeWindow({time_window_start}, {time_window_end}), Travel Time({travel_time}), Additional Time({add_time}), Arrival Diff({arrival_diff}) \n"
                            )
                        except:
                            f.write(
                                f" {customer} Load({load}) Quantity({quantity}) End Time({end_time}) Arrival({arrival}) TimeWindow({time_window_start}, {time_window_end}),  Travel Time({travel_time}), Additional Time({add_time})\n"
                            )
                            print(f"Exception at {i} : Customer {customer}")
                        if i < len(route["route"]) - 1:
                            f.write(" -> ")
                    f.write("\n")
                    f.write(f"Distance of the route: {route['distance']:.1f} m\n")
                    sol_load += load
                    f.write(f"Load of the route: {load}\n")

                f.write(f"Total distance of all routes without break distance: {total_distance_value:.1f} m\n")
                # f.write(f"Total distance of all routes with break distance: {act_dist_value:.1f} m\n")
                f.write(f"Total load of all routes: {sol_load}\n")


def write_solution_to_file(
    sol_file,
    chain_id,
    solution_data,
    ell,
    earliest_start_data,
    latest_end_data,
    tau,
    add_time_for_break,
    total_distance_value,
):
    """
    Writes the solution to a specified file.

    Args:
        sol_file (str): Path to the output file.
        chain_id (str): Identifier for the solution chain.
        solution_data (list): List of dictionaries containing route details for each vehicle.
        ell (dict): Dictionary containing load information for customers.
        earliest_start_data (list): Earliest start times for each customer.
        latest_end_data (list): Latest end times for each customer.
        tau (dict): Travel time matrix.
        add_time_for_break (dict): Additional time for breaks matrix.
        total_distance_value (float): Total distance traveled without break distance.
    """
    if sol_file is None:
        return

    with open(sol_file, "w") as f:
        f.write(f"Chain Id: {chain_id}\n")
        sol_load = 0

        for route in solution_data:
            load = 0
            f.write(f"Route for vehicle {route['vehicle']}:\n")

            for i in range(len(route["route"])):
                customer = route["route"][i]
                load += ell[customer]
                end_time = route["ends"][i]
                arrival = route["arr_times"][i]

                time_window_start = earliest_start_data[customer - 1]
                time_window_end = latest_end_data[customer - 1]
                travel_time = (
                    tau.get((route["route"][i - 1], route["route"][i]), tau.get((0, route["route"][i]), 0))
                    if i > 0
                    else 0
                )
                add_time = (
                    add_time_for_break.get(
                        (route["route"][i - 1], route["route"][i]), add_time_for_break.get((0, route["route"][i]), 0)
                    )
                    if i > 0
                    else 0
                )
                arrival_diff = (arrival - route["arr_times"][i - 1] - travel_time) if i > 0 else 0
                quantity = route["quantity"][i]

                try:
                    station_need = route["stations"][i]
                    f.write(
                        f" {customer}  End Time({end_time}) Arrival({arrival}) Quantity({quantity}) "
                        f"Load({load}) Station Needed ({station_need}) TimeWindow({time_window_start}, {time_window_end}), "
                        f"Travel Time({travel_time}), Additional Time({add_time}), Arrival Diff({arrival_diff}) \n"
                    )
                except Exception:
                    f.write(
                        f" {customer} Load({load}) Quantity({quantity}) End Time({end_time}) Arrival({arrival}) "
                        f"TimeWindow({time_window_start}, {time_window_end}), Travel Time({travel_time}), Additional Time({add_time})\n"
                    )
                    print(f"Exception at {i} : Customer {customer}")

                if i < len(route["route"]) - 1:
                    f.write(" -> ")

            f.write("\n")
            f.write(f"Distance of the route: {route['distance']:.1f} m\n")
            sol_load += load
            f.write(f"Load of the route: {load}\n")

        f.write(f"Total distance of all routes without break distance: {total_distance_value:.1f} m\n")
        f.write(f"Total load of all routes: {sol_load}\n")


def read_input_pdptw(
    n,
    P,
    D,
    C,
    tau,
    add_time_for_break,
    c,
    add_cost_for_break,
    a,
    b,
    s,
    ell,
    num_vehicles,
    str_time_limit,
    sol_file,
    chain_id,
    delta,
):

    nb_vehicles = int(num_vehicles)
    vehicle_capacity = C[1]
    max_horizon = 86399

    customers_x = []
    customers_y = []
    customers = []
    demands = []
    earliest_start = []
    latest_end = []
    service_time = []
    pick_up_index = []
    delivery_index = []

    for i in range(1, n + 1):
        customers.append(i)
        demands.append(ell[i])
        earliest_start.append(a[i])
        latest_end.append(b[i])
        service_time.append(s[i])
        pick_up_index.append(-1)
        delivery_index.append(i + n - 1)

    for i in range(n + 1, 2 * n + 1):
        customers.append(i)
        demands.append(ell[i])
        earliest_start.append(a[i])
        latest_end.append(b[i])
        service_time.append(s[i])
        pick_up_index.append(i - n - 1)
        delivery_index.append(-1)

    nb_customers = 2 * n

    distance_matrix = compute_distance_matrix(customers, c)
    distance_depots = compute_distance_depots(customers, c)
    time_matrix = compute_time_matrix(customers, tau)
    time_depots = compute_time_depots(customers, tau)
    add_cost_matrix = compute_add_cost_matrix(customers, add_cost_for_break)
    add_cost_depots = compute_add_cost_depots(customers, add_cost_for_break)
    add_time_matrix = compute_add_time_matrix(customers, add_time_for_break)
    add_time_depots = compute_add_time_depots(customers, add_time_for_break)

    return (
        nb_customers,
        nb_vehicles,
        vehicle_capacity,
        distance_matrix,
        distance_depots,
        time_matrix,
        time_depots,
        add_cost_matrix,
        add_cost_depots,
        add_time_matrix,
        add_time_depots,
        demands,
        service_time,
        earliest_start,
        latest_end,
        pick_up_index,
        delivery_index,
        max_horizon,
    )


# Compute the distance matrix
def compute_distance_matrix(customers, c):
    nb_customers = len(customers)
    distance_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(nb_customers):
            if i == j:
                distance_matrix[i][j] = 0
                continue
            distance_matrix[i][j] = c[customers[i], customers[j]]
            distance_matrix[j][i] = c[customers[j], customers[i]]
    return distance_matrix


# Compute the time matrix
def compute_time_matrix(customers, tau):
    nb_customers = len(customers)
    time_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(nb_customers):
            if i == j:
                time_matrix[i][j] = 0
                continue
            time_matrix[i][j] = tau[customers[i], customers[j]]
            time_matrix[j][i] = tau[customers[j], customers[i]]
    return time_matrix


# Compute the additional cost matrix
def compute_add_cost_matrix(customers, add_cost_for_break):
    nb_customers = len(customers)
    add_cost_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(nb_customers):
            if i == j:
                add_cost_matrix[i][j] = 0
                continue
            add_cost_matrix[i][j] = add_cost_for_break[customers[i], customers[j]]
            add_cost_matrix[j][i] = add_cost_for_break[customers[j], customers[i]]
    return add_cost_matrix


# Compute the additional time matrix
def compute_add_time_matrix(customers, add_time_for_break):
    nb_customers = len(customers)
    add_time_matrix = [[None for i in range(nb_customers)] for j in range(nb_customers)]
    for i in range(nb_customers):
        for j in range(nb_customers):
            if i == j:
                add_time_matrix[i][j] = 0
                continue
            add_time_matrix[i][j] = add_time_for_break[customers[i], customers[j]]
            add_time_matrix[j][i] = add_time_for_break[customers[j], customers[i]]
    return add_time_matrix


# Compute the distances to the depot
def compute_distance_depots(customers, c):
    nb_customers = len(customers)
    distance_depots = [None] * nb_customers
    for i in range(nb_customers):
        distance_depots[i] = c[0, customers[i]]
    return distance_depots


# Compute the times to the depot
def compute_time_depots(customers, tau):
    nb_customers = len(customers)
    time_depots = [None] * nb_customers
    for i in range(nb_customers):
        time_depots[i] = tau[0, customers[i]]
    return time_depots


# Compute the additional costs to the depot
def compute_add_cost_depots(customers, add_cost_for_break):
    nb_customers = len(customers)
    add_cost_depots = [None] * nb_customers
    for i in range(nb_customers):
        add_cost_depots[i] = add_cost_for_break[0, customers[i]]
    return add_cost_depots


# Compute the additional times to the depot
def compute_add_time_depots(customers, add_time_for_break):
    nb_customers = len(customers)
    add_time_depots = [None] * nb_customers
    for i in range(nb_customers):
        add_time_depots[i] = add_time_for_break[0, customers[i]]
    return add_time_depots


def parse_results(file_path):
    routes = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    # Print and extract date
    chain_id = lines[0].strip()
    print(chain_id)

    current_route = []

    for line in lines[2:]:
        line = line.strip()

        if line.startswith("Route for vehicle"):
            # If we are already building a route, save it
            if current_route:
                # Add depot (0) at the beginning and end of the route
                current_route = [0] + current_route + [0]
                routes.append(current_route)
            # Start a new route
            print(line)
            current_route = []
        elif line.startswith("Distance of the route") or line.startswith("Load of the route"):
            print(line)
        elif line.startswith("Total distance of all routes"):
            total_distance = float(line.split(":")[1].strip().replace(" m", ""))
            print(line)
        elif line.startswith("Total load of all routes"):
            total_load = int(line.split(":")[1].strip())
            print(line)
        else:
            # Process the route detaioptimizer
            segments = line.split("->")
            for segment in segments:
                customer = int(segment.split()[0])
                current_route.append(customer)
            print(line)

    # Add the last route
    if current_route:
        # Add depot (0) at the beginning and end of the route
        current_route = [0] + current_route + [0]
        routes.append(current_route)

    return routes


def main(chain_id, num_vehicles, vehicle_capacity, delta, num_stations, station_nodes):
    n, P, D, C, tau, add_time_for_break, c, add_cost_for_break, a, b, s, ell, depot = get_data(
        chain_id, num_vehicles, vehicle_capacity, delta, num_stations, station_nodes
    )
    RESULTS_DIR = "./results"
    SOLUTIONS_DIR = f"{RESULTS_DIR}/PDPTW_DB_Hexaly_Solutions"
    # Save detaioptimizer to text file in solutions directory
    if not os.path.exists(SOLUTIONS_DIR):
        os.makedirs(SOLUTIONS_DIR)
    if not os.path.exists(f"{SOLUTIONS_DIR}/{chain_id}"):
        os.makedirs(f"{SOLUTIONS_DIR}/{chain_id}")

    filename = f"{SOLUTIONS_DIR}/{chain_id}/data.txt"
    with open(filename, "w") as file:
        file.write(f"Chain Id: {chain_id}\n")
        file.write(f"Number of requests: {n}\n")
        file.write(f"Pickup nodes: {P}\n")
        file.write(f"Delivery nodes: {D}\n")
        file.write(f"Vehicle capacity: {C}\n")
        for i in range(1, n + 1):
            file.write(f"Load at pickup node {i}: {ell[i]}, Time window: [{a[i]},{b[i]}], Service time: {s[i]}\n")

        for i in range(n + 1, 2 * n + 1):
            file.write(f"Load at delivery node {i}: {ell[i]}, Time window: [{a[i]},{b[i]}], Service time: {s[i]}\n")

    str_time_limit = "10"
    sol_file = f"{SOLUTIONS_DIR}/{chain_id}/routes.txt"
    pdptw_solver(
        n,
        P,
        D,
        C,
        tau,
        add_time_for_break,
        c,
        add_cost_for_break,
        a,
        b,
        s,
        ell,
        num_vehicles,
        str_time_limit,
        sol_file,
        chain_id,
        delta,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process chains for Hexaly solver.")

    parser.add_argument("--c", type=int, help="Chain ID", default=0)
    parser.add_argument("--v", type=int, help="Num Vehicles", default=2)
    parser.add_argument("--cap", type=int, help="Vehicle Capacity", default=12)
    parser.add_argument("--d", type=int, help="Delta", default=900)
    parser.add_argument("--s", type=int, help="Num Stations", default=5)
    parser.add_argument(
        "--sn",
        type=int,
        nargs="+",
        help="Station Node IDs",
        default=[8537, 4638, 9284, 3607, 6703],
    )

    args = parser.parse_args()

    # Access the arguments
    print(f"Chain ID: {args.c}")
    print(f"Num Vehicles: {args.v}")
    print(f"Vehicle Capacity: {args.cap}")
    print(f"Delta: {args.d}")
    print(f"Num Stations: {args.s}")
    print(f"Station Node IDs: {args.sn}")

    main(args.c, args.v, args.cap, args.d, args.s, args.sn)
