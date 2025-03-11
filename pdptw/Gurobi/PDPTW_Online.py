import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os
import argparse
import folium
import io
from PIL import Image

# import load_data
import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
import time
from MPC_relocation import MPC
import ast
import pickle
from pdptw_gurobi import pdptw_solver, pdptw_solver_station
import csv
import copy

# from hexaly_station import pdptw_solver as hexaly_pdptw_solver


def load_source_data():
    # Load travel time matrix
    file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.npy"
    # TRAVEL_TIME_MATRIX = pd.read_csv(file_path, header=None)
    # TRAVEL_TIME_MATRIX = TRAVEL_TIME_MATRIX.round().astype(np.int32)
    travel_time = np.load(file_path)
    # # Load nodes
    # file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/nodes.csv"
    # nodes = pd.read_csv(file_path
    # Load chains
    DF_TRAIN = pd.read_csv("/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/train_chains.csv")
    DF_TEST = pd.read_csv("/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/test_chains.csv")
    ########### Getting station
    station_path = (
        "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/station_mapping_medium.pkl"
    )

    with open(station_path, "rb") as f:
        node_to_station = pickle.load(f)
    station_id_file = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/stations_medium.csv"
    df = pd.read_csv(station_id_file)  # Replace with the actual file path

    # Extract the 'station_id' column as a list
    station_ids = df["station_id"].tolist()
    return travel_time, DF_TEST, DF_TRAIN, node_to_station, station_ids


def generate_CARTA_data(
    CarNumber=5,
    Capacity=8,
    request_df=None,
    depot_start_location=None,
    v_start_time=None,
    OD_dropoff_request=None,
    TRAVEL_TIME_MATRIX=None,
    parking_nodes=[229, 5346, 1905, 5139],
    Park=True,
):  # C: car capacity; depot_start_dict: depot of each car
    data = request_df
    if data is None:
        data = [  # pickup node, dropoff node, pickup time, dropoff time
            [229, 2898, 18900, 19169, 0],
            [5346, 2898, 19800, 19878, 1],
            [1905, 10243, 23400, 23513, 2],
            [5139, 4446, 26100, 26634, 3],
            [8879, 1446, 26400, 27324, 4],
            [6361, 2887, 27000, 27221, 5],
            [2770, 3603, 27030, 27342, 6],
        ]
    # Initialize dictionaries for points, a (earliest arrival), and b (latest arrival)
    points = {}
    a = {}
    b = {}
    P = []
    D = []
    W = []
    # Loop through the data and set values for pickup and corresponding drop-off nodes
    i = 1
    n = len(data)
    nodes_id = []
    # Loop through the data and set values for pickup and corresponding drop-off nodes
    for idx, request in enumerate(data):
        pickup_node_id, dropoff_node_id, pickup_time, dropoff_time, chain_order = request

        # Placeholder coordinates for pickup and dropoff nodes
        points[idx] = (pickup_node_id, chain_order)  # Assign coordinates for the pickup node
        points[idx + n] = (dropoff_node_id, chain_order)  # Assign coordinates for the dropoff node

        # Pickup time windows
        a[idx] = pickup_time - 15 * 60  # Early arrival time
        b[idx] = pickup_time + 15 * 60  # Late arrival time

        # Dropoff time windows
        a[idx + n] = dropoff_time - 15 * 60  # Early arrival time
        b[idx + n] = dropoff_time + 15 * 60  # Late arrival time
        # Track node IDs
        nodes_id.append(pickup_node_id)
        nodes_id.append(dropoff_node_id)
    P = list(range(0, n))
    D = list(range(n, n * 2))
    ell = {i: 1 for i in P}
    ell.update({i: -1 for i in D})  # Update load update
    ######################## Getting Vehicle information
    K = list(range(1, CarNumber + 1))
    ################################## Generate testing data
    # if OD_dropoff_request is None:
    #     # OD_dropoff_request = [
    #     #     [
    #     #         random.sample(nodes_id, 1)[0],
    #     #         random.sample(range(int(np.mean(list(b.values()))), max(list(b.values()))), 1)[0],
    #     #         -1,
    #     #         random.sample(K, 1)[0],
    #     #     ]
    #     #     for i in range(1, random.randint(2, 6))
    #     # ]  # 5 cars, 8 capacity [node_id, _dropoff_time]
    if depot_start_location is None:
        depot_start_location = {k: 3607 for k in K}  # deport start locations
    C = {i: Capacity for i in K}  # 5 cars, 8 capacity
    if v_start_time is None:
        # v_start_time = {k: random.sample(list(range(10000)), 1)[0] for k in K}
        v_start_time = {k: 0 for k in K}

    ############################################
    # ell.update({n*2+k-1: 0 for k in K})
    depot_start_points = {n * 2 + k - 1: (depot_start_location[k], -1) for k in K}
    depot_start_dict = {k: n * 2 + k - 1 for k in K}
    depot_end_points = {n * 2 + len(C): (3607, -1)}
    # ell.update({n*2+len(C): 0})
    depot_end = n * 2 + len(C)
    points.update(depot_end_points)
    points.update(depot_start_points)
    ##################################################################################
    #### Update for unfinished request (dropoff request )
    # give a fake for debugging:
    # OD_dropoff_request=[[6361, 26384, -1, 4], [4446, 26875, -1, 2], [8879, 26934, -1, 5], [2898, 26026, -1, 1], [6361, 27698, -1, 5]]
    # OD_dropoff_request=[[] for i in range(5)]
    cur_idx = len(points)
    i = 0
    OD = []
    OD_dict = {k: [] for k in K}
    v_load = {i: 0 for i in K}
    if OD_dropoff_request is not None:
        for values in OD_dropoff_request:
            if len(values) > 0:
                node_id, dropoff_time, el_load, v, chian_order = values
                points.update({cur_idx + i: (node_id, chian_order)})
                a.update({cur_idx + i: dropoff_time - 15 * 60})
                b.update({cur_idx + i: dropoff_time + 15 * 60})
                OD.append(cur_idx + i)
                ell.update({cur_idx + i: el_load})
                OD_dict[v].append(cur_idx + i)
                v_load[v] = v_load[v] - el_load
                i = i + 1
    N = P + D
    ################# If consider parking
    W = []
    cur_idx = len(points)
    i = 0
    # if Park:  # If decied to Park
    #     for node_id in parking_nodes:
    #         points[cur_idx + i] = (node_id, -1)
    #         W.append(cur_idx + i)
    #         ell.update({cur_idx + i: 0})
    #         i = i + 1
    #############################################
    all_node = list(points.keys())
    V = all_node
    # Calculate travel times and costs using Euclidean distance
    tau = {
        (i, j, k): TRAVEL_TIME_MATRIX[points[i][0], points[j][0]]
        for i in all_node
        for j in all_node
        if i != j
        for k in C.keys()
    }
    go_station = {
        (i, j): min(
            parking_nodes,
            key=lambda station: TRAVEL_TIME_MATRIX[points[i][0], station] + travel_time_matrix[station, points[j][0]],
        )
        for i in all_node
        for j in all_node
        if i != j
    }
    tau_s = {
        (i, j, k): TRAVEL_TIME_MATRIX[points[i][0], go_station[i, j]]
        + TRAVEL_TIME_MATRIX[go_station[i, j], points[j][0]]
        - tau[i, j, k]
        for i in all_node
        for j in all_node
        if i != j
        for k in C.keys()
    }
    ############################ Update for depots nodes pickup
    a.update({n * 2 + k - 1: v_start_time[k] for k in K})
    b.update({n * 2 + k - 1: max(b.values()) for k in K})
    a.update({n * 2 + len(C): min(v_start_time.values()) + 20 * 60})
    b.update({n * 2 + len(C): max(b.values()) + max(tau.values())})
    c = {(i, j, k): tau[i, j, k] for i, j, k in tau.keys()}  # Assuming cost is proportional to travel time
    s = {i: 1 for i in all_node}
    # Set of vehicles
    return (
        n,
        P,
        D,
        C,
        tau,
        c,
        a,
        b,
        s,
        ell,
        K,
        points,
        depot_end,
        OD,
        depot_start_dict,
        v_load,
        OD_dict,
        V,
        OD_dropoff_request,
        v_start_time,
        points,
        W,
        tau_s,
        go_station,
    )


def simulate(chain_id, t):
    # date = [2022,9,5]
    # year, month, day = date[0], date[1], date[2]
    #
    # n, P, D, C, tau, omega, c, c_omega, a, b, s, ell, K, points, depot_start, depot_end = load_data.get_data_no_break(2022, 9, 5)
    # n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end = load_data.get_data_no_break(2022, 9, 5)
    # n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end, OD, depot_start_dict, depot_end_dict, v_load,OD_dict , V = generate_example_data_new()
    # file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.csv"
    travel_time, DF_TEST, DF_TRAIN, node_to_station, station_ids = load_source_data()
    (
        n,
        P,
        D,
        C,
        tau,
        c,
        a,
        b,
        s,
        ell,
        K,
        points,
        depot_end,
        OD,
        depot_start_dict,
        v_load,
        OD_dict,
        V,
        OD_dropoff_request,
        v_start_time,
        points,
        W,
        tau_s,
        go_station,
    ) = generate_CARTA_data(
        depot_start_location=None, OD_dropoff_request=None, travel_time_matrix=travel_time, Park=True
    )

    routes, node_times, node_loads, nodetravel_times, optimality_gap = pdptw_solver_station(
        n,
        P,
        D,
        C,
        tau,
        tau_s,
        go_station,
        c,
        a,
        b,
        s,
        ell,
        K,
        depot_end,
        OD,
        v_load=v_load,
        depot_start_dict=depot_start_dict,
        OD_dict=OD_dict,
        V=V,
        v_start_time=v_start_time,
    )
    # def pdptw_solver(n, P, D, C, tau, c, a, b, s, ell, K, depot_end, OD, depot_start_dict=None, depot_end_dict=None, v_load=None, OD_dict=None, V=None):
    print(routes)
    print(OD_dict)
    print(P)
    print(D)
    # print(OD_dropoff_request)
    # plot_routes(routes, points, year, month, day, SOLUTIONS_DIR)
    # save_routes(chain, t, routes, optimality_gap, SOLUTIONS_DIR)


def update_state(
    new_time,
    a,
    points,
    n,
    depot_start_location,
    all_routes,
    idle_v,
):
    # new_time=1790
    P_N = n
    unvisited_nodes = defaultdict(list)
    NODE_ID_INDEX = 0
    ARRIVAL_TIME_INDEX = 1
    load_index = 2
    TRAVL_TIME_INDEX = 3
    A_INDEX = 4
    NODE_SEQ_INDEX = 5
    TYPE_INDEX = 6
    CHAIN_ORDER_INDEX = 7
    finished_request = []

    for k, route_v in all_routes.items():
        visited_nodes = []
        # Iterate through nodes and their times
        OD_dropoff_request = []
        for plnode in route_v:
            node_time = plnode[ARRIVAL_TIME_INDEX]
            if node_time <= new_time:
                visited_nodes.append(plnode)
            else:
                unvisited_nodes[k].append(plnode)
        # Sort the nodes (if needed, based on order in the route)
        visited_nodes = sorted(visited_nodes, key=lambda x: x[ARRIVAL_TIME_INDEX])  # Sorting by time
        unvisited_nodes[k] = sorted(unvisited_nodes[k], key=lambda x: x[ARRIVAL_TIME_INDEX])
        v_start_time[k] = new_time  # Update current time
        if len(visited_nodes) > 1:  # If visited node greater than 0, then update depot start location.
            v_current_node = visited_nodes[-1]
            if len(unvisited_nodes[k]) > 1:
                next_node = unvisited_nodes[k][0]
                if new_time + v_current_node[TRAVL_TIME_INDEX] >= next_node[A_INDEX] or next_node[TYPE_INDEX] == "RS":
                    v_current_node = next_node
                    v_new_time = next_node[ARRIVAL_TIME_INDEX]
                    v_start_time[k] = v_new_time
                    # unvisited_nodes[k] = unvisited_nodes[k]
            depot_start_location[k] = v_current_node[NODE_ID_INDEX]
        all_unvisited = [(i[NODE_SEQ_INDEX], i[CHAIN_ORDER_INDEX], i[TYPE_INDEX]) for i in unvisited_nodes[k]]
        unvisited_node_id = []
        for info in visited_nodes:
            node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = info
            if type == "D":
                finished_request.append(chain_order)
        for info in unvisited_nodes[k]:
            node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = info
            if type == "P":
                unvisited_node_id.append(node)
                unvisited_node_id.append(node + P_N)
        for info in all_unvisited:
            node_id, chain_order, type = info
            if node_id not in unvisited_node_id and type == "D":
                OD_dropoff_request.append([points[node_id][0], a[node_id] + 15 * 60, -1, k, points[node_id][1]])
    return (v_start_time, OD_dropoff_request, depot_start_location, finished_request)


def log_route_plan(time, routes):
    # Column headers
    route_file = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/results/route_planning.csv"
    # Open the CSV file in write mode ('w'), with newline='' to prevent extra blank lines
    rows_to_write = []
    for vehicle_id, v_route in routes.items():
        for data in v_route:
            # Create a row with the vehicle's data
            vehicle_info = [
                time,
                vehicle_id,  # vehicle_id
                data[0],
                data[1],  # node_id
                data[2],  # arrival_time
                data[3],  # load
                data[4],  # travel_time
                data[5],  # early_arrival
                data[6],  # node
                data[7],  # type
            ]
            rows_to_write.append(vehicle_info)
    with open(route_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_write)  # Write all rows at once


def get_MPC_input(
    c_t,
    v_cur_location=None,
    data=None,
    OD_dropoff_request=None,
    travel_time=None,
    node_to_station=None,
    station_ids=None,
    all_routes=None,
    idle_car_ids=None,
):
    T = 10 * 60  # 20 minutes
    time_step = 1 * 60
    Price = 10
    beta = 1
    G = {i: 0 for i in station_ids}
    T_s = list(range(0, int(T / time_step)))
    for v, node_id in idle_car_ids.items():
        G[node_to_station[node_id]] = G[node_to_station[node_id]] + 1
    V = [(i, j, t) for i in station_ids for j in station_ids for t in T_s]
    E = [(i, j) for i in station_ids for j in station_ids]

    count_numbder = {v: 0 for v in V}
    for info in data:
        pickup_node_id, dropoff_node_id, pickup_time, dropoff_time, chain_order = info
        if pickup_time <= c_t + T:
            link = (
                node_to_station[pickup_node_id],
                node_to_station[dropoff_node_id],
                max(int((pickup_time - c_t) / time_step) - 1, 0),
            )
            count_numbder[link] = count_numbder[link] + 1
    tripAttr = []
    for v in V:
        for t in T_s:
            tripAttr.append([v[0], v[1], t, count_numbder[v[0], v[1], t], (max(T_s) + 1 - t) * Price])
    # G={i for i in }
    # Trip attributes: List of trips in the form (origin, destination, time, demand, price)
    # tripAttr = [(1, 2, 0, 10, 5), (1, 2, 1, 10, 3), (2, 3, 0, 20, 8), (2, 3, 1, 20, 8)]
    # Demand times: Mapping of (origin, destination) to a dictionary of demand times (time -> demand)
    # demandTime = {(1, 2): {0: 1, 1: 2}, (2, 3): {0: 1, 1: 1}}
    demandTime = {e: {t: int(travel_time[e[0], e[1]] / time_step) for t in T_s} for e in E}
    # Rebalancing times: Mapping of (origin, destination) to a dictionary of rebalance times (time -> time)
    ###### Travel time between each stations
    # rebTime = {(1, 2): {0: 1, 1: 1}, (2, 3): {0: 1, 1: 2}}
    rebTime = {e: {t: int(travel_time[e[0], e[1]] / time_step) for t in T_s} for e in E}

    # Initialize demand (origin, destination) -> time -> demand
    demand = defaultdict(dict)
    for (
        i,
        j,
        tt,
        d,
        _,
    ) in tripAttr:
        demand[i, j][tt] = d
    # Initialize price (origin, destination) -> time -> price
    price = defaultdict(dict)
    for i, j, tt, _, p in tripAttr:
        price[i, j][tt] = p
    # Accumulated number of vehicles (region, time) -> count of vehicles
    acc = defaultdict(dict)
    for n in G:
        acc[n][0] = G[n]  # Initial vehicle count for time 0
    # Arriving vehicles (region, time) -> count of arriving vehicles
    dacc = defaultdict(dict)
    dacc = {e: {t: 0 for t in T_s} for e in station_ids}
    # dacc = {1: {0: 1, 1: 1}, 2: {0: 1, 1: 0}, 3: {0: 1, 1: 0}}
    # Rebalancing vehicle flow (origin, destination) -> time -> number of rebalancing vehicles
    edges = E
    # Extract demand attributes for the next T time steps
    demandAttr = [
        (i, j, tt, demand[i, j][tt], demandTime[i, j][tt], price[i, j][tt])
        for i, j in demand
        for tt in T_s
        if demand[i, j][tt] > 1e-3
    ]
    # Accumulated number of vehicles per region at time t
    accTuple = [(n, acc[n][0]) for n in acc]
    # Accumulated vehicles arriving at each region for the next T time steps
    daccTuple = [(n, tt, dacc[n][tt]) for n in acc for tt in T_s]
    # Rebalancing edge attributes (origin, destination, rebalance time at time t)
    edgeAttr = [(i, j, rebTime[i, j][0]) for i, j in edges]

    return demandAttr, accTuple, daccTuple, edgeAttr, edges, int(T / time_step), beta


def insert_station(
    routes,
    node_times,
    node_loads,
    nodetravel_times,
    a,
    P,
    D,
    points,
    travel_time,
    station_ids,
):
    delta = 15 * 30
    all_routes = defaultdict(list)
    for k, route in routes.items():
        v_routes = []
        idle_v = []
        # Iterate through nodes and their times
        OD_dropoff_request = []
        if len(node_times[k]) == 2:
            idle_v.append(k)
        for i, node_time in enumerate(node_times[k][:-1]):
            if points[route[i]][0] in station_ids and points[route[i]][1] == -1:
                type = "S"  # if node is in station
            elif route[i] in P:
                type = "P"  # Pickup node
            elif route[i] in D:
                type = "D"  # Delivery node
            elif points[route[i]][1] == -1:
                type = "S"
            else:
                type = "D"  # left delivery node
            v_routes.append(
                [
                    points[route[i]][0],
                    node_time,
                    node_loads[k][i],
                    nodetravel_times[k][i],
                    a[route[i]],
                    route[i],
                    type,
                    points[route[i]][1],
                ]
            )
        all_routes[k] = v_routes
    return all_routes, idle_v
    # if type != "S":
    #     if node_time + nodetravel_times[k][i] + delta < a[route[i + 1]] or (
    #         len(node_times[k]) == 2 and points[route[i]][0] not in station_ids
    #     ):
    #         distance = [
    #             (v, travel_time[points[route[i]][0], v] + travel_time[v, points[route[i + 1]][0]])
    #             for v in station_ids
    #         ]
    #         station, tt = min(distance, key=lambda x: x[1])
    #         if (
    #             node_time + nodetravel_times[k][i] + delta + tt < a[route[i + 1]] and node_loads[k][i] == 0
    #         ) or (len(node_times[k]) == 2 and points[route[i]][0] not in station_ids):
    #             v_routes.append(
    #                 [
    #                     station,
    #                     travel_time[points[route[i]][0], station],
    #                     node_loads[k][i],
    #                     travel_time[station, points[route[i + 1]][0]],
    #                     0,
    #                     -1,
    #                     "S",
    #                     -1,
    #                 ]
    #             )


def get_idle_cars(c_t, all_routes, depot_start_location):
    if all_routes is None:
        return False, [], depot_start_location
    idle_cars = {}
    v_cur_location = {}
    for k, route_v in all_routes.items():
        if len(route_v) <= 1:  # If no requested planned
            node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = route_v[0]
            idle_cars[k] = node_id
            v_cur_location[k] = node_id
        else:  # if already visited all nodes
            node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = route_v[-1]
            if c_t >= arrival_time and load == 0:
                idle_cars[k] = node_id
                v_cur_location[k] = node_id
            else:  # If planned but haven't visited the first node
                track_v = []
                for v in route_v:
                    node_id, arrival_time, load, travel_time, early_arrival, node, type, chain_order = v
                    if arrival_time <= c_t + travel_time:
                        track_v.append(v)
                cur_node_v = track_v[-1][0]
                if len(track_v) <= 1 and load == 0 and type == "S":  # If still in the first node, haven't start moving
                    idle_cars[k] = cur_node_v
                v_cur_location[k] = cur_node_v

    if len(idle_cars) > 0:
        return True, idle_cars, v_cur_location
    else:
        return False, [], v_cur_location


def get_reallocation(c_t, paxAction, rebAction, all_routes, idle_car_ids, node_to_station, travel_time, edges):

    if len(rebAction) > 0:
        rellocation_edges = [[i, j, rebAction[i, j]] for i, j in edges if (i, j) in rebAction]
        for reloc in rellocation_edges:
            st_from, st_to, num = reloc
            source_v = []
            for v, node_id in idle_car_ids.items():
                if node_to_station[node_id] == st_from:
                    source_v.append([v, travel_time[node_id, st_to], node_id])
            sorted_v = sorted(source_v, key=lambda x: x[1])
            move_v = sorted_v[: int(min(num, len(sorted_v)))]  #  Choose the idle cars closest to the station to move
            for v_info in move_v:
                v, travel_time_to, node_id = v_info
                all_routes[v].append(
                    [
                        st_to,
                        c_t + travel_time_to,
                        0,
                        travel_time_to,
                        -1,  # No arrival time requriment
                        -1,  # no ID sequence,
                        "RS",
                        -1,
                    ]
                )
    return all_routes


def PDPTW_Plan(
    CarNumber=1,
    Capacity=8,
    request_df=None,
    depot_start_location=None,
    OD_dropoff_request=None,
    v_start_time=None,
    TRAVEL_TIME_MATRIX=None,
    Park=False,
):
    (
        n,
        P,
        D,
        C,
        tau,
        c,
        a,
        b,
        s,
        ell,
        K,
        points,
        depot_end,
        OD,
        depot_start_dict,
        v_load,
        OD_dict,
        V,
        OD_dropoff_request,
        v_start_time,
        points,
        W,
        _,
        _,
    ) = generate_CARTA_data(
        CarNumber=CarNumber,
        Capacity=Capacity,
        request_df=data,
        depot_start_location=depot_start_location,
        OD_dropoff_request=OD_dropoff_request,
        v_start_time=v_start_time,
        TRAVEL_TIME_MATRIX=travel_time,
        Park=False,
    )
    routes, node_times, node_loads, nodetravel_times, optimality_gap = pdptw_solver(
        n,
        P,
        D,
        C,
        tau,
        c,
        a,
        b,
        s,
        ell,
        K,
        depot_end,
        OD,
        v_load=v_load,
        depot_start_dict=depot_start_dict,
        OD_dict=OD_dict,
        V=V,
        v_start_time=v_start_time,
    )
    print(
        f" at time {cur_time/60} minute --RunTime {time.time()-cur_time_clock}----------------- Finish Routing Planning "
    )
    print(routes, optimality_gap)
    print(len(P), P)
    print(len(D), D)
    all_routes, idle_v = insert_station(
        routes,
        node_times,
        node_loads,
        nodetravel_times,
        a,
        P,
        D,
        points,
        travel_time,
        station_ids,
    )
    log_route_plan(cur_time, all_routes)
    return (routes, node_times, node_loads, nodetravel_times, optimality_gap, a, points, n, idle_v)


if __name__ == "__main__":

    route_file = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/results/route_planning.csv"
    headers = [
        "time",
        "car_id",
        "node_id",
        "arrival_time",
        "load",
        "travel_time",
        "early_arrival",
        "node",
        "type",
        "chain_order",
    ]
    # Open the CSV file in write mode ('w'), with newline='' to prevent extra blank lines
    with open(route_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(headers)

    NODE_ID_INDEX = 0
    ARRIVAL_TIME_INDEX = 1
    load_index = 2
    TRAVL_TIME_INDEX = 3
    A_INDEX = 4
    NODE_SEQ_INDEX = 5
    SECONDS_PER_DAY = 60 * 60 * 24
    SECONDS_PER_DAY = 60 * 60 * 24
    mpc_solver = MPC()
    travel_time, DF_TEST, DF_TRAIN, node_to_station, station_ids = load_source_data()
    data = DF_TEST
    data["come_time"] = data["pickup_time_since_midnight"] - 1 * 60 * 60  # 1 hour in seconds
    chain_id = 0
    # Define the 30-minute interval in seconds
    interval = 5 * 60  # 5 minutes in seconds
    MPC_interval = 20 * 60
    cur_time = 0
    CarNumber = 5
    Capacity = 8

    ###Initialize all variables
    K = list(range(1, CarNumber + 1))
    v_start_time = {k: cur_time for k in K}
    OD_dropoff_request = [[] for i in K]  # Initial
    test_data = data[data["chain_id"] == chain_id]
    depot_start_location = {k: 3607 for k in K}  # deport start locations
    unfinished_demand = []
    simulate(chain_id, 0)

    All_finished_request = []
    all_routes = None
    while cur_time <= SECONDS_PER_DAY:
        # Use only test_data in the condition if data and test_data refer to the same DataFrame
        df = test_data[
            (test_data["come_time"] > cur_time) & (test_data["come_time"] <= cur_time + interval)
        ].reset_index(drop=True)
        if len(df) > 0:  # If there is request in this slot
            req_data = df[
                [
                    "pickup_node_id",
                    "dropoff_node_id",
                    "pickup_time_since_midnight",
                    "dropoff_time_since_midnight",
                    "chain_order",
                ]
            ].values.tolist()
            data = req_data + unfinished_demand  # Update request with unfinished request
            cur_time_clock = time.time()
            # Getting MPC for rebalancing Check if there are idle vehicles
            Has_Idle, idle_car_ids, v_cur_location = get_idle_cars(cur_time, all_routes, depot_start_location)
            # At this point, check if need relloaction?
            if cur_time > 0 and cur_time % MPC_interval == 0 and Has_Idle:  # Current request > 0
                demandAttr, accTuple, daccTuple, edgeAttr, edges, T, beta = get_MPC_input(
                    c_t=cur_time,
                    v_cur_location=v_cur_location,
                    data=data,
                    OD_dropoff_request=OD_dropoff_request,
                    travel_time=travel_time,
                    node_to_station=node_to_station,
                    station_ids=station_ids,
                    idle_car_ids=idle_car_ids,
                )
                paxAction, rebAction = mpc_solver.MPC_run(
                    0, demandAttr, accTuple, daccTuple, edgeAttr, edges, T, beta
                )  # vehicle location,
                # print(f" empty relocation {rebAction}")
                # print(f"carry passenges {paxAction}")
                # paxAction = {(3607, 4638): 1.0}
                if len(paxAction) + len(rebAction) > 0:
                    all_routes = get_reallocation(
                        cur_time, paxAction, rebAction, all_routes, idle_car_ids, node_to_station, travel_time, edges
                    )
            ########## Given MPC Results, Reallocation idle vehilces
            else:
                routes, node_times, node_loads, nodetravel_times, optimality_gap = PDPTW_Plan(
                    CarNumber=CarNumber,
                    Capacity=Capacity,
                    request_df=data,
                    depot_start_location=depot_start_location,
                    OD_dropoff_request=OD_dropoff_request,
                    v_start_time=v_start_time,
                    TRAVEL_TIME_MATRIX=travel_time,
                    Park=False,
                )
            ################### Update Current State
            v_start_time, OD_dropoff_request, depot_start_location, finished_request, a, points, n, idle_v = (
                update_state(
                    cur_time + interval,
                    a,
                    points,
                    n,
                    depot_start_location,
                    all_routes,
                    idle_v,
                )
            )
            All_finished_request = All_finished_request + finished_request
            # Remove the leftdropoff request (which pickup already passed)
            unfinished_demand = [
                i for i in data if i[-1] not in All_finished_request + [i[-1] for i in OD_dropoff_request]
            ]
        else:
            print(f" at time {cur_time/60} minute --No new Results ")
        cur_time = cur_time + interval
