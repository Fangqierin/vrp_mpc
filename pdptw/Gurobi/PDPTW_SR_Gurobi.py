# Pickup and Delivery with Time Windows.

# Objective - minimize cost of routes.
# Number of requests - n
# Pickup - node i, Delviery - node n+i
# P = {1,2,...,n}
# D = {n+1,...,2n}
# N = P U D
# Request i consists of transporting d_i units from i to n + i. 
# l_i = d_i and l_{n+i} = - d_i
# K is the set of vehicles
# Each k in K has a N_k = P_k U D_k
# k also has a network G_k = (V_k,A_k)
# V_k = N_k U {o(k),d(k)} where o(k) is the origin depot and d(k) is the destination depot
# A_k is a subset of Vk x V_k which is the set of all feasible arcs
# Capacity of k is given by C_k and the travel time and cost between i,j os given by t_{ijk} and c_{ijk}
# Vehicle k leaves from the depot unloaded at time a_{o(k)}=b{o(k)}
# Feasible path starts at o(k) visits any node at most once and ends at d(k) while staying within G_k
# If a vehicle visits i it must happen during the time window [a_i,b_i] when servise s_i must begin. Vehicle can wait if it is too early.


# Three types of variables, x_ijk equal to 1 if (i,j) belonging to A_k is used by vehicle k, 0 otherwise. 
# T_ik specifying when vehicle k starts the service at node i in V_k.
# L_ik giving the load of vehicle k after service at node i has been completed.

# Objective it to minimize the cost (distance): min (sum over k)((sum over (i,j) in A _k) c_ijk*x_ijk)

# Constraint 1: (sum over k in K) ((sum over j in N_k U {d(k)}) x_ijk) = 1    for i in P             Service once   

# Constraint 2: (sum over j in N_k) x_{ijk} - (sum over j in N_k)x_{j(n+i)k} = 0 for all k in K and i in P_k            Service by same vehicle

# Constraint 3: (sum over j in P_k U {d(k)}) x_{o(k)jk} =1 for all k in K            Start from depot

# Constraint 4: (sum over i in N_k U {o(k)}) x_{ijk} - (sum over i in N_k U {d(k)}) x_{ijk} = 0  for all k in K and j in N_k            Maintain flow of vehicle

# Constraint 5: (sum over i in D_k U {o(k)}) x_{id(k)k} = 1 for all k in K       End at depot

# Constraint 6: x_ijk(T_ik _ s_i + t_ijk - T_jk) <= 0 for all k in K and (i,j) in A_k        Time Constraint

# Constraint 7: a_i <= T_ik <= b_i for all k in K and i in V_k          Time Window Constraint

# Constraint 8: T_ik + t_i,n+i,k <= T_n+i,k  for all k in K and i in P_k            Pickup and Delivery order

# Constraint 9: x_ijk(L_ik + l_j - L_jk)= 0 for all k in K and (i,j) in A_k     Load flow constraint

# Constraint 10: l_i <=L_i<= C_k for all k in K and i in P_k        Load at pickup

# Constraint 11: 0<=L_n+i,k <=C_k - l_i for all k in K and n+i in D_i           Load at delivery

# Constraint 12: L_o(k),k = 0 for all k in K       Load at origin

#x_ijk >=0 for all k in K and (i,j) in A_k

# x_ijk is binary for all k in K and (i,j)  in A_k



#### Computation of T_jk: If x_ijk =1 then T_jk = max{a_j,T_ik + s_i + t_ijk}

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import os
import argparse
import folium
import io
from PIL import Image
import load_data
import pandas as pd
import random
from collections import defaultdict
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def load_source_data(file_path=None):
    
    # Load travel time matrix
    file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.npy"
    # TRAVEL_TIME_MATRIX = pd.read_csv(file_path, header=None)
    # TRAVEL_TIME_MATRIX = TRAVEL_TIME_MATRIX.round().astype(np.int32)
    travel_time=np.load(file_path) 
    # # Load nodes
    # file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/nodes.csv"
    # nodes = pd.read_csv(file_path 
    # Load chains
    DF_TRAIN = pd.read_csv("/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/train_chains.csv")
    DF_TEST = pd.read_csv("/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/test_chains.csv") 
    return travel_time, DF_TEST, DF_TRAIN


def generate_example_data_new():
    # Example vehicle capacity
    # C = {1: 10, 2: 10}
    C = {1: 5, 2: 5}

    # Example coordinates for plotting
    points = {
        0: (0, 0),   # Depot start
        1: (1, 3),   # Pickup 1
        2: (4, 4),   # Pickup 2
        3: (5, 1),   # Pickup 3
        4: (2, 6),   # Pickup 4
        5: (7, 3),   # Pickup 5
        6: (3, 1),   # Delivery 1
        7: (6, 5),   # Delivery 2
        8: (7, 2),   # Delivery 3
        9: (1, 5),   # Delivery 4
        10: (8, 4),  # Delivery 5
        11: (0, 0),  # Depot end (same as start)
        14: (0, 5),  # Depot end (same as start) 
    }

    OD_points={
        12: (1, 9),  # The delivery node remaning!
        13: (1, 2),  # The delivery node remaning!
    }
    n = int((len(points)-2)/2)

    points.update(OD_points)
    # Set of pickup nodes
    P = list(range(1, n + 1))
    # Set of delivery nodes
    D = list(range(n + 1, 2 * n + 1))
    depot_start=0
    depot_end=12
    ########### ---- FQ I change 
    OD=[11,13]
    depot_start_dict={1:14, 2:0}
    depot_end_dict={1:11, 2:12}
    v_load={1:2, 2:1}
    OD_dict={1:[11,13], 2:[]} 
    # All nodes
    N = P + D
    # all_node=N + [depot_start,depot_end]+list({item for sublist in OD_dict.values() for item in sublist})
    all_node=list(points.keys())
    V=all_node
    # Calculate travel times and costs using Euclidean distance
    tau = {
        (i, j, k): euclidean_distance(points[i], points[j])
        for i in all_node
        for j in all_node
        if i != j
        for k in C.keys()
    }
    c = {(i, j, k): tau[i, j, k] for i, j, k in tau.keys()}  # Assuming cost is proportional to travel time 
    # # Example time windows 4  
    # FQ added for reaming departures 
    a_OD={12: 0, 13: 60}
    b_OD={12: 500, 13: 120}
    a = {0: 0, 1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60, 7: 90, 8: 80, 9: 80, 10: 100, 11: 100, 14:0}
    b = {0: 200, 1: 50, 2: 60, 3: 70, 4: 80, 5: 90, 6: 100, 7: 130, 8: 120, 9: 120, 10: 140, 11: 120, 14:200}
    a.update(a_OD)
    b.update(b_OD)
    # Example service times
    s = {i: 1 for i in all_node}  
    # Example load to be picked up or delivered at each node
    # ell = {1: 3, 2: 2, 3: 4, 4: 1, 5: 2}
    # ell.update({6: -3, 7: -2, 8: -4, 9: -1, 10: -2})
    ell_OD={12:0, 13:-1, 11:-1}
    ell = {1: 3, 2: 3, 3: 4, 4: 2, 5: 3}
    ell.update({6: -3, 7: -3, 8: -4, 9: -2, 10: -3})
    ell.update(ell_OD)
    # Set of vehicles
    K = list(C.keys()) 
    print("The number of nodes is n = ", n)
    return n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end,OD, depot_start_dict, depot_end_dict, v_load, OD_dict, V


def generate_CARTA_data(t=0, C=None, depot_start_location=None, OD_dict=None, OD_dropoff_request=None, travel_time_matrix=None): #C: car capacity; depot_start_dict: depot of each car 
    # Example vehicle capacity
    # C = {1: 10, 2: 10} 
    CarNumber=5
    Capacity=8
    # Load request data
    data = pd.DataFrame({
        'pickup_node_id': [229, 5346, 1905, 5139, 8879, 6361, 2770],
        'dropoff_node_id': [2898, 2898, 10243, 4446, 1446, 2887, 3603],
        'pickup_time_since_midnight': [18900, 19800, 23400, 26100, 26400, 27000, 27030],
        'dropoff_time_since_midnight': [19169, 19878, 23513, 26634, 27324, 27221, 27342],
        'chain_id': [0, 0, 0, 0, 0, 0, 0],
        'chain_order': [0, 1, 2, 3, 4, 5, 6]
    })
    # Initialize dictionaries for points, a (earliest arrival), and b (latest arrival)
    points = {}
    a = {}
    b = {} 
    P=[]
    D=[]
    # Loop through the data and set values for pickup and corresponding drop-off nodes
    i=1
    n=len(data)
    nodes_id=[]
    
    for idx, row in data.iterrows():
        pickup_node_id = row['pickup_node_id']
        dropoff_node_id = row['dropoff_node_id']
        pickup_time = row['pickup_time_since_midnight']
        dropoff_time = row['dropoff_time_since_midnight']
        # Placeholder coordinates for pickup and dropoff nodes
        points[idx] = pickup_node_id         # Assign coordinates for the pickup node
        points[idx+n] =   dropoff_node_id   # Assign coordinates for the dropoff node
        
        # Pickup time windows
        a[idx] = pickup_time - 15 * 60        # Early arrival time
        b[idx] = pickup_time + 15 * 60        # Late arrival time
        
        # Dropoff time windows
        a[idx+n] = dropoff_time - 15 * 60      # Early arrival time
        b[idx+n] = dropoff_time + 15 * 60      # Late arrival time 
        nodes_id.append(pickup_node_id)
        nodes_id.append(dropoff_node_id)

    P=list(range(0, n))
    D=list(range(n, n*2))
    ell={i: 1 for i in P}
    ell.update({i: -1 for i in D}) # Update load update 
    ######################## Getting Vehicle information 
    V=list(range(1, CarNumber+1))
    if C is None:
        C = {i: 8 for i in range(1, CarNumber+1)}   # 5 cars, 8 capacity 
    K=list(C.keys()) 
    if  depot_start_location is None and t==0: 
        depot_start_location={k: 3607 for k in K}   # 5 cars, 8 capacity 
        # ell.update({n*2+k-1: 0 for k in K})
    depot_start_points={n*2+k-1:  depot_start_location[k] for k in K}
    depot_start_dict={k: n*2+k-1 for k in K}
    depot_end_points={n*2+len(C): 3607}
    # ell.update({n*2+len(C): 0})

    depot_end=n*2+len(C) 
    points.update(depot_end_points)
    points.update(depot_start_points)
   
    ##################################################################################
    #### Update for unfinished request (dropoff request )
    if OD_dropoff_request is None: 
        #Create a fake request 
        OD_dropoff_request=[[random.sample(nodes_id, 1)[0], random.sample(range(int(np.mean(list(b.values()))),max(list(b.values()))), 1)[0], -1, random.sample(V, 1)[0]] for i in range(1, random.randint(2, 10))]   # 5 cars, 8 capacity [node_id, _dropoff_time]
    # give a fake for debugging: 
    # OD_dropoff_request=[[6361, 26384, -1, 4], [4446, 26875, -1, 2], [8879, 26934, -1, 5], [2898, 26026, -1, 1], [6361, 27698, -1, 5]]
    # OD_dropoff_request=[[] for i in range(5)]
    cur_idx=len(points)
    i=0
    OD=[]
    OD_dict={i:[] for i in V}
    v_load={i: 0 for i in V}
    for values in OD_dropoff_request:
        if len(values)>0:
            node_id, dropoff_time, el_load, v=values
            points.update({cur_idx+i: node_id})
            a.update({cur_idx+i:dropoff_time-15*60})
            b.update({cur_idx+i:dropoff_time+15*60})
            OD.append(cur_idx+i)
            ell.update({cur_idx+i: el_load})
            OD_dict[v].append(cur_idx+i)
            v_load[v]=v_load[v]-el_load
            i=i+1
    N = P + D
    all_node=list(points.keys())
    V=all_node
    # Calculate travel times and costs using Euclidean distance
    tau = {
        (i, j, k): travel_time_matrix[points[i], points[j]]
        for i in all_node
        for j in all_node
        if i != j
        for k in C.keys()
    }
    ############################ Update for depots nodes pickup
    a.update({n*2+k-1: t for k in K})
    b.update({n*2+k-1: max(b.values()) for k in K})
    a.update({n*2+len(C): t})
    b.update({n*2+len(C): max(b.values())+max(tau.values())})
    c = {(i, j, k): tau[i, j, k] for i, j, k in tau.keys()}  # Assuming cost is proportional to travel time 
    s = {i: 1 for i in all_node}   
    # Set of vehicles
    print("The number of nodes is n = ", n)
    return n, P, D, C, tau, c, a, b, s, ell, K, points, depot_end,OD, depot_start_dict, v_load, OD_dict, V, OD_dropoff_request


def pdptw_solver(n, P, D, C, tau, c, a, b, s, ell, K, depot_end, OD, depot_start_dict=None, depot_end_dict=None, v_load=None, OD_dict=None, V=None):
    model = gp.Model("PDPTW")

    model.Params.MIPFocus = 1
    model.setParam('Heuristics', 0.5)
    model.setParam('TimeLimit', 10*3600)
    # model.setParam('TimeLimit', 1000)
    model.Params.CliqueCuts = 2

    # Sets
    N = P + D  # all nodes
    # if V is None:
    #     V = N + list(set([ depot_end]))+OD  # all nodes including depots

    A = [(i, j) for i in V for j in V if i != j]  # all arcs including depots
    M = 100000  # Big M constant
    depot_end_set=[depot_end]
    epsilon = 1e-5  # Small value
    # penalty
    penalty = 1e7

    # Ensure tau and c dictionaries are populated for all necessary keys
    for i, j in A:
        for k in K:
            if (i, j, k) not in tau:
                tau[(i, j, k)] = 1  # Default value or handle as needed
            if (i, j, k) not in c:
                c[(i, j, k)] = 1  # Default value or handle as needed
    # Decision variables
    x = model.addVars(A, K, vtype=GRB.BINARY, name="x")
    t = model.addVars(V, K, vtype=GRB.CONTINUOUS, name="t")
    y = model.addVars(V, K, vtype=GRB.CONTINUOUS, name="y")
    r = model.addVars(P, vtype=GRB.BINARY, name="r")

    # Objective: Minimize the total cost
    model.setObjective(
        gp.quicksum(c[i, j, k] * x[i, j, k] for i, j in A for k in K) + penalty * (n-gp.quicksum(r[i] for i in P)),
        GRB.MINIMIZE
    )

    # Constraint 1: Each node is visited not more once
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N+OD if i != j) <= r[i] for i in P),
        name="ServiceOnce",
    )

    # Constraint 1a: Each pickup node is visited not more once
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N+OD if i != j) == r[i] for i in P),
        name="pickup_visited_once",
    )
    # Constraint 1b: Each delivery node is visited as many times as the corresponding pickup node
    model.addConstrs(
        (
            gp.quicksum(x[i, j, k] for k in K for j in N +depot_end_set+OD if i != j) == r[i - n] 
            for i in D
        ),
        name="delivery_visited_once",
    )
    #################################
    #Constraint add  FQ:
    # if OD_dict is not None:
    for v, _OD in OD_dict.items():
        # if depot_start_dict is not None:
        _depot_start=depot_start_dict[v]
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for i in N+[_depot_start] if j!=i for k in [v]) == 1 for j in _OD), name="Pass OD 1",)
        #Constraint 5: Pass 11    ############# FQ  remove that one
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for j in N+depot_end_set if j!=i for k in [v]) == 1 for i in _OD), name="Pass OD",)
            # else:
            #     model.addConstrs(
            #     (gp.quicksum(x[i, j, k] for i in N+[depot_start] if j!=i for k in [v]) == 1 for j in _OD), name="Pass OD 1",)
            #     #Constraint 5: Pass 11    ############# FQ  remove that one
            #     model.addConstrs(
            #         (gp.quicksum(x[i, j, k] for j in N+depot_end_set if j!=i for k in [v]) == 1 for i in _OD), name="Pass OD",)

    # Constraint 2: Service by the same vehicle ############### FQ add this
    # model.addConstrs(
    #     (
    #       x[i, 11, k] == gp.quicksum(x[j, i, k] for j in N+[depot_start] if j != i )
    #         for k in K
    #         for i in N+[depot_start]
    #     ),
    #     name="Sever 11 with the same vehicle",
    # ) 
#########################################################################
    # Constraint 2: Service by the same vehicle
    if OD_dict is not None:
        for v, _OD in OD_dict.items():
            model.addConstrs(
                (
                    gp.quicksum(x[i, j, k] for j in N+_OD if i != j)
                    == gp.quicksum(x[j, n + i, k] for j in N+_OD if j != n + i)
                    for k in [v]
                    for i in P
                ),
                name="SameVehicle",
            )
    else:
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] for j in N+OD if i != j)
                == gp.quicksum(x[j, n + i, k] for j in N+OD if j != n + i)
                for k in K
                for i in P
            ),
            name="SameVehicle",
        )
    # Constraint 3: Start from depot # FQ try
    # if depot_start_dict is None:
    #     model.addConstrs(
    #         (gp.quicksum(x[depot_start, j, k] for j in P + depot_end_set+OD) == 1 for k in K), name="StartDepot",)
    # else:
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
        (gp.quicksum(x[_depot_start, j, k] for j in P + depot_end_set+OD) == 1 for k in [v]), name="StartDepot",)
        model.addConstrs(
        (gp.quicksum(x[_depot_start, j, k] for j in D) == 0 for k in [v]), name="StartDepot",)
    # Constraint 4: Maintain flow of vehicle
    # if depot_start_dict is None:
    #     model.addConstrs(
    #         (
    #             gp.quicksum(x[i, j, k] for i in N + [depot_start]+OD if i != j)
    #             == gp.quicksum(x[j, i, k] for i in N + depot_end_set+OD if j != i)
    #             for k in K
    #             for j in N+OD
    #         ),
    #         name="MaintainFlow",
    #     )
    # else:
    for v, _depot_start in depot_start_dict.items():
        _OD=OD_dict[v]
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] for i in N + [_depot_start]+_OD if i != j)
                == gp.quicksum(x[j, i, k] for i in N + depot_end_set+_OD if j != i)
                for k in [v]
                for j in N+_OD
            ),
            name="MaintainFlow",
        )
    #Constraint 5: End at depot    ############# FQ  remove that one
    # if depot_start_dict is None: 
    #     model.addConstrs(
    #         (gp.quicksum(x[i, depot_end, k] for i in D+OD + [depot_start]) == 1 for k in K), name="EndDepot",)
    # else:
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
            (gp.quicksum(x[i, depot_end, k] for i in D+OD + [_depot_start]) == 1 for k in [v]), name="EndDepot",)
    ###################################################################################
    # Constraint 6: Time constraint
    model.addConstrs(
        (t[i, k] + s[i] + tau[i, j, k] - t[j, k] <= M * (1 - x[i, j, k])
         for k in K for i, j in A),
        name="TimeConstraint",
    )
    # Constraint 7: Time window constraint (split into two constraints)
    model.addConstrs(
        (t[i, k]>=a[i] for k in K for i in V if i not in depot_end_set), name="TimeWindow_Lower"
    )
    model.addConstrs(
        (t[i, k] <= b[i] for k in K for i in V if i not in depot_end_set), name="TimeWindow_Upper"
    )
    # model.addConstrs(
    #     (a[i] <= t[i, k] for k in K for i in V ), name="TimeWindow_Lower"
    # )
    # model.addConstrs(
    #     (t[i, k] <= b[i] for k in K for i in V), name="TimeWindow_Upper"
    # )
    # Constraint 8: Pickup and delivery order
    model.addConstrs(
        (t[i, k] + tau[i, n + i, k] <= t[n + i, k]
         for k in K for i in P),
        name="PickupDeliveryOrder",
    )

    # Constraint 9: Load flow constraint (linearized)
    # for v, _depot_start in depot_start_dict.items():
    #     _OD=OD_dict[v]
    #     model.addConstrs(
    #         (
    #             y[i, k] + ell[j] - y[j, k] <= M * (1 - x[i, j, k])
    #             for k in [v]
    #             for i, j in A
    #             if i in [_depot_start]+N+_OD and j in N+OD
    #         ),
    #         name="LoadFlow_Upper",
    #     )
    #     model.addConstrs(
    #         (
    #             y[i, k] + ell[j] - y[j, k] >= -M * (1 - x[i, j, k])
    #             for k in [v]
    #             for i, j in A
    #             if i in [_depot_start]+N+_OD and j in N+OD 
    #         ),
    #         name="LoadFlow_Lower",
    #     )
    model.addConstrs(
            (
                y[i, k] + ell[j] - y[j, k] <= M * (1 - x[i, j, k])
                for k in K
                for i, j in A
                if i in N+OD and j in N+OD
            ),
            name="LoadFlow_Upper",
        )
    model.addConstrs(
        (
            y[i, k] + ell[j] - y[j, k] >= -M * (1 - x[i, j, k])
            for k in K
            for i, j in A
            if i in N+OD and j in N+OD 
        ),
        name="LoadFlow_Lower",
    )

    # Constraint 10: Load at pickup (split into two constraints)
    model.addConstrs(
        (ell[i] <= y[i, k] for k in K for i in P), name="LoadPickup_Lower"
)
    model.addConstrs(
        (y[i, k] <= C[k] for k in K for i in P), name="LoadPickup_Upper")

    # Constraint 11: Load at delivery (split into two constraints) ### FQ change this
    model.addConstrs(
        (0 <= y[i, k] for k in K for i in D+OD),
        name="LoadDelivery_Lower",
    )
    model.addConstrs(
        (y[i, k] <= C[k] for k in K for i in D+OD),
        name="LoadDelivery_Upper",
    )
    # Constraint 12: Load at origin   #################### Load at orgin FQ change it
    # if v_load is not None: 
    for v, load in v_load.items():
        # if depot_start_dict is not None:
        _depot_start=depot_start_dict[v]
        model.addConstrs((y[_depot_start, k] == load for k in [v]), name="LoadOrigin")
            # else:
            #     model.addConstrs((y[depot_start, k] == load for k in [v]), name="LoadOrigin")

    # else:
    #     model.addConstrs((y[depot_start, k] == 0 for k in K), name="LoadOrigin")

    # if depot_start_dict is None:
    #     model.addConstrs(
    #     (
    #         t[depot_start,k] == 0 for k in K), name="StartingTimeDepot", )
    # else:
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
        (
        t[_depot_start,k] == 0 for k in [v]), name="StartingTimeDepot", )

    model.optimize() 
    if model.SolCount > 0:
        solution = model.getAttr("x", x)
        time_solution = model.getAttr("x", t)
        Load_solution = model.getAttr("x", y)
        # Function to extract routes and break information
        def extract_route(k):
            route = []
            current_node = depot_start_dict[k]
            node_times = []
            node_loads = []
            nodetravel_times = []
            while True:
                route.append(current_node)
                node_times.append(time_solution[current_node, k])
                node_loads.append(Load_solution[current_node, k])
                next_node = [
                    j
                    for j in V
                    if (current_node, j, k) in solution
                    and solution[current_node, j, k] > 0.5
                ]
                if (current_node, next_node[0], k) in tau:
                    nodetravel_times.append(tau[current_node, next_node[0], k])
                if  next_node[0] == depot_end: # or len(next_node) == 0:
                    route.append(next_node[0])
                    node_times.append(time_solution[depot_end, k])
                    break
                current_node = next_node[0]
            return route, node_times, nodetravel_times, node_loads

        for k in K:
            route, node_times, nodetravel_times, node_loads = extract_route(k)
            print(f"\nRoute for vehicle {k}:")
            for i in range(len(route) - 1):
                if route[i] in ell.keys():
                    print(f"Node {route[i]}: Total Time {node_times[i]:.2f}, Travel Time {nodetravel_times[i]:.2f}, Time Window [{a[route[i]]}, {b[route[i]]}], Load {node_loads[i]:.2f} load update {ell[route[i]]}")
                else:
                    print(f"Node {route[i]}: Total Time {node_times[i]:.2f}, Travel Time {nodetravel_times[i]:.2f}, Time Window [{a[route[i]]}, {b[route[i]]}], Load {node_loads[i]:.2f}")
                print(f" -> {route[i + 1]}") 
            print(f"Node {route[-1]}: Time {node_times[-1]:.2f}, Total Time {int(node_times[i]+s[i])}")

        routes = {k: extract_route(k)[0] for k in K}
        return routes, model.MIPGap
    
    else:
        print("No optimal solution found.")
        return None


def save_routes(year, month, day, routes, optimality_gap, SOLUTIONS_DIR):

    filename = f"{SOLUTIONS_DIR}/{year}_{month}_{day}/routes.txt"
    with open(filename, "w") as file:
        file.write(f"Date: {year}_{month}_{day}\n")
        for k, route in routes.items():
            file.write(f"\nRoute for vehicle {k}:\n")
            file.write(" -> ".join(map(str, route)) + "\n")
        file.write(f"\nOptimality gap: {optimality_gap:.2f}%\n")


def main(day, month, year):

    # date = [2022,9,5]
    # year, month, day = date[0], date[1], date[2]
    #
    #n, P, D, C, tau, omega, c, c_omega, a, b, s, ell, K, points, depot_start, depot_end = load_data.get_data_no_break(2022, 9, 5)
    # n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end = load_data.get_data_no_break(2022, 9, 5)
    # n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end, OD, depot_start_dict, depot_end_dict, v_load,OD_dict , V = generate_example_data_new()
    file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.csv"
    travel_time, DF_TEST, DF_TRAIN=load_source_data(file_path)
    n, P, D, C, tau, c, a, b, s, ell, K, points, depot_end,OD, depot_start_dict, v_load, OD_dict, V, OD_dropoff_request=generate_CARTA_data(t=0, C=None, depot_start_location=None, OD_dict=None, OD_dropoff_request=None, travel_time_matrix=travel_time)
    # n_, P_, D_, C_, tau_, c_, a_, b_, s_, ell_, K_, points_, depot_end_,OD_, depot_start_dict_, v_load_, OD_dict_, V_, OD_dropoff_request=generate_CARTA_data(t=0, C=None, depot_start_location=None, OD_dict=None, OD_dropoff_request=None, travel_time_matrix=travel_time)

    year = 2022
    month = 9
    day = 5
    RESULTS_DIR = "../../results"
    SOLUTIONS_DIR = f"{RESULTS_DIR}/PDPTW_SR_Gurobi_solutions"

    # Save details to text file in solutions directory
    if not os.path.exists(SOLUTIONS_DIR):
        os.makedirs(SOLUTIONS_DIR)
    if not os.path.exists(f"{SOLUTIONS_DIR}/{year}_{month}_{day}"):
        os.makedirs(f"{SOLUTIONS_DIR}/{year}_{month}_{day}")

    filename = f"{SOLUTIONS_DIR}/{year}_{month}_{day}/data.txt"
    with open(filename, "w") as file:
        file.write(f"Date: {year}_{month}_{day}\n")
        file.write(f"Number of requests: {n}\n")
        file.write(f"Pickup nodes: {P}\n")
        file.write(f"Delivery nodes: {D}\n")
        file.write(f"Vehicle capacity: {C}\n")
        for i in P:   # FQ change it to 0 to n For Pickup Nodes 
            file.write(f"Load at pickup node {i}: {ell[i]}, Time window: [{a[i]},{b[i]}], Service time: {s[i]}, Coordinates: {points[i]}\n")
        
        for i in D:
            file.write(f"Load at delivery node {i}: {ell[i]}, Time window: [{a[i]},{b[i]}], Service time: {s[i]}, Coordinates: {points[i]}\n")

    
    routes, optimality_gap = pdptw_solver(
       n, P, D, C, tau, c, a, b, s, ell, K, depot_end,OD, v_load=v_load, depot_start_dict=depot_start_dict, OD_dict=OD_dict, V=V
    )
# def pdptw_solver(n, P, D, C, tau, c, a, b, s, ell, K, depot_end, OD, depot_start_dict=None, depot_end_dict=None, v_load=None, OD_dict=None, V=None):

    print(routes)
    print(OD_dict)
    print(P)
    print(D)
    # print(OD_dropoff_request)
    # plot_routes(routes, points, year, month, day, SOLUTIONS_DIR)

    save_routes(year, month, day, routes, optimality_gap, SOLUTIONS_DIR) 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a Parquet file and filter by routes.")
    
    parser.add_argument("--d", type=int, help="The day (1-31)", default=[])
    parser.add_argument("--m", type=int, help="The month (1-12)", default=[])
    parser.add_argument("--y", type=int, help="The year", default=[]) 
    args = parser.parse_args() 
    route_number = args.d
    print(args.d, args.m, args.y)
    main(args.d, args.m, args.y)
