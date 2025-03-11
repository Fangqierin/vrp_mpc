import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import random
from collections import defaultdict
from pathlib import Path
from MPC_relocation import MPC


def pdptw_solver(
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
    depot_start_dict=None,
    depot_end_dict=None,
    v_load=None,
    OD_dict=None,
    V=None,
    v_start_time=None,
):
    model = gp.Model("PDPTW")
    model.setParam("OutputFlag", 0)

    model.Params.MIPFocus = 1
    model.setParam("Heuristics", 0.7)
    model.setParam("TimeLimit", 30)
    model.setParam("PoolSearchMode", 2)  # Continue searching for feasible solutions after time limit is reached

    # model.setParam('TimeLimit', 1000)
    model.Params.CliqueCuts = 2

    # Sets
    N = P + D  # all nodes
    # if V is None:
    #     V = N + list(set([ depot_end]))+OD  # all nodes including depots
    A = [(i, j) for i in V for j in V if i != j]  # all arcs including depots
    M = 100000  # Big M constant
    depot_end_set = [depot_end]
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
        gp.quicksum(c[i, j, k] * x[i, j, k] for i, j in A for k in K) + penalty * (n - gp.quicksum(r[i] for i in P)),
        GRB.MINIMIZE,
    )

    # Constraint 1: Each node is visited not more once
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N + OD if i != j) <= r[i] for i in P),
        name="ServiceOnce",
    )

    # Constraint 1a: Each pickup node is visited not more once
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N + OD if i != j) == r[i] for i in P),
        name="pickup_visited_once",
    )
    # Constraint 1b: Each delivery node is visited as many times as the corresponding pickup node
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N + depot_end_set + OD if i != j) == r[i - n] for i in D),
        name="delivery_visited_once",
    )
    ##### FQ add it same car visit both Pick up node  and delivery node
    for k in K:
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for k in K for j in N + depot_end_set + OD if i != j) == r[i - n] for i in D),
            name="delivery_visited_once",
        )
    #################################
    # if OD_dict is not None:
    for v, _OD in OD_dict.items():
        # if depot_start_dict is not None:
        _depot_start = depot_start_dict[v]
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for i in N + [_depot_start] + _OD if j != i for k in [v]) == 1 for j in _OD),
            name="Pass OD 1",
        )
        # Constraint 5: Pass 11    ############# FQ  remove that one
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for j in N + depot_end_set + _OD if j != i for k in [v]) == 1 for i in _OD),
            name="Pass OD",
        )
    #########################################################################
    # Constraint 2: Service by the same vehicle
    if OD_dict is not None:
        for v, _OD in OD_dict.items():
            model.addConstrs(
                (
                    gp.quicksum(x[i, j, k] for j in N + _OD if i != j)
                    == gp.quicksum(x[j, n + i, k] for j in N + _OD if j != n + i)
                    for k in [v]
                    for i in P
                ),
                name="SameVehicle",
            )
    else:
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] for j in N + OD if i != j)
                == gp.quicksum(x[j, n + i, k] for j in N + OD if j != n + i)
                for k in K
                for i in P
            ),
            name="SameVehicle",
        )
    # Constraint 3: Start from depot # FQ try
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
            (gp.quicksum(x[_depot_start, j, k] for j in P + depot_end_set + OD) == 1 for k in [v]),
            name="StartDepot",
        )
        model.addConstrs(
            (gp.quicksum(x[_depot_start, j, k] for j in D) == 0 for k in [v]),
            name="StartDepot",
        )
    # Constraint 4: Maintain flow of vehicle
    for v, _depot_start in depot_start_dict.items():
        _OD = OD_dict[v]
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] for i in N + [_depot_start] + _OD if i != j)
                == gp.quicksum(x[j, i, k] for i in N + depot_end_set + _OD if j != i)
                for k in [v]
                for j in N + _OD
            ),
            name="MaintainFlow",
        )
    # Constraint 5: End at depot    ############# FQ  remove that one
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
            (gp.quicksum(x[i, depot_end, k] for i in D + OD + [_depot_start]) == 1 for k in [v]),
            name="EndDepot",
        )
    ###################################################################################
    # Constraint 6: Time constraint
    model.addConstrs(
        (t[i, k] + s[i] + tau[i, j, k] - t[j, k] <= M * (1 - x[i, j, k]) for k in K for i, j in A),
        name="TimeConstraint",
    )
    # Constraint 7: Time window constraint (split into two constraints)
    model.addConstrs((t[i, k] >= a[i] for k in K for i in V if i not in depot_end_set), name="TimeWindow_Lower")
    model.addConstrs((t[i, k] <= b[i] for k in K for i in V if i not in depot_end_set), name="TimeWindow_Upper")
    # Constraint 8: Pickup and delivery order
    model.addConstrs(
        (t[i, k] + tau[i, n + i, k] <= t[n + i, k] for k in K for i in P),
        name="PickupDeliveryOrder",
    )
    # Constraint 9: workload constraint
    model.addConstrs(
        (y[i, k] + ell[j] - y[j, k] <= M * (1 - x[i, j, k]) for k in K for i, j in A if i in N + OD and j in N + OD),
        name="LoadFlow_Upper",
    )
    model.addConstrs(
        (y[i, k] + ell[j] - y[j, k] >= -M * (1 - x[i, j, k]) for k in K for i, j in A if i in N + OD and j in N + OD),
        name="LoadFlow_Lower",
    )
    # Constraint 10: Load at pickup (split into two constraints)
    model.addConstrs((ell[i] <= y[i, k] for k in K for i in P), name="LoadPickup_Lower")
    model.addConstrs((y[i, k] <= C[k] for k in K for i in P), name="LoadPickup_Upper")

    # Constraint 11: Load at delivery (split into two constraints) ### FQ change this
    model.addConstrs(
        (0 <= y[i, k] for k in K for i in D + OD),
        name="LoadDelivery_Lower",
    )
    model.addConstrs(
        (y[i, k] <= C[k] for k in K for i in D + OD),
        name="LoadDelivery_Upper",
    )
    # Constraint 12: Load at origin   #################### Load at orgin FQ change it
    for v, load in v_load.items():
        # if depot_start_dict is not None:
        _depot_start = depot_start_dict[v]
        model.addConstrs((y[_depot_start, k] == load for k in [v]), name="LoadOrigin")

    for v, _depot_start in depot_start_dict.items():
        _start_time = v_start_time[v]
        model.addConstrs(
            (t[_depot_start, k] == _start_time for k in [v]),
            name="StartingTimeDepot",
        )

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
                next_node = [j for j in V if (current_node, j, k) in solution and solution[current_node, j, k] > 0.5]
                if (current_node, next_node[0], k) in tau:
                    nodetravel_times.append(tau[current_node, next_node[0], k])
                if next_node[0] == depot_end:  # or len(next_node) == 0:
                    route.append(next_node[0])
                    node_times.append(time_solution[depot_end, k])
                    break
                current_node = next_node[0]
            return route, node_times, nodetravel_times, node_loads

        routes = {k: extract_route(k)[0] for k in K}
        node_times = {k: extract_route(k)[1] for k in K}
        node_loads = {k: extract_route(k)[3] for k in K}
        nodetravel_times = {k: extract_route(k)[2] for k in K}
        return routes, node_times, node_loads, nodetravel_times, model.MIPGap
    else:
        print("No optimal solution found.")
        return None


def pdptw_solver_station(
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
    depot_start_dict=None,
    depot_end_dict=None,
    v_load=None,
    OD_dict=None,
    V=None,
    v_start_time=None,
):
    model = gp.Model("PDPTW")
    model.setParam("OutputFlag", 0)

    model.Params.MIPFocus = 1
    model.setParam("Heuristics", 0.7)
    model.setParam("TimeLimit", 30)
    model.setParam("PoolSearchMode", 2)  # Continue searching for feasible solutions after time limit is reached
    # model.setParam('TimeLimit', 1000)
    model.Params.CliqueCuts = 2
    # Sets
    N = P + D  # all nodes
    # if V is None:
    #     V = N + list(set([ depot_end]))+OD  # all nodes including depots
    A = [(i, j) for i in V for j in V if i != j]  # all arcs including depots
    M = 100000  # Big M constant
    depot_end_set = [depot_end]
    epsilon = 1e-5  # Small value
    delta = 15 * 60
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
    beta = model.addVars(A, K, vtype=GRB.BINARY, name="r")
    eta = model.addVars(A, K, vtype=GRB.BINARY, name="r")

    # Objective: Minimize the total cost
    model.setObjective(
        gp.quicksum(c[i, j, k] * x[i, j, k] for i, j in A for k in K) + penalty * (n - gp.quicksum(r[i] for i in P)),
        GRB.MINIMIZE,
    )

    # Constraint 1: Each node is visited not more once
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N + OD if i != j) <= r[i] for i in P),
        name="ServiceOnce",
    )
    # Constraint 1a: Each pickup node is visited not more once
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N + OD if i != j) == r[i] for i in P),
        name="pickup_visited_once",
    )
    # Constraint 1b: Each delivery node is visited as many times as the corresponding pickup node
    model.addConstrs(
        (gp.quicksum(x[i, j, k] for k in K for j in N + depot_end_set + OD if i != j) == r[i - n] for i in D),
        name="delivery_visited_once",
    )
    ##### FQ add it same car visit both Pick up node  and delivery node
    for k in K:
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for k in K for j in N + depot_end_set + OD if i != j) == r[i - n] for i in D),
            name="delivery_visited_once",
        )
    #################################
    # if OD_dict is not None:
    for v, _OD in OD_dict.items():
        # if depot_start_dict is not None:
        _depot_start = depot_start_dict[v]
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for i in N + [_depot_start] + _OD if j != i for k in [v]) == 1 for j in _OD),
            name="Pass OD 1",
        )
        # Constraint 5: Pass 11    ############# FQ  remove that one
        model.addConstrs(
            (gp.quicksum(x[i, j, k] for j in N + depot_end_set + _OD if j != i for k in [v]) == 1 for i in _OD),
            name="Pass OD",
        )
    #########################################################################
    # Constraint 2: Service by the same vehicle
    if OD_dict is not None:
        for v, _OD in OD_dict.items():
            model.addConstrs(
                (
                    gp.quicksum(x[i, j, k] for j in N + _OD if i != j)
                    == gp.quicksum(x[j, n + i, k] for j in N + _OD if j != n + i)
                    for k in [v]
                    for i in P
                ),
                name="SameVehicle",
            )
    else:
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] for j in N + OD if i != j)
                == gp.quicksum(x[j, n + i, k] for j in N + OD if j != n + i)
                for k in K
                for i in P
            ),
            name="SameVehicle",
        )
    # Constraint 3: Start from depot # FQ try
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
            (gp.quicksum(x[_depot_start, j, k] for j in P + depot_end_set + OD) == 1 for k in [v]),
            name="StartDepot",
        )
        model.addConstrs(
            (gp.quicksum(x[_depot_start, j, k] for j in D) == 0 for k in [v]),
            name="StartDepot",
        )
    # Constraint 4: Maintain flow of vehicle
    for v, _depot_start in depot_start_dict.items():
        _OD = OD_dict[v]
        model.addConstrs(
            (
                gp.quicksum(x[i, j, k] for i in N + [_depot_start] + _OD if i != j)
                == gp.quicksum(x[j, i, k] for i in N + depot_end_set + _OD if j != i)
                for k in [v]
                for j in N + _OD
            ),
            name="MaintainFlow",
        )
    # Constraint 5: End at depot    ############# FQ  remove that one
    for v, _depot_start in depot_start_dict.items():
        model.addConstrs(
            (gp.quicksum(x[i, depot_end, k] for i in D + OD + [_depot_start]) == 1 for k in [v]),
            name="EndDepot",
        )
    ###################################################################################
    # Constraint 6: Time constraint #-----------FQ add beta for stationing
    model.addConstrs(
        (
            t[i, k] + s[i] + tau[i, j, k] + beta[i, j, k] * tau_s[i, j, k] - t[j, k] <= M * (1 - x[i, j, k])
            for k in K
            for i, j in A
        ),
        name="TimeConstraint",
    )
    # Constraint 7: Time window constraint (split into two constraints)
    model.addConstrs((t[i, k] >= a[i] for k in K for i in V if i not in depot_end_set), name="TimeWindow_Lower")
    model.addConstrs((t[i, k] <= b[i] for k in K for i in V if i not in depot_end_set), name="TimeWindow_Upper")

    # Constraint 8: Pickup and delivery order # FQ add beta for stationing
    model.addConstrs(
        (t[i, k] + tau[i, n + i, k] + beta[i, n + i, k] * tau_s[i, n + i, k] <= t[n + i, k] for k in K for i in P),
        name="PickupDeliveryOrder",
    )
    # Constraint 9: workload constraint
    model.addConstrs(
        (y[i, k] + ell[j] - y[j, k] <= M * (1 - x[i, j, k]) for k in K for i, j in A if i in N + OD and j in N + OD),
        name="LoadFlow_Upper",
    )
    model.addConstrs(
        (y[i, k] + ell[j] - y[j, k] >= -M * (1 - x[i, j, k]) for k in K for i, j in A if i in N + OD and j in N + OD),
        name="LoadFlow_Lower",
    )
    # Constraint 10: Load at pickup (split into two constraints)
    model.addConstrs((ell[i] <= y[i, k] for k in K for i in P), name="LoadPickup_Lower")
    model.addConstrs((y[i, k] <= C[k] for k in K for i in P), name="LoadPickup_Upper")

    # Constraint 11: Load at delivery (split into two constraints) ### FQ change this
    model.addConstrs(
        (0 <= y[i, k] for k in K for i in D + OD),
        name="LoadDelivery_Lower",
    )
    model.addConstrs(
        (y[i, k] <= C[k] for k in K for i in D + OD),
        name="LoadDelivery_Upper",
    )
    # Constraint 12: Load at origin   #################### Load at orgin FQ change it
    for v, load in v_load.items():
        # if depot_start_dict is not None:
        _depot_start = depot_start_dict[v]
        model.addConstrs((y[_depot_start, k] == load for k in [v]), name="LoadOrigin")

    for v, _depot_start in depot_start_dict.items():
        _start_time = v_start_time[v]
        model.addConstrs(
            (t[_depot_start, k] == _start_time for k in [v]),
            name="StartingTimeDepot",
        )
    ### Constraint 13 for stationing traveling time: if beta=0 then t_jk-(t_ik +s_i+tau_ijk)-delta <0
    model.addConstrs(
        (
            (t[j, k] - (t[i, k] + s[i] + tau[i, j, k]) - delta) <= -1 * epsilon + M * eta[i, j, k]
            for k in K
            for i, j in A
        ),
        name="TimeStation",
    )
    model.addConstrs(
        ((t[j, k] - (t[i, k] + s[i] + tau[i, j, k]) - delta) >= -1 * M * (1 - eta[i, j, k]) for k in K for i, j in A),
        name="TimeStation",
    )
    model.addConstrs(
        (beta[i, j, k] >= epsilon - M * (1 - x[i, j, k]) - M * (1 - eta[i, j, k]) for k in K for i, j in A),
        name="TimeStation",
    )
    model.addConstrs(
        (beta[i, j, k] <= epsilon + M * x[i, j, k] for k in K for i, j in A),
        name="TimeStation",
    )
    model.addConstrs(
        (beta[i, j, k] <= epsilon + M * eta[i, j, k] for k in K for i, j in A),
        name="TimeStation",
    )
    ### Constraint 14: For stationing vehicle load: if beta_ijk=1, then y_ik=0
    model.addConstrs(
        (y[i, k] <= M * (1 - beta[i, j, k]) for k in K for i, j in A),
        name="LoadStation",
    )
    ################################################
    model.optimize()
    if model.SolCount > 0:
        solution = model.getAttr("x", x)
        time_solution = model.getAttr("x", t)
        Load_solution = model.getAttr("x", y)
        station_solution = model.getAttr("x", beta)
        time_gap_solution = model.getAttr("x", eta)

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
                next_node = [j for j in V if (current_node, j, k) in solution and solution[current_node, j, k] > 0.5]
                if (current_node, next_node[0], k) in tau:
                    nodetravel_times.append(tau[current_node, next_node[0], k])
                if next_node[0] == depot_end:  # or len(next_node) == 0:
                    route.append(next_node[0])
                    node_times.append(time_solution[depot_end, k])
                    break
                current_node = next_node[0]
            return route, node_times, nodetravel_times, node_loads

        for k in K:
            route, node_times, nodetravel_times, node_loads = extract_route(k)
            print(f"\nRoute for vehicle {k}:")
            for i in range(len(route) - 1):
                if route[i] in a.keys() and route[i] in ell.keys():
                    print(
                        f"Node {route[i]}: Total Time {node_times[i]:.2f}, Travel Time {nodetravel_times[i]:.2f}, Time Window [{a[route[i]]}, {b[route[i]]}], Load {node_loads[i]:.2f} load update {ell[route[i]]}"
                    )
                else:
                    print(
                        f"Node {route[i]}: Total Time {node_times[i]:.2f}, Travel Time {nodetravel_times[i]:.2f}, Load {node_loads[i]:.2f}"
                    )

                if station_solution[route[i], route[i + 1], k] >= 0.5:
                    print(f"-->Go to station {go_station[route[i], route[i+1]]}-->  ")
                print(f" -> {route[i + 1]}")

            print(f"Node {route[-1]}: Total Time {node_times[-1]:.2f}, Total Time {int(node_times[i]+s[i])}")

        routes = {k: extract_route(k)[0] for k in K}
        node_times = {k: extract_route(k)[1] for k in K}
        node_loads = {k: extract_route(k)[3] for k in K}
        nodetravel_times = {k: extract_route(k)[2] for k in K}
        return routes, node_times, node_loads, nodetravel_times, model.MIPGap
    else:
        print("No optimal solution found.")
        return None
