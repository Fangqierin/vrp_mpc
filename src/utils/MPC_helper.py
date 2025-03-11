from collections import defaultdict


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
