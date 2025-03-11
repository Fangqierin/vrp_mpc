def generate_CARTA_data(
    CarNumber=5,
    Capacity=8,
    request_df=None,
    depot_start_location=None,
    v_start_time=None,
    a=None,
    TRAVEL_TIME_MATRIX=None,
    OD_dropoff_request=None,
    station_ids=[229, 5346, 1905, 5139],
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
    #     for node_id in station_ids:
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
            station_ids,
            key=lambda station: TRAVEL_TIME_MATRIX[points[i][0], station] + TRAVEL_TIME_MATRIX[station, points[j][0]],
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
