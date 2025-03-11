# %%
import numpy as np
import pandas as pd
import os


def get_data_Hexlay(
    CarNumber=0,
    Capacity=8,
    request_df=None,
    depot_start_location=None,  # Added for Online
    v_start_time=None,  # Added for Online
    OD_dropoff_request=None,  # Added for Online
    TRAVEL_TIME_MATRIX=None,
    parking_nodes=[229, 5346, 1905, 5139],
    Park=True,
):
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
    n = int(len(data))
    ########## Creating the required data ##########
    # Set of pickup nodes for chain
    P = list(range(1, n + 1))
    # Set of delivery nodes for chain
    D = list(range(n + 1, 2 * n + 1))
    # All nodes for chain
    N = P + D
    # Number of vehicles for chain
    C = {i: Capacity for i in range(1, CarNumber + 1)}
    # Setting up loads for chain
    ell = {}
    ell[0] = 0
    for i in range(1, n + 1):
        ell[i] = 1
    for i in range(n + 1, 2 * n + 1):
        ell[i] = -1
    ell[2 * n + 1] = 0
    # Service time at each node
    s = {i: 0 for i in range(0, 2 * n + 2)}
    # Depot start and end node
    depot = 0
    # Set of vehicles
    K = list(C.keys())
    # # time window for each node
    a = {}
    b = {}
    points = {}
    nodes_id = []
    node_dict = {}
    # Loop through the data and set values for pickup and corresponding drop-off nodes
    for idx, request in enumerate(data):
        ########
        idx = idx + 1  # FQ Add this one 2/15 Need change
        ########
        pickup_node_id, dropoff_node_id, pickup_time, dropoff_time, chain_order = request
        # Placeholder coordinates for pickup and dropoff nodes
        points[idx] = (pickup_node_id, chain_order)  # Assign coordinates for the pickup node
        points[idx + n] = (dropoff_node_id, chain_order)  # Assign coordinates for the dropoff node
        # Pickup time windows
        a[idx] = max(pickup_time - 15 * 60, 0)  # Early arrival time
        b[idx] = pickup_time + 15 * 60  # Late arrival time
        # Dropoff time windows
        a[idx + n] = dropoff_time - 15 * 60  # Early arrival time
        b[idx + n] = dropoff_time + 15 * 60  # Late arrival time
        # Track node IDs
        node_dict[idx] = pickup_node_id
        node_dict[idx + n] = dropoff_node_id
    #### Need to change! for Online Decision
    a[0] = 0
    b[0] = 86399
    a[2 * n + 1] = 0
    b[2 * n + 1] = 86399
    node_dict[0] = 3607  # FQ Add this one 2/15 Need change
    ######################################
    # file_path = f"{folder_path}/travel_time_matrix/travel_time_matrix.npy"
    if TRAVEL_TIME_MATRIX is None:
        file_path = (
            "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.npy"
        )
        TRAVEL_TIME_MATRIX = np.load(file_path)
        TRAVEL_TIME_MATRIX = np.round(TRAVEL_TIME_MATRIX)
        TRAVEL_TIME_MATRIX = TRAVEL_TIME_MATRIX.astype(np.int32)
    ###################
    # Travel time between nodes
    tau = {
        (i, j): TRAVEL_TIME_MATRIX[node_dict[i], node_dict[j]]
        for i in range(2 * n + 1)
        for j in range(2 * n + 1)
        if i != j
    }

    # Distance matrix
    c = tau
    results = []
    omega = {}
    go_station = {}
    tau_s = {}
    for i in node_dict.keys():
        for j in node_dict.keys():
            if i != j:
                min_value = float("inf")  # Start with infinity for minimization

                min_station = None  # To track the station with the minimum value
                for station in parking_nodes:
                    value = max(
                        0,
                        (
                            TRAVEL_TIME_MATRIX[node_dict[i], station]
                            + TRAVEL_TIME_MATRIX[station, node_dict[j]]
                            - TRAVEL_TIME_MATRIX[node_dict[i], node_dict[j]]
                        ),
                    )
                    if value < min_value:
                        min_value = value
                        min_station = station
                go_station[(i, j)] = min_station  # Store the station for the pair (i, j)
                # Append the result as a dictionary for each pair

                results.append(
                    {"i_node_id": node_dict[i], "j_node_id": node_dict[j], "min_add_time_station_node": min_station}
                )
                omega[(i, j)] = min_value
                tau_s[(i, j)] = min_value
    df = pd.DataFrame(results)
    ####### For test ########
    RESULTS_DIR = "./results"
    chain_id = "fq_test_2025"
    SOLUTIONS_DIR = f"{RESULTS_DIR}/PDPTW_DB_Hexaly_Solutions"
    if not os.path.exists(SOLUTIONS_DIR):
        os.makedirs(SOLUTIONS_DIR)
    if not os.path.exists(f"{SOLUTIONS_DIR}/{chain_id}"):
        os.makedirs(f"{SOLUTIONS_DIR}/{chain_id}")

    output_file = f"{SOLUTIONS_DIR}/{chain_id}/min_add_time_station_nodes_chain.csv"
    df.to_csv(output_file, index=False)

    # Additional distance for taking a break
    c_omega = omega
    print(f"\nDone.")
    return n, P, D, C, tau, omega, c, c_omega, a, b, s, ell, depot
