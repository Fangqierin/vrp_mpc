# %%
import numpy as np
import pandas as pd
import os


# %%
def get_data(chain_id, num_vehicles, vehicle_capacity, delta, num_stations, station_nodes):
    # chain_id, num_vehicles, vehicle_capacity, delta, num_stations, station_nodes = 0, 4, 12, 600, 5, [8537,4638, 9284, 3607, 6703]
    print(f"Processing chain {chain_id} ... ")
    folder_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/"
    data_fp = "./data_clean/CARTA/processed/test_chains.csv"
    data_fp = f"{folder_path}/CARTA/processed/test_chains.csv"
    # Load the data
    data = pd.read_csv(data_fp)
    data = data[data["chain_id"] == chain_id]
    # Number of requests for chain
    n = int(len(data))

    ########## Creating the required data ##########

    # Set of pickup nodes for chain
    P = list(range(1, n + 1))
    # Set of delivery nodes for chain
    D = list(range(n + 1, 2 * n + 1))
    # All nodes for chain
    N = P + D
    # Number of vehicles for chain
    C = {i: vehicle_capacity for i in range(1, num_vehicles + 1)}

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
    a[0] = 0
    b[0] = 86399
    for i in range(1, n + 1):
        a[i] = (
            int(data["pickup_time_since_midnight"].iloc[i - 1] - 900)
            if data["pickup_time_since_midnight"].iloc[i - 1] - 900 > 0
            else 0
        )
        b[i] = (
            int(data["pickup_time_since_midnight"].iloc[i - 1] + 900)
            if data["pickup_time_since_midnight"].iloc[i - 1] + 900 < 86399
            else 86399
        )
        a[i + n] = (
            int(data["dropoff_time_since_midnight"].iloc[i - 1] - 900)
            if data["dropoff_time_since_midnight"].iloc[i - 1] - 900 > 0
            else 0
        )
        b[i + n] = (
            int(data["dropoff_time_since_midnight"].iloc[i - 1] + 900)
            if data["dropoff_time_since_midnight"].iloc[i - 1] + 900 < 86399
            else 86399
        )
    a[2 * n + 1] = 0
    b[2 * n + 1] = 86399

    file_path = "./data_clean/travel_time_matrix/travel_time_matrix.csv"
    file_path = f"{folder_path}/travel_time_matrix/travel_time_matrix.csv"
    file_path = f"{folder_path}/travel_time_matrix/travel_time_matrix.npy"

    # TRAVEL_TIME_MATRIX = pd.read_csv(file_path, header=0)
    TRAVEL_TIME_MATRIX = np.load(file_path)
    # TRAVEL_TIME_MATRIX = TRAVEL_TIME_MATRIX.values
    TRAVEL_TIME_MATRIX = np.round(TRAVEL_TIME_MATRIX)
    TRAVEL_TIME_MATRIX = TRAVEL_TIME_MATRIX.astype(np.int32)

    nodes = pd.concat([data["pickup_node_id"], data["dropoff_node_id"]])
    node_dict = {i + 1: node for i, node in enumerate(nodes)}
    node_dict[0] = 3607
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

    for i in node_dict.keys():
        for j in node_dict.keys():
            if i != j:
                min_value = float("inf")  # Start with infinity for minimization

                min_station = None  # To track the station with the minimum value
                for station in station_nodes:
                    value = (
                        TRAVEL_TIME_MATRIX[node_dict[i], station]
                        + TRAVEL_TIME_MATRIX[station, node_dict[j]]
                        - TRAVEL_TIME_MATRIX[node_dict[i], node_dict[j]]
                    )
                    if value < min_value:
                        min_value = value
                        min_station = station

                # Append the result as a dictionary for each pair
                results.append(
                    {"i_node_id": node_dict[i], "j_node_id": node_dict[j], "min_add_time_station_node": min_station}
                )
                omega[(i, j)] = min_value

    df = pd.DataFrame(results)

    RESULTS_DIR = "./results"
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
