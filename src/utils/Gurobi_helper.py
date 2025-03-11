from collections import defaultdict
from src.policies.mpc.MPC_VRP.src.utils.load_data_Gurobi import generate_CARTA_data

from src.policies.mpc.MPC_VRP.src.utils.pdptw_gurobi import pdptw_solver as pdptw_solver_gurobi
from src.policies.mpc.MPC_VRP.src.utils.pdptw_gurobi import pdptw_solver_station as pdptw_solver_station_gurobi
from src.policies.mpc.MPC_VRP.src.utils.log import log_route_plan

import time


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


def PDPTW_Gurobi_Plan(
    CarNumber=1,
    Capacity=8,
    request_df=None,
    depot_start_location=None,
    OD_dropoff_request=None,
    v_start_time=None,
    TRAVEL_TIME_MATRIX=None,
    Park=False,
    station_ids=None,
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
        request_df=request_df,
        depot_start_location=depot_start_location,
        OD_dropoff_request=OD_dropoff_request,
        v_start_time=v_start_time,
        TRAVEL_TIME_MATRIX=TRAVEL_TIME_MATRIX,
        Park=False,
    )
    routes, node_times, node_loads, nodetravel_times, optimality_gap = pdptw_solver_gurobi(
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

    all_routes, idle_v = insert_station(
        routes,
        node_times,
        node_loads,
        nodetravel_times,
        a,
        P,
        D,
        points,
        TRAVEL_TIME_MATRIX,
        station_ids,
    )
    print(routes, optimality_gap)
    print(len(P), P)
    print(len(D), D)
    return (routes, node_times, node_loads, nodetravel_times, optimality_gap, all_routes, a, points, n, idle_v)


def simulate(chain_id, t):  # Simualte one time PDPTW Solver
    # date = [2022,9,5]
    # year, month, day = date[0], date[1], date[2]
    #
    # n, P, D, C, tau, omega, c, c_omega, a, b, s, ell, K, points, depot_start, depot_end = load_data.get_data_no_break(2022, 9, 5)
    # n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end = load_data.get_data_no_break(2022, 9, 5)
    # n, P, D, C, tau, c, a, b, s, ell, K, points, depot_start, depot_end, OD, depot_start_dict, depot_end_dict, v_load,OD_dict , V = generate_example_data_new()
    # file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.csv"
    def load_source_data():
        # Load travel time matrix
        file_path = (
            "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/travel_time_matrix.npy"
        )
        # TRAVEL_TIME_MATRIX = pd.read_csv(file_path, header=None)
        # TRAVEL_TIME_MATRIX = TRAVEL_TIME_MATRIX.round().astype(np.int32)
        travel_time = np.load(file_path)
        # # Load nodes
        # file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/travel_time_matrix/nodes.csv"
        # nodes = pd.read_csv(file_path
        # Load chains
        DF_TRAIN = pd.read_csv(
            "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/train_chains.csv"
        )
        DF_TEST = pd.read_csv(
            "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/test_chains.csv"
        )
        ########### Getting station
        station_path = (
            "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/station_mapping_medium.pkl"
        )

        with open(station_path, "rb") as f:
            node_to_station = pickle.load(f)
        station_id_file = (
            "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/stations_medium.csv"
        )
        df = pd.read_csv(station_id_file)  # Replace with the actual file path

        # Extract the 'station_id' column as a list
        station_ids = df["station_id"].tolist()
        return travel_time, DF_TEST, DF_TRAIN, node_to_station, station_ids

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
        depot_start_location=None, OD_dropoff_request=None, TRAVEL_TIME_MATRIX=travel_time, Park=True
    )

    routes, node_times, node_loads, nodetravel_times, optimality_gap = pdptw_solver_station_gurobi(
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
