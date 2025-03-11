from load_data_hexlay import get_data_Hexlay
from hexaly_station import pdptw_hexlay_solver


def PDPTW_Hexlay_Plan(
    CarNumber=1,
    Capacity=8,
    request_df=None,
    depot_start_location=None,
    OD_dropoff_request=None,
    v_start_time=None,
    TRAVEL_TIME_MATRIX=None,
    Park=False,
    station_ids=None,
    str_time_limit="10",  # For Hexlay execution time
    sol_file=None,
    delta=900,
    chain_id=0,
):
    # (
    #     n,
    #     P,
    #     D,
    #     C,
    #     tau,
    #     c,
    #     a,
    #     b,
    #     s,
    #     ell,
    #     K,
    #     points,
    #     depot_end,
    #     OD,
    #     depot_start_dict,
    #     v_load,
    #     OD_dict,
    #     V,
    #     OD_dropoff_request,
    #     v_start_time,
    #     points,
    #     W,
    #     _,
    #     _,
    # )
    (n, P, D, C, tau, add_time_for_break, c, add_cost_for_break, a, b, s, ell, depot) = get_data_Hexlay(
        CarNumber=CarNumber,
        Capacity=Capacity,
        request_df=request_df,
        depot_start_location=depot_start_location,  # Add for Online
        v_start_time=v_start_time,  # Add for Online
        OD_dropoff_request=OD_dropoff_request,  # Add for Online
        TRAVEL_TIME_MATRIX=TRAVEL_TIME_MATRIX,
        station_ids=station_ids,
        Park=False,
    )

    # routes, node_times, node_loads, nodetravel_times, optimality_gap = pdptw_solver_gurobi(
    #     n,
    #     P,
    #     D,
    #     C,
    #     tau,
    #     c,
    #     a,
    #     b,
    #     s,
    #     ell,
    #     K,
    #     depot_end,
    #     OD,
    #     v_load=v_load,
    #     depot_start_dict=depot_start_dict,
    #     OD_dict=OD_dict,
    #     V=V,
    #     v_start_time=v_start_time,
    # )
    ##########Need finish this!!!! FQ!----->>>Stop here. 17

    routes, node_times, node_loads, nodetravel_times, optimality_gap = pdptw_hexlay_solver(
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
        CarNumber,
        str_time_limit,
        sol_file,
        chain_id,
        delta,
    )

    # return (routes, node_times, node_loads, nodetravel_times, optimality_gap, all_routes, a, points, n, idle_v)
