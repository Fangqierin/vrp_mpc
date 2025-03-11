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
from src.policies.mpc.MPC_VRP.src.utils.pdptw_gurobi import pdptw_solver as pdptw_solver_gurobi
from src.policies.mpc.MPC_VRP.src.utils.pdptw_gurobi import pdptw_solver_station as pdptw_solver_station_gurobi

import csv
import copy
# from src.policies.mpc.MPC_VRP.src.utils.hexaly_station import pdptw_solver as hexaly_pdptw_solver
from src.policies.mpc.MPC_VRP.src.utils.load_data_hexlay import get_data_Hexlay
from src.policies.mpc.MPC_VRP.src.utils.log import log_route_plan
from utils.MPC_helper import get_MPC_input, get_idle_cars, get_reallocation
from utils.Gurobi_helper import PDPTW_Gurobi_Plan
from utils.load_data_Gurobi import generate_CARTA_data
from env.simulate import update_state


def load_source_data():
    # Load travel time matrix
    file_path = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/data/CARTA/processed/travel_time_matrix.npy" 
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
    # simulate(chain_id, 0)
    ######## Simulate for Single day Online Routing Iteratively
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
            ########## Given MPC Results, Reallocation idle vehilces, Then Run PDPTW!
            # else:
            routes, node_times, node_loads, nodetravel_times, optimality_gap, all_routes, a, points, n, idle_v = (
                PDPTW_Gurobi_Plan(
                    CarNumber=CarNumber,
                    Capacity=Capacity,
                    request_df=data,
                    depot_start_location=depot_start_location,
                    OD_dropoff_request=OD_dropoff_request,
                    v_start_time=v_start_time,
                    TRAVEL_TIME_MATRIX=travel_time,
                    Park=False,
                    station_ids=station_ids,
                )
            )
            print(
                f" at time {cur_time/60} minute --RunTime {time.time()-cur_time_clock}----------------- Finish Routing Planning "
            )

            log_route_plan(cur_time, all_routes)
            ################### Update Current State
            v_start_time, OD_dropoff_request, depot_start_location, finished_request = update_state(
                cur_time + interval,
                a,
                points,
                n,
                depot_start_location,
                all_routes,
                idle_v,
            )
            All_finished_request = All_finished_request + finished_request
            # Remove the leftdropoff request (which pickup already passed)
            unfinished_demand = [
                i for i in data if i[-1] not in All_finished_request + [i[-1] for i in OD_dropoff_request]
            ]
        else:
            print(f" at time {cur_time/60} minute --No new Results ")
        cur_time = cur_time + interval
