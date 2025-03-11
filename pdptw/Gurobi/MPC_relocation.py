from collections import defaultdict
import subprocess
import os
import sys
import re
import numpy as np
import copy


def mat2str(mat):
    return str(mat).replace("'", '"').replace("(", "<").replace(")", ">").replace("[", "{").replace("]", "}")


class MPC:
    def __init__(self):
        """
        :param cplexpath: Path to the CPLEX solver.
        """
        self.cplexpath = "/home/fangqil2/IBM/opl/bin/x86-64_linux/"
        self.directory = "mpc/nyc_brooklyn"
        self.policy_name = "mpc"
        # self.T = kwargs.get("T")
        self.platform = None

    def get_fake_input(self):
        t = 0
        T = 1
        G = {1: 15, 2: 5, 3: 8}
        # Trip attributes: List of trips in the form (origin, destination, time, demand, price)
        tripAttr = [(1, 2, 0, 10, 5), (1, 2, 1, 10, 3), (2, 3, 0, 20, 8), (2, 3, 1, 20, 8)]
        # Demand times: Mapping of (origin, destination) to a dictionary of demand times (time -> demand)
        demandTime = {(1, 2): {0: 1, 1: 2}, (2, 3): {0: 1, 1: 1}}
        # Rebalancing times: Mapping of (origin, destination) to a dictionary of rebalance times (time -> time)
        rebTime = {(1, 2): {0: 1, 1: 1}, (2, 3): {0: 1, 1: 2}}
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

        dacc = {1: {0: 1, 1: 1}, 2: {0: 1, 1: 0}, 3: {0: 1, 1: 0}}
        # Rebalancing vehicle flow (origin, destination) -> time -> number of rebalancing vehicles
        rebFlow = defaultdict(dict)
        # Set of regions
        region = list(G)
        # Number of regions
        nregion = len(G)
        # Set of edges for rebalancing
        edges = [(1, 2), (2, 3)]
        # Extract demand attributes for the next T time steps
        demandAttr = [
            (i, j, tt, demand[i, j][tt], demandTime[i, j][tt], price[i, j][tt])
            for i, j in demand
            for tt in range(t, t + T)
            if demand[i, j][tt] > 1e-3
        ]
        # Accumulated number of vehicles per region at time t
        accTuple = [(n, acc[n][t]) for n in acc]

        # Accumulated vehicles arriving at each region for the next T time steps
        daccTuple = [(n, tt, dacc[n][tt]) for n in acc for tt in range(t, t + T)]

        # Rebalancing edge attributes (origin, destination, rebalance time at time t)
        edgeAttr = [(i, j, rebTime[i, j][t]) for i, j in edges]

        # Accumulated number of vehicles per region at time t
        accTuple = [(n, acc[n][t]) for n in acc]

        # Accumulated vehicles arriving at each region for the next T time steps
        daccTuple = [(n, tt, dacc[n][tt]) for n in acc for tt in range(t, t + T)]

        # Rebalancing edge attributes (origin, destination, rebalance time at time t)
        edgeAttr = [(i, j, rebTime[i, j][t]) for i, j in edges]
        beta = 2

        return t, demandAttr, accTuple, daccTuple, edgeAttr, edges, T, beta

    def test(self):
        pax_action, reb_action = self.MPC_run()

    def MPC_run(self, t, demandAttr, accTuple, daccTuple, edgeAttr, edges, T, beta):
        ####################
        # t, demandAttr, accTuple, daccTuple, edgeAttr, edges, T, beta = self.get_fake_input()
        # ---------------------------------------------------------------------------
        modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
        MPCPath = os.getcwd().replace("\\", "/") + "/mpc_results/cplex_logs/" + self.directory + "/"
        if not os.path.exists(MPCPath):
            os.makedirs(MPCPath)
        datafile = MPCPath + "data_2_{}.dat".format(t)
        resfile = MPCPath + "res_2_{}.dat".format(t)
        with open(datafile, "w") as file:
            file.write('path="' + resfile + '";\r\n')
            file.write("t0=" + str(t) + ";\r\n")
            file.write("T=" + str(T) + ";\r\n")
            file.write("beta=" + str(beta) + ";\r\n")
            file.write("demandAttr=" + mat2str(demandAttr) + ";\r\n")
            file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
            file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
            file.write("daccAttr=" + mat2str(daccTuple) + ";\r\n")
        file.close()
        modfile = "/home/fangqil2/optimus_fix/src/policies/mpc/MPC_VRP/src/cplex_mod/" + "MPC.mod"
        my_env = os.environ.copy()
        if self.platform == None:
            my_env["LD_LIBRARY_PATH"] = self.cplexpath
        else:
            my_env["DYLD_LIBRARY_PATH"] = self.cplexpath
        out_file = MPCPath + "out_{}.dat".format(t)

        with open(out_file, "w") as output_f:
            subprocess.check_call(
                [self.cplexpath + "oplrun", modfile, datafile],
                stdout=output_f,
                # env=my_env,
            )
        output_f.close()
        paxFlow = defaultdict(float)
        rebFlow = defaultdict(float)
        with open(resfile, "r", encoding="utf8") as file:
            for row in file:
                item = row.replace("e)", ")").strip().strip(";").split("=")
                if item[0] == "flow":
                    values = item[1].strip(")]").strip("[(").split(")(")
                    for v in values:
                        if len(v) == 0:
                            continue
                        i, j, f1, f2 = v.split(",")
                        f1 = float(re.sub("[^0-9e.-]", "", f1))
                        f2 = float(re.sub("[^0-9e.-]", "", f2))
                        paxFlow[int(i), int(j)] = float(f1)
                        rebFlow[int(i), int(j)] = float(f2)
        paxAction = [paxFlow[i, j] if (i, j) in paxFlow else 0 for i, j in edges]
        rebAction = [rebFlow[i, j] if (i, j) in rebFlow else 0 for i, j in edges]

        return paxFlow, rebFlow


if __name__ == "__main__":
    mpc = MPC()
    paxAction, rebAction = mpc.MPC_run()
    print(paxAction, rebAction)

# def MPC_exact(self, env, sumo=False):
#     tstep = env.tstep
#     t = env.time  #   # Current time step

#     flows = list()
#     demandAttr = list()
#     if sumo:
#         pass
#     else:

#         t = env.time
#         demandAttr = [
#             (
#                 i,
#                 j,
#                 tt,
#                 env.demand[i, j][tt],
#                 env.demandTime[i, j][tt],
#                 env.price[i, j][tt],
#             )
#             for i, j in env.demand
#             for tt in range(t, t + self.T)
#             if env.demand[i, j][tt] > 1e-3
#         ]

#         accTuple = [(n, env.acc[n][t]) for n in env.acc]
#         daccTuple = [(n, tt, env.dacc[n][tt]) for n in env.acc for tt in range(t, t + self.T)]
#         edgeAttr = [(i, j, env.rebTime[i, j][t]) for i, j in env.edges]
#         ### Give a fake value:-------------------------------------------------
#         t, demandAttr, accTuple, daccTuple, edgeAttr = self.get_fake_input()

#         # ---------------------------------------------------------------------------
#         modPath = os.getcwd().replace("\\", "/") + "/src/cplex_mod/"
#         MPCPath = os.getcwd().replace("\\", "/") + "/saved_files/cplex_logs/" + self.directory + "/"
#     if not os.path.exists(MPCPath):
#         os.makedirs(MPCPath)
#     datafile = MPCPath + "data_{}.dat".format(t)
#     resfile = MPCPath + "res_{}.dat".format(t)
#     with open(datafile, "w") as file:
#         file.write('path="' + resfile + '";\r\n')
#         file.write("t0=" + str(t) + ";\r\n")
#         file.write("T=" + str(self.T) + ";\r\n")
#         file.write("beta=" + str(env.beta) + ";\r\n")
#         file.write("demandAttr=" + mat2str(demandAttr) + ";\r\n")
#         file.write("edgeAttr=" + mat2str(edgeAttr) + ";\r\n")
#         file.write("accInitTuple=" + mat2str(accTuple) + ";\r\n")
#         file.write("daccAttr=" + mat2str(daccTuple) + ";\r\n")

#     modfile = modPath + "MPC.mod"
#     my_env = os.environ.copy()
#     if self.platform == None:
#         my_env["LD_LIBRARY_PATH"] = self.cplexpath
#     else:
#         my_env["DYLD_LIBRARY_PATH"] = self.cplexpath
#     out_file = MPCPath + "out_{}.dat".format(t)

#     with open(out_file, "w") as output_f:
#         subprocess.check_call(
#             [self.cplexpath + "oplrun", modfile, datafile],
#             stdout=output_f,
#             env=my_env,
#         )
#     output_f.close()
#     paxFlow = defaultdict(float)
#     rebFlow = defaultdict(float)
#     with open(resfile, "r", encoding="utf8") as file:
#         for row in file:
#             item = row.replace("e)", ")").strip().strip(";").split("=")
#             if item[0] == "flow":
#                 values = item[1].strip(")]").strip("[(").split(")(")
#                 for v in values:
#                     if len(v) == 0:
#                         continue
#                     i, j, f1, f2 = v.split(",")
#                     f1 = float(re.sub("[^0-9e.-]", "", f1))
#                     f2 = float(re.sub("[^0-9e.-]", "", f2))
#                     paxFlow[int(i), int(j)] = float(f1)
#                     rebFlow[int(i), int(j)] = float(f2)
#     paxAction = [paxFlow[i, j] if (i, j) in paxFlow else 0 for i, j in env.edges]
#     rebAction = [rebFlow[i, j] if (i, j) in rebFlow else 0 for i, j in env.edges]

#     return paxAction, rebAction
