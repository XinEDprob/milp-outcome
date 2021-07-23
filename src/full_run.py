import argparse
from collections import OrderedDict

# import cplex
# from cplex.callbacks import BranchCallback
# from cplex.callbacks import NodeCallback
# from cplex.exceptions import CplexSolverError
import sys
import os.path
import gurobipy as gy
import time
import shared
import utilities
from gurobipy import GRB, GurobiError
import datetime

""" 
Run (instance, seed) with single (longest specified) time limit (2h).
Collect basic info on the run at predefined time stamps (tau, depending on tl and rho).
Note: these are not the basic info that are used to build features, but will be used to define subsequent runs.

It is not possible to control the invocation of MIPInfoCallback, hence
basic stats on the run are collected via BranchCallback.

Run as
>>> python full_run.py --instance air04.mps.gz --dataset Benchmark78

Default random seed:
- 12.8.0.0 : 201709013
- 12.7.1.0 : 201610271

"""


""" 
Argparser definition.
"""

class Experiment:
    def __init__(self, instance='air04.mps.gz', seed=201610271, dataset='Benchmark78', time_limit=7200):
        self.instance = instance
        self.seed = seed
        self.dataset = dataset
        self.time_limit = time_limit
        self.tls = [1200., 2400., 3600., 7200.]
        self.rhos = [5, 10, 15, 20, 25]
        self.data_path = shared.DATA_PATH
        self.inst_path = shared.INST_PATH

ARGS = Experiment(instance='app1-2.mps')


"""
Time limits and rho-stamps definition.
"""

# TLs = [1200., 2400., 3600., 7200.]
# RHOs = [5, 10, 15, 20, 25]

# TLs = [1500., 6000.]
# RHOs = [20]

TLs = ARGS.tls
RHOs = ARGS.rhos

TL_dict = {}
for tl in TLs:
    TL_dict[tl] = [rho * tl/100 for rho in RHOs]

RHO_dict = {}
for rho in RHOs:
    RHO_dict[rho] = [rho * tl/100 for tl in TLs]

STAMPS = [rho * tl/100 for rho in RHOs for tl in TLs]
STAMPS.extend(TLs)
STAMPS = list(set(STAMPS))  # remove duplicates and sort
STAMPS.sort()

ALL_STAMPS = []
for k in STAMPS:  # TLs are also considered
    ALL_STAMPS.extend([k/100.*p for p in [25, 50, 75, 100]])

ALL_STAMPS_U = list(set(ALL_STAMPS))
ALL_STAMPS_U.sort()

ALL_STAMPS_flags = OrderedDict()
for t_stamp in ALL_STAMPS_U:
    ALL_STAMPS_flags[t_stamp] = False


""" 
Callbacks definition.
"""


# class MyEmptyNode(NodeCallback):
#     """
#     Empty callback. Custom subclass of NodeCallback.
#     This callback will be used *before CPLEX enters a node*.
#     """
#     def __init__(self, env):
#         NodeCallback.__init__(self, env)
#         self.times_called = 0
#
#     def __call__(self):
#         self.times_called += 1


# callback function for gurobi
def mycallback(model, where):
    global ARGS
    global stats_append
    global ALL_STAMPS_flags
    if where == GRB.Callback.MIP:
        elapsed = model.cbGet(GRB.Callback.RUNTIME)
        for stamp in ALL_STAMPS_flags.keys():
            if elapsed >= stamp and not ALL_STAMPS_flags[stamp]:
                ALL_STAMPS_flags[stamp] = True  # might not be hit if stamp == timelimit

                # No ELAPSED_TICKS, B_CALLS, INCUMBENT, OBJECTIVE for gurobi

                #  "NAME", "SEED", "TIME_STAMP", "NODES", "ITCNT", "GAP",
                #  "ELAPSED_SECS", "ELAPSED_TICKS", "B_CALLS",
                #  "BEST_BOUND", "INCUMBENT", "OBJECTIVE",
                #  "STATUS", "SOL_TIME", "END_LINE"
                name = ARGS.instance.split('.')[0]
                objbst = model.cbGet(GRB.Callback.MIP_OBJBST)
                objbnd = model.cbGet(GRB.Callback.MIP_OBJBND)
                gap = round(abs(objbst - objbnd)/(1.0 + abs(objbst)), 4)
                line = [name, ARGS.seed, stamp, model.cbGet(GRB.Callback.MIP_NODCNT),
                        model.cbGet(GRB.Callback.MIP_ITRCNT), gap, elapsed,
                        objbst, None, None, False]
                # "STATUS" and "SOL_TIME" are filled in after solve() call is over, with "END_LINE" True (in the main)
                for entry in line:
                    stats_append.write("%s\t" % entry)
                stats_append.write("\n")
                break



if __name__ == "__main__":



    cwd = os.getcwd()

    name = ARGS.instance.split('.')[0]
    inst_info_str = name + '_' + str(ARGS.seed)
    dir_info_str = ARGS.dataset + '_' + str(ARGS.seed) + '/'

    # data directory setup
    utilities.dir_setup(parent_path_inst_dir=ARGS.data_path, rhos=ARGS.rhos)


    try:
        os.mkdir(ARGS.data_path + "/OUT/" + dir_info_str)
    except OSError:
        if not os.path.isdir(ARGS.data_path + "/OUT/" + dir_info_str):
            raise

    os.chdir(ARGS.data_path + "/OUT/" + dir_info_str)
    sys.stdout = open(inst_info_str + '.out', 'w')  # output file

    os.chdir(ARGS.inst_path)
    pb = gy.read(name)

    print("\nName: {}".format(pb.ModelName))
    print("# Variables: {}".format(pb.NumVars))
    print("# Constraints: {}".format(pb.NumConstrs))
    print("Seed: {}".format(ARGS.seed))

    utilities.set_solver_full_run_parameters(pb, inst_info_str, dir_info_str, ARGS)

    # register callback
    # branch_instance = pb.register_callback(MyStatBranch)
    # node_instance = pb.register_callback(MyEmptyNode)

    header = [
        "NAME", "SEED", "TIME_STAMP", "NODES", "ITCNT", "GAP",
        "ELAPSED_SECS",
        "BEST_BOUND",
        "STATUS", "SOL_TIME", "END_LINE"
    ]

    try:
        os.mkdir(ARGS.data_path + "/STAT/" + dir_info_str)
    except OSError:
        if not os.path.isdir(ARGS.data_path + "/STAT/" + dir_info_str):
            raise

    with open(os.path.join(ARGS.data_path + "/STAT/" + dir_info_str, inst_info_str + '.stat'), 'a') as stats_append:
        for item in header:
            stats_append.write("%s\t" % item)
        stats_append.write("\n")

        try:
            t0 = datetime.datetime.now()
            print("\nInitial time-stamps: {} ".format(t0))
            pb.optimize(mycallback)
            sol_time_secs = datetime.datetime.now() - t0
            print("\nFinal elapsed: {}".format(sol_time_secs))
            objbst = pb.ObjVal
            objbnd = pb.ObjBound
            gap = abs(objbst - objbnd)/(1.0 + abs(objbst))
            # termination line
            end_line = [
                name,
                ARGS.seed,
                None,
                int(pb.NodeCount),
                int(pb.IterCount),
                round(pb.MIPGap,4) ,
                round(sol_time_secs.seconds,4),
                objbst,
                pb.Status,
                float(round(sol_time_secs.seconds,4)),
                True
            ]

            for item in end_line:
                stats_append.write("%s\t" % item)
            stats_append.write("\n")

            print("\nStatus: {}".format(pb.Status))

        except GurobiError as e:  # end-line data will be all zeros if exception is raised
            print("Exception raised during solve")
            end_line = [
                name, ARGS.seed, None, None, None, None,
                None, None, None,
                None, None, None,
                e.errno, None, True
            ]

            for item in end_line:
                stats_append.write("%s\t" % item)
            stats_append.write("\n")

        # print("\nMyStatBranch # calls: {}".format(branch_instance.times_called))
        # print("MyEmptyNode # calls: {}".format(node_instance.times_called))
        print("\nMyStatBranch flags: ")
        print(ALL_STAMPS_flags)

        # pb.end()
    stats_append.close()
