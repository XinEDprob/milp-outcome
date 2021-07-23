import numpy as np
import pandas as pd
import os
# import cplex
# from cplex.exceptions import CplexSolverError
# from cplex.callbacks import BranchCallback
# from cplex.callbacks import NodeCallback
import gurobipy as gy
from gurobipy import GRB, GurobiError
import pickle
import argparse
from collections import OrderedDict
import datetime
import shared
import utilities


"""
Run as
>>> python data_run.py --instance air04.mps.gz --seed 20180329 --node_limit $NODELIM

Extract information from the B&B tree via BranchCB and NodeCB.

Callbacks are invoked by the solver at *every* branched node, and at each node selection, but we extract information 
only up to a certain number of nodes (--node_limit), before stopping the run. 
Node limit is computed from a full run, as the # of nodes processed up to rho% of a certain time_limit 
(see full_run.py).

In particular, we trigger

    > BranchCB features: at every node in the partial tree, i.e., until optimization stops
    > NodeCB features: at few points in time, {25, 50, 75, 100}% of rho time stamp.
    
    (e.g., when rho=20, we collect NodeCB features at {5, 10, 15, 20}% of time_limit, 
    where 20 = 100% of rho means just before stopping data collection)
    Note: there is no collection after the root node (the time-stamp is not mapped in TIME_NODES_DICT)
    
    In this way we can have more info about problems that are solved *before* tau, 
    for which the observed tree is the whole tree.
    
The information is saved directly into a (global) list, for the entire (partial) run.
At the end of the run, the list is converted to a single array and saved it in DATA_PATH/NPY/.

NOTE: for instances that are fully solved before tau (`trivial' ones), the last node will *not* be mapped by callbacks.
      This happens naturally, because solve() ends, and there is no way that node is going to enter 
      the BranchCB or the NodeCB. 
      In particular, the last datum about the gap will not be 0.0, and we will not have NodeCB as last row of the data.
      This is why final_info are collected and stored (for analysis and checks), though the information about the
      end of the resolution process is *not explicit* in the collected data.

Now the order of the (31) basic extracted features is important, and fixed as follows:

    GENERAL
0:  'global_cb_count' Y
1:  'times_called'
2:  'get_time' 
3:  'get_dettime'
4:  'elapsed' Y
5:  'det_elapsed'    

    BRANCH_CB ONLY
6:  'index'
7:  'parent_id'
8:  'parent_index'
9:  'node'
10: 'nodes_left'
11: 'objective'
12: 'iinf'                  **expensive**
13: 'best_integer'
14: 'best_bound'
15: 'itCnt'
16: 'gap'
17: 'num_nodes'
18: 'depth'
19: 'num_fixed_vars'        **expensive**
20: 'is_integer_feasible'
21: 'has_incumbent'          

    NODE_CB ONLY
22: 'open_nodes_len' Y
23: 'open_nodes_avg'
24: 'open_nodes_min'
25: 'open_nodes_max'
26: 'num_nodes_at_min'  
27: 'num_nodes_at_max'  
28: 'num_cuts' Y

    GENERAL
29: 'cb_time'
30: 'cb_dettime'

"""

"""
Parser definition.
"""

class Experiment:
    def __init__(self, instance='air04.mps.gz', seed=201610271, dataset='Benchmark78', time_limit=7200):
        self.instance = instance
        self.seed = seed
        self.dataset = dataset
        self.time_limit = time_limit
        self.node_limit = 6000
        # self.tls = [1200., 2400., 3600., 7200.]
        self.rho = 10
        self.label_time = 1200
        # for storing data
        self.label = 1
        self.trivial = 0
        self.tau = 120
        self.sol_time = 459
        self.data_path = shared.DATA_PATH
        self.inst_path = shared.INST_PATH

# ARGS = Experiment(instance='app1-2.mps')
ARGS = Experiment()

"""
Custom classes definition.
"""


class GlobalCBCount:
    """
    A class setting a global count on Callbacks calls.
    """
    __counter = 0  # global counter of callbacks calls

    def __init__(self):
        GlobalCBCount.__counter += 1
        self.__cb_count = GlobalCBCount.__counter

    def get_cb_count(self):
        return self.__cb_count


class UserNodeIndex:
    """
    A class setting a global counter as unique index to each created node.
    """
    __index = -1  # so that root has index 0

    def __init__(self):
        UserNodeIndex.__index += 1
        self.__node_index = UserNodeIndex.__index

    def get_node_index(self):
        return self.__node_index


class UserNodeData:
    """
    A class defining a custom data-handle for nodes in the B&B tree.
    """
    def __init__(self, user_index, user_depth, user_parent_id, user_parent_index):
        self.user_index = user_index  # UserNodeIndex counter
        self.user_depth = user_depth  # depth will be updated recursively from parent to children
        self.user_parent_id = user_parent_id  # id of parent node
        self.user_parent_index = user_parent_index  # index of parent node


"""
Global definitions.
"""

# global definition of list, which will contain lists of length 31, i.e., the rows of the final array.
GLOBAL_LIST = list()

# the correct dictionary RHO_TAU_NODES is name_seed.pickle in ARGS.data_path/TIME_NODES_DICT/dataset_seed/
dict_dir = ARGS.data_path + "/TIME_NODES_DICT/" + ARGS.dataset + "_" + str(ARGS.seed) + "/"
dict_name = ARGS.instance.split('.')[0] + "_" + str(ARGS.seed) + ".pkl"

with open(os.path.join(dict_dir, dict_name), "rb") as p:
    RHO_TAU_NODES_DICT = pickle.load(p)
    nodeCB_FLAGS = OrderedDict()
    for eta in RHO_TAU_NODES_DICT[ARGS.rho][ARGS.tau]:
        if eta:  # map only node marks that are not None
            nodeCB_FLAGS[eta] = False
p.close()

# additional last NodeCB call at the end of the data collection
# (not met if node_limit corresponds to final number of nodes)
nodeCB_FLAGS[ARGS.node_limit] = False

# global instance of UserNodeData for the root node
root = UserNodeIndex()
print("\tRoot index is: {}".format(root.get_node_index()))
root_user_data = UserNodeData(user_index=root.get_node_index(), user_depth=0, user_parent_id=-1,
                              user_parent_index=None)

"""
Callbacks definition.
"""

# NOTE: there is no 'real' time_limit for data_run (enforce at 2h, with node_limit being effectively used).
# So it makes no sense to measure time_to_end.
# Elapsed time is meaningful, but accounts for data collection overhead as well.
# A more deterministic metric should be given by the number of nodes.

# Gurobi does not support the collection of the following information
# 1. collect the obj. of the unexplored nodes
# 2. number of cuts for different category

#
def data_run_callback(model, where):
        global nodeCB_FLAGS
        global GLOBAL_LIST
        global ARGS
        gcb = GlobalCBCount()
        if where == GRB.Callback.MIP:
            for n in nodeCB_FLAGS.keys():
                if model.cbGet(GRB.Callback.MIP_NODCNT) >= n and not nodeCB_FLAGS[n]:
                    print("*** NodeCB data call at num_nodes {}".format(model.cbGet(GRB.Callback.MIP_NODCNT)))
                    nodeCB_FLAGS[n] = True
                    node_cb_list = list()
                    node_cb_list.append(gcb.get_cb_count())
                    # node_cb_list.append(self.times_called)
                    node_cb_list.extend([model.cbGet(GRB.Callback.RUNTIME)])

                    # 7 features from MIP callback
                    node_cb_list.extend([model.cbGet(GRB.Callback.MIP_OBJBST), model.cbGet(GRB.Callback.MIP_OBJBND),
                                         model.cbGet(GRB.Callback.MIP_NODCNT), model.cbGet(GRB.Callback.MIP_SOLCNT),
                                         model.cbGet(GRB.Callback.MIP_CUTCNT), model.cbGet(GRB.Callback.MIP_NODLFT),
                                         model.cbGet(GRB.Callback.MIP_ITRCNT)])

                    node_cb_list.extend([None] * 10)

                    GLOBAL_LIST.append(node_cb_list)
                    break

        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_NODCNT) < ARGS.node_limit:
                branch_cb_list = list()
                branch_cb_list.append(gcb.get_cb_count())
                branch_cb_list.extend([model.cbGet(GRB.Callback.RUNTIME)])

                branch_cb_list.extend([None] * 7)
                # 5 features from MIPNODE callback
                branch_cb_list.extend([model.cbGet(GRB.Callback.MIPNODE_STATUS), model.cbGet(GRB.Callback.MIPNODE_OBJBST),
                                       model.cbGet(GRB.Callback.MIPNODE_OBJBND), model.cbGet(GRB.Callback.MIPNODE_NODCNT),
                                       model.cbGet(GRB.Callback.MIPNODE_SOLCNT)])
                branch_cb_list.extend([None] * 5)
                GLOBAL_LIST.append(branch_cb_list)

        if where == GRB.Callback.MIPSOL:
            sol_cb_list = []
            sol_cb_list.append(gcb.get_cb_count())
            sol_cb_list.extend([model.cbGet(GRB.Callback.RUNTIME)])

            sol_cb_list.extend([None] * 12)
            # 5 features from MIPSOL callback
            sol_cb_list.extend([model.cbGet(GRB.Callback.MIPSOL_OBJ), model.cbGet(GRB.Callback.MIPSOL_OBJBST),
                                model.cbGet(GRB.Callback.MIPSOL_OBJBND), model.cbGet(GRB.Callback.MIPSOL_NODCNT),
                                model.cbGet(GRB.Callback.MIPSOL_SOLCNT)])
            GLOBAL_LIST.append(sol_cb_list)

if __name__ == "__main__":

    import sys
    import time
    import os

    name = ARGS.instance.split('.')[0]
    inst_name_info = name + '_' + str(ARGS.seed) + '_' + str(ARGS.label_time) + '_' + str(ARGS.tau) + '_' + str(ARGS.rho)

    # sys.stdout = None

    os.chdir(ARGS.inst_path)
    pb = gy.read(name)

    print("\nName: {}".format(pb.ModelName))
    print("# Variables: {}".format(pb.NumVars))
    print("# Constraints: {}".format(pb.NumConstrs))

    print("\nNode limit = {}".format(ARGS.node_limit))
    print("(rho, tl) = ({}, {}), tau = {}".format(ARGS.rho, ARGS.label_time, ARGS.tau))
    print("Solution time = {}".format(ARGS.sol_time))  # solution time of the full run
    print("Label = {}".format(ARGS.label))
    print("Trivial = {}".format(ARGS.trivial))

    # set solver parameters
    utilities.set_solver_data_run_parameters(pb, ARGS)

    # register callback
    # branch_instance = pb.register_callback(MyDataBranch)
    # node_instance = pb.register_callback(MyDataNode)

    try:
        t0 = datetime.datetime.now()
        print("\nInitial time-stamps: {} ".format(t0))
        pb.optimize(data_run_callback)
        elapsed = datetime.datetime.now() - t0
        status = pb.Status
        num_nodes = int(pb.NodeCount)
        itCnt = int(pb.IterCount)
        gap = pb.MIPGap
        print("\nStatus: {}".format(pb.Status ))
    except GurobiError as e:  # data will be all zeros if exception is raised
        print("Exception raised during solve: {}".format(e.args[2]))
        elapsed = None
        elapsed_ticks = None
        status = e.args[2]  # error code
        num_nodes = None
        itCnt = None
        gap = None

    final_info = [elapsed, status, num_nodes, itCnt, gap]

    # print("\nMyDataBranch was called {} times".format(branch_instance.times_called))
    # print("MyDataBranch count data: {}".format(branch_instance.count_data))
    # print("\nMyDataNode was called {} times".format(node_instance.times_called))
    # print("MyDataNode count data: {}".format(node_instance.count_data))

    print("\nNodeCB flags: ")
    print(nodeCB_FLAGS)

    print("\nFinal info of data_run: {}".format(final_info))

    # conversion to array
    global_arr = np.asarray(GLOBAL_LIST)
    print("Array shape is: ", global_arr.shape)

    # use np.savez to save into inst_name_info:
    npy_rho_dir = ARGS.data_path + '/NPY_RHO_' + str(int(ARGS.rho)) + '/' + ARGS.dataset + '_' + str(ARGS.seed) + '/'
    try:
        os.mkdir(npy_rho_dir)
    except OSError:
        if not os.path.isdir(npy_rho_dir):
            raise

    os.chdir(npy_rho_dir)
    np.savez_compressed(
        inst_name_info,
        ft_matrix=global_arr,
        name=name,
        seed=ARGS.seed,
        dataset=ARGS.dataset,
        rho=ARGS.rho,
        tau=ARGS.tau,
        node_limit=ARGS.node_limit,
        label_time=ARGS.label_time,
        label=ARGS.label,
        trivial=ARGS.trivial,
        sol_time=ARGS.sol_time,
        data_final_info=final_info,
    )
