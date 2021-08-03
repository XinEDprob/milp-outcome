import numpy as np
import pandas as pd
import argparse
import math
import sys
import os
import glob
from collections import OrderedDict
import gurobipy as gy
from gurobipy import GRB, GurobiError
import shared
import time
import os
# import cplex
import random
import pickle
"""
Process data collected with callbacks during (partial) B&B run (data_run) into a single array of features.
Build a data point of (features, label).

In this script, we just consider the 37 features that were selected for learning in the CPAIOR 2019 paper.
For the complete feature set see features_all.py and the additional ones in learning.ipynb.

Each stored npz file contains 

    "ft_matrix"     : matrix of raw data from BranchCB and NodeCB
    "name"          : problem name
    "seed"          : the seed of the computation
    "dataset"       : name of the dataset to which the problem belongs
    "rho"           : rho
    "tau"           : tau
    "node_limit"    : node limit used in data collection
    "label_time"    : time with respect to which a label was computed
    "label"         : the computed label in {0, 1}
    "trivial"       : boolean flag
    "sol_time"      : time of resolution (or timelimit)
    "data_final_info: array of final information on the run (useful to check trivial instances!)
    
"""

"""
Parser and auxiliary functions
"""

class Experiment:
    def __init__(self, npy_path, learning_path, filename, inst_path=shared.INST_PATH):
        self.npy_path = npy_path
        self.learning_path = learning_path
        self.inst_path = inst_path
        self.filename = filename

eps = 1e-5

""" Routines """


# NOTE: this is *not* the same definition as the MIP Relative Gap implemented by CPLEX
def primal_dual_gap(best_ub, best_lb):
    """
    :param best_ub: current best upper bound: c^Tx~
    :param best_lb: current best lower bound: z_
    """
    if (abs(best_ub) == 0) & (abs(best_lb) == 0):
        return 0
    elif best_ub * best_lb < 0:
        return 1
    else:
        return abs(best_ub - best_lb) / max([abs(best_ub), abs(best_lb)])


def primal_gap(best_sol, feas_sol):
    """
    :param best_sol: optimal or best known solution value: c^Tx*,
    :param feas_sol: feasible solution value: c^Tx~
    """
    if (abs(best_sol) == 0) & (abs(feas_sol) == 0):
        return 0
    elif best_sol * feas_sol < 0:
        return 1
    else:
        return abs(best_sol - feas_sol) / max([abs(best_sol), abs(feas_sol)])


"""
Selected features (37)
"""


def get_37_features(branch_df, num_discrete_vars, num_all_vars):
    """
    :param branch_df:
    :param node_df:
    :param num_discrete_vars:
    :param num_all_vars:
    :return:
    """
    global eps
    # last_branch = branch_df.iloc[-1]
    # last_node = node_df.iloc[-1]

    mip_df = branch_df.loc[~branch_df.MIP_OBJBND.isnull()]
    mipnode_df = branch_df.loc[~branch_df.MIPNODE_STATUS.isnull()]
    mipsol_df = branch_df.loc[~branch_df.MIPSOL_OBJ.isnull()]

    last_node = mipnode_df.iloc[-1]

    fts_list = []

    # first set of features: last observed global measure (2)
    # gap
    gap = primal_dual_gap(last_node['MIPNODE_OBJBND'], last_node['MIPNODE_OBJBST'])
    # global bound ratio
    # TODO: negative positive
    bound_ratio = last_node['MIPNODE_OBJBST']/last_node['MIPNODE_OBJBND'] \
        if last_node['MIPNODE_OBJBST'] and last_node['MIPNODE_OBJBND'] else None

    fts_list.extend([gap, bound_ratio])

    # second set of features: number of pruned nodes (2)
    pruned = (mipnode_df['MIPNODE_NODCNT'].diff(1) - 1).dropna().tolist()  # correction!
    cumulative = sum([item for item in pruned if item > 0])
    p1 = cumulative / mipnode_df['MIPNODE_NODCNT'].iloc[-1] if branch_df['MIPNODE_NODCNT'].iloc[-1] != 0 else None

    fts_list.extend([p1])
    # TODO not a good indicator
    if not mip_df['MIP_ITRCNT'].empty:
        iterCNT = mip_df['MIP_ITRCNT'].iloc[-1]/mipnode_df['MIPNODE_NODCNT'].iloc[-1] if mip_df['MIP_ITRCNT'].iloc[-1] and \
                    mipnode_df['MIPNODE_NODCNT'].iloc[-1] else None
    else:
        iterCNT = None
    fts_list.extend([iterCNT])

    # third set of features: about iinf (4)
    quantile_df = mipnode_df.loc[mipnode_df['MIPNODE_iif'] < mipnode_df['MIPNODE_iif'].quantile(.05)][['MIPNODE_iif', 'times_called']]
    iinf1 = mipnode_df['MIPNODE_iif'].max() / float(num_discrete_vars)
    iinf2 = mipnode_df['MIPNODE_iif'].min() / float(num_discrete_vars)
    iinf3 = mipnode_df['MIPNODE_iif'].mean() / float(num_discrete_vars)
    iinf4 = quantile_df.shape[0] / float(mipnode_df.times_called.max())

    fts_list.extend([iinf1, iinf2, iinf3, iinf4])


    # forth set of features: about incumbent (4)
    abs_improvement = pd.Series(abs(mipnode_df['MIPNODE_OBJBST'].diff(1).dropna()))
    bool_updates = pd.Series((abs_improvement != 0))

    num_updates = bool_updates.sum()  # real number of updates (could be 0)
    inc1 = float(num_updates) / mipnode_df['MIPNODE_NODCNT'].iloc[-1] if branch_df['MIPNODE_NODCNT'].iloc[-1] != 0 else None
    inc2 = abs_improvement.mean() / mipnode_df['MIPNODE_OBJBST'].iloc[-1]  if mipnode_df['MIPNODE_OBJBST'].iloc[-1]  else None

    # add dummy 1 (update) at the end of bool_updates
    bool_updates[bool_updates.shape[0]] = 1.
    non_zeros = bool_updates.values == 1
    zeros = ~non_zeros
    zero_counts = np.cumsum(zeros)[non_zeros]
    zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive incumbent updates
    zeros_to_last = zero_counts[-1]
    zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg
    try:
        inc3 = zero_counts.mean()
        inc4 = zeros_to_last
    except ValueError:
        inc3 = None
        inc4 = None
    # TODO: think of a better denominator
    incA3 = inc3 / mipnode_df['MIPNODE_NODCNT'].iloc[-1] if inc3 and mipnode_df['MIPNODE_NODCNT'].iloc[-1] else None
    incA4 = inc4 / inc3 if inc3 else None

    fts_list.extend([inc1, inc2, incA3, incA4])

    # fifth set of features best bound (4)
    abs_improvement = pd.Series(abs(mipnode_df['MIPNODE_OBJBND'].diff(1).dropna()))
    bool_updates = pd.Series((abs_improvement != 0))
    avg_improvement = abs_improvement.sum() / bool_updates.sum() if bool_updates.sum() != 0 else None

    num_updates = bool_updates.sum()  # real number of updates (could be 0)
    bb1 = float(num_updates) / mipnode_df['MIPNODE_NODCNT'].iloc[-1] if mipnode_df['MIPNODE_NODCNT'].iloc[-1] != 0 else None
    bb2 = abs_improvement.mean() / mipnode_df['MIPNODE_OBJBND'].iloc[-1]  if mipnode_df['MIPNODE_OBJBND'].iloc[-1]  else None

    # add dummy 1 (update) at the end of bool_updates
    bool_updates[bool_updates.shape[0]] = 1.
    non_zeros = bool_updates.values == 1
    zeros = ~non_zeros
    zero_counts = np.cumsum(zeros)[non_zeros]
    zero_counts[1:] -= zero_counts[:-1].copy()  # distance between two successive best_bound updates
    zeros_to_last = zero_counts[-1]
    zero_counts = zero_counts[:-1]  # removes last count (to the end) to compute max, min, avg
    try:
        bb3 = zero_counts.mean()
        bb4 = zeros_to_last
    except ValueError:
        bb3 = None
        bb4 = None
    bbA3 = bb3 / mipnode_df['MIPNODE_NODCNT'].iloc[-1] if bb3 and mipnode_df['MIPNODE_NODCNT'].iloc[-1] else None
    bbA4 = bb4 / bb3 if bb3 else None

    fts_list.extend([bb1, bb2, bbA3, bbA4])

    # sixth set of features about objective (3)
    feasible_obj = mipnode_df.query('MIPNODE_STATUS == 2')
    quantile_df = feasible_obj.loc[feasible_obj['MIPNODE_relobj'] > feasible_obj['MIPNODE_relobj'].quantile(.95)][['MIPNODE_relobj',
                                                                                                'times_called']]
    obj1 = quantile_df.shape[0] / float(mipnode_df.times_called.max())
    obj2 = abs(mipnode_df['MIPNODE_relobj'].quantile(0.95) - mipnode_df.iloc[-1]['MIPNODE_OBJBST'])
    obj3 = abs(mipnode_df['MIPNODE_relobj'].quantile(0.95) - mipnode_df.iloc[-1]['MIPNODE_OBJBND'])

    fts_list.extend([obj1, obj2, obj3])

    # seventh set of features about fixed variables (4)
    # fix1 = mipnode_df['num_fixed_vars'].max() / float(num_all_vars)
    # fix2 = mipnode_df['num_fixed_vars'].min() / float(num_all_vars)
    # quantile_df = branch_df.loc[branch_df['num_fixed_vars'] >
    #                             branch_df['num_fixed_vars'].quantile(.95)][['num_fixed_vars', 'times_called']]
    # fix4 = quantile_df.shape[0] / float(branch_df.times_called.max())
    # fix6 = (branch_df.times_called.max() - quantile_df.times_called.max()) / float(branch_df.times_called.max())
    #
    # fts_list.extend([fix1, fix2, fix4, fix6])

    # about integral (1)
    total_time = mipnode_df.iloc[-1]['times_called']
    # opt = float(miplib_df.loc[miplib_df['Name'] == inst_name]['Objective'])  # best known objective

    # copy part of branch_df
    use_cols = ['times_called', 'MIPNODE_OBJBST', 'MIPNODE_OBJBND']
    copy_branch_df = mipnode_df[use_cols].copy()
    copy_branch_df['inc_changes'] = abs(copy_branch_df['MIPNODE_OBJBST'].diff(1))
    copy_branch_df['inc_bool'] = copy_branch_df['inc_changes'] != 0
    copy_branch_df['bb_changes'] = abs(copy_branch_df['MIPNODE_OBJBND'].diff(1))
    copy_branch_df['bb_bool'] = copy_branch_df['bb_changes'] != 0

    pd_dict = OrderedDict()  # {t_i: pd(t_i)} for t_i with incumbent change
    pd_dict[0] = 1
    for idx, row in copy_branch_df.loc[(copy_branch_df['inc_bool'] != 0) | (copy_branch_df['bb_bool'] != 0)].iterrows():
        pd_dict[row['times_called']] = primal_dual_gap(row['MIPNODE_OBJBST'], row['MIPNODE_OBJBND'])
    pd_dict[total_time] = None

    pd_times = list(pd_dict.keys())
    pd_integrals = list(pd_dict.values())

    pdi = 0
    for i in range(len(pd_times) - 1):
        pdi += pd_integrals[i] * (pd_times[i + 1] - pd_times[i])

    fts_list.extend([pdi])

    return fts_list


def get_features_label(data_file, data_cols, num_discrete_vars, num_all_vars, num_all_constr, known_opt_value=None):
    """
    :param data_file: file .npz containing data for an (instance, seed) pair
    :param data_cols: column names for data in data_file
    :param num_discrete_vars: number of discrete variables for the problem at hand
    :param num_all_vars: total number of variables
    :param num_all_constr: total number of constraints
    :param known_opt_value: value of optimal solution (if known) else None
    :return: extract features vector for a single data-point from .npz data_file.
    """
    # data_file contains 'ft_matrix', 'label_time', 'label', 'name', 'seed', 'data_final_info' (and others)
    loaded_file = np.load(data_file, allow_pickle=True)
    loaded_data = loaded_file['ft_matrix']

    # check for empty data in is main loop (outside this function)

    # define DataFrame from data
    all_df = pd.DataFrame(loaded_data, columns=data_cols)
    all_df.set_index(all_df['global_cb_count'].values, inplace=True)
    # branch_df = all_df.loc[~all_df.index.isnull()]
    # node_df = all_df.loc[all_df.index.isnull()]

    # idx_split, chunk_num, num_nodes_list = get_chunks(node_df, branch_df)

    all_features = get_37_features(branch_df=all_df,
                                   num_discrete_vars=num_discrete_vars, num_all_vars=num_all_vars)
    all_features.insert(0, float(loaded_file['label_time']))
    # all_features.insert(1, float(chunk_num))
    all_features.append(float(loaded_file['label']))

    print("Len of ft vector: {}".format(len(all_features)))
    return all_features


def features_select_37(ARGS):
    src_dir = os.getcwd()

    """
    Callback data
    """

    # COLS = [
    #     'global_cb_count', 'times_called', 'get_time', 'get_dettime', 'elapsed', 'det_elapsed',
    #     'index', 'parent_id', 'parent_index',
    #     'node', 'nodes_left', 'objective', 'iinf', 'best_integer', 'best_bound', 'itCnt', 'gap',
    #     'num_nodes', 'depth', 'num_fixed_vars', 'is_integer_feasible', 'has_incumbent',
    #     'open_nodes_len', 'open_nodes_avg', 'open_nodes_min', 'open_nodes_max',
    #     'num_nodes_at_min', 'num_nodes_at_max', 'num_cuts',
    #     'cb_time', 'cb_dettime'
    # ]
    COLS = [
        'global_cb_count', 'times_called', 'MIP_OBJBST', 'MIP_OBJBND', 'MIP_NODCNT', 'MIP_SOLCNT', 'MIP_CUTCNT',
        'MIP_NODLFT', 'MIP_ITRCNT', 'MIPNODE_STATUS', 'MIPNODE_OBJBST', 'MIPNODE_OBJBND', 'MIPNODE_NODCNT',
        'MIPNODE_SOLCNT', 'MIPNODE_iif', 'MIPNODE_relobj', 'MIPSOL_OBJ', 'MIPSOL_OBJBST', 'MIPSOL_OBJBND', 'MIPSOL_NODCNT', 'MIPSOL_SOLCNT'
    ]

    """
    Dataset split (on names and directly on npz files)
    """

    os.chdir(ARGS.inst_path)

    names_list = list()
    inst_count = 0

    for inst in glob.glob('*.mps.gz'):  # extension dependency
        inst_count += 1
        names_list.append(inst)
    print("\nTotal # instances: {}".format(inst_count))

    # remove extension from names
    ext = lambda x: x.rstrip('.mps.gz')  # extension dependency
    names_list = [ext(name) for name in names_list]

    # shuffle names and split them
    random.shuffle(names_list)
    # other proportion for train and test split
    train_size = int(math.ceil(3 * len(names_list)) / 5.)  # test_size is 2/5
    list_1 = names_list[:train_size]
    list_2 = names_list[train_size:]
    print("Disjoint lists: {}. Len: {} {}".format(set(list_1).isdisjoint(list_2), len(list_1), len(list_2)))

    os.chdir(src_dir)
    # save list_1 and list_2 for future reference
    # with open(os.path.join(ARGS.learning_path, '1_' + ARGS.filename.rstrip('.npz') + '_names.pkl'), 'wb') as f1:
    #     pickle.dump(list_1, f1)
    # f1.close()

    # with open(os.path.join(ARGS.learning_path, '2_' +ARGS.filename.rstrip('.npz') + '_names.pkl'), 'wb') as f2:
    #     pickle.dump(list_2, f2)
    # f2.close()

    print("Lengths of splits: {} {}".format(len(list_1), len(list_2)))

    # in subsequent runs, do not perform split but read instead names from loaded lists
    # with open('1_' + ARGS.filename.split('_')[0] + '_names.pkl', 'rb') as f1:
    #     list_1 = pickle.load(f1)
    # f1.close()
    # with open('2_' + ARGS.filename.split('_')[0] + '_names.pkl', 'rb') as f2:
    #     list_2 = pickle.load(f2)
    # f2.close()

    # select file names in npy_path
    os.chdir(ARGS.npy_path)

    npz_1 = [npz for npz in glob.glob('*.npz') if npz.split('_')[0] in list_1]
    npz_2 = [npz for npz in glob.glob('*.npz') if npz.split('_')[0] in list_2]

    # now process one list at a time

    """
    Sets (split) processing
    """

    sets = [npz_1, npz_2]
    set_idx = 0

    for split in sets:
        set_idx += 1
        print("\nProcessing split # {}".format(set_idx))

        os.chdir(ARGS.npy_path)

        count = 0
        count_empty = 0
        glob_list = []

        for f in split:

            count += 1

            # check if data in f is empty
            loaded = np.load(f, allow_pickle=True)  # contains 'ft_matrix', 'label_time', 'label', 'name', 'seed'
            data = loaded['ft_matrix']
            name = str(loaded['name'])  # w/o extension .mps.gz
            print("\n{} {} data type and shape: {} {}".format(count, name, data.dtype, data.shape))

            # check if data is empty
            if data.shape[0] == 0:
                count_empty += 1
                continue  # go to next file

            # read the instance to gather basic variables/constraints info
            os.chdir(ARGS.inst_path)
            c = gy.read(name + '.mps.gz')
            # c = cplex.Cplex(name + '.mps.gz')
            # c.set_results_stream(None)
            # c.set_error_stream(None)
            # c.set_log_stream(None)
            # c.set_warning_stream(None)

            num_discrete = c.NumIntVars
            num_vars = c.NumVars
            num_constr = c.NumConstrs
            # c.end()
            os.chdir(ARGS.npy_path)

            glob_list.append(np.asarray(get_features_label(
                data_file=f,
                data_cols=COLS,
                num_discrete_vars=num_discrete,
                num_all_vars=num_vars,
                num_all_constr=num_constr,
                known_opt_value=None
            ),
                dtype=np.float))

        print("\nCount: {}".format(count))
        print("Empty data count: {}".format(count_empty))
        print("Len of glob_list: {}".format(len(glob_list)))
        global_arr = np.asarray(glob_list, dtype=np.float)
        print("Shape of global_arr: {}".format(global_arr.shape))

        # print("\nSample: ")
        # print(global_arr[0])

        os.chdir(ARGS.learning_path)
        np.savez(str(set_idx) + '_37_' + ARGS.filename, data=global_arr)


if __name__ == "__main__":
    data_path = shared.DATA_PATH
    rhos = [5, 10, 15, 20, 25]
    dataset = 'Benchmark78'
    seed = '201610271'
    name = 'bab5'
    label_times = [1200., 2400., 3600., 7200.]
    for rho in rhos:
        for label_time in label_times:
            tau = rho*label_time/100
            print(f'process rho: {rho} and label_time: {label_time}')
            npy_rho_dir = data_path + 'NPY_RHO_' + str(rho) + '/' + dataset + '_' + str(seed) + '/'
            learning_path = data_path + 'LEARNING/'
            filename = name + '_' + str(seed) + '_' + str(label_time) + '_' + str(tau) + '_' + str(rho)

            ARGS = Experiment(npy_path=npy_rho_dir, learning_path=learning_path, filename=filename)

            features_select_37(ARGS)


