from cllm.reg.io import *
from cllm.io import split_data
from cllm.reg.basecp import BaseRegCP, OptimalSolution, ConformalScore2Norm, ConformalScoreRightSide, RawPredSolution
from cllm.reg.groupcp import GroupCP
from cllm.reg.necp import NEWCP, GroupNEWCP, ConditionNEWCP, LearnedNEWCP
from cllm.reg.fgroupcp import FixedGroupCP
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from cllm.utils import setup_logging
import logging
import pandas as pd
from pathlib import Path
import torch
import pickle
import numpy as np
from cllm.reg.dummy import DefaultModel
from cllm.reg.threscp import AvgSizeThresCP

def evaluate_once(methods, pdf, target_path, seed=None, balance=False, split_train=True, alphas=[0.5, 0.4, 0.3, 0.2, 0.1]):
    if split_train:
        split_arr = [0.4, 0.3, 0.3]
    else:
        split_arr = [0, 0.5, 0.5]
    if balance: 
        pdf = balance_tbe_data(pdf, seed=seed)
        # shuffle.
        train_data, validate_data, test_data = split_data(pdf, split_arr)
    else:
        if seed is None:
            seed = 42
        train_data, validate_data, test_data = split_data(pdf, split_arr, shuffle=True, random_seed=seed)
    if seed is None:
            seed = 42
    torch.manual_seed(seed)
    methods[0][1].train(train_data)
    for name, method in methods:
        data = []
        for alpha in alphas:
            print('--------', name, '-------')
            # method.train(train_data)
            method.calibrate(validate_data, alpha)
            res = method.test(test_data)
            print(res)
            for k, v in res.items():
                data.append((name, k, v, alpha, seed))
        pd.DataFrame(data, columns=['method', 'name', 'value', 'alpha', 'seed']).to_csv(target_path / '{}_{}.csv'.format(name, seed))


def run_onebatch(dataset, target_name, balance=False, reg_model=None, split_train=True, specify_alpha=None):
    # mgt: multiple groud-truth.
    gt_func_num = 0
    if dataset == 'aug_mgt_as_one':
        pdf = load_std_mgt_as_one_data(dataset, base_path='./datasets/STD')
        gt_func_num = 60
    elif dataset == 'aug':
        pdf = load_tbe2_data(dataset)
        gt_func_num = 225
    elif dataset == 'ds1k':
        pdf = load_ds1k_data(dataset)
        gt_func_num = 1000
    elif dataset == 'ds10kbase':
        pdf = load_ds10k_data('base')
        gt_func_num = 416
    elif dataset == 'ds10kall':
        pdf = load_ds10k_data('all')
        gt_func_num = 833
    elif dataset == 'ds10kbase_v2':
        pdf = load_ds10k_data_v2('base')
        gt_func_num = 390
    elif dataset == 'ds10kall_v2':
        pdf = load_ds10k_data_v2('all')
        gt_func_num = 782
    elif dataset == 'ds10kbase_deepseek':
        pdf = load_ds10k_data_v2_deepseek('base')
        gt_func_num = 377
    elif dataset == 'ds10kall_deepseek':
        pdf = load_ds10k_data_v2_deepseek('all')
        gt_func_num = 771
    elif dataset == 'ds10kbase_deepseek_iodesc':
        pdf = load_ds10k_data_v2_deepseek('base', embed_key='iodesc')
        gt_func_num = 377
    elif dataset == 'ds10kall_deepseek_iodesc':
        pdf = load_ds10k_data_v2_deepseek('all', embed_key='iodesc')
        gt_func_num = 771
    elif dataset == 'ds10kbase_deepseek_iodesc_st':
        pdf = load_ds10k_data_v2_deepseek('base', embed_key='iodesc', use_st_embedding=True)
        gt_func_num = 377
    elif dataset == 'ds10kall_deepseek_iodesc_st':
        pdf = load_ds10k_data_v2_deepseek('all', embed_key='iodesc', use_st_embedding=True)
        gt_func_num = 771
    else:
        raise NotImplementedError()
    # else:
    #     pdf = load_std_data(dataset, base_path='./datasets/STD')
    pass
    if reg_model is None:
        model = DefaultModel()
    elif reg_model == 'SVR':
        model = SVR()
    elif reg_model == 'Neural':
        # use nerual network from sklearn.
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
    else:
        raise NotImplementedError()
    methods = [
        ('base', BaseRegCP(model, score=ConformalScore2Norm())),
        ('base_weight', ConditionNEWCP(model, score=ConformalScore2Norm(), method='softmax')),
        # ('group_2', GroupCP(model, 2, score=ConformalScore2Norm())),
        # ('group_3', GroupCP(model, 3, score=ConformalScore2Norm())),
        # ('fixed_group_learnq_2_8_2', FixedGroupCP(model, group_ratios=[0.8, 0.2], num_epochs_pinball=500)),
        # ('fixed_group_computeq_2_8_2', FixedGroupCP(model, group_ratios=[0.8, 0.2])),
        # ('fixed_group_computeq_weight_2_8_2', FixedGroupCP(model, group_ratios=[0.8, 0.2], weight_method='softmax')),
        # ('size_group_k_3', AvgSizeThresCP(model, 3, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        # ('size_group_weight_k_2', AvgSizeThresCP(model, 2, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=1, use_weight_calibration=False)),
        # ('size_group_weight_k_2_T2', AvgSizeThresCP(model, 2, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=2, use_weight_calibration=False)),
        # ('size_group_weight_k_3', AvgSizeThresCP(model, 3, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=1, use_weight_calibration=False)),
        # ('size_group_weight_k_3_T2', AvgSizeThresCP(model, 3, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=2, use_weight_calibration=False)),
        # ('size_group_k_4', AvgSizeThresCP(model, 4, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        # ('size_group_weight_k_4', AvgSizeThresCP(model, 4, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=1, use_weight_calibration=False)),
        # ('size_group_k_2', AvgSizeThresCP(model, 2, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        # ('size_group_k_1', AvgSizeThresCP(model, 1, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),

        ('size_group_kp_1', AvgSizeThresCP(model, 0.01*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        ('size_group_kp_05', AvgSizeThresCP(model, 0.005*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        ('size_group_kp_2', AvgSizeThresCP(model, 0.02*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        # ('size_group_kp_4', AvgSizeThresCP(model, 0.04*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        # ('size_group_kp_6', AvgSizeThresCP(model, 0.06*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),
        # ('size_group_kp_8', AvgSizeThresCP(model, 0.08*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', T=1)),

        # ('size_group_weight_kp_4', AvgSizeThresCP(model, 0.04*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=1, use_weight_calibration=False)),
        # ('size_group_weight_kp_6', AvgSizeThresCP(model, 0.06*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=1, use_weight_calibration=False)),
        # ('size_group_weight_kp_8', AvgSizeThresCP(model, 0.08*gt_func_num, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', T=1, use_weight_calibration=False)),

        # you cannot set the following param, since we do not support exchangebility for learned q.
        # ('fixed_group_softmax_2_8_2', FixedGroupCP(model, group_ratios=[0.8, 0.2], num_epochs_pinball=500, weight_method='softmax')),

        # ('fixed_group_computeq_2_9_1', FixedGroupCP(model, group_ratios=[0.9, 0.1])),
        # ('fixed_group_learnq_2_9_1', FixedGroupCP(model, group_ratios=[0.9, 0.1], num_epochs_pinball=500)),
        # ('fixed_group_computeq_2_8_2', FixedGroupCP(model, group_ratios=[0.8, 0.2])),
        # ('fixed_group_learnq_2_8_2', FixedGroupCP(model, group_ratios=[0.8, 0.2], num_epochs_pinball=500)),
        # ('fixed_group_computeq_2_7_3', FixedGroupCP(model, group_ratios=[0.7, 0.3])),
        # ('fixed_group_learnq_2_7_3', FixedGroupCP(model, group_ratios=[0.7, 0.3], num_epochs_pinball=500)),
        # ('fixed_group_computeq_2_6_4', FixedGroupCP(model, group_ratios=[0.6, 0.4])),
        # ('fixed_group_learnq_2_6_4', FixedGroupCP(model, group_ratios=[0.6, 0.4], num_epochs_pinball=500)),
        # ('fixed_group_computeq_2_5_5', FixedGroupCP(model, group_ratios=[0.5, 0.5])),
        # ('fixed_group_learnq_2_5_5', FixedGroupCP(model, group_ratios=[0.5, 0.5], num_epochs_pinball=500)),

        # ('fixed_group_computeq_2_a2', FixedGroupCP(model, group_ratios=[0.8, 0.2])),
        # ('fixed_group_computeq_3_a2', FixedGroupCP(model, group_ratios=[0.4, 0.4, 0.2])),
        # ('fixed_group_computeq_4_a2', FixedGroupCP(model, group_ratios=[0.26, 0.27, 0.27, 0.2])),
        # ('fixed_group_computeq_5_a2', FixedGroupCP(model, group_ratios=[0.2, 0.2, 0.2, 0.2, 0.2])),


        # ('fixed_group_learnq_2_a2', FixedGroupCP(model, group_ratios=[0.8, 0.2], num_epochs_pinball=500)),
        # ('fixed_group_learnq_3_a2', FixedGroupCP(model, group_ratios=[0.4, 0.4, 0.2], num_epochs_pinball=500)),
        # ('fixed_group_learnq_4_a2', FixedGroupCP(model, group_ratios=[0.26, 0.27, 0.27, 0.2], num_epochs_pinball=500)),
        # ('fixed_group_learnq_5_a2', FixedGroupCP(model, group_ratios=[0.2, 0.2, 0.2, 0.2, 0.2], num_epochs_pinball=500)),

    ]

    target_path = Path('./data_out/paper_base/exp_result/{}/'.format(target_name))
    target_path.mkdir(parents=True, exist_ok=True)

    seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]
    # seeds = [11, 22, 33]
    # seeds = [22]
    if specify_alpha:
        alphas = [specify_alpha]
    else:
        alphas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    for seed in seeds:
        evaluate_once(methods, pdf, target_path, seed=seed, balance=balance, split_train=split_train, alphas=alphas)


if __name__ == '__main__':
    setup_logging(level=logging.INFO, to_file=False)

    exp_pairs = [
        # ('aug_mgt_as_one', 'std_struct_mgt_as_one_test_reg', False, 'SVR'),
        # ('aug_mgt_as_one', 'std_struct_mgt_as_one_test_noreg', False, None, True, False),
        # ('aug', 'tde2_struct_test_reg', False, 'SVR', True, False),
        # ('aug', 'tde2_struct_test_noreg', False, None, True, False),
        # ('aug', 'tde2_struct_test_reg_neural', False, 'Neural', True, False),
        # ('aug_mgt_as_one', 'std_base_noreg_all_alpha_36', False, None, False, 0.36),
        # ('aug', 'tde2_base_noreg_all_alpha_36', False, None, False, 0.36),

        # ('aug_mgt_as_one', 'std_base_noreg_all', False, None, False, False),
        # ('aug', 'tde2_base_noreg_all', False, None, False, False),

        # ('aug_mgt_as_one', 'std_fix_alpha_noreg_all', False, None, False, 0.2),
        # ('aug', 'tde2_fix_alpha_noreg_all', False, None, False, 0.2),

        # ('ds1k', 'ds1k_base_noreg_all', False, None, False, False),
        # ('ds1k', 'ds1k_fix_alpha_noreg_all', False, None, False, 0.2),

        # ('ds10kbase', 'ds10kbase_base_noreg_all', False, None, False, False),
        # ('ds10kbase', 'ds10kbase_fix_alpha_noreg_all', False, None, False, 0.2),

        # ('ds10kall', 'ds10kall_base_noreg_all', False, None, False, False),
        # ('ds10kall', 'ds10kall_fix_alpha_noreg_all', False, None, False, 0.2),

        # ('ds10kbase_v2', 'ds10kbase_v2_base_noreg_all', False, None, False, False),
        # ('ds10kbase_v2', 'ds10kbase_v2_fix_alpha_noreg_all', False, None, False, 0.2),

        # ('ds10kall_v2', 'ds10kall_v2_base_noreg_all', False, None, False, False),
        # ('ds10kall_v2', 'ds10kall_v2_fix_alpha_noreg_all', False, None, False, 0.2),


        # deepseek
        # ('ds10kbase_deepseek', 'ds10kbase_deepseek_base_noreg_all', False, None, False, False),
        # ('ds10kbase_deepseek', 'ds10kbase_deepseek_fix_alpha_noreg_all', False, None, False, 0.2),

        # ('ds10kall_deepseek', 'ds10kall_deepseek_base_noreg_all', False, None, False, False),
        # ('ds10kall_deepseek', 'ds10kall_deepseek_fix_alpha_noreg_all', False, None, False, 0.2),    

        # ('ds10kbase_deepseek_iodesc', 'ds10kbase_deepseek_base_noreg_all_iodesc', False, None, False, False),
        # ('ds10kbase_deepseek_iodesc', 'ds10kbase_deepseek_fix_alpha_noreg_all_iodesc', False, None, False, 0.2),
        ('ds10kall_deepseek_iodesc', 'ds10kall_deepseek_base_noreg_all_iodesc', False, None, False, False),
        # ('ds10kall_deepseek_iodesc', 'ds10kall_deepseek_fix_alpha_noreg_all_iodesc', False, None, False, 0.2),

        # ('ds10kbase_deepseek_iodesc_st', 'ds10kbase_deepseek_base_noreg_all_iodesc_st', False, None, False, False),
        # ('ds10kbase_deepseek_iodesc_st', 'ds10kbase_deepseek_fix_alpha_noreg_all_iodesc_st', False, None, False, 0.2),
        # ('ds10kall_deepseek_iodesc_st', 'ds10kall_deepseek_base_noreg_all_iodesc_st', False, None, False, False),
        # ('ds10kall_deepseek_iodesc_st', 'ds10kall_deepseek_fix_alpha_noreg_all_iodesc_st', False, None, False, 0.2),
    ]

    for dataset, target, balance, use_reg_model, split_train, specify_alpha in exp_pairs:
        print('processing:', dataset, target, balance, use_reg_model)
        run_onebatch(dataset, target, balance, use_reg_model, split_train=split_train, specify_alpha=specify_alpha)
