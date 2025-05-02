import numpy as np
import pandas as pd
from cllm.reg.io import load_pickle
from pathlib import Path

def compute_coverage_from_interval(gt_arr, pred_intervals, agg=True):
    match_arr = []
    for gt, interval in zip(gt_arr, pred_intervals):
        # print(gt, interval)
        # if gt < 0:
        #     exit()
        if gt > 0 and gt >= interval[0] and gt <= interval[1]:
            match_arr.append(1)
        else:
            match_arr.append(0)
    # print(match_arr)
    # print('count from group', _count, len(gt_arr))
    match_arr = np.array(match_arr)
    if not agg:
        return match_arr
    return match_arr.mean()
    # return _count / len(gt_arr)

def eval_metrics(pdf, pred_arr, gt_arr, dist_col='dist_arr', pred_type='interval'):
    if pred_type == 'interval':
        # select_dist 
        pred_dist = []
        # print(pred_arr.shape)
        for _id, (dist_arr, (lb, ub), gt_v) in enumerate(zip(pdf[dist_col], pred_arr, gt_arr)):
            selected_distances = dist_arr[(dist_arr >= lb) & (dist_arr <= ub)]
            # if len(selected_distances) == 0 and gt_v >= lb and gt_v <= ub:
            #     print(lb, ub, gt_v)
            #     print(dist_arr)
            #     print(_id)
            #     print(pdf.iloc[_id])
            #     x = pdf.iloc[_id]
            #     # compute embed.
            #     print('dist', np.linalg.norm(np.array(x['emb']) - np.array(x['gt_emb']).astype(np.float32)))
            #     # compute all emb.
            #     base_path = Path('./datasets/STD/')
            #     embedding_path = base_path / 'embedding'
            #     gt_embedding = load_pickle(embedding_path / 'embedding_struct_funcs_desc_3.5.pkl')
            #     # compute distances.
            #     all_gt_embeddings = np.array([gt_embedding[x] for x in sorted(gt_embedding.keys())])
            #     arr = np.linalg.norm(all_gt_embeddings - np.array(x['emb']).reshape(1, -1), axis=1)
            #     print(arr)
            #     print(arr[46])
            #     exit()
            pred_dist.append(selected_distances)

    else:
        assert pred_type == 'list'
        pred_dist = pred_arr
    # compute.
    # method1 = compute_coverage_from_interval(gt_arr, pred_arr, agg=False)
    # method2 = compute_accuracy(pred_dist, gt_arr, agg=False)
    # for m1, m2, inter, plist, gt in zip(method1, method2, pred_arr, pred_dist, gt_arr):
    #     if m1 != m2:
    #         print('match result', m1, m2)
    #         print(inter, plist, gt)
    return {
        # 'accuracy': compute_accuracy(pred_dist, gt_arr),
        'accuracy': compute_coverage_from_interval(gt_arr, pred_arr),
        'avg_size': compute_avg_size(pred_dist),
        # 'coverage': compute_coverage_rates(pred_dist, gt_arr).mean(),
        'fdr': compute_false_discovery_rates(pred_dist, gt_arr).mean(),
    }

def eval_metrics_per_row(pdf, pred_arr, gt_arr, dist_col='dist_arr', pred_type='interval'):
    ub_num_arr = []
    lb_num_arr = []
    oor_num_arr = []
    if pred_type == 'interval':
        # select_dist 
        pred_dist = []
        # print(pred_arr.shape)
        for dist_arr, (lb, ub) in zip(pdf[dist_col], pred_arr):
            pred_dist.append(dist_arr[(dist_arr >= lb) & (dist_arr <= ub)])
            ub_num_arr.append(len(dist_arr[dist_arr > ub]))
            lb_num_arr.append(len(dist_arr[dist_arr < lb]))
            oor_num_arr.append(ub_num_arr[-1] + lb_num_arr[-1])
    else:
        assert pred_type == 'list'
        pred_dist = pred_arr
        ub_num_arr = [np.nan] * len(pred_dist)
        lb_num_arr = [np.nan] * len(pred_dist)
        for dist_arr, _row in zip(pdf[dist_col], pred_arr):
            oor_num_arr.append(len(dist_arr) - len(_row))
        
    
    # return a pandas frame.
    acc_arr = compute_coverage_from_interval(gt_arr, pred_arr, agg=False)
    # count mismatch num.
    data = []
    for _id, (_pred, _ori_pred, _acc, _lbn, _ubn, _oor) in enumerate(zip(pred_dist, pred_arr, acc_arr, lb_num_arr, ub_num_arr, oor_num_arr)):
        data.append((_id, len(_pred), _ori_pred, _acc, _lbn, _ubn, _oor))
    return pd.DataFrame(data, columns=['id', 'size', 'pred', 'acc', 'lb_num', 'ub_num', 'oor_size'])

# NOTE: this is wrong, since float values could have errors, use intervals instead.
# def compute_accuracy(pred_dist, gt_dist, agg=True):
#     print('acc compute')
#     acc_arr = []
#     for pred_arr, gt in zip(pred_dist, gt_dist):
#         # print(pred_arr)
#         # print(gt)
#         # if gt < 0:
#         #     exit()
#         # if gt = -1, means we don't have gt.
#         if len(pred_arr) > 0 and gt > 0 and gt >= np.min(pred_arr) and gt <= np.max(pred_arr):
#             acc_arr.append(1)
#         else:
#             acc_arr.append(0)
#     print('acc from all', np.array(acc_arr).sum(), len(acc_arr))
#     if not agg:
#         return acc_arr
#     return np.array(acc_arr).mean()

def compute_avg_size(preds):
    sizes = []
    for arr in preds:
        sizes.append(len(arr))
    # print(sizes)
    return np.array(sizes).mean()

def compute_coverage_rates(pred_dist, gt_dist):
    # coverage is accuracy.
    acc_arr = []
    for pred_arr, gt in zip(pred_dist, gt_dist):
        if len(pred_arr) > 0 and gt >= np.min(pred_arr) and gt <= np.max(pred_arr):
            acc_arr.append(1)
        else:
            acc_arr.append(0)
    return np.array(acc_arr)

def compute_false_discovery_rates(pred_dist, gt_dist):
    acc_arr = []
    for pred_arr, gt in zip(pred_dist, gt_dist):
        if len(pred_arr) > 0 and gt >= np.min(pred_arr) and gt <= np.max(pred_arr):
            acc_arr.append(1/len(pred_arr))
        else:
            acc_arr.append(0)
    return np.array(acc_arr)