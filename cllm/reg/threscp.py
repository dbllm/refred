from cllm.reg.basecp import BaseRegCP, transform_x_y, ConformalScore2Norm
from cllm.reg.fgroupcp import FixedGroupCP, min_max_normlize, objective
from cllm.reg.utils import get_q_hat, get_q_hat_weighted
from sklearn.svm import SVR
from typing import List
import logging
import pandas as pd
import torch
import numpy as np
from torch import optim
from tqdm import tqdm

from cllm.reg.eval import compute_avg_size, eval_metrics

class AvgSizeThresCP(FixedGroupCP):
    def __init__(self, model: SVR, target_avg_size: float,
                reg_feat_name: str = 'emb', h_feat_name: str='emb', gt_name: str = 'dist',
            num_epochs=500, num_epochs_pinball=0, score=None, 
            weight_method='none', T = 0.1, epsilon=0, use_weight_calibration=False):
        self.target_avg_size = target_avg_size
        self.group_m = 2
        self.h_feat_name = h_feat_name
        self.num_epochs = num_epochs
        self.num_epochs_pinball = num_epochs_pinball
        self.weight_method = weight_method
        self.T = T
        self.epsilon=epsilon
        self.use_weight_calibration = use_weight_calibration

        self.model = model
        self.feat_name = reg_feat_name
        self.gt_name = gt_name
        if score is None:
            self.score = ConformalScore2Norm()
        else:
            self.score = score

    def compute_group_labels_iterative(self, pdf: pd.DataFrame, nf_scores, feats, preds, alpha):
        # 
        # compute the distance to each prediction.
        # get init_q_hat.
        # all_q_hats = []
        # for dist_arr, pred in zip(pdf['dist_arr'], preds):
        #     abs_arr = np.abs(np.array(dist_arr) - pred)
        #     all_q_hats.append(abs_arr)
        # sorted_q_values = np.concatenate(all_q_hats)
        # sorted_q_values = np.sort(sorted_q_values)
        # target_q_hat = np.quantile(sorted_q_values, self.target_avg_size * len(preds) / len(sorted_q_values))
        # # avg_size = target_size/total quantile.
        # # print(q_hat_candidates)
        # # q_hat = min(q_hat_candidates)
        # q_hat = target_q_hat


        def compute_len_arr(preds, q_hat):
            lb_arr = np.array(preds) - q_hat
            ub_arr = np.array(preds) + q_hat
            len_num = []
            for _i in range(len(pdf)):
                _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
                _inrange_num = _dist_arr[(_dist_arr >= lb_arr[_i]) & (_dist_arr <= ub_arr[_i])]
                len_num.append(len(_inrange_num))
            return np.array(len_num)
        
        # 1. compute cf_score, split it into two groups.
        len_num_base = compute_len_arr(preds, nf_scores)

        init_group_labels = (len_num_base > self.target_avg_size).astype(int)

        # compute q_hat for the first group.
        def compute_q_hat_and_len_num(_group_labels):
            if self.weight_method == 'none' or not self.use_weight_calibration:
                # print(len(nf_scores[_group_labels==0]), alpha)
                current_q_hat = get_q_hat(nf_scores[_group_labels==0], alpha)
                # compute avg size under this q_hat
                current_len_num = compute_len_arr(preds, current_q_hat)
            else:
                # q_hat is meaningless, 
                current_q_hat = []
                g_feats = [feats[_group_labels == 0], feats[_group_labels == 1]]
                g_nf_scores = [nf_scores[_group_labels==0], nf_scores[_group_labels==1]]
                # compute distances.
                for _feat, _label in zip(feats, _group_labels):
                    # computes
                    # weights = np.linalg.norm(g_feats[_label] - _feat.reshape(1, -1), axis=1)
                    x = _feat
                    all_gt = g_feats[_label]
                    x_norm = x / np.linalg.norm(x)
                    all_gt_norm = all_gt / np.linalg.norm(all_gt, axis=1, keepdims=True)

                    # Step 2: Compute cosine similarity
                    cosine_similarities = np.dot(all_gt_norm, x_norm)

                    # Step 3: Compute cosine distance
                    weights = 1 - cosine_similarities
            
                    if self.weight_method == 'softmax':
                        weights += 1e-6
                        weights = -1/weights
                        exp_weights = np.exp(weights / self.T)
                        # print('sum', sum(exp_weights))
                        exp_weights += 1e-6
                        weights = exp_weights / sum(exp_weights)
                    weights+= 1e-6
                    weights = min_max_normlize(weights)
                    _q_hat = get_q_hat_weighted(g_nf_scores[_label], weights, self.alpha)
                    current_q_hat.append(_q_hat)

                current_len_num = compute_len_arr(preds, current_q_hat)
            return current_q_hat, current_len_num
        
        current_q_hat, current_len_num = compute_q_hat_and_len_num(init_group_labels)
        current_avg_size = current_len_num[init_group_labels == 0].mean()

        print(current_avg_size)
        # return init_group_labels
        group_labels = init_group_labels
        q_hat = current_q_hat
        len_num = current_len_num
        avg_size = current_avg_size

        if current_avg_size > self.target_avg_size - self.epsilon:
            # sort by cf_score, and iteratively remove on.
            sorted_scores = sorted(nf_scores[init_group_labels==0], reverse=True)
            for _s in sorted_scores:
                group_labels = (~((nf_scores<_s) & (init_group_labels==0))).astype(int)
            # sorted_preds = sorted(preds[init_group_labels==0], reverse=True)
            # for _s in sorted_preds:
            #     group_labels = (~((preds < _s) & (init_group_labels ==0))).astype(int)
                q_hat, len_num = compute_q_hat_and_len_num(group_labels)
                # q_hat = get_q_hat(nf_scores[group_labels==0], alpha)
                # len_num = compute_len_arr(preds, q_hat)
                avg_size = len_num[group_labels==0].mean()
                if avg_size + self.epsilon < self.target_avg_size:
                    break
        elif current_avg_size + self.epsilon < self.target_avg_size:
            # same, but for the other group.
            sorted_scores = sorted(nf_scores[init_group_labels==1])
            for _s in sorted_scores:
                tmp_group_labels = ((nf_scores>_s) & (init_group_labels==1)).astype(int)
            # sorted_preds = sorted(preds[init_group_labels==1])
            # for _s in sorted_preds:
            #     tmp_group_labels = ((preds>_s) & (init_group_labels==1)).astype(int)
                tmp_q_hat, tmp_len_num = compute_q_hat_and_len_num(tmp_group_labels)
                # q_hat = get_q_hat(nf_scores[group_labels==0], alpha)
                # len_num = compute_len_arr(preds, q_hat)
                tmp_avg_size = tmp_len_num[tmp_group_labels==0].mean()
                if tmp_avg_size + self.epsilon > self.target_avg_size:
                    break
                # update.
                group_labels, q_hat, len_num, avg_size = tmp_group_labels, tmp_q_hat, tmp_len_num, tmp_avg_size
        print(avg_size)
        self.target_q_hat = q_hat
        return group_labels
        
    
    def compute_group_labels_binary(self, pdf: pd.DataFrame, nf_scores, preds, alpha):
        # 2 groups.
        # finds all points that in the given size. 
        # compute sizes.
        # lb_arr = np.array(pdf[self.gt_name] - nf_scores)
        
        # init_lb_arr = np.zeros(len(pdf) )
        init_lb_arr = np.array(preds + nf_scores)
        init_ub_arr = np.array(preds + nf_scores)

        def compute_len_arr(lb_arr, ub_arr):
            len_num = []
            for _i in range(len(pdf)):
                _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
                _inrange_num = _dist_arr[(_dist_arr >= lb_arr[_i]) & (_dist_arr <= ub_arr[_i])]
                len_num.append(len(_inrange_num))
            return np.array(len_num)

        def split_group(len_num):
            target_avg_size = self.target_avg_size
            group_labels = (len_num > target_avg_size).astype(int)
            prev_group_labels = group_labels
            avg_size = len_num[group_labels==0].mean()
            while avg_size < self.target_avg_size:
                if np.all(group_labels==0): 
                    prev_group_labels = group_labels
                    break
                target_avg_size += 1
                prev_group_labels = group_labels
                group_labels = (len_num > target_avg_size).astype(int)
                avg_size = len_num[group_labels==0].mean()
            return prev_group_labels

        # len_num = compute_len_arr(init_lb_arr, init_ub_arr)
        # group_labels = split_group(len_num)

        # group_nf_scores = nf_scores[group_labels == 0]
        # q_hat = get_q_hat(group_nf_scores, alpha)
        group_labels = np.zeros(len(nf_scores))
        lb = 0
        ub = max(nf_scores)

        precision = 0.0001

        while ub - lb > precision:
            # print(lb, ub)
            # computes current avg size.
            current_q_hat = (ub - lb) / 2 + lb
            # print('qhat', current_q_hat)
            # print('pred:', preds[:10])
            new_lb_arr = np.array(preds - [current_q_hat]* len(preds))
            # print('new_lb:', new_lb_arr[:10])
            new_ub_arr = np.array(preds + [current_q_hat]* len(preds))
            len_num = compute_len_arr(new_lb_arr, new_ub_arr)
            avg_size = len_num[group_labels == 0].mean()
            group_labels = split_group(len_num)

            # print('expect size', avg_size)

            # print('preds', preds[group_labels==0][:10])
            # print('cq:', current_q_hat)
            # # computes q _hats
            # _pred_inter = self.score.predict(preds, current_q_hat)
            # print('predsinter', _pred_inter[:10])
            # convert to dist arr.

            # _size = eval_metrics(_pred_inter)['avg_size']

            # print((group_labels==0)[:10])

            # len_x = []
            # preds_inter2 = []
            # for _i, _select, _inter in zip(range(len(pdf)), group_labels==0, _pred_inter):
            #     if not _select: continue
            #     if _i < 10:
            #         print('range', new_lb_arr[_i], new_ub_arr[_i])
            #     _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
            #     print(_inter, 'expect:', new_lb_arr[_i], new_ub_arr[_i])
            #     _inrange_num = _dist_arr[(_dist_arr >= _inter[0]) & (_dist_arr <= _inter[1])]
            #     preds_inter2.append(_inrange_num)
            #     len_x.append(len(_inrange_num))

            # _size = compute_avg_size(preds_inter2)
            # print('eval size', _size)
            # print(np.array(len_x).mean())
            
            # for _pi, _l in zip(_pred_inter, preds_inter2):
            #     print(_pi, 'expect:', _l)

            # exit()

            if avg_size < self.target_avg_size:
                lb = current_q_hat
            else:
                ub = current_q_hat
        # final 
        q_hat = (ub + lb) / 2
        new_lb_arr = np.array(preds - [q_hat]* len(preds))
        new_ub_arr = np.array(preds + [q_hat]* len(preds))
        len_num = compute_len_arr(new_lb_arr, new_ub_arr)
        avg_size = len_num[group_labels == 0].mean()

        group_labels = split_group(len_num)
        print('final size:', avg_size)


        preds = []
        len_x = []
        for _i in range(len(pdf)):
            _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
            _inrange_num = _dist_arr[(_dist_arr >= new_lb_arr[_i]) & (_dist_arr <= new_ub_arr[_i])]
            if group_labels[_i] == 0:
                preds.append(_inrange_num)
                len_x.append(len(_inrange_num))
                # print('compare', len(_inrange_num), len_num[_i])
        
        # print(compute_avg_size(preds))
        # print(len_x)

        # print((group_labels== 0).sum(), (group_labels == 1).sum())
        # print(avg_size)
        # exit()
        return group_labels

    def compute_group_labels(self, pdf: pd.DataFrame, nf_scores, preds, alpha):
        # 2 groups.
        # finds all points that in the given size. 
        # compute sizes.
        # lb_arr = np.array(pdf[self.gt_name] - nf_scores)
        
        init_lb_arr = np.zeros(len(pdf))
        # ub_arr = np.array(pdf[self.gt_name] + nf_scores)
        init_ub_arr = np.array(preds + nf_scores)

        def compute_len_arr(lb_arr, ub_arr):
            len_num = []
            for _i in range(len(pdf)):
                _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
                _inrange_num = _dist_arr[(_dist_arr >= lb_arr[_i]) & (_dist_arr <= ub_arr[_i])]
                len_num.append(len(_inrange_num))
            return np.array(len_num)

        def split_group(len_num):
            return (len_num > self.target_avg_size).astype(int)

        len_num = compute_len_arr(init_lb_arr, init_ub_arr)
        group_labels = split_group(len_num)

        terminate = False

        while not terminate:
            # get 
            # compute
            # calibrate.
            # compute q_hat.
            group_nf_scores = nf_scores[group_labels == 0]
            q_hat = get_q_hat(group_nf_scores, alpha)

            # validate.
            new_ub_arr = np.array(preds + [q_hat]* len(preds))
            
            len_num = compute_len_arr(init_lb_arr, new_ub_arr)
            # compute_avg_size
            avg_size = len_num[group_labels == 0].mean()
            print('avg size', avg_size)
            if avg_size <= self.target_avg_size:
                terminate = True
            # update group labels.
            group_labels = split_group(len_num)
        # for _ln, _l in zip(len_num, group_labels):
        #     print(_ln, _l)
        print((group_labels== 0).sum(), (group_labels == 1).sum())
        print(avg_size)
        # exit()
        return group_labels

    def calibrate(self, pdf: pd.DataFrame, alpha: float):
        self.alpha = alpha
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        gt_values = gt_values.reshape(-1)
        h_feats, _ = transform_x_y(pdf, self.h_feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        nf_scores = self.score.calibrate(preds, gt_values)

        # print(score_qs)
        # print(nf_scores)
        # group_labels = self.compute_group_labels_binary(pdf, nf_scores, preds, alpha)
        group_labels = self.compute_group_labels_iterative(pdf, nf_scores, feats, preds, alpha)

        group_labels = np.array(group_labels)

        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        feats_tensor = torch.tensor(h_feats).type(torch.float32)
        group_tensor = torch.tensor(group_labels).type(torch.long)

        print(group_tensor.shape[0], nf_tensor.shape[0], nf_scores.shape)
        assert group_tensor.shape[0] == nf_tensor.shape[0]

        h_y = self.train_group_net_ce(feats_tensor, group_tensor)
        # check if there is values for each group.
        attempts = 3
        while len(np.unique(h_y)) != self.group_m and attempts > 0:
            logging.info('invalid results, try to retrain...')
            h_y = self.train_group_net_ce(feats_tensor, group_tensor)
            attempts -= 1
        
        # compute difference.
        count = 0
        for pred_l, gt_l in zip(h_y, group_labels):
            if pred_l != gt_l:
                count +=1 
        print('diff label count:', count)
        # exit()
        
        # collect nf scores.
        self.q_hats = []
        if self.weight_method == 'none':
            for group_i in range(self.group_m):
                group_nf_scores = nf_scores[h_y == group_i]
                if len(group_nf_scores) == 0:
                    self.q_hats.append(-1)
                else:
                    # self.q_hats.append(np.quantile(group_nf_scores, 1-alpha))
                    self.q_hats.append(get_q_hat(group_nf_scores, alpha))
            self.q_hats[0] = self.target_q_hat
        else:
            # get group points.
            self.group_feats = []
            self.group_nfscores = []
            for i in range(self.group_m):
                self.group_feats.append(feats[h_y==i])
                self.group_nfscores.append(nf_scores[h_y==i])
        print(self.q_hats)

        if self.num_epochs_pinball > 0:
            self.train_group_net_pinball(feats_tensor, nf_tensor, alpha)
        


    def train_group_net_pinball(self, feats_tensor, nf_tensor, alpha):
        self.h_net.train()
        q_hats_tensor = torch.tensor(self.q_hats)
        optimizer_h = optim.Adam(self.h_net.parameters(), lr=0.01)
        optimizer_q = optim.Adam([q_hats_tensor], lr=0.01)
        for epoch in tqdm(range(self.num_epochs_pinball)):
            optimizer_h.zero_grad()
            loss_h = objective(self.h_net, q_hats_tensor, feats_tensor, nf_tensor, alpha)
            loss_h.backward()
            optimizer_h.step()

            # Weight clipping
            for param in self.h_net.parameters():
                param.data.clamp_(min=1e-6) 
            
            optimizer_q.zero_grad()
            loss_q = objective(self.h_net, q_hats_tensor, feats_tensor, nf_tensor, alpha)
            loss_q.backward()
            optimizer_q.step()
            # set the first q_hat back.
            q_hats_tensor[0] = self.target_q_hat

        self.q_hats = q_hats_tensor.numpy().tolist()
        self.h_net.eval()
        print(self.q_hats)

if __name__ == '__main__':
    # train to split the data points into two groups.
    import torch
    from torch import nn
    from cllm.reg.io import load_std_data
    from cllm.io import split_data
    import numpy as np
    from sklearn.svm import SVR
    from cllm.reg.basecp import transform_x_y
    from IPython.utils import io
    from cllm.reg.fgroupcp import FixedGroupCP, PredCondFixedCP
    from cllm.reg.basecp import BaseRegCP, OptimalSolution, ConformalScore2Norm, ConformalScoreRightSide, RawPredSolution

    pdf = load_std_data('aug_onegt', base_path='./datasets/STD')
    train_data, validate_data, test_data = split_data(pdf, [0.4, 0.3, 0.3], shuffle=True, random_seed=42)
    train_x, train_y = transform_x_y(train_data, 'emb', 'dist')
    model = SVR()

    # group_cp = AvgSizeThresCP(model, 1, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', epsilon=0)
    # group_cp = AvgSizeThresCP(model, 1, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', epsilon=0)
    # group_cp = AvgSizeThresCP(model, 2, num_epochs=500, num_epochs_pinball=500, 
    #                           score=ConformalScore2Norm(), weight_method='none', epsilon=0)
    # group_cp = AvgSizeThresCP(model, 2, num_epochs=1000, score=ConformalScore2Norm(), weight_method='none', epsilon=0)
    group_cp = AvgSizeThresCP(model, 2, num_epochs=1000, score=ConformalScore2Norm(), weight_method='softmax', epsilon=0, T=0.1)
    group_cp.train(train_data)
    group_cp.calibrate(validate_data, 0.1)
    res = group_cp.test(validate_data)
    print(res)
    res = group_cp.test(test_data)
    print(res)