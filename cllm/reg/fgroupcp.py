# fixed group cp.
from sklearn.svm import SVR
from cllm.reg.basecp import *
from cllm.reg.utils import get_q_hat_weighted
from cllm.reg.eval import compute_coverage_from_interval

import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import time

import logging

from cllm.reg.groupcp import TrailEval, pinball_loss
from typing import List

class PredCondFixedCP(BaseRegCP):
    '''Prediction-conditioned fixed cp.
    Simply splits data points into groups based on the prediction value.
    '''
    def __init__(self, model: SVR, group_ratios: List[float],
                 reg_feat_name: str = 'emb', gt_name: str = 'dist',
                score=None, weight_method='none'):
        self.group_ratios = group_ratios
        assert sum(self.group_ratios) == 1
        self.group_m = len(self.group_ratios)
        self.weight_method = weight_method
        super().__init__(model, reg_feat_name, gt_name, score)
    
    def predict_labels(self, pdf: pd.DataFrame):
        feats, _ = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds= self.model.predict(feats)
        # get labels.
        group_idx = []
        for pred in preds:
            added=False
            for i, _rang in enumerate(zip(self.preds_arr[:-1], self.preds_arr[1:])):
                if pred > _rang[0] and pred <= _rang[1]:
                    group_idx.append(i)
                    added=True
            if not added:
                # out-of-range, assign to the last group.
                group_idx.append(self.group_m-1)
            
        group_idx = np.array(group_idx)
        return group_idx
    
    def calibrate(self, pdf: pd.DataFrame, alpha: float):
        self.alpha = alpha
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        gt_values = gt_values.reshape(-1)
        preds = self.model.predict(feats).reshape(-1)
        nf_scores = self.score.calibrate(preds, gt_values)

        # # print(score_qs)
        # # print(nf_scores)
        # # compute sizes.

        # lb_arr = np.array(pdf[self.gt_name] - nf_scores)
        # ub_arr = np.array(pdf[self.gt_name] + nf_scores)
        # len_num = []
        # for _i in range(len(pdf)):
        #     _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
        #     _inrange_num = _dist_arr[(_dist_arr >= lb_arr[_i]) & (_dist_arr <= ub_arr[_i])]
        #     len_num.append(len(_inrange_num))
        # pdf['pred_size'] = len_num

        sorted_preds = sorted(preds)

        len_qs = [-1]
        for q_pos in np.cumsum(self.group_ratios):
            len_qs.append(np.quantile(sorted_preds, q_pos))
        
        group_labels = []
        for s in sorted_preds:
            # reverse
            for i, range_pair in enumerate(zip(len_qs[:-1], len_qs[1:])):
                if s > range_pair[0] and s <= range_pair[1]:
                    group_labels.append(i)
        
        self.preds_arr = len_qs
        
        group_labels = np.array(group_labels)

        # collect nf scores.
        self.q_hats = []
        if self.weight_method == 'none':
            for group_i in range(self.group_m):
                group_nf_scores = nf_scores[group_labels == group_i]
                if len(group_nf_scores) == 0:
                    self.q_hats.append(-1)
                else:
                    # self.q_hats.append(np.quantile(group_nf_scores, 1-alpha))
                    self.q_hats.append(get_q_hat(group_nf_scores, alpha))
        else:
            # get group points.
            self.group_feats = []
            self.group_nfscores = []
            for i in range(self.group_m):
                self.group_feats.append(feats[group_labels==i])
                self.group_nfscores.append(nf_scores[group_labels==i])

        print(self.q_hats)
    
    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr', use_recalibrated=True):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        
        # check preds fall into which group.
        group_idx = self.predict_labels(pdf)

        if self.weight_method == 'none':
            q_hats = self.q_hats
            q_hats = np.array([q_hats[g] for g in group_idx.reshape(-1)])
            # reshape
            q_hats = q_hats.reshape(-1)
            pred_intervals = self.score.predict(preds, q_hats)
        else:
            # compute q_hat for each datapoint.
            q_hats = []
            for _gid, _feat in zip(group_idx, feats):
                weights = np.linalg.norm(self.group_feats[_gid] - _feat.reshape(1, -1), axis=1)
                if self.weight_method == 'softmax':
                    weights += 1e-6
                    weights = -1/weights
                    exp_weights = np.exp(weights / 0.1)
                    weights = exp_weights / sum(exp_weights)
                weights = min_max_normlize(weights)
                _q_hat = get_q_hat_weighted(self.group_nfscores[_gid], weights, self.alpha)
                q_hats.append(_q_hat)
            q_hats = np.array(q_hats).reshape(-1)
            pred_intervals = self.score.predict(preds, q_hats)
        if return_pred:
            return pred_intervals
        # group_idx

        group_coverages = []
        group_avgsizes = []
        group_nums = []
        for i in range(self.group_m):
            group_nums.append((group_idx==i).astype(int).sum().item())
            # get intervals. 
            group_mask = (group_idx == i).tolist()
            # print(len(group_mask))
            # print(pred_intervals)
            if np.sum(group_mask) == 0:
                group_coverages.append(0)
                group_avgsizes.append(0)
            else:
                group_pdf = pdf[group_mask]
                group_inters = pred_intervals[group_mask]
                group_gts = gt_values[group_mask]
                group_coverages.append(compute_coverage_from_interval(group_gts, group_inters))
                # compute average size.
                group_metrics = eval_metrics(group_pdf, group_inters, group_gts)
                group_avgsizes.append(group_metrics['avg_size'])
        
        # add q 
        return {
            **eval_metrics(pdf, pred_intervals, pdf[self.gt_name], dist_arr_col),
            'gp_num': group_nums,
            'q_hats': self.q_hats,
            # compute coverage & avg_size per group.
            'gp_coverages': group_coverages,
            'gp_avg_sizes': group_avgsizes
        }


def min_max_normlize(weights):
    min_w = np.min(weights)
    max_w = np.max(weights)
    return (weights - min_w) / (max_w - min_w + 1e-6)


class GroupNet(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(input_dim, output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def objective(h, q, X, S, alpha):
    h_X = h(X)  # Shape: (n, m)
    h_X = torch.softmax(h_X, dim=1)
    m = h_X.shape[1]
    n = len(S)
    
    # Expand q and S to enable broadcasting
    q = q.unsqueeze(0).expand(n, m)  # Shape: (n, m)
    S = S.unsqueeze(1).expand(n, m)  # Shape: (n, m)
    
    # Compute pinball_loss for the entire batch
    losses = pinball_loss(q, S, alpha)  # Shape: (n, m)
    
    # Compute the total loss
    total_loss = torch.sum(h_X * losses) / n
    
    return total_loss

class FixedGroupCP(BaseRegCP):
    def __init__(self, model: SVR, group_ratios: List[float],
                 reg_feat_name: str = 'emb', h_feat_name: str='emb', gt_name: str = 'dist',
                num_epochs=500, num_epochs_pinball=0, score=None, 
                weight_method='none', T = 0.1,
                split_by='score'):
        self.group_ratios = group_ratios
        assert sum(self.group_ratios) == 1
        self.group_m = len(self.group_ratios)
        self.h_feat_name = h_feat_name
        self.num_epochs = num_epochs
        self.num_epochs_pinball = num_epochs_pinball
        self.weight_method = weight_method
        assert split_by in ['score', 'size', 'pred']
        self.split_by = split_by
        self.T = T
        super().__init__(model, reg_feat_name, gt_name, score)
    
    def train_group_net_ce(self, feats_tensor, group_tensor):
        self.h_net = GroupNet(feats_tensor.shape[1], self.group_m)
        self.h_net.train()

        # pretrain.
        optimizer_ce = optim.Adam(self.h_net.parameters(), lr=0.01)
        
        loss_h_func = torch.nn.CrossEntropyLoss()
        for epoch in tqdm(range(self.num_epochs)):
            optimizer_ce.zero_grad()
            h_y = self.h_net(feats_tensor)
            loss_h = loss_h_func(h_y, group_tensor)
            loss_h.backward()
            optimizer_ce.step()
            
            # Weight clipping
            for param in self.h_net.parameters():
                param.data.clamp_(min=1e-6) 
        
        # calibrate.
        self.h_net.eval()
        h_y = self.h_net(feats_tensor)
        h_y = torch.argmax(torch.softmax(h_y, dim=1), dim=1).detach().numpy()
        return h_y
    
    def train_group_net_pinball(self, feats_tensor, nf_tensor, alpha):
        self.h_net.train()
        q_hats_tensor = torch.tensor(self.q_hats)
        optimizer_h = optim.Adam(self.h_net.parameters(), lr=0.01)
        optimizer_q = optim.Adam([q_hats_tensor], lr=0.01)
        for epoch in tqdm(range(self.num_epochs_pinball)):
            # freeze group.
            # optimizer_h.zero_grad()
            # loss_h = objective(self.h_net, q_hats_tensor, feats_tensor, nf_tensor, alpha)
            # loss_h.backward()
            # optimizer_h.step()

            # # Weight clipping
            # for param in self.h_net.parameters():
            #     param.data.clamp_(min=1e-6) 
            
            optimizer_q.zero_grad()
            loss_q = objective(self.h_net, q_hats_tensor, feats_tensor, nf_tensor, alpha)
            loss_q.backward()
            optimizer_q.step()

        self.q_hats = q_hats_tensor.numpy().tolist()
        self.h_net.eval()
        print(self.q_hats)
    
    def compute_group_labels(self, pdf: pd.DataFrame, nf_scores, preds):
        
        if self.split_by == 'size':
            # compute sizes.
            # lb_arr = np.array(pdf[self.gt_name] - nf_scores)
            lb_arr = np.zeros(len(pdf))
            ub_arr = np.array(pdf[self.gt_name] + nf_scores)
            len_num = []
            for _i in range(len(pdf)):
                _dist_arr = np.array(pdf.iloc[_i]['dist_arr'])
                _inrange_num = _dist_arr[(_dist_arr >= lb_arr[_i]) & (_dist_arr <= ub_arr[_i])]
                len_num.append(len(_inrange_num))
            pdf['pred_size'] = len_num

            sorted_len_num = sorted(len_num)

            len_qs = [-1]
            for q_pos in np.cumsum(self.group_ratios):
                len_qs.append(np.quantile(sorted_len_num, q_pos))
            
            group_labels = []
            for s in len_num:
                # reverse
                for i, range_pair in enumerate(zip(len_qs[:-1], len_qs[1:])):
                    if s > range_pair[0] and s <= range_pair[1]:
                        group_labels.append(i)
        elif self.split_by == 'score':
            # split by nf_score.
            sorted_nf_scores = sorted(nf_scores)

            score_qs = [-1]
            for q_pos in np.cumsum(self.group_ratios):
                score_qs.append(np.quantile(sorted_nf_scores, q_pos))

            group_labels = []
            for s in nf_scores:
                # reverse
                for i, range_pair in enumerate(zip(score_qs[:-1], score_qs[1:])):
                    if s > range_pair[0] and s <= range_pair[1]:
                        group_labels.append(i)
        elif self.split_by == 'pred':
            
            sorted_preds = sorted(preds)

            pred_qs = [-1]
            for q_pos in np.cumsum(self.group_ratios):
                pred_qs.append(np.quantile(sorted_preds, q_pos))

            group_labels = []
            for s in preds:
                # reverse
                for i, range_pair in enumerate(zip(pred_qs[:-1], pred_qs[1:])):
                    if s > range_pair[0] and s <= range_pair[1]:
                        group_labels.append(i)
        else:
            raise NotImplementedError()
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
        group_labels = self.compute_group_labels(pdf, nf_scores, preds)

        group_labels = np.array(group_labels)

        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        feats_tensor = torch.tensor(h_feats).type(torch.float32)
        group_tensor = torch.tensor(group_labels).type(torch.long)

        print(group_tensor.shape[0], nf_tensor.shape[0], nf_scores.shape)
        assert group_tensor.shape[0] == nf_tensor.shape[0]

        h_y = self.train_group_net_ce(feats_tensor, group_tensor)
        # check if there is values for each group.
        while len(np.unique(h_y)) != self.group_m:
            logging.info('invalid results, try to retrain...')
            h_y = self.train_group_net_ce(feats_tensor, group_tensor)
        
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


    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr', use_recalibrated=True):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        feats_tensor = torch.tensor(feats).type(torch.float32)
        with torch.no_grad():
            probs = self.h_net(feats_tensor).detach()
            probs = torch.softmax(probs, dim=1)
        group_idx = torch.argmax(probs, dim=1)
        if self.weight_method == 'none':
            q_hats = self.q_hats
            q_hats = np.array([q_hats[g] for g in group_idx.reshape(-1)])
            # reshape
            q_hats = q_hats.reshape(-1)
            pred_intervals = self.score.predict(preds, q_hats)
        else:
            # compute q_hat for each datapoint.
            q_hats = []
            for _gid, _feat in zip(group_idx, feats):
                # weights = np.linalg.norm(self.group_feats[_gid] - _feat.reshape(1, -1), axis=1)
                x = _feat
                all_gt = self.group_feats[_gid]
                x_norm = x / np.linalg.norm(x)
                all_gt_norm = all_gt / np.linalg.norm(all_gt, axis=1, keepdims=True)

                # Step 2: Compute cosine similarity
                cosine_similarities = np.dot(all_gt_norm, x_norm)

                # Step 3: Compute cosine distance
                weights = 1 - cosine_similarities

                if self.weight_method == 'softmax':
                    # weights += 1e-6
                    # weights = -1/weights
                    weights = - weights
                    exp_weights = np.exp(weights / self.T)
                    weights = exp_weights / (sum(exp_weights) + 1)
                weights = min_max_normlize(weights)
                _q_hat = get_q_hat_weighted(self.group_nfscores[_gid], weights, self.alpha)
                q_hats.append(_q_hat)
            q_hats = np.array(q_hats).reshape(-1)
            pred_intervals = self.score.predict(preds, q_hats)
        if return_pred:
            return pred_intervals
        # group_idx

        group_coverages = []
        group_avgsizes = []
        group_nums = []
        for i in range(self.group_m):
            group_nums.append((group_idx==i).int().sum().item())
            # get intervals. 
            group_mask = (group_idx == i).tolist()
            # print(len(group_mask))
            # print(pred_intervals)
            if np.sum(group_mask) == 0:
                group_coverages.append(0)
                group_avgsizes.append(0)
            else:
                group_pdf = pdf[group_mask]
                group_inters = pred_intervals[group_mask]
                group_gts = gt_values[group_mask]
                group_coverages.append(compute_coverage_from_interval(group_gts, group_inters))
                # compute average size.
                group_metrics = eval_metrics(group_pdf, group_inters, group_gts)
                group_avgsizes.append(group_metrics['avg_size'])
        
        # add q 
        return {
            **eval_metrics(pdf, pred_intervals, pdf[self.gt_name], dist_arr_col),
            'gp_num': group_nums,
            'q_hats': self.q_hats,
            # compute coverage & avg_size per group.
            'gp_coverages': group_coverages,
            'gp_avg_sizes': group_avgsizes
        }
