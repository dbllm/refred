# non-exchangeable conformal prediction.

from pandas.core.api import DataFrame as DataFrame
from sklearn.svm import SVR
from cllm.reg.basecp import *
from cllm.reg.utils import get_q_hat_weighted
from cllm.reg.groupcp import pinball_loss, GroupNet, objective

import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import time

import logging

def min_max_normlize(weights):
    min_w = np.min(weights)
    max_w = np.max(weights)
    return (weights - min_w) / (max_w - min_w)


class NEWCP(BaseRegCP):

    def __init__(self, model: SVR, 
                 feat_name: str = 'emb', gt_name: str = 'dist', score=None) -> None:
        super().__init__(model, feat_name, gt_name, score)
    
    def calibrate(self, pdf: DataFrame, alpha: float):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        nf_scores = self.score.calibrate(preds, gt_values)
        # compute weights.
        abs_err = np.abs(preds - gt_values)
        # weights should \in [0, 1]
        # smaller error gives higher weights.
        weights = 1 - abs_err / max(abs_err)
        # we compute weights as distances.
        self.q_hat = get_q_hat_weighted(nf_scores, weights, alpha)
    
    def test(self, pdf: pd.DataFrame, return_pred: bool =False, dist_arr_col='dist_arr'):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        pred_values = self.score.predict(preds, self.q_hat)
        if return_pred:
            return pred_values
        return {
            **eval_metrics(pdf, pred_values, gt_values, dist_arr_col),
            'q_hats': [self.q_hat],
        }


class ConditionNEWCP(BaseRegCP):

    def __init__(self, model: SVR, 
                 feat_name: str = 'emb', gt_name: str = 'dist', score=None, 
                 method='softmax', T = 1) -> None:
        super().__init__(model, feat_name, gt_name, score)
        self.weight_method = method
        # T is used for softmax
        self.T = T
    
    def calibrate(self, pdf: DataFrame, alpha: float):
        # no calibration.
        self.alpha = alpha
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        nf_scores = self.score.calibrate(preds, gt_values)
        self.nf_scores = nf_scores
        self.cal_feats = feats
        pass
        # # compute weights.
        # abs_err = np.abs(preds - gt_values)
        # # weights should \in [0, 1]
        # # smaller error gives higher weights.
        # weights = 1 - abs_err / max(abs_err)
        # # we compute weights as distances.
        # self.q_hat = get_q_hat_weighted(nf_scores, weights, alpha)
    
    def test(self, pdf: pd.DataFrame, return_pred: bool =False, dist_arr_col='dist_arr'):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        # print(' pred shape', preds.shape)
        # compute distance
        q_hats = []
        for _feat in feats:
            # weights = np.linalg.norm(self.cal_feats - _feat.reshape(1, -1), axis=1)
            x = _feat
            all_gt = self.cal_feats
            x_norm = x / np.linalg.norm(x)
            all_gt_norm = all_gt / np.linalg.norm(all_gt, axis=1, keepdims=True)

            # Step 2: Compute cosine similarity
            cosine_similarities = np.dot(all_gt_norm, x_norm)

            # Step 3: Compute cosine distance
            weights = 1 - cosine_similarities
            
            #  Reciprocal Softmax: w=e^{-\frac{1}/{d_j T}} / \sum_{...}
            # Softmax-Based Weighting w=e^{-\frac{d_j}{T}} / \sum_{...}
            if self.weight_method == 'softmax':
                weights = - weights
                exp_weights = np.exp(weights/self.T)
                weights = exp_weights / (sum(exp_weights) + 1)
            if self.weight_method == 'softmax2':
                weights += 1e-6
                weights = - 1 / weights
                exp_weights = np.exp(weights/self.T)
                weights = exp_weights / (sum(exp_weights) + 1)
            elif self.weight_method == 'simple':
                # normalize them to [0,1]
                weights = -weights
                pass
            else:
                assert NotImplementedError()
            # smaller weights means close, so we need to negate the distance.
            weights = min_max_normlize(weights)
            # weights = 1-weights

            # print(weights)
            # get q_hats.
            _q_hat = get_q_hat_weighted(self.nf_scores, weights, self.alpha)
            q_hats.append(_q_hat)
        pred_values = self.score.predict(preds, q_hats)
        if return_pred:
            return pred_values
        return {
            **eval_metrics(pdf, pred_values, gt_values, dist_arr_col),
        }


def weighted_objective(h, q, w, X, S, alpha):
    h_X = h(X)  # Shape: (n, m)
    # w_X = w(X)
    m = h_X.shape[1]
    n = len(S)
    total_loss = 0
    for j in range(n):
        for i in range(m):
            total_loss += h_X[j, i] * pinball_loss(q[i], S[j], alpha) * w[j]
    total_loss /= (n * m)
    return total_loss

class GroupNEWCP(BaseRegCP):
    def __init__(self, model: SVR, mgroup: int, 
                 feat_name: str = 'emb', gt_name: str = 'dist', score=None, 
                 num_epochs=200, lr=0.002) -> None:
        self.mgroup = mgroup
        super().__init__(model, feat_name, gt_name, score)
        self.num_epochs = num_epochs
        self.lr = lr
        self.group_net = None

    def calibrate(self, pdf: pd.DataFrame, alpha: float):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        nf_scores = self.score.calibrate(preds, gt_values)
        # solve the optimization problem.
        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        feats_tensor = torch.tensor(feats).type(torch.float32)
        h_net = GroupNet(feats.shape[1], self.mgroup)
        # w_net = WNet(feats.shape[1]*2)
        
        # q = torch.tensor([torch.median(nf_tensor)] * self.mgroup, requires_grad=True)
        q = torch.rand(self.mgroup).requires_grad_()
        # w = torch.tensor([.5] * len(nf_scores), requires_grad=True)

        # Optimizers for the model parameters and quantiles
        optimizer_h = optim.Adam(h_net.parameters(), lr=self.lr)
        # optimizer_w = optim.Adam(w_net.parameters(), lr=self.lr)
        optimizer_q = optim.Adam([q], lr=self.lr)

        # compute W.
        abs_err = np.abs(preds - gt_values)
        # weights should \in [0, 1]
        # smaller error gives higher weights.
        w = 1 - abs_err / max(abs_err)
        

        for epoch in tqdm(range(self.num_epochs)):
            # Step 1: Optimize q while keeping h fixed
            optimizer_q.zero_grad()
            # loss_q = weighted_objective(h_net, q, w_net, feats_tensor, nf_tensor, 1-alpha)
            loss_q = weighted_objective(h_net, q, w, feats_tensor, nf_tensor, 1-alpha)
            loss_q.backward()
            optimizer_q.step()

            # optimizer_w.zero_grad()
            # loss_w = weighted_objective(h_net, q, w_net, feats_tensor, nf_tensor, 1-alpha)
            # loss_w.backward()
            # optimizer_w.step()

            # # Step 2: Optimize h while keeping q fixed
            optimizer_h.zero_grad()
            # loss_h = weighted_objective(h_net, q, w_net, feats_tensor, nf_tensor, 1-alpha)
            loss_h = weighted_objective(h_net, q, w, feats_tensor, nf_tensor, 1-alpha)
            loss_h.backward()
            optimizer_h.step()
            
            # if (epoch + 1) % 10 == 0:
            #     logging.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss_q.item()}, q: {q.detach().numpy()}")

        self.group_net = h_net
        self.q_hats = q.detach().numpy()
        logging.info('q: %s', q)
        # with torch.no_grad():
        #     logging.info('w peek: %s', w_net(feats_tensor)[:20])

    
    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr'):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        feats_tensor = torch.tensor(feats).type(torch.float32)
        with torch.no_grad():
            probs = self.group_net(feats_tensor).detach()
        group_idx = torch.argmax(probs, dim=1)
        q_hats = [self.q_hats[g] for g in group_idx.reshape(-1)]
        pred_intervals = self.score.predict(preds, q_hats)
        if return_pred:
            return pred_intervals
        return eval_metrics(pdf, pred_intervals, pdf[self.gt_name], dist_arr_col)


def temperture_weighted_objective(q, w, tsmodel, S, alpha):
    n = len(S)
    w = tsmodel(w)
    total_loss = 0
    for j in range(n):
        total_loss += pinball_loss(q, S[j], alpha) * w[j]
    total_loss /= n
    # time.sleep(1)
    # logging.info('loss %s', total_loss.item())
    return total_loss

class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.0):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, logits):
        return self.softmax(logits / self.temperature)

class LearnedNEWCP(BaseRegCP):
    # learn two params: \tau & q
    
    def __init__(self, model: SVR, 
                 feat_name: str = 'emb', gt_name: str = 'dist', score=None, 
                 num_epochs=100, lr=0.01) -> None:
        super().__init__(model, feat_name, gt_name, score)
        self.num_epochs = num_epochs
        self.lr = lr
        self.w = None

    def calibrate(self, pdf: pd.DataFrame, alpha: float):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        nf_scores = self.score.calibrate(preds, gt_values)
        # solve the optimization problem.
        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        feats_tensor = torch.tensor(feats).type(torch.float32)
        
        tau_model = TemperatureScaling()
        q = torch.rand(1).requires_grad_()
        
        optimizer_q = optim.Adam([tau_model.temperature, q], lr=self.lr)

        # compute W.
        abs_err = np.abs(preds - gt_values)
        # weights should \in [0, 1]
        # smaller error gives higher weights.
        w = 1 - abs_err / max(abs_err)
        w = torch.tensor(w)
        self.w = w
        
        # perform scaling.


        for epoch in tqdm(range(self.num_epochs)):
            # Step 1: Optimize q while keeping h fixed
            optimizer_q.zero_grad()
            loss_q = temperture_weighted_objective(q, w, tau_model, nf_tensor, 1-alpha)
            loss_q.backward()
            optimizer_q.step()
            
            # if (epoch + 1) % 10 == 0:
            #     logging.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss_q.item()}, q: {q.detach().numpy()}")

        self.q_hats = q.detach().numpy()
        logging.info('q: %s', q.item())
        self.tau_model = tau_model
        logging.info('tau: %s', tau_model.temperature.item())
        # with torch.no_grad():
        #     logging.info('w peek: %s', w_net(feats_tensor)[:20])


    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr'):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        # feats_tensor = torch.tensor(feats).type(torch.float32)
        # weights = nn.functional.softmax(self.w / self.tau)

        pred_intervals = self.score.predict(preds, self.q_hats)
        if return_pred:
            return pred_intervals
        return eval_metrics(pdf, pred_intervals, gt_values, dist_arr_col)
    
    def profile(self, pdf: pd.DataFrame, test_pdf: pd.DataFrame, alpha, target_path, num_epochs=30, lr=0.002):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats)
        nf_scores = self.score.calibrate(preds, gt_values)
        # solve the optimization problem.
        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        feats_tensor = torch.tensor(feats).type(torch.float32)
        
        tau_model = TemperatureScaling()
        q = torch.rand(1).requires_grad_()
        
        optimizer_q = optim.Adam([tau_model.temperature, q], lr=lr)

        # compute W.
        abs_err = np.abs(preds - gt_values)
        # weights should \in [0, 1]
        # smaller error gives higher weights.
        w = 1 - abs_err / max(abs_err)
        w = torch.tensor(w)
        self.w = w
        
        # perform scaling.


        for epoch in tqdm(range(num_epochs)):
            # Step 1: Optimize q while keeping h fixed
            optimizer_q.zero_grad()
            loss_q = temperture_weighted_objective(q, w, tau_model, nf_tensor, 1-alpha)
            loss_q.backward()
            optimizer_q.step()

            logging.info('q: %s, tau %s', q.item(), tau_model.temperature.item())
            
            # if (epoch + 1) % 10 == 0:
            #     logging.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss_q.item()}, q: {q.detach().numpy()}")

        self.q_hats = q.detach().numpy()
        logging.info('q: %s', q.item())
        self.tau_model = tau_model
        logging.info('tau: %s', tau_model.temperature.item())
