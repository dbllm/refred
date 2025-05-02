from sklearn.svm import SVR
from cllm.reg.basecp import *
from cllm.reg.eval import compute_coverage_from_interval

import torch.nn as nn
import torch
import torch.optim as optim
from tqdm import tqdm
import time

import logging

from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple

class GroupNet4QA(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 4)
        self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(4, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        # print(x[:5])
        soft_m = torch.softmax(x, dim=1)
        # print(soft_m[:5])
        # exit()
        return soft_m
        # return torch.softmax(self.fc3(x), dim=1)


class GroupNet(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        # self.fc1 = nn.Linear(input_dim, 500)
        # self.fc2 = nn.Linear(500, 200)
        # self.dropout = nn.Dropout(p=0.5)
        # self.fc3 = nn.Linear(200, output_dim)

        self.fc1 = nn.Linear(input_dim, output_dim * 2)
        # self.fc2 = nn.Linear(10, 2)
        # self.dropout = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(output_dim * 2, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        # print(x[:5])
        soft_m = torch.softmax(x, dim=1)
        # print(soft_m[:5])
        # exit()
        return soft_m
        # return torch.softmax(self.fc3(x), dim=1)

def pinball_loss(q, s, alpha):
    error = q - s
    loss = torch.where(error > 0, alpha * error, (1 - alpha) * -error)
    return loss

def objective(h, q, X, S, alpha):
    h_X = h(X)  # Shape: (n, m)
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

# def objective(h, q, X, S, alpha):
#     # print()
#     h_X = h(X)  # Shape: (n, m)
#     # print(h_X[:5])
#     # time.sleep(1)
#     m = h_X.shape[1]
#     n = len(S)
#     total_loss = 0
#     for j in range(n):
#         for i in range(m):
#             tl =  h_X[j, i] * pinball_loss(q[i], S[j], alpha)
#             # print(tl.shape)
#             total_loss += tl
#     # total_loss /= (n * m)
#     total_loss /= n

#     # entropy loss
#     # entropy = - torch.mean(torch.sum(h_X * torch.log(h_X+1e-10), dim=1))

#     # return total_loss.mean() - entropy * 0.5
#     return total_loss.mean()


TrailEval = namedtuple('TrailEval', ['q_hats', 'groups', 'loss', 'coverage', 'group_coverages', 'group_avgsizes', 'seed'])

class GroupCP(BaseRegCP):
    def __init__(self, model: SVR, mgroup: int, 
                 reg_feat_name: str = 'emb', h_feat_name: str='emb', gt_name: str = 'dist', score=None, 
                 num_epochs=50, lr_h=0.01, lr_q =0.01, model_arch=GroupNet) -> None:
        self.mgroup = mgroup
        super().__init__(model, reg_feat_name, gt_name, score)
        self.h_feat_name = h_feat_name
        self.num_epochs = num_epochs
        self.lr_h = lr_h
        self.lr_q = lr_q
        self.group_net = None
        self.auto_tune_history = []
        self.q_hats_prime = None
        self.model_arch = model_arch

    def auto_tune(self,  pdf: pd.DataFrame, alpha: float, num_epochs = None, num=10):
        logging.info('begin auto tune, alpha %s', alpha)
        logging.info('epochs: [%s, %s]', self.num_epochs, num_epochs)
        logging.info('lr_h: %s, lr_q: %s', self.lr_h, self.lr_q)
        for i in range(num):
            _seed = np.random.randint(-1e6, 1e6)
            torch.manual_seed(_seed)
            self.calibrate(pdf, alpha, num_epochs)
            # evaluate coverage.
            _res = self.eval(pdf, alpha)
            _res = _res._replace(seed=_seed)
            print('_res', _res)
            logging.info('trail %s: %s', i, _res)
            self.auto_tune_history.append(_res)

    def recalibrate_q_hats(self, pdf: pd.DataFrame, alpha: float, group_net=None):
        # feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        h_feats, _ = transform_x_y(pdf, self.h_feat_name, self.gt_name)
        # preds = self.model.predict(feats)
        # nf_scores = self.score.calibrate(preds, gt_values)
        feats_tensor = torch.tensor(h_feats).type(torch.float32)
        # for each group, perform base conformal prediction.
        base_cp = BaseRegCP(self.model, self.feat_name, self.gt_name, self.score)
        # apply classification.
        if group_net is None:
            group_net = self.group_net
        h_x = group_net(feats_tensor)
        ans = torch.argmax(h_x, dim=1)

        group_qs = []
        for gi in range(self.mgroup):
            mask = (ans == gi).reshape(-1).tolist()
            group_pdf = pdf[mask]
            if len(group_pdf) == 0:
                group_qs.append(-1)
                continue
            _, group_q = base_cp.calibrate(group_pdf, alpha)
            group_qs.append(group_q)
        self.q_hats_prime = np.array(group_qs)
        return group_qs

    def eval(self, pdf: pd.DataFrame, alpha: float, h_net=None, q=None):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        h_feats, _ = transform_x_y(pdf, self.h_feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        nf_scores = self.score.calibrate(preds, gt_values)
        feats_tensor = torch.tensor(h_feats).type(torch.float32)
        h_net = h_net if h_net is not None else self.group_net
        q = q if q is not None else self.q_hats
        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        with torch.no_grad():
            h_x = h_net(feats_tensor)
            # print(h_x)
            loss_h = objective(h_net, q, feats_tensor, nf_tensor, alpha)
            # count how many items per group.
            ans = torch.argmax(h_x, dim=1)
            groups = []

            # compute coverage.
            group_idx = torch.argmax(h_x, dim=1).detach().numpy()
            q_hats = np.array([q[g].item() for g in group_idx.reshape(-1)]).reshape(-1)
            
            pred_intervals = self.score.predict(preds, q_hats)
            
            group_coverages = []
            group_avgsizes = []
            for i in range(self.mgroup):
                groups.append((ans==i).int().sum().item())
                # get intervals. 
                group_mask = (ans == i).tolist()
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

            coverage = compute_coverage_from_interval(gt_values, pred_intervals)

            return TrailEval(q.tolist(), groups, loss_h.item(), coverage, group_coverages, group_avgsizes, None)

    def calibrate(self, pdf: pd.DataFrame, alpha: float, num_epochs=None, report_every=None, report_func=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        gt_values = gt_values.reshape(-1)
        h_feats, _ = transform_x_y(pdf, self.h_feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        nf_scores = self.score.calibrate(preds, gt_values)
        # solve the optimization problem.
        nf_tensor = torch.tensor(nf_scores).type(torch.float32)
        feats_tensor = torch.tensor(h_feats).type(torch.float32)
        # h_net = GroupNet(feats.shape[1], self.mgroup)
        h_net = self.model_arch(feats_tensor.shape[1], self.mgroup)
        
        # q = torch.tensor([torch.median(nf_tensor)] * self.mgroup, requires_grad=True)
        # q = (torch.rand(self.mgroup)).clone().requires_grad_() 
        
        min_score, max_score = min(nf_scores.reshape(-1)), max(nf_scores.reshape(-1))
        q = (min_score + torch.rand(self.mgroup) * (max_score - min_score)).requires_grad_()
        # print('min, max', min_score, max_score)
        # q = torch.tensor(np.linspace(min_score, max_score, self.mgroup)).requires_grad_()

        # q_intervals = np.linspace(min_score, max_score, self.mgroup+1)
        # q = []
        # for r, lb, ub in zip(torch.rand(self.mgroup), q_intervals[:-1], q_intervals[1:]):
        #     q.append(r * (ub-lb)+lb)
        # q = torch.stack(q).requires_grad_()

        # Optimizers for the model parameters and quantiles
        optimizer_h = optim.Adam(h_net.parameters(), lr=self.lr_h)
        optimizer_q = optim.Adam([q], lr=self.lr_q)

        for epoch in tqdm(range(num_epochs)):
            if report_every and report_func is not None:
                if epoch % report_every == 0:
                    report_func(epoch, h_net, q.detach().numpy())
            # Step 2: Optimize h while keeping q fixed
            optimizer_h.zero_grad()
            # q.requires_grad = False
            # h_net.requires_grad_(True)
            loss_h = objective(h_net, q, feats_tensor, nf_tensor, alpha)
            loss_h.backward()
            optimizer_h.step()

            # Step 1: Optimize q while keeping h fixed
            optimizer_q.zero_grad()
            # q.requires_grad = True
            # h_net.requires_grad_(False)
            loss_q = objective(h_net, q, feats_tensor, nf_tensor, alpha)
            loss_q.backward()
            optimizer_q.step()
            
            # if (epoch + 1) % 10 == 0:
            #     logging.info(f"Epoch [{epoch + 1}/{self.num_epochs}], Loss: {loss_q.item()}, q: {q.detach().numpy()}")
        self.group_net = h_net
        self.q_hats = q.detach().numpy()
        logging.info('q: %s', q)
    
    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr', use_recalibrated=True):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        feats_tensor = torch.tensor(feats).type(torch.float32)
        with torch.no_grad():
            probs = self.group_net(feats_tensor).detach()
        group_idx = torch.argmax(probs, dim=1)
        q_hats = self.q_hats
        if use_recalibrated and self.q_hats_prime is not None:
            q_hats = self.q_hats_prime
        q_hats = np.array([q_hats[g].item() for g in group_idx.reshape(-1)])
        # reshape
        q_hats = q_hats.reshape(-1)
        pred_intervals = self.score.predict(preds, q_hats)
        if return_pred:
            return pred_intervals
        # group_idx


        group_coverages = []
        group_avgsizes = []
        group_nums = []
        for i in range(self.mgroup):
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
            'q_hats': self.q_hats.tolist(),
            # compute coverage & avg_size per group.
            'gp_coverages': group_coverages,
            'gp_avg_sizes': group_avgsizes
        }


    # def profile(self, pdf: pd.DataFrame, test_pdf: pd.DataFrame, alpha: float, out_path: str, num_epochs=200, lr=0.002, report_interval=10):
    #     feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
    #     preds = self.model.predict(feats)
    #     nf_scores = self.score.calibrate(preds, gt_values)
    #     # solve the optimization problem.
    #     nf_tensor = torch.tensor(nf_scores).type(torch.float32)
    #     feats_tensor = torch.tensor(feats).type(torch.float32)
    #     h_net = GroupNet(feats.shape[1], self.mgroup)
        
    #     # q = torch.tensor([torch.median(nf_tensor)] * self.mgroup, requires_grad=True)
    #     q = (torch.rand(self.mgroup) * self.rand_factor).clone().requires_grad_()

    #     # Optimizers for the model parameters and quantiles
    #     optimizer_h = optim.Adam(h_net.parameters(), lr=lr)
    #     optimizer_q = optim.Adam([q], lr=lr)

    #     # writer = SummaryWriter(out_path)

    #     for epoch in tqdm(range(num_epochs)):
    #         # if (epoch) % report_interval == 0:
    #         with torch.no_grad():
    #             h_x = h_net(feats_tensor)
    #             loss_h = objective(h_net, q, feats_tensor, nf_tensor, 1-alpha)
    #             # count how many items per group.
    #             ans = torch.argmax(h_x, dim=1)
    #             groups = []
    #             for i in range(self.mgroup):
    #                 groups.append((ans==i).int().sum())
    #                 # writer.add_scalar('group_num', groups[-1], epoch)
    #             logging.info('group nums %s', groups)
    #         # writer.add_scalar('training loss', loss_h.item(), epoch)
    #         # writer.add_scalars('q values', dict([('q values-{}'.format(i), v)for i, v in enumerate(q.detach().numpy())]), epoch)
    #         # evaluate it.
    #         logging.info('loss %s', loss_h.item())
    #         logging.info('q values %s', q.detach().numpy())

    #         for i in range(3):
    #             # Step 1: Optimize q while keeping h fixed
    #             optimizer_q.zero_grad()
    #             # q.requires_grad = True
    #             # h_net.requires_grad_(False)
    #             loss_q = objective(h_net, q, feats_tensor, nf_tensor, 1-alpha)
    #             loss_q.backward()
    #             optimizer_q.step()
    #             print('q loss', loss_q.item())

    #         for i in range(3):
    #             # Step 2: Optimize h while keeping q fixed
    #             optimizer_h.zero_grad()
    #             # q.requires_grad = False
    #             # h_net.requires_grad_(True)
    #             loss_h = objective(h_net, q, feats_tensor, nf_tensor, 1-alpha)
    #             loss_h.backward()
    #             optimizer_h.step()
    #             print('h loss', loss_h.item())


            