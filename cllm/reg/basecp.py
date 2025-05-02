
import pandas as pd
from cllm.reg.utils import get_q_hat
from numpy.typing import ArrayLike
import numpy as np
from sklearn.svm import SVR
from cllm.reg.eval import eval_metrics, eval_metrics_per_row

class ConformalScore:
    def calibrate(self, preds: ArrayLike, gt_values: ArrayLike):
        pass

    def predict(self, preds: ArrayLike, q: float):
        pass

class ConformalScore2Norm(ConformalScore):

    def calibrate(self, preds: ArrayLike, gt_values: ArrayLike):
        return np.abs(preds - gt_values)
    
    def predict(self, preds: ArrayLike, q: float):
        arr1 = preds-q
        arr2 = preds+q
        return np.hstack([arr1.reshape(-1, 1), arr2.reshape(-1, 1)])
    
class ConformalScoreRightSide(ConformalScore):
    def calibrate(self, preds: ArrayLike, gt_values: ArrayLike):
        return preds - gt_values
    
    def predict(self, preds: ArrayLike, q:float):
        # range.
        inf_array = np.full_like(preds, -np.inf)
        return np.hstack([inf_array.reshape(-1, 1), (preds + q).reshape(-1, 1)])

def transform_x_y(pdf: pd.DataFrame, x_col: str, y_col: str):
    x = np.array(pdf[x_col].to_list())
    # reshape it.
    y = np.array(pdf[y_col].to_list()).reshape(-1)
    return x, y

class BaseRegCP:
    def __init__(self, model: SVR, feat_name: str='emb', gt_name: str = 'dist', score=None) -> None:
        self.model = model
        self.feat_name = feat_name
        self.gt_name = gt_name
        if score is None:
            self.score = ConformalScore2Norm()
        else:
            self.score = score
    
    def train(self, pdf: pd.DataFrame):
        x, y = transform_x_y(pdf, self.feat_name, self.gt_name)
        self.model.fit(x, y)

    def calibrate(self, pdf: pd.DataFrame, alpha: float):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        nf_scores = self.score.calibrate(preds, gt_values)
        self.q_hat = get_q_hat(nf_scores, alpha)
        # print('q', self.q_hat)
        return nf_scores, self.q_hat
        
    
    def test(self, pdf: pd.DataFrame, return_pred=False, dist_arr_col='dist_arr'):
        feats, gt_values = transform_x_y(pdf, self.feat_name, self.gt_name)
        preds = self.model.predict(feats).reshape(-1)
        # print('pred', preds.shape)
        pred_values = self.score.predict(preds, self.q_hat)
        # print('pred interval', pred_values.shape)
        # print('pred interval', pred_values[:5])
        if return_pred:
            return pred_values
        return {
            **eval_metrics(pdf, pred_values, gt_values, dist_arr_col),
            'q_hats': [self.q_hat],
        }
    
        # masks = (cf_values <= self.q_hat)
        # if return_mask:
        #     return masks
        # return eval_metrics(masks, gt_indices, reject_if_empty)
    
    def test_per_question(self, pdf: pd.DataFrame, dist_arr_col='dist_arr'):
        pred_values = self.test(pdf, return_pred=True, dist_arr_col=dist_arr_col)
        pdf = eval_metrics_per_row(pdf, pred_values, pdf[self.gt_name], dist_arr_col)
        pdf['q_hat'] = self.q_hat
        return pdf

class RawPredSolution(BaseRegCP):
    def __init__(self, model: SVR, feat_name: str='emb', gt_name: str = 'dist', both_sides=False) -> None:
        self.model = model
        self.feat_name = feat_name
        self.gt_name = gt_name
        self.both_sides = both_sides

    def calibrate(self, pdf: pd.DataFrame, *args, **kvargs):
        pass

    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr'):
        feats = np.array(pdf[self.feat_name].to_list())
        preds = self.model.predict(feats).reshape(-1, 1)
        if self.both_sides:
            # means range.
            # ub = lb = pred
            pred_intervals = np.hstack([preds, preds])
        else:
            pred_intervals = np.hstack([np.full_like(preds, -np.inf), preds])
        # print('pred interval', pred_intervals[:5])
        if return_pred:
            return pred_intervals
        return eval_metrics(pdf, pred_intervals, pdf[self.gt_name], dist_arr_col)

class OptimalSolution(BaseRegCP):

    def __init__(self, feat_name: str='emb', gt_name: str = 'dist', both_sides=False) -> None:
        self.feat_name = feat_name
        self.gt_name = gt_name
        self.both_sides = both_sides

    def train(self, pdf: pd.DataFrame):
        pass

    def calibrate(self, pdf: pd.DataFrame, *args, **kvargs):
        pass

    def test(self, pdf: pd.DataFrame, return_pred: bool=False, dist_arr_col='dist_arr'):
        preds = np.array(pdf[self.gt_name].to_list()).reshape(-1, 1)
        if self.both_sides:
            # means range.
            # ub = lb = pred
            pred_intervals = np.hstack([preds, preds])
        else:
            pred_intervals = np.hstack([np.full_like(preds, -np.inf), preds])
        # print('pred interval', pred_intervals[:5])
        if return_pred:
            return pred_intervals
        return eval_metrics(pdf, pred_intervals, pdf[self.gt_name], dist_arr_col)