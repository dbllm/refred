import numpy as np
from numpy.typing import ArrayLike
import math

def get_q_hat(cf_scores: ArrayLike, alpha: float):
    cf_scores = cf_scores.reshape(-1)
    n = len(cf_scores)
    if n == 0:
        return 0
    quantiles = math.ceil((n+1) * (1-alpha)) / n
    if quantiles > 1: quantiles=1
    # print(quantiles)
    q_hat = np.quantile(cf_scores, quantiles, method='higher')
    return q_hat


def get_q_hat_weighted(cf_scores: ArrayLike, weights: ArrayLike, alpha: float):
    cf_scores = cf_scores.reshape(-1)
    n = len(cf_scores)
    quantiles = math.ceil((n+1) * (1-alpha)) / n
    if quantiles > 1: quantiles=1
    # normalize weights.
    assert np.all((weights>=0) & (weights <=1))
    weights_normalized = weights / (np.sum(weights) + 1)

    # get 1-alpha quantiles.
    sorter = np.argsort(cf_scores)
    # print(cf_scores.shape)
    values = cf_scores[sorter]
    # print(sorter.shape)
    weights_normalized = weights_normalized[sorter]
    weighted_cumsum = np.cumsum(weights_normalized)
    ind = np.where(weighted_cumsum >= quantiles)[0]
    # print(ind)
    if len(ind)== 0:
        return values[-1]
    ind_threshold = np.min(ind)
    return values[ind_threshold]