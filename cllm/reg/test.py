from cllm.reg.io import load_tbe_data
from cllm.io import split_data
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_pinball_loss
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars,
    OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor,
    PassiveAggressiveRegressor, HuberRegressor, QuantileRegressor,
    TheilSenRegressor, RANSACRegressor
)
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    BaggingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor,
    StackingRegressor, VotingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.cross_decomposition import PLSRegression, PLSCanonical
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def train_regression_model(train_data, test_data, model):
    x = np.array(train_data['emb'].to_list())
    y = np.array(train_data['dist'].to_list())

    model.fit(x, y)

    test_x = np.array(test_data['emb'].to_list())
    test_y = np.array(test_data['dist'].to_list())

    pred_y = model.predict(test_x)

    mse = mean_squared_error(test_y, pred_y)
    print("Mean squared error:", mse)

    # Calculate R^2 score
    r2 = r2_score(test_y, pred_y)
    print("R^2 score:", r2)
    # print(x, y)

if __name__ == '__main__':
    # pdf = load_tbe_data('origin')
    pdf = load_tbe_data()
    # train, validate, test
    train_data, validate_data, test_data = split_data(pdf, [0.4, 0.3, 0.3], shuffle=False)
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge()),
        ("Lasso Regression", Lasso()),
        ("Elastic Net", ElasticNet()),
        ("SGD Regressor", SGDRegressor()),
        ("Support Vector Regression (SVR)", SVR()),
        # ("Random Forest Regressor", RandomForestRegressor()),
        # ("Gradient Boosting Regressor", GradientBoostingRegressor()),
        ("AdaBoost Regressor", AdaBoostRegressor()),
        ("K-Neighbors Regressor", KNeighborsRegressor())
    ]

    for name, model in models:
        print('----')
        print(name)
        train_regression_model(train_data, test_data, model)
    
    pass