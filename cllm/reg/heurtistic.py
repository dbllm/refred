from cllm.reg.io import load_std_data, balance_tbe_data
from cllm.io import split_data
from pathlib import Path
import pandas as pd

class TopKHeuristic:

    def __init__(self, k) -> None:
        self.k = k

    def test(self, pdf: pd.DataFrame):
        for _id, _row in pdf.iterrows():
            pass
        pass

class ClosestK:

    def __init__(self, k) -> None:
        self.k = k

    def calibrate(self, pdf: pd.DataFrame):
        # sort all 
        pass

    def test(self, pdf: pd.DataFrame):
        pass

def evaluate_one(dataset, balance, k, seed=11):
    pdf = load_std_data(dataset)

    pdf['sample_id'] = pdf['id'].apply(lambda x: int(x.split('-')[1]))
    if balance:
        pdf = balance_tbe_data(pdf, seed=seed)
        # shuffle. 
        train_data, validate_data, test_data = split_data(pdf, [0.4, 0.3, 0.3], shuffle=False)
        # train_data, validate_data, test_data = reshape_open_dataset(train_data, validate_data, test_data, seed=42)
    else:
        if seed is None:
            seed = 42
        train_data, validate_data, test_data = split_data(pdf, [0.4, 0.3, 0.3], shuffle=True, random_seed=seed)
        # train_data, validate_data, test_data = reshape_open_dataset(train_data, validate_data, test_data, seed=seed)
    if seed is None:
        seed = 42
    
    # finds evaluation.
    
    # finds multiple.


if __name__ == '__main__':
    dataset='aug'
    evaluate_one(dataset, False, 3)