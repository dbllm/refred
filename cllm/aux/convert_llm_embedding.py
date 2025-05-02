import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike
import pickle

def convert_llm_embedding(input_file, output_file, convert_func):
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    with open(input_file, 'rb') as rf:
        arr = pickle.load(rf)

    converted = []
    for item in tqdm(arr):
        converted.append(convert_func(item))
    np.save(output_file, np.array(converted))

def mean_pooling(arr: ArrayLike):
    return np.mean(arr, axis=0)

def max_pooling(arr: ArrayLike):
    return np.max(arr, axis=0)

def min_pooling(arr: ArrayLike):
    return np.min(arr, axis=0)

def last_token(arr: ArrayLike):
    return arr[-1]

METHODS = {
    'last': last_token,
    'mean': mean_pooling,
    'max': max_pooling,
    'min': mean_pooling
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--method', required=True, choices=METHODS.keys())
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    convert_llm_embedding(args.input, args.output, METHODS[args.method])
