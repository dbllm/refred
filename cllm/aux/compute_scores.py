'''
compute scores based on the llm output.
'''

import re
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from collections import Counter


def normalize_text(s):
    # Convert to lowercase
    s = s.lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', '', s)
    # Remove punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # Replace duplicate whitespace with a single space
    s = re.sub(r'\s+', ' ', s)
    # Strip leading/trailing whitespace
    s = s.strip()
    return s


def exact_match(prediction, answers):
    """True if prediction matches any answer."""
    return float(any([prediction == a for a in answers]))


def normalized_likelihood(log_probs, alpha=0.6):
    """Likelihood with length penalty."""
    total_log_probs = np.sum(np.clip(log_probs, -1e5, 0))
    penalty = (5 + len(log_probs)) ** alpha / (5 + 1) ** alpha
    return np.exp(total_log_probs / penalty)


def extract_string(string, stop=['\n', ',','[.]']):
    """Cut off the text as soon as any stop words occur."""
    return re.split("|".join(stop), string, maxsplit=1)[0]

def compute_scores(item):
    answers = item['answers']
    answers = [normalize_text(a) for a in answers]
    prob_list = []
    loss_list = []
    pred_list = []

    for pred_obj in item['pred']:
        pred = pred_obj['response']
        prediction = re.split(r'[.,\n]', pred, maxsplit=1)[0]
        prediction = normalize_text(prediction)
        prob = normalized_likelihood(pred_obj['probs'])
        loss = 1 - exact_match(prediction, answers)
        prob_list.append(prob)
        loss_list.append(loss)
        pred_list.append(prediction)
    
    pred_counter = Counter(pred_list)
    freq_list = []
    for pred in pred_list:
        freq_list.append(pred_counter.get(pred) / len(pred_list))
        
    return {'total_prob': prob_list, 'exact_loss': loss_list, 'freq': freq_list}


def compute_scores_for_file(input_path, output_path):
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    with open(input_path, 'r') as rf, open(output_path, 'w') as wf:
        for i, line in enumerate(tqdm(rf)):
            item = json.loads(line)
            score_obj = compute_scores(item)
            wf.write('{}\n'.format(json.dumps(score_obj)))



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    # input_string = "The quick, brown fox jumps over a lazy dog."
    # input_string = extract_string(input_string)
    # print('input string', input_string)
    # standardized_string = normalize_text(input_string)
    # print(standardized_string)
    args = parser_args()
    compute_scores_for_file(args.input, args.output)