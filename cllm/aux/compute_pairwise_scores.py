import re
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
from cllm.aux.compute_scores import normalize_text

import spacy

class SimCompute:
    def compute(self, predict_list):
        pass

class ExactMatchCompute:

    def __init__(self) -> None:
        pass

    def compute(self, predict_list):
        g = len(predict_list)
        arr = np.zeros((g, g))
        for j in range(g):
            # always 1 for itself.
            arr[j,j] = 1
            for k in range(j, g):
                _match = (predict_list[j] == predict_list[k])
                arr[j,k], arr[k, j] = _match, _match
        return arr
    
class SpacyCompute:
    # https://stackoverflow.com/questions/72067294/how-to-speed-up-computing-sentence-similarity-using-spacy-in-python
    def __init__(self, model_name) -> None:
        self.nlp = spacy.load(model_name, exclude=["tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"])
    
    def compute(self, predict_list):
        docs = [self.nlp(x) for x in predict_list]
        g = len(predict_list)
        arr = np.zeros((g, g))
        for j in range(g):
            # always 1 for itself.
            arr[j,j] = 1
            for k in range(j, g):
                _match = docs[j].similarity(docs[k])
                arr[j,k], arr[k, j] = _match, _match
        return arr

METHOD_MAPPING = {
    'exact_match': ExactMatchCompute(),
    'spacy.cosine.v1': SpacyCompute('en_core_web_md')
}

def compute_pairwise_scores_for_file(input_path, output_path, matching_func: SimCompute):
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    with open(input_path, 'r') as rf:
        arr = []
        for i, line in enumerate(tqdm(rf)):
            item = json.loads(line)
            predict_list = [normalize_text(x['response']) for x in item['pred']]
            arr.append(matching_func.compute(predict_list))
        np.save(output_path, np.array(arr))


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sim', default='exact_match', choices=METHOD_MAPPING.keys())
    return parser.parse_args()

if __name__ == '__main__':
    # input_string = "The quick, brown fox jumps over a lazy dog."
    # input_string = extract_string(input_string)
    # print('input string', input_string)
    # standardized_string = normalize_text(input_string)
    # print(standardized_string)
    args = parser_args()
    compute_pairwise_scores_for_file(args.input, args.output, METHOD_MAPPING[args.sim])