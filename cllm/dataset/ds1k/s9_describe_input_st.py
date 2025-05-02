
from cllm.aux.st_embed import SentenceTransformerEmbeddingClient
import json
import argparse
import numpy as np
import os
import tqdm
def compute_desc_embedding(input_file, input_key, output_file):
    '''
    Compute the embedding of the description of the function.
    Parameters:
        input_file: the input file.
        input_key: the key of the description in the input file.
        output_file: the output file.
    '''
    # create if not exists
    if not os.path.exists(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    client = SentenceTransformerEmbeddingClient()
    arr = []
    with open(input_file, 'r') as rf:
        for line in tqdm.tqdm(rf):
            item = json.loads(line)
            desc = item[input_key]
            e1 = client.get_embedding(desc)
            arr.append(e1)
    np.save(output_file, np.array(arr))

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--input_key', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    compute_desc_embedding(args.input, args.input_key, args.output)
