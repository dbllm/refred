from pathlib import Path
from cllm.io import load_data
import json
import pandas as pd
from cllm.dataset.tde.common import SKIP_IDS

def transform_all(input_path, output_path):
    with open(output_path, 'w') as wf:
        for obj in load_data(input_path):
            if obj['id'] in SKIP_IDS: continue
            table = pd.DataFrame(obj['table'], columns=['a', 'b'])
            wf.write('{}\n'.format(json.dumps({
                'id': obj['id'],
                'oq_id': obj['id'],
                'seq_a': table['a'].to_list(),
                'seq_b': table['b'].to_list(),
                'tip': obj['tip']
            })))

if __name__ == '__main__':
    base_path = Path('./datasets/TDE/')
    input_path = base_path / 'raw_datasets.jsonl'
    output_path = base_path / 'raw_datasets_transformed.jsonl'
    transform_all(input_path, output_path)