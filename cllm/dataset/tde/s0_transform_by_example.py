from pathlib import Path
import os
from tqdm import tqdm
import pandas as pd
import json
import re

def read_question(file_path):
    tips = None
    data = []
    print('processing file', file_path)
    with open(file_path, 'r', encoding='ISO 8859-7') as rf:
        for line in rf.readlines():
            if line.strip().startswith('//'):
                # tips
                tips = line.strip()[2:]
            else:
                _tuple = re.split(r'\t\t', line.strip())
                if len(_tuple) == 1:
                    # treat the second as empty string.
                    _tuple = (_tuple[0], '')
                assert len(_tuple) == 2, 'error for file : {}'.format(file_path)
                data.append(_tuple)
    return tips, data

if __name__ == '__main__':
    dataset = Path('../Transform-Data-by-Example/Benchmark')
    save_path = Path('./datasets/TDE/raw_datasets.jsonl')
    save_path.parent.mkdir(exist_ok=True, parents=True)
    
    _id = 0
    with open(save_path, 'w') as wf:
        for category in tqdm(sorted(os.listdir(dataset))):
            if category.startswith('.'): continue
            for file_name in tqdm(sorted(os.listdir(dataset / category))):
                if file_name.startswith('.'): continue
                tip, table = read_question(dataset / category / file_name)
                obj = {'id': _id, 'category': category, 'file': file_name, 'tip':tip, 'table':table}
                wf.write('{}\n'.format(json.dumps(obj)))
                _id += 1