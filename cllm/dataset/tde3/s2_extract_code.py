from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import re
from collections import defaultdict

def parse_responses(file_path):
    _cache = None
    responses = {}
    with open(file_path, 'r') as rf:
        for line in rf:
            if re.match(r'^-+response:#[0-9]+-+$', line):
                r_id = re.search(r'[0-9]+', line).group()
                assert r_id is not None
                if _cache is not None:
                    responses[int(r_id)-1]= _cache
                _cache = []
            elif line.strip() != '':
                _cache.append(line)
    if _cache is not None:
        responses[int(r_id)]= _cache
    return responses

def extract_code_from_response(response):
    # finds ```python`
    _match = re.search(r'```python\s+(.*?)```', response, re.DOTALL)
    if _match is not None:
        response = _match.group()[9:-3]
    return response


def merge_responses(response_list):
    unique_dict = defaultdict(list)
    for _id, _r in response_list.items():
        _str = ''.join(_r)
        unique_dict[_str].append(_id)
    return unique_dict.values()

def extract_code(raw_path, input_folder, output_folder):
    with open(raw_path, 'r') as rf:
        for line in tqdm(rf.readlines()):
            obj = json.loads(line)
            file_path = Path(input_folder) / '{}.txt'.format(obj['id'])
            responses = parse_responses(file_path)
            # responses = list(merge_responses(responses))
            assert len(responses) == 1
            code_lines = responses[0]
            code = extract_code_from_response(''.join(code_lines))
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            with open(Path(output_folder) / '{}.py'.format(obj['id']), 'w') as wf:
                wf.write(code)

if __name__ == '__main__':
    extract_code(
        './datasets/TDE/raw_datasets.jsonl', 
        './datasets/TDE_eval/complex_gpt_answers_raw_4/',
        './datasets/TDE_eval/complex_gpt_answers_code_4/'
    )
