import pandas as pd
from cllm.aux.open_ai import OpenAIClient
from pathlib import Path 
from tqdm import tqdm
import re
import json
import itertools

def generate_more_input_prompt(code, test_case):
    prompt = f"""Please generate 10 more input cases for the following function:
Function:
{code}
For your information, below is one example of the input:
{test_case}
Please write the code that assigns the `test_input` variable that could produce the above output.
For example, if the input consists of a single value of 1, then the code should be like `test_input = 1`; 
if the input consits of two values, then the code should be like `value1 = ...\nvalue2= ...\ntest_input = (value1, value2)`.
Please write all your input cases below, and separate each case by a horizontal line (---).
Do not generate trivial input cases, for example, empty input, empty tensor, etc. The input should be meaningful and useful in the context of the given function.
Please only include your code in the answer, and do not include any other text. do not use ```python or ``` in the code.
Answer:
A1: 
"""
    return prompt

def generate_input_code(code, test_case, model="gpt-3.5-turbo-0125", max_tokens=4096):
    prompt = f"""Please generate 10 more input cases for the following function:
Function:
{code}
For your information, below is one example of the input:
{test_case}
Please write the code that assigns the `test_input` variable that could produce the above output.
For example, if the input consists of a single value of 1, then the code should be like `test_input = 1`; 
if the input consits of two values, then the code should be like `value1 = ...\nvalue2= ...\ntest_input = (value1, value2)`.
Please write all your input cases below, and separate each case by a horizontal line (---).
Do not generate trivial input cases, for example, empty input, empty tensor, etc. The input should be meaningful and useful in the context of the given function.
Answer:
A1: 
"""
    max_tokens = 8192
    model = 'deepseek-chat'
    client = OpenAIClient(model_name=model, endpoint='deepseek')
    result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=max_tokens, batch_size=1, stop=[])
    client.close()
    return result[0]['response']

def parse_output(output):
    parts = output.split('---')
    results = []
    for part in parts:
        part = part.strip()
        # Fix: replace r'A?*\n' with r'^A\d+:\s*\n' to match lines like 'A1:', 'A2:', etc.
        part = re.sub(r'^A\d+:\s*\n', '', part)
        if part == '': continue
        results.append(part)
    return results

def get_more_inputs():
    with open('datasets/DS-1000/test_input_code_error_problems.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    dest_path = Path('datasets/DS-10k_deepseek/test_input_more_inputs_v2.jsonl')
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    uid_generator = itertools.count(0)

    max_tokens = 8192
    model = 'deepseek-chat'
    client = OpenAIClient(model_name=model, endpoint='deepseek')

    with open(dest_path, 'w') as wf:
        for _idx, row in tqdm(df.iterrows(), total=len(df)):
            code = row['gt_function']
            test_case = row['test_case_code']
            # do not handle error cases.
            # if _idx > 10:break
            if test_case == 'error':
                continue
            try:
                prompt = generate_more_input_prompt(code, test_case)
                result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=max_tokens, batch_size=1, stop=[])
                result = result[0]['response']
                more_inputs = parse_output(result)
                for _input in more_inputs:
                    wf.write(json.dumps({'problem_id': row['problem_id'], 'origin_uid': row['uid'], 'input': _input, 'uid': next(uid_generator)}) + '\n')
            except Exception as e:
                print(e)
                print(f'Error in row {_idx}')
                wf.write(json.dumps({'problem_id': row['problem_id'], 'origin_uid': row['uid'], 'input': 'error', 'uid': next(uid_generator)}) + '\n')
            # break
    client.close()

def create_input_files():
    with open('datasets/DS-10k_deepseek/test_input_more_inputs_v2.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)
    
    base_folder = Path('data_out/deepseek_code_tmp_more_inputs_txt_v2')
    if base_folder.exists():
        # warning
        print(f'Warning: {base_folder} already exists.')
        return
    base_folder.mkdir(parents=True, exist_ok=True)
    # group by problem_id.
    grouped = df.groupby('problem_id')
    for problem_id, group in grouped:
        # sort by uid.
        group = group.sort_values(by='uid') 
        # write to file.
        with open(base_folder / f'{problem_id}.txt', 'w') as wf:
            for _idx, row in group.iterrows():
                wf.write(row['input'])
                wf.write('\n')
                wf.write('---\n')


def re_assemble_jsonl():
    with open('datasets/DS-10k_deepseek/test_input_more_inputs_v2.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    dest_path = Path('datasets/DS-10k_deepseek/test_input_more_inputs_v2_refined.jsonl')

    base_folder = Path('data_out/deepseek_code_tmp_more_inputs_txt_v2')
    
    uid_generator = itertools.count(0)
    with open(dest_path, 'w') as wf:
        # group by problem_id.
        grouped = df.groupby('problem_id')
        for problem_id, group in grouped:
            # sort by uid.
            # write to file.
            read_path = base_folder / f'{problem_id}.txt'
            if not read_path.exists():
                continue
            with open(read_path, 'r') as rf:
                more_inputs = rf.read()
            more_inputs = parse_output(more_inputs)
            for _input in more_inputs:
                if 'test_input' in _input and 'None' in _input:
                    # skip meaningless input.
                    continue
                wf.write(json.dumps({'problem_id': problem_id, 
                                    'input': _input, 'uid': next(uid_generator)}) + '\n')


if __name__ == '__main__':
#    get_more_inputs()
    # create_input_files()
    re_assemble_jsonl()