import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path

def load_ds1k():
    df = pd.read_json("hf://datasets/xlangai/DS-1000/test.jsonl", lines=True)
    print(df.head())
    print(df.columns)

    for i in range(100):
        reg = r"exec_context\s*=\s*r?\"\"\"(.*?)\"\"\""
        res = re.search(reg, df.iloc[i]['code_context'], re.DOTALL)
        print(f'------------ item {i} ------------')
        # print(df.iloc[i]['code_context'])
        if res:
            print(res.group(1))
        else:
            print('no match')


        # print('prompt:')
        # print(df.iloc[i]['prompt'])
        # print('reference_code:')
        # print(df.iloc[i]['reference_code'])
        # print('metadata:')
        # print(df.iloc[i]['metadata'])
        # print('code_context:')
        # print(df.iloc[i]['code_context'])
        # print()


def test_code_function():
    df = pd.read_json("hf://datasets/xlangai/DS-1000/test.jsonl", lines=True)
    matched_num = 0
    input_matched_num = 0
    output_matched_num = 0
    for i in tqdm(range(len(df)), total=len(df)):
        # test if there is a exec_context part in the code_context
        reg = r'exec_context\s*=\s*r?\"\"\"[\s\S]*?\"\"\"'
        if re.search(reg, df.iloc[i]['code_context']):
            matched_num += 1
            input_reg = r'test_input'
            output_reg = r'result'
            if input_reg in df.iloc[i]['code_context']:
                input_matched_num += 1
            if output_reg in df.iloc[i]['code_context']:
                output_matched_num += 1
    print(f'matched_num: {matched_num}')
    print(f'input_matched_num: {input_matched_num}')
    print(f'output_matched_num: {output_matched_num}')
    print(f'total_num: {len(df)}')


def synthesize_code_function():
    df = pd.read_json("hf://datasets/xlangai/DS-1000/test.jsonl", lines=True)
    gt_func_list = []
    for i in tqdm(range(len(df)), total=len(df)):
        reg = r"exec_context\s*=\s*r?\"\"\"(.*?)\"\"\""
        res = re.search(reg, df.iloc[i]['code_context'], re.DOTALL)
        assert res, f'item {i} no match'

        func_template = res.group(1)
        # add reference code to [insert]
        reference_code = df.iloc[i]['reference_code']
        func_template = func_template.replace('[insert]', reference_code)
        
        # update the func_template as a new column in df 
        # this is the gt function.
        gt_func_list.append(func_template)
    df['gt_function'] = gt_func_list

    target_path = Path('./datasets/DS-1000/test_gt_function.jsonl')
    target_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(target_path, orient='records', lines=True)


def read_local_gt_function():
    df = pd.read_json('./datasets/DS-1000/test_gt_function.jsonl', lines=True)
    print(df.head())
    # test if reference code is in the gt function
    for i in tqdm(range(len(df)), total=len(df)):
        if i > 10:
            break
        reference_code = df.iloc[i]['reference_code']
        gt_function = df.iloc[i]['gt_function']
        print(f'------------ item {i} ------------')
        print(gt_function)
        assert reference_code in gt_function, f'item {i} reference code not in gt function'

if __name__ == "__main__":
    # load_ds1k()
    # synthesize_code_function()
    read_local_gt_function()