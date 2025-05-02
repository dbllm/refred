import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os
import subprocess
import json

def write_code_for_io():
    '''
    run code to get input and output.
    '''
    with open('./datasets/DS-1000/test_exec_code.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    output_folder = Path('./data_out/code_tmp/')
    output_folder.mkdir(parents=True, exist_ok=True)


    for i in tqdm(range(len(df)), total=len(df)):
        # write to file.
        with open(output_folder / f'{df.iloc[i]["problem_id"]}.py', 'w') as f:
            f.write(df.iloc[i]['code_context'])

def collect_output():
    # Note: we use conda env: ds1000-3.10
    # execute: 'conda activate ds1000-3.10'
    base_folder = Path('./data_out/code_tmp/')
    problem_ids = []
    outputs = []
    # collect all file names first.
    files = [file for file in base_folder.iterdir() if file.is_file()]
    # sort files based on file name.
    files.sort(key=lambda x: int(x.name.split('.')[0]))

    # Create a copy of the current environment
    env = os.environ.copy()

    # Modify or add a new environment variable
    env['MKL_THREADING_LAYER'] = 'GNU'

    with open('./datasets/DS-1000/test_exec_code_output.jsonl', 'w') as f:  
        for _id, file in tqdm(enumerate(files), total=len(files)):
            problem_id = file.name.split('.')[0]
            print(problem_id)
            # if _id < 898: continue
            # if _id in [862, 863, 864, 865, 897, 898]:
            #     f.write(json.dumps({'problem_id': problem_id, 'output': 'error'}) + '\n')
            #     continue
            try:
                #  to avoid MKL threading error.
                proc = subprocess.check_output(['python', file], env=env)
                output = proc.decode()
            except subprocess.CalledProcessError as e:
                output = str(e)
            outputs.append(output)
            problem_ids.append(problem_id)
            f.write(json.dumps({'problem_id': problem_id, 'output': output}) + '\n')
            # error: 862


def identify_null_values():
    with open('./datasets/DS-1000/test_exec_code_output.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)
    df['null_output'] = df['output'].apply(lambda x: x.find('output: None') != -1)

    df.to_json('./datasets/DS-1000/test_exec_code_output_null.jsonl', lines=True, orient='records')
    print('null values: ', len(df[df['null_output'] == True]))

if __name__ == '__main__':
    # write_code_for_io()
    # collect_output()
    # identify Null values.
    identify_null_values()