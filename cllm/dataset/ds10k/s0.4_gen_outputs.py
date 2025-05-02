import json
import pandas as pd
import os
from pathlib import Path
import subprocess
from tqdm import tqdm

def gen_outputs_jsonl():
    with open('datasets/DS-1000/test_gt_function.jsonl', 'r') as f:
        gt_df = pd.read_json(f, lines=True)
    gt_df['problem_id'] = gt_df['metadata'].apply(lambda x: x['problem_id'])

    with open('datasets/DS-10k_deepseek/test_input_more_inputs_v2_refined.jsonl', 'r') as f:
        input_df = pd.read_json(f, lines=True)

    merged_df = pd.merge(gt_df, input_df, on='problem_id', how='inner')

    with open('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined.jsonl', 'w') as f:
        for _idx, row in merged_df.iterrows():
            # merge code.
            print('uid:', row['uid'])
            # finds test_input in code.
            code_content = row['gt_function']
            pos = code_content.find('test_input')
            # find the first \n before pos
            pos = code_content.rfind('\n', 0, pos)

            code_content_p1, code_content_p2 = code_content[:pos], code_content[pos:]

            output_str = "print('input:', test_input)\nprint('output:', result)"
            code = '\n'.join([code_content_p1, '\n', row['input'], code_content_p2, output_str])
            print(row['problem_id'])
            print(code)
            f.write(json.dumps({**row, 'code': code}) + '\n')
            # break

    # merged_df.to_json('datasets/DS-1000/test_gt_function_more_inputs.jsonl', lines=True, orient='records')

def gen_outputs_code_folder():
    with open('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    base_path = Path('data_out/code_tmp_more_inputs_v2_refined')
    base_path.mkdir(parents=True, exist_ok=True)
    for _idx, row in df.iterrows():
        with open(base_path / f'{_idx}.py', 'w') as f:
            f.write(row['code'])


def gen_output_values():
    with open('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    base_path = Path('data_out/code_tmp_more_inputs_v2_refined')
    # collect all file names first.
    files = [file for file in base_path.iterdir() if file.is_file()]
    # sort files based on file name.
    files.sort(key=lambda x: int(x.name.split('.')[0]))

    # Create a copy of the current environment
    env = os.environ.copy()

    # Modify or add a new environment variable
    env['MKL_THREADING_LAYER'] = 'GNU'

    
    output_path = Path('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled.jsonl')

    # execute: 'conda activate ds1000-3.10'
    with open(output_path, 'w') as f:
        for _idx, row in tqdm(df.iterrows(), total=len(df)):
            code_path = base_path / f'{_idx}.py'
            try:
                # Capture both stdout and stderr
                proc = subprocess.run(['python', code_path], 
                                   env=env,
                                   capture_output=True,
                                   text=True)
                if proc.returncode == 0:
                    output = proc.stdout
                else:
                    output = f"Exit code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            except Exception as e:
                pass
            f.write(json.dumps({**row, 'output': output}) + '\n')

def count_errors_1():
    with open('datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)
    print(df['output'].apply(lambda x: 'non-zero exit' in x).sum())

    # try to fix the errors.
    df['error_1'] = df['code'].apply(lambda x: 'python' in x)
    print(df[df['error_1'] == True].head())
    print(df['error_1'].sum())

def count_errors_2():
    with open('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)
    print(df['output'].apply(lambda x: 'Error' in x).sum())

    # try to fix the errors.
    # df['error_1'] = df['code'].apply(lambda x: 'python' in x)
    # print(df[df['error_1'] == True].head())
    # print(df['error_1'].sum())    
    # row = df[df['error_1'] == True].iloc[0]
    # print(row['output'])
    # print(row)

    # get A?
    

    error_df = df[df['output'].apply(lambda x: 'Error' in x)]
    error_df.to_json('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_error.jsonl', lines=True, orient='records')

    # get unique error ids.
    error_ids = error_df['problem_id'].unique()
    print(len(error_ids))
    print(error_ids)

    noerror_df = df[df['output'].apply(lambda x: 'Error' not in x)]
    print(len(noerror_df))

    # for _idx, row in error_df.iterrows():
    #     print(row['problem_id'])
    #     print(row['uid'])
    #     print(row['output'])
    #     print(row['code'])
    #     print('-' * 100)

if __name__ == '__main__':
    # gen_outputs_jsonl()
    # gen_outputs_code_folder()
    # gen_output_values()
    count_errors_2()