import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm
import json
import re
from contextlib import redirect_stdout
from io import StringIO
import os
from cllm.utils import setup_logging
from collections import defaultdict
import subprocess

def validate_answer(file_path, expected_output):
    proc = subprocess.Popen(['python', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    output = stdout.decode()
    error = stderr.decode()
    # f = StringIO()
    # with redirect_stdout(f):
    #     try:
    #         exec(code_block)
    #     except Exception as e: 
    #         logging.warning('exception %s', e)
    #         pass
    # output = f.getvalue()
    if error != '':
        logging.info('error: %s', error)
    logging.info('output: %s', output)
    logging.info('expect: %s', expected_output)
    ans = output.strip() == str(expected_output).strip()
    logging.info('check result: %s', ans)
    return ans, error != ''


def auto_validate(raw_path, code_folder, correct_output, incorrect_list_path):
    # skip_ids = [
    #     61, 81, 94, 105, 124, 134, 143, 180, 211, 225, 238
    # ]
    assert not Path(incorrect_list_path).exists()
    setup_logging(level=logging.INFO, to_file=True, log_name='tde_auto_validate_complex_4.log')
    code_folder = Path(code_folder)
    correct_output = Path(correct_output)

    correct_output.mkdir(parents=True, exist_ok=True)

    count_gt_id = read_gt_id()

    correct_num = 0
    total_num = 0
    error_num = 0
    incorrect_list = []
    status = []
    with open(raw_path, 'r') as rf:
        for line in tqdm(rf.readlines()):
            obj = json.loads(line)
            if obj['id'] not in count_gt_id: continue
            logging.info('validating %s', obj['id'])
            total_num += 1
            pdf = pd.DataFrame(obj['table'], columns=('seq_a', 'seq_b'))
            file_path = code_folder / '{}.py'.format(obj['id'])
            logging.info('processing file: %s', file_path)
            # execute it.
            is_correct, has_error = validate_answer(file_path, pdf['seq_b'].to_list())
            incorrect_list.append(obj['id'])
            if is_correct:
                correct_num += 1
                status.append('correct')
            else:
                status.append('incorrect')
            if has_error:
                error_num += 1
                # save it.
                # with open(correct_output / '{}.py'.format(obj['id']), 'w') as wf:
                #     wf.writelines(responses[_id])
                # break
            # exit()
    pdf = pd.DataFrame(incorrect_list, columns=['id'])
    pdf['status'] = status
    pdf['category'] = None
    pdf.to_csv(incorrect_list_path, index=False)
    logging.info('done, total num: %s, correct num: %s, error num: %s', total_num, correct_num, error_num)
    print('done, total num: %s, correct num: %s, error num: %s', total_num, correct_num, error_num)

# def validate_correct_code(stop_when_error=True, offset=0):
#     skip_ids = [
#         61, 81, 94, 105, 124, 134, 143, 180, 211, 225, 238
#     ]
#     setup_logging(level=logging.INFO, to_file=False)
#     input_path = Path('./datasets/TDE/')
#     correct_output = input_path / 'gpt_answers_correct_merged'
#     correct_output.mkdir(parents=True, exist_ok=True)
#     manual_source_input = input_path / 'gpt_answers_raw_4'

#     correct_num = 0
#     total_num = 0
#     with open(input_path / 'raw_datasets.jsonl', 'r') as rf1:
#         for line in tqdm(rf1.readlines()):
#             total_num += 1
#             obj = json.loads(line)
#             if obj['id'] < offset or obj['id'] in skip_ids: continue
#             pdf = pd.DataFrame(obj['table'], columns=('seq_a', 'seq_b'))
#             file_path = correct_output / '{}.txt'.format(obj['id'])
#             logging.info('checking file: %s', file_path)
#             is_correct = False
#             if file_path.exists():
#                 with open(file_path, 'r') as rf:
#                     is_correct = validate_answer(''.join(rf.readlines()), pdf['seq_b'].to_list())
#             else:
#                 logging.info('file does not exist, will create a manual template')
#                 with open(file_path, 'w') as wf:
#                     wf.write('// modified manually\n')
#                     with open(manual_source_input / '{}.txt'.format(obj['id']), 'r') as _rf2:
#                         wf.writelines(_rf2.readlines())
#             if is_correct:
#                 correct_num += 1
#             elif stop_when_error:
#                 exit()
    
#     logging.info('done, total num: %s, correct num: %s', total_num, correct_num)


def compute_ratio(incorrect_list_path):
    df = pd.read_csv(incorrect_list_path)
    total_num = len(df)
    error_num = len(df[df['status'] == 'incorrect'])
    sums = []
    for category in df['category'].unique():
        num = len(df[df['category'] == category])
        ratio = num / total_num
        sums.append(ratio)
        print('{}: {}, error: {}'.format(category, ratio, num/error_num))
    print('error ratio: {}'.format(len(df[df['status'] == 'incorrect']) / total_num))
    print('sum: {}'.format(sum(sums)))
    print('error percent ratio: {}'.format(sum(sums)))

def read_gt_id():
    file = './datasets/TDE2/outputs/all.jsonl'
    df = pd.read_json(Path(file), lines=True)
    return df['oq_id'].unique().tolist()

if __name__ == '__main__':
    # auto_validate(
    #     './datasets/TDE/raw_datasets.jsonl',
    #     './datasets/TDE_eval/complex_gpt_answers_code_4/',
    #     './datasets/TDE_eval/complex_gpt_answers_code_4_correct/',
    #     './datasets/TDE_eval/incorrect_list_4.txt',
    # )
    compute_ratio('./datasets/TDE_eval/incorrect_list_4.txt')
    # validate_correct_code(stop_when_error=False)
    # validate_correct_code(offset=238)
    # print(len(read_gt_id()))
