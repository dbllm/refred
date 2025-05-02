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

def validate_answer(code_lines, expected_output):
    code_block = ''.join(code_lines).strip()
    # print(code_block)
    # finds ```python`
    _match = re.search(r'```python\s+(.*?)```', code_block, re.DOTALL)
    if _match is not None:
        code_block = _match.group()[9:-3]
    # create a separate file.
    with open('validate_tmp.py', 'w') as wf:
        wf.write(code_block)
    proc = subprocess.Popen(['python', 'validate_tmp.py'], stdout=subprocess.PIPE)
    output = proc.communicate()[0].decode()
    # f = StringIO()
    # with redirect_stdout(f):
    #     try:
    #         exec(code_block)
    #     except Exception as e: 
    #         logging.warning('exception %s', e)
    #         pass
    # output = f.getvalue()
    logging.info('output: %s', output)
    logging.info('expect: %s', expected_output)
    ans = output.strip() == str(expected_output).strip()
    logging.info('check result: %s', ans)
    return ans

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

def merge_responses(response_list):
    unique_dict = defaultdict(list)
    for _id, _r in response_list.items():
        _str = ''.join(_r)
        unique_dict[_str].append(_id)
    return unique_dict.values()

def auto_validate():
    input_path = Path('./datasets/TDE/')
    skip_ids = [
        61, 81, 94, 105, 124, 134, 143, 180, 211, 225, 238
    ]
    # setup_logging(level=logging.INFO, to_file=False)
    # setup_logging(level=logging.INFO, to_file=True, log_name='tde_auto_validate_complex_4.log')
    # gpt_raw_folder = 'complex_gpt_answers_raw_4'
    # correct_output = input_path / 'complex_gpt_answers_correct'
    setup_logging(level=logging.INFO, to_file=True, log_name='tde_auto_validate_complex_3.5.log')
    gpt_raw_folder = 'complex_gpt_answers_raw_3'
    correct_output = input_path / 'complex_gpt_answers_correct_3'

    # setup_logging(level=logging.INFO, to_file=True, log_name='tde_auto_validate_4.log')
    # gpt_raw_folder = 'gpt_answers_raw_4'
    # correct_output = input_path / 'gpt_answers_correct_4'

    correct_output.mkdir(parents=True, exist_ok=True)

    correct_num = 0
    total_num = 0
    with open(input_path / 'raw_datasets.jsonl', 'r') as rf:
        for line in tqdm(rf.readlines()):
            obj = json.loads(line)
            if obj['id'] in skip_ids: continue
            total_num += 1
            pdf = pd.DataFrame(obj['table'], columns=('seq_a', 'seq_b'))
            file_path = input_path / gpt_raw_folder / '{}.txt'.format(obj['id'])
            logging.info('processing file: %s', file_path)
            responses = parse_responses(file_path)
            unique_response_ids = merge_responses(responses)
            logging.info('unique_answers: %s', unique_response_ids)
            for _idl in unique_response_ids:
                _id = _idl[0]
                # execute it.
                logging.info('validating answer %s', _id)
                is_correct = validate_answer(responses[_id], pdf['seq_b'].to_list())
                if is_correct:
                    correct_num += 1
                    # save it.
                    with open(correct_output / '{}.txt'.format(obj['id']), 'w') as wf:
                        wf.writelines(responses[_id])
                    break
            # exit()
    
    logging.info('done, total num: %s, correct num: %s', total_num, correct_num)

def validate_correct_code(stop_when_error=True, offset=0):
    skip_ids = [
        61, 81, 94, 105, 124, 134, 143, 180, 211, 225, 238
    ]
    setup_logging(level=logging.INFO, to_file=False)
    input_path = Path('./datasets/TDE/')
    correct_output = input_path / 'gpt_answers_correct_merged'
    correct_output.mkdir(parents=True, exist_ok=True)
    manual_source_input = input_path / 'gpt_answers_raw_4'

    correct_num = 0
    total_num = 0
    with open(input_path / 'raw_datasets.jsonl', 'r') as rf1:
        for line in tqdm(rf1.readlines()):
            total_num += 1
            obj = json.loads(line)
            if obj['id'] < offset or obj['id'] in skip_ids: continue
            pdf = pd.DataFrame(obj['table'], columns=('seq_a', 'seq_b'))
            file_path = correct_output / '{}.txt'.format(obj['id'])
            logging.info('checking file: %s', file_path)
            is_correct = False
            if file_path.exists():
                with open(file_path, 'r') as rf:
                    is_correct = validate_answer(''.join(rf.readlines()), pdf['seq_b'].to_list())
            else:
                logging.info('file does not exist, will create a manual template')
                with open(file_path, 'w') as wf:
                    wf.write('// modified manually\n')
                    with open(manual_source_input / '{}.txt'.format(obj['id']), 'r') as _rf2:
                        wf.writelines(_rf2.readlines())
            if is_correct:
                correct_num += 1
            elif stop_when_error:
                exit()
    
    logging.info('done, total num: %s, correct num: %s', total_num, correct_num)

if __name__ == '__main__':
    auto_validate()
    # validate_correct_code(stop_when_error=False)
    # validate_correct_code(offset=238)
