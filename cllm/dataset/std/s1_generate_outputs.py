from pathlib import Path
from cllm.utils import setup_logging
from cllm.io import load_data
from cllm.dataset.tde.common import *
import logging
from cllm.aux.open_ai import *
import pandas as pd
import subprocess
import json
from collections import defaultdict

def read_inputs(file_path):
    data = []
    with open(file_path, 'r') as rf:
        for line in rf:
            # skip empty values.
            if line.strip() == '': continue
            data.append(parse_one(line))
    return data

def parse_one(line: str):
    number_match = re.match(r'[0-9]*[.]', line)
    if number_match:
        line = line[number_match.end():]
    line = line.strip()

    point_match = re.match(r'[-*]', line)
    if point_match:
        line = line[point_match.end():]
    line = line.strip()

    replace_words = {
        "t's": "t\\'s",
        "n's": "n\\'s",
        "n't": "n\\'t",
        "N'T": "N\\'T",
        "I'm": "I\\'m",
        "It's": "It\\'s",
        "I've": "I\\'ve",
        "I'll": "I\\'ll",
        "it'll": "it\\'ll",
    }
    for w in replace_words:
        if w in line:
            line = line.replace(w, replace_words[w])
    line = line.strip()

    if line[0] != '[':
        line = '['+line+']'
    # print(line)
    p =  eval(line)
    return p
        

def replace_input(code_block, new_seq):
    # finds seq_a ???
    lines = code_block.split('\n')
    replace_mode = False
    replace_part = None
    for i, line in enumerate(lines):
        line = line.strip()
        if replace_mode:
            # whether to end the mode.
            if len(line) > 0 and line[-1] == ']': 
                replace_mode = False
                replace_part = (replace_part[0], i+1)
        else:
            # whether to start the mode.
            if line.startswith('seq_a '): 
                if line[-1] == ']':
                    replace_mode = False
                    replace_part = (i, i+1)
                else:
                    replace_mode = True
                    replace_part = (i, )
    
    # replace.
    new_str = 'seq_a = {}\n'.format(str(new_seq))
    replaced_code = '\n'.join([
        *lines[:replace_part[0]], 
        new_str, 
        *lines[replace_part[1]:]
        ])
    return replaced_code

def execute_code(code_block, i):
    # create a separate file.
    with open('validate_tmp.py', 'w') as wf:
        wf.write(code_block)
    try:
        proc = subprocess.check_output(['python', 'validate_tmp.py'])
        output = proc.decode()
        logging.info('line: [%s], output: %s', i, output)
    except subprocess.CalledProcessError as e:
        logging.error('error: %s', str(e))
        exit()
    # f = StringIO()
    # with redirect_stdout(f):
    #     try:
    #         exec(code_block)
    #     except Exception as e: 
    #         logging.warning('exception %s', e)
    #         pass
    # output = f.getvalue()
    return output.strip()

def execute_all(offset=0):
    input_folder = Path('./datasets/STD/')
    code_path = input_folder / 'functions'
    input_path = input_folder / 'inputs'

    output_path = input_folder / 'outputs'
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_folder / 'all_funcs.jsonl')
    
    with open(output_path / 'all.jsonl', 'w') as wf:
        for given_qid in tqdm(sorted(data.keys())):
            obj = data[given_qid]
            if given_qid < offset: continue
            # if given_qid in SKIP_IDS:
            #     logging.info('ignoring id: %s', given_qid)
            #     continue
            in_file = input_path / '{}.txt'.format(obj['name'])
            logging.info('checking file %s', in_file)
            code = read_code(code_path / '{}.py'.format(obj['name']))
            inputs = read_inputs(in_file)
            # print(inputs)
            # outputs = []
            for i, seq_a in enumerate(inputs):
                complete_code = replace_input(code, seq_a)
                seq_b = execute_code(complete_code, i+1)
                # outputs.append(seq_b)
                obj = {
                    'id': '{}-{}'.format(given_qid, i),
                    'oq_id': given_qid,
                    'seq_a': seq_a,
                    'seq_b': seq_b,
                    'tip': None
                }
                wf.write('{}\n'.format(json.dumps(obj)))

        # with open(input_path / '{}.txt'.format(given_qid), 'w') as wf:
        #     wf.writelines([str(o) for o in outputs])

if __name__ == '__main__':
    setup_logging(level=logging.INFO, to_file=False)
    # execute_all(offset=231)
    execute_all()
    # execute_all(offset=53)
    