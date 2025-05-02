# Steps:
# 1. describe the transformation
# 2. finds the relevant functions from the repository.
from pathlib import Path
from cllm.utils import setup_logging
from cllm.dataset.tde.common import *
import logging

from cllm.dataset.tde.s1_tde_gpt_answers import ask_gpt
import pandas as pd
import re
import random

import shutil

LLAMA2_PROMPT_1 = """Please describe how to transform seq_a to seq_b below. Your description should be short, concrete and reflect the core logic of the transformation.{}
Q: seq_a=[1, 2, 3, 4], seq_b=[10, 20, 30, 40]
A: Multiply each element from the given sequence by 10, and output the result as a list.
Q: seq_a=['5 b', '8 x', '98x'], seq_b = ['5', '8', '98']
A: Extract number from each element in the given sequence, and format each number as a string in the result list.
"""

LLAMA2_PROMPT_2 = """Please describe how to transform seq_a to seq_b below. Your description should be short, concrete and reflect the core logic of the transformation.{}
Q: seq_a=[1, 2, 3, 4], seq_b=[10, 20, 30, 40]; seq_a=[4, 5, 2, 7], seq_b=[40, 50, 20, 70]
A: Multiply each element from the given sequence by 10, and output the result as a list.
Q: seq_a=['5 b', '8 x', '98x'], seq_b = ['5', '8', '98']; seq_a=['8 b', '11 x', '9x'], seq_b = ['8', '11', '9']; 
A: Extract number from each element in the given sequence, and format each number as a string in the result list.
"""

LLAMA2_PROMPT_3 = """Please describe how to transform seq_a to seq_b below. Your description should be short, concrete and reflect the core logic of the transformation.{}
Q: seq_a=[1, 2, 3, 4], seq_b=[10, 20, 30, 40]; seq_a=[4, 5, 2, 7], seq_b=[40, 50, 20, 70]; seq_a=[41, 6, 1, 21], seq_b=[410, 60, 10, 210]
A: Multiply each element from the given sequence by 10, and output the result as a list.
Q: seq_a=['5 b', '8 x', '98x'], seq_b = ['5', '8', '98']; seq_a=['8 b', '11 x', '9x'], seq_b = ['8', '11', '9'];  seq_a=['81 b', '5 b', '3 i'], seq_b = ['81', '5', '3']; 
A: Extract number from each element in the given sequence, and format each number as a string in the result list.
"""

def format_q(obj_list):
    # flatten sequences.
    values = []
    for obj in obj_list:
        if 'table' in obj:
            data = pd.DataFrame(data, columns=['a', 'b'])
            values.extend([data['a'].to_list(), data['b'].to_list()])
        elif 'seq_a' in obj:
            values.extend([obj['seq_a'], obj['seq_b']])
        else:
            raise NotImplementedError()
    # format.
    num = len(values)
    assert num%2==0
    str_list = ["Q: "]
    element = ['seq_a={}, seq_b={}']
    str_list.extend(';'.join(element * (num//2)))
    str_list.append('\nA:')
    return ''.join(str_list).format(*values)


def construct_question(obj_list, given_template):
    obj = obj_list[0]
    if obj['tip'] is not None:
        tip = re.sub(r'https?:\S*', '', obj['tip'])
        q = given_template.format(' Tip: {}\n'.format(tip))
    else:
        q = given_template.format('')
    q = '{}{}'.format(q, format_q(obj_list))
    return q


def ask_all_gpt(input_path, output_path, seq_num, model='gpt-3.5-turbo', offset=0):
    from cllm.aux.open_ai import OpenAIClient
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path)
    data = pd.DataFrame(data.values())

    client = OpenAIClient(model)

    seq_num_prompt_dict = {
        1: LLAMA2_PROMPT_1,
        2: LLAMA2_PROMPT_2,
        3: LLAMA2_PROMPT_3
    }

    random.seed(42)

    for _id, _row in tqdm(data.iterrows(), total=len(data)):
        if _row['oq_id'] in SKIP_IDS or _id < offset:
            logging.info('ignoring id: %s', _row['id'])
            continue
        prompt = seq_num_prompt_dict[seq_num]

        obj_list = [_row]
        # get all other ids, with the same 
        if seq_num > 1:
            # finds other rows that share the same oq_id.
            candidates = data[data['oq_id'] == _row['oq_id']].reset_index(drop=True)
            # finds the location of the current one.
            idx = None
            for _idx, c in candidates.iterrows():
                if c['id'] == _row['id']: idx = _idx
            more_ids = []
            for _ in range(seq_num-1):
                idx += 1
                if idx >= len(candidates): idx -= len(candidates)
                more_ids.append(idx)

            # print(candidates)
            # print(more_ids)
            for _idx in more_ids: obj_list.append(candidates.iloc[_idx])
        q = construct_question(obj_list, prompt)
        print(q)
        with open(output_path / '{}.txt'.format(_row['id']), 'w') as wf:
            res = ask_gpt(client, q, num=1)
            wf.write(res[0]['response'])


def ask_all_gpt_fix_mode(origin_output_path, input_path, output_path, seq_num, model='gpt-3.5-turbo', offset=0):
    from cllm.aux.open_ai import OpenAIClient
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path)
    data = pd.DataFrame(data.values())

    client = OpenAIClient(model)

    seq_num_prompt_dict = {
        1: LLAMA2_PROMPT_1,
        2: LLAMA2_PROMPT_2,
        3: LLAMA2_PROMPT_3
    }

    random.seed(42)

    copied, queried = 0, 0

    for _id, _row in tqdm(data.iterrows(), total=len(data)):
        if _row['oq_id'] in SKIP_IDS or _id < offset:
            logging.info('ignoring id: %s', _row['id'])
            continue
        prompt = seq_num_prompt_dict[seq_num]

        obj_list = [_row]
        # get all other ids, with the same 
        if seq_num > 1:
            # finds other rows that share the same oq_id.
            candidates = data[data['oq_id'] == _row['oq_id']].reset_index(drop=True)
            # finds the location of the current one.
            idx = None
            for _idx, c in candidates.iterrows():
                if c['id'] == _row['id']: idx = _idx
            more_ids = []
            for _ in range(seq_num-1):
                idx += 1
                if idx >= len(candidates): idx -= len(candidates)
                more_ids.append(idx)

            # print(candidates)
            # print(more_ids)
            for _idx in more_ids: obj_list.append(candidates.iloc[_idx])
        if _row['oq_id'] in REVISED_MORE_INPUT_IDS:
            q = construct_question(obj_list, prompt)
            print(q)
            with open(output_path / '{}.txt'.format(_row['id']), 'w') as wf:
                res = ask_gpt(client, q, num=1)
                wf.write(res[0]['response'])
            queried += 1
        else:
            # copy from the origin file.
            file_name = '{}.txt'.format(_row['id'])
            shutil.copyfile(origin_output_path / file_name, output_path / file_name)
            copied += 1
    print('copied: ', copied, 'queried:', queried)

# def ask_all_hf(input_path, output_path, model='meta-llama/Llama-2-7b-chat-hf'):
#     from cllm.aux.huggingface import LlamaHFClient
#     output_path.mkdir(parents=True, exist_ok=True)

#     data = read_dataset(input_path)
#     client = LlamaHFClient(model)
#     for given_qid in tqdm(sorted(data.keys())):
#         if given_qid in SKIP_IDS:
#             logging.info('ignoring id: %s', given_qid)
#             continue
#         q = construct_question(data[given_qid], LLAMA2_PROMPT)
#         # print(q)

#         # client = LlamaHFClient(model, bits_and_bytes=True)
#         client.set_stop_tokens(stop=['\n'], additional_stop_ids=[[13]])

#         with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
#             res = client.sample_answer(q, seed=42, max_tokens=1000)
#             wf.write(res[0]['response'])

if __name__ == '__main__':
    input_path = Path('./datasets/TDE/')

    # source_path = input_path / 'raw_datasets.jsonl'
    # output_path = input_path / 'gpt_seqdesc_desc_3.5'

    # source_path = input_path / 'gpt_more_outputs_3.5'/ 'all.jsonl'

    source_path = input_path / 'gpt_more_outputs_20_3.5_fix'/ 'all.jsonl'
    num = 3
    output_path = input_path / 'gpt_more_outputs_20_num{}_desc_3.5_fix'.format(num)

    origin_output_path = input_path / 'gpt_more_outputs_20_num{}_desc_3.5'.format(num)
    # ask_all_gpt(source_path, output_path, num, offset=3256)
    # ask_all_gpt(source_path, output_path, num, offset=3533)
    ask_all_gpt_fix_mode(origin_output_path, source_path, output_path, num)

    # output_path = input_path / 'gpt_more_outputs_desc_llama2-7b'
    # output_path = input_path / 'gpt_seqdesc_desc_llama2-7b'
    # model = 'meta-llama/Llama-2-7b-chat-hf'

    # output_path = input_path / 'gpt_seqdesc_desc_llama2-13b'
    # model = 'meta-llama/Llama-2-13b-chat-hf'
    # ask_all_hf(source_path, output_path, model=model)
