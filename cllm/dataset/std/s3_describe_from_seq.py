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
import shutil

SIMPLE_PROMPT = """Please provide a short description for the function that could transform the following seq_a to seq_b. Please fulfill the following requirements:
1. The description should reflect the core logic of the function.
2. The decsription should be short and concrete."""

LLAMA2_PROMPT = """Please describe how to transform seq_a to seq_b below. Your description should be short, concrete and reflect the core logic of the transformation.{}
Q: seq_a=[1, 2, 3, 4], seq_b=[10, 20, 30, 40]
A: Multiply each element from the given sequence by 10, and output the result as a list.
Q: seq_a=['5 b', '8 x', '98x'], seq_b = ['5', '8', '98']
A: Extract number from each element in the given sequence, and format each number as a string in the result list.
"""

LLAMA2_STRUCT_PROMPT = """Please describe how to transform seq_a to seq_b below. \
Please first describe the type of the input and output. E.g., whether it's string, numerical, or complex text extraction operations.
Followed by the description of the transformation.\
Your description should be short, concrete and reflect the core logic of the transformation.{}
Q: seq_a=[1, 2, 3, 4], seq_b=[10, 20, 30, 40]
A: Input: a list of numerical values.
Output: a list of numerical values.
Functionality: The transformation multiplies each element from the given sequence by 10, and output the result as a list.
Q: seq_a=['5 b', '8 x', '98x'], seq_b = ['5', '8', '98']
A: Input: Input: a list of string values, with numerical values in each element.
Output: a list of numerical values.
Functionality: The transformation extracts number from each element in the given sequence, and format each number as a string in the result list.
"""

def construct_question(obj, given_template):
    if obj['tip'] is not None:
        tip = re.sub(r'https?:\S*', '', obj['tip'])
        q = given_template.format(' Tip: {}\n'.format(tip))
    else:
        q = given_template.format('')
    if 'table' in obj:
        data = obj['table']
        data = pd.DataFrame(data, columns=['a', 'b'])
        q = '{}\nQ: seq_a={}, seq_b={}\nA:'.format(q, data['a'].to_list(), data['b'].to_list())
    elif 'seq_a' in obj:
        q = '{}\nQ: seq_a={}, seq_b={}\nA:'.format(q, obj['seq_a'], obj['seq_b'])
    else:
        raise NotImplementedError()
    return q


def ask_all_gpt(input_path, output_path, model='gpt-3.5-turbo'):
    from cllm.aux.open_ai import OpenAIClient
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path)
    client = OpenAIClient(model)

    for given_qid in tqdm(sorted(data.keys())):
        # if given_qid in SKIP_IDS:
        #     logging.info('ignoring id: %s', given_qid)
        #     continue
        q = construct_question(data[given_qid], LLAMA2_STRUCT_PROMPT)
        # print(q)

        with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
            res = ask_gpt(client, q, num=1)
            wf.write(res[0]['response'])


def ask_all_hf(input_path, output_path, model='meta-llama/Llama-2-7b-chat-hf'):
    from cllm.aux.huggingface import LlamaHFClient
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path)
    client = LlamaHFClient(model)
    for given_qid in tqdm(sorted(data.keys())):
        if given_qid in SKIP_IDS:
            logging.info('ignoring id: %s', given_qid)
            continue
        q = construct_question(data[given_qid], LLAMA2_PROMPT)
        # print(q)

        # client = LlamaHFClient(model, bits_and_bytes=True)
        client.set_stop_tokens(stop=['\n'], additional_stop_ids=[[13]])

        with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
            res = client.sample_answer(q, seed=42, max_tokens=1000)
            wf.write(res[0]['response'])

# def ask_all_gpt_fix_mode(origin_output_path, input_path, output_path, model='gpt-3.5-turbo', offset=0):
#     from cllm.aux.open_ai import OpenAIClient
#     output_path.mkdir(parents=True, exist_ok=True)

#     data = read_dataset(input_path)
#     data = pd.DataFrame(data.values())

#     client = OpenAIClient(model)


#     copied, queried = 0, 0

#     for _id, _row in tqdm(data.iterrows(), total=len(data)):
#         if _row['oq_id'] in SKIP_IDS or _id < offset:
#             logging.info('ignoring id: %s', _row['id'])
#             continue

#         if _row['oq_id'] in REVISED_MORE_INPUT_IDS:
#             q = construct_question(_row, LLAMA2_STRUCT_PROMPT)
#             print(q)
#             with open(output_path / '{}.txt'.format(_row['id']), 'w') as wf:
#                 res = ask_gpt(client, q, num=1)
#                 wf.write(res[0]['response'])
#             queried += 1
#         else:
#             # copy from the origin file.
#             file_name = '{}.txt'.format(_row['id'])
#             shutil.copyfile(origin_output_path / file_name, output_path / file_name)
#             copied += 1
#     print('copied: ', copied, 'queried:', queried)

if __name__ == '__main__':
    input_path = Path('./datasets/STD/')

    source_path = input_path / 'outputs'/ 'all.jsonl'
    output_path = input_path / 'struct_more_outputs_gpt_desc_3.5'
    ask_all_gpt(source_path, output_path)
    # ask_all_gpt_fix_mode(origin_output_path, fixed_source_path, output_path)
