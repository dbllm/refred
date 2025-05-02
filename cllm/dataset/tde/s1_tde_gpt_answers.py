import json
import pandas as pd
import re
from cllm.aux.open_ai import *
from tqdm import tqdm
from cllm.utils import setup_logging
import logging

SIMPLE_PROMPT_NO_TIP='''Below are two sequences, please convert the first sequence into the second one using pandas and python.'''
SIMPLE_PROMPT_TIP='''Below are two sequences, please convert the first sequence into the second one using pandas and python based on the given description.'''

COMPLEX_PROMPT_TIP="""Please generate python code for the task below. When generating, please follow these guidelines:
1. Please only generate one python code snippet surrounded by (```python```) blocks that contains the complete implementation for the task.
2. In the code, please only print the transformed sequence as a list to the console, and do not add any additional context to the output.
3. Please choose suitable formating for numeric values, including but not limited to using apis with round, math.ceil, math.floor, and Decimal to gaurantee the output format produced by the code aligns with the expected output.
"""

def construct_question(obj, use_complex_tip=False):
    data = obj['table']
    data = pd.DataFrame(data, columns=['a', 'b'])
    q_template = ""
    if use_complex_tip:
        q_template = COMPLEX_PROMPT_TIP+'\n'
    if obj['tip'] is None:
        q_template += SIMPLE_PROMPT_NO_TIP
        return '{}\nseq_a:{}\nseq_b:{}'.format(q_template, data['a'].to_list(), data['b'].to_list())
    else:
        q_template += SIMPLE_PROMPT_TIP
        tip = obj['tip']
        # remove url.
        tip = re.sub(r'https?://(?:www\\.)?[ a-zA-Z0-9./]+', '', tip)
        return '{}\nDescription:{}\nseq_a:{}\nseq_b:{}'.format(q_template, tip, data['a'].to_list(), data['b'].to_list())

def ask_gpt(client: OpenAIClient, q:str, num:int=3, seed:int=42, max_tokens=1000):
    prompt_messages = [{
        'role': 'user', 'content': q.strip()
    }]
    # use different seeds for the same ans.
    res = []
    for i in range(num):
        _r = client.complete(prompt_messages, seed=seed+i, batch_size=num, stop=[], max_tokens=max_tokens)
        res.append(_r[0])
    return _r

def ask_all(complex_prompt=False):

    setup_logging(level=logging.INFO, to_file=True)
    input_path = Path('./datasets/TDE/raw_datasets.jsonl')
    # output_path = Path('./datasets/TDE/gpt_answers_raw_4/')
    output_path = Path('./datasets/TDE/complex_gpt_answers_raw_3/')
    client = OpenAIClient('gpt-3.5-turbo')
    # output_path = Path('./datasets/TDE/complex_gpt_answers_raw_4/')
    # client = OpenAIClient('gpt-4-turbo')
    # output_path = Path('./datasets/TDE/gpt_answers_raw/')
# 
    output_path.mkdir(exist_ok=True, parents=True)

    # gpt4 is expensive.
    num = 1

    with open(input_path, 'r') as rf:
        for line in tqdm(rf.readlines()):
            obj = json.loads(line)
            q = construct_question(obj, use_complex_tip=complex_prompt)
            res = ask_gpt(client, q, num=num)
            with open(output_path / '{}.txt'.format(obj['id']), 'w') as wf:
                # write response.
                for _i, one_response in enumerate(res):
                    wf.write('\n---------response:#{}-----------\n'.format(_i))
                    wf.write(res[_i]['response'])

def ask_one(given_qid):
    setup_logging(to_file=False)
    input_path = Path('./datasets/TDE/raw_datasets.jsonl')
    output_path = Path('./datasets/TDE/complex_gpt_answers_raw_4/')
# 
    output_path.mkdir(exist_ok=True, parents=True)

    client = OpenAIClient('gpt-4-turbo')
    # gpt4 is expensive.
    num = 1

    with open(input_path, 'r') as rf:
        for line in tqdm(rf.readlines()):
            obj = json.loads(line)
            if obj['id'] != given_qid: continue
            q = construct_question(obj)
            logging.info('ask question: %s', q)
            res = ask_gpt(client, q, num=num, max_tokens=3000)
            with open(output_path / '{}.txt'.format(obj['id']), 'w') as wf:
                # write response.
                for _i, one_response in enumerate(res):
                    wf.write('\n---------response:#{}-----------\n'.format(_i))
                    wf.write(res[_i]['response'])

if __name__ == '__main__':
    # ask_one(41)
    ask_all(True)