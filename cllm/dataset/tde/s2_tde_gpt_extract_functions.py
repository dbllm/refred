from cllm.utils import setup_logging
from cllm.dataset.tde.common import *
from cllm.aux.open_ai import *
from cllm.dataset.tde.s1_tde_gpt_answers import ask_gpt
from pathlib import Path
import logging

SIMPLE_PROMPT = """Please refactor the following code snippet to a function, which transforms seq_a to seq_b. 
Please meet the following requirements when rewriting:
1. The rewritten function should accept a list as the input (seq_a), and return the transformed output (seq_b).
2. Please also give the function an appropriate name to reflect the intent of the transform."""

def construct_question(code):
    return '{}\n```python\n{}```'.format(SIMPLE_PROMPT, code)

def ask_one(given_qid):
    if given_qid in SKIP_IDS:
        print("the qid should be skipped")
        return
    input_path = Path('./datasets/TDE/')
    # dataset_path = input_path / 'raw_datasets.jsonl'
    code_path = input_path / 'gpt_answers_correct_merged'
    output_path = input_path / 'gpt_answers_func_3.5'
    output_path.mkdir(parents=True, exist_ok=True)

    # questions = read_question(dataset_path)
    # read code.
    code = read_code(code_path / '{}.txt'.format(given_qid))
    q = construct_question(code)
    # print(q)

    client = OpenAIClient('gpt-3.5-turbo')

    with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
        res = ask_gpt(client, q, num=1)
        wf.write(res[0]['response'])


def ask_all():
    input_path = Path('./datasets/TDE/')
    # dataset_path = input_path / 'raw_datasets.jsonl'
    code_path = input_path / 'gpt_answers_correct_merged'
    output_path = input_path / 'gpt_answers_func_3.5'
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path / 'raw_datasets.jsonl')
    for given_qid in tqdm(sorted(data.keys())):
        if given_qid in SKIP_IDS:
            logging.info('ignoring id: %s', given_qid)
            continue
        # questions = read_question(dataset_path)
        # read code.
        code = read_code(code_path / '{}.txt'.format(given_qid))
        q = construct_question(code)
        # print(q)

        client = OpenAIClient('gpt-3.5-turbo')

        with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
            res = ask_gpt(client, q, num=1)
            wf.write(res[0]['response'])

if __name__ == '__main__':
    for i in range(9):
        ask_one(i)
    # ask_all()