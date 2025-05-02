from pathlib import Path
from cllm.utils import setup_logging
from cllm.dataset.tde.common import *
import logging
from cllm.aux.open_ai import *
from cllm.dataset.tde.s1_tde_gpt_answers import ask_gpt

SIMPLE_PROMPT = """Please provide a short description for the function in the given code snippets. Please fulfill the following requirements:
1. The description should reflect the core logic of the function.
2. The description should be short and concrete.
"""

STRUCT_PROMPT = """For the given code, please describe its functionality to reflect the core logic. \
Please first describe the type of the input and output. E.g., whether it's string, numerical, or complex text extraction operations.\
The description should be short and concrete. The following is an example of the output format:
Input: a list of numerical values.
Output: a list of strings representing dates.
Functionality: transform the input values into date with format "yyyy-MM-dd hh:mm:ss".
"""

def construct_question(code, prompt):
    return '{}\n```python\n{}```'.format(prompt, code)

def ask_all():
    input_path = Path('./datasets/STD/')
    # dataset_path = input_path / 'raw_datasets.jsonl'
    code_path = input_path / 'functions'
    output_path = input_path / 'function_answers_desc_3.5'
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path / 'all_funcs.jsonl')
    for given_qid in tqdm(sorted(data.keys())):
        obj = data[given_qid]
        # if given_qid in SKIP_IDS:
        #     logging.info('ignoring id: %s', given_qid)
        #     continue
        # questions = read_question(dataset_path)
        # read code.
        code = read_code(code_path / '{}.py'.format(obj['name']))
        q = construct_question(code, STRUCT_PROMPT)
        # print(q)

        client = OpenAIClient('gpt-3.5-turbo')

        with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
            res = ask_gpt(client, q, num=1)
            wf.write(res[0]['response'])

if __name__ == '__main__':
    setup_logging(level=logging.INFO, to_file=False)
    ask_all()