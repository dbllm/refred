from pathlib import Path
from cllm.utils import setup_logging
from cllm.dataset.tde.common import *
import logging
from cllm.aux.open_ai import *
from cllm.dataset.tde.s1_tde_gpt_answers import ask_gpt
import pandas as pd


SIMPLE_PROMPT = """Below is a function that transform a sequence to another sequence. 
```python
{}
```
This is one possible input sequence for the above functions: {}.
Please generate 20 additional unique input sequences (with varying number of elements) below, one per line:
"""

def construct_question(obj, code):
    data = obj['table']
    data = pd.DataFrame(data, columns=['a', 'b'])
    return SIMPLE_PROMPT.format(code, data['a'].to_list())

def ask_all():
    input_path = Path('./datasets/TDE/')
    code_path = input_path / 'gpt_answers_func_manual'
    output_path = input_path / 'gpt_more_inputs_20_3.5'
    output_path.mkdir(parents=True, exist_ok=True)

    data = read_dataset(input_path / 'raw_datasets.jsonl')
    for given_qid in tqdm(sorted(data.keys())):
        if given_qid in SKIP_IDS:
            logging.info('ignoring id: %s', given_qid)
            continue
        code = read_code(code_path / '{}.txt'.format(given_qid))
        q = construct_question(data[given_qid], code)
        # print(q)

        client = OpenAIClient('gpt-3.5-turbo')

        with open(output_path / '{}.txt'.format(given_qid), 'w') as wf:
            res = ask_gpt(client, q, num=1, max_tokens=4096)
            wf.write(res[0]['response'])


if __name__ == '__main__':
    ask_all()