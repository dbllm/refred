from cllm.aux.open_ai import *
from cllm.dataset.tde.s1_tde_gpt_answers import construct_question
from cllm.utils import setup_logging
import logging
from cllm.dataset.tde.s1_tde_gpt_answers import *
from cllm.dataset.tde.common import read_dataset
import numpy as np

def create_html_file(tokens, probs, output_file, template='$PH1$'):
    with open(output_file, 'w') as wf:
        items = ["{{word: {}, score: {}}}".format(repr(t), p) for t, p in zip(tokens, probs)]
        new_str = 'const textWithScores = [{}];'.format(', '.join(items))
        wf.write(template.replace('$PH1$', new_str))

def _ask_or_load(cache_path: Path, given_qid: int, obj: Any, client):
    cache_file = cache_path / '{}.json'.format(given_qid)
    data = None
    if cache_file.exists():
        # load from cache.
        with open(cache_file, 'r') as rf:
            data = json.loads(rf.read())
    else:
        q = construct_question(obj)
        logging.info('ask question: %s', q)
        data = ask_gpt(client, q, num=1, max_tokens=3000)
        with open(cache_file, 'w') as wf:
            wf.write(json.dumps(data))
    return data

def ask_all():
    setup_logging(to_file=False)
    input_path = Path('./datasets/tde/raw_datasets.jsonl')
    template_path = Path('./template/llmout_template.html')
    output_path = Path('./data_out/tde/visualize-3.5/')
    cache_path = Path('./data_out/tde/vis_cache-3.5/')
    
    output_path.mkdir(exist_ok=True, parents=True)
    cache_path.mkdir(exist_ok=True, parents=True)

    obj_dict = read_dataset(input_path)

    client = OpenAIClient('gpt-3.5-turbo')

    for given_qid, obj in tqdm(obj_dict.items()):
        data = _ask_or_load(cache_path, given_qid, obj, client)
        # to json.
        # print(data)
        # transform log prob to prob.
        # only one response.
        data = data[0]
        tokens = data['tokens']
        probs = np.exp(np.array(data['probs']))
        create_html_file(tokens, probs, 
                        output_path / '{}.html'.format(given_qid), 
                        open(template_path).read()
                        )

def ask_one(given_qid):
    setup_logging(to_file=False)
    input_path = Path('./datasets/tde/raw_datasets.jsonl')
    template_path = Path('./template/llmout_template.html')
    output_path = Path('./data_out/tde/visualize-3.5/')
    cache_path = Path('./data_out/tde/vis_cache-3.5/')
    
    output_path.mkdir(exist_ok=True, parents=True)
    cache_path.mkdir(exist_ok=True, parents=True)

    obj_dict = read_dataset(input_path)

    client = OpenAIClient('gpt-3.5-turbo')

    data = _ask_or_load(cache_path, given_qid, obj_dict[given_qid], client)
    # to json.
    # print(data)
    # transform log prob to prob.
    # only one response.
    data = data[0]
    tokens = data['tokens']
    probs = np.exp(np.array(data['probs']))
    create_html_file(tokens, probs, 
                     output_path / '{}.html'.format(given_qid), 
                     open(template_path).read()
                    )
    

if __name__ == '__main__':
    
    # ask_one(2)
    ask_all()