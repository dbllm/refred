from openai import OpenAI, RateLimitError
import os
from dotenv import load_dotenv
import tiktoken
load_dotenv()
from typing import List, Any, Dict
import backoff
from pathlib import Path
import math
import json
from tqdm import tqdm

class OpenAIClient:
    
    def __init__(self, model_name, endpoint='openai') -> None:
        self.model_name = model_name
        if endpoint == 'openai':
            self.client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))
        elif endpoint == 'deepseek':
            self.client = OpenAI(api_key = os.environ.get("DEEPSEEK_API_KEY"), base_url='https://api.deepseek.com/')
        else:
            raise ValueError(f"Invalid endpoint: {endpoint}")
        # self.encoding = tiktoken.encoding_for_model(model_name)

    def close(self):
        self.client.close()

    # see https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    @backoff.on_exception(backoff.expo, RateLimitError)
    def complete(self, messages, seed, batch_size=20, max_tokens=100, stop=['\n', '.', ',']):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            n = batch_size,
            stream = False,
            temperature=1.0,
            max_tokens=max_tokens,
            top_p = 1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            logprobs=True,
            seed=seed,
            stop = stop
        )
        results = []
        for res in response.choices:
            results.append({
                'response': res.message.content,
                'tokens': [i.token for i in res.logprobs.content],
                'probs': [i.logprob for i in res.logprobs.content],
            })
        return results

def convert_qa_to_openai_template(prompt: str):
    messages = []
    qas = prompt.split('Q:')
    for qa in qas:
        # skip first empty string
        if len(qa.strip()) == 0: continue
        _q, _a = qa.split('A:')
        
        # do not consider current question
        if _q.strip() == '{Question}': break
        messages.extend([
            {"role": "user", "content": _q.strip()},
            {"role": "assistant", "content": _a.strip()}
        ])
    return messages

def append_message(messages: List[str], question: str):
    return [*messages, {"role": "user", "content": question}]

def run_and_save(model_name, prompt_template: str, dataset: List[Any], 
                 run_config: Dict[str, Any], dataset_name: str, run_id: str, resume:bool = True):
    '''
    n: number of results obtained from each question.
    '''
    client = OpenAIClient(model_name)

    dest_folder = Path('data_out') / dataset_name / run_id
    dest_folder.mkdir(parents=True, exist_ok=True)

    n = run_config['generation_num']
    base_seed = run_config['seed']
    batch_size = run_config['batch_size']
    prompt_messages = convert_qa_to_openai_template(prompt_template)

    def _runsave(_dss: List[Any], _dest_file: Path, resume: bool):
        if resume:
            with open(_dest_file, 'r') as rf:
                previous_res_num = len(rf.readlines())

        # _res_json = list()
        with open(_dest_file, 'a' if resume else 'w') as wf:
            for _idx, item in tqdm(enumerate(_dss), total=len(_dss)):
                if resume and _idx < previous_res_num:
                    continue
                res = dict(item)
                rets = []
                q = item['question']
                prompt = append_message(prompt_messages, q)
                batch_num = math.ceil(n / batch_size)
                for i in range(batch_num):
                    sequence_num = batch_size if i+1 < batch_num else n - batch_size * i
                    rets.extend(client.complete(prompt, seed=base_seed + i, batch_size= sequence_num))
                res['pred'] = rets
                _json_s = json.dumps(res)
                wf.write('{}\n'.format(_json_s))
    _runsave(dataset, dest_folder / 'all_raw.jsonl', resume)

if __name__ == '__main__':
    # client = OpenAIClient('gpt-3.5-turbo')

    # client.complete() 
    with open('prompts/usstock_mon.txt') as rf:
        prompt_template = '\n'.join(rf.readlines())

    import json
    # get one sample.
    with open('datasets/usstock_llm_mon/all.jsonl') as rf:
        obj = json.loads(rf.readline())
    
    message_template = convert_qa_to_openai_template(prompt_template)
    message_prompt = append_message(message_template, obj['question'])
    client = OpenAIClient('gpt-3.5-turbo-0125')
    res = client.complete(message_prompt, 42, 2)
    print(res)