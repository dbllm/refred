import argparse
from pathlib import Path
from tqdm import tqdm
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import os
import numpy as np
import pickle
from numpy.typing import ArrayLike

device = os.environ['DEVICE']

class LlamaHFClient:

    def __init__(self, model_path, save_mem=False) -> None:
        extra_params = {}
        if save_mem:
            ttype = torch.float16
            if 'cuda' in device:
                ttype = torch.bfloat16
            extra_params['torch_dtype']=ttype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', **extra_params)
        self.stop_ids = []
        self.stopping_criteria = None
    
    def get_embedding(self, question) -> torch.Tensor:
        input_ids = self.tokenizer([question], return_tensors='pt').input_ids.to(device)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, max_new_tokens=1,
                                    do_sample=True, top_k=0, top_p=1,
                                    stopping_criteria=self.stopping_criteria,
                                    return_dict_in_generate=True, output_hidden_states=True)
        # get hidden states
        # print(outputs['hidden_states'])
        # print(len(outputs['hidden_states']), len(outputs['hidden_states'][0]))
        # print('input', input_ids.shape)
        # print('embedding', outputs['hidden_states'][0][-1].shape)
        return outputs['hidden_states'][0][-1].squeeze().cpu().detach()
    
    
def compute_llm_embedding(input_file, output_file, model_name, prompt_file, save_mem, method='last'):
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    client = LlamaHFClient(model_name, save_mem)

    if prompt_file is not None:
        with open('prompts/{}'.format(prompt_file), 'r') as rf:
            prompt_template = '\n'.join(rf.readlines())
            # convert to messages
    
    convert_func = METHODS[method]

    with open(input_file, 'r') as rf:
        arr = []
        for i, line in enumerate(tqdm(rf)):
            item = json.loads(line)
            
            if prompt_file is not None:
                q = prompt_template.format(Question= item['question'])
            else:
                q = item['question']
            e1 = client.get_embedding(q).float().numpy()
            e1 = convert_func(e1)
            arr.append(e1)
        
        arr = np.array(arr)
        with open(output_file, 'wb') as wf:
            pickle.dump(arr, wf)

def mean_pooling(arr: ArrayLike):
    return np.mean(arr, axis=0)

def max_pooling(arr: ArrayLike):
    return np.max(arr, axis=0)

def min_pooling(arr: ArrayLike):
    return np.min(arr, axis=0)

def last_token(arr: ArrayLike):
    return arr[-1]

METHODS = {
    'last': last_token,
    'mean': mean_pooling,
    'max': max_pooling,
    'min': mean_pooling
}

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--save_mem', required=True)
    parser.add_argument('--method', required=True, choices=METHODS.keys())
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    compute_llm_embedding(args.input, args.output, args.model_name, args.prompt, args.save_mem == 'True', args.method)
