from dotenv import load_dotenv
load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteriaList, StoppingCriteria, set_seed, BitsAndBytesConfig
import torch
import torch.nn.functional as F

from datasets import load_dataset, DatasetDict, Dataset
import os
from pathlib import Path
from tqdm import tqdm
import json
import math
from typing import Dict, Any

device = os.environ['DEVICE']


class StoppingCriteriaOnTokens(StoppingCriteria):
  def __init__(self, stops = []) -> None:
    super().__init__()
    self.stops = [stop.to(device) for stop in stops]
  
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
    for seq in input_ids:
      for stop in self.stops:
        if len(seq) >= len(stop) and torch.equal(stop,seq[-len(stop):]):
          return True
    return False

class LlamaHFClient:

    def __init__(self, model_path, save_mem=True, bits_and_bytes=False) -> None:
        extra_params = {}
        if save_mem:
            ttype = torch.float16
            if 'cuda' in device:
                ttype = torch.bfloat16
            extra_params['torch_dtype']=ttype
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # if self.tokenizer.pad_token_id is None:
        if bits_and_bytes:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                # bnb_4bit_compute_dtype = torch.float16,
                # bnb_4bit_quant_type='nf4',
                # bnb_4bit_use_double_quant = False
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb_config, 
                device_map='auto', **extra_params
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', **extra_params)
        self.pipeline = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)
        self.stop_ids = []
        self.stopping_criteria = None
    
    def set_stop_tokens(self, stop=['\n', '.', ','], additional_stop_ids = [
        [29899], # .
        [29892], # ,
        [13], # \n
    ]):
        '''
        Note: for special tokens that cannot encode from raw symbols, please put them into the additional_stop_ids.
        '''
        if stop is not None and len(stop) > 0:
            stop_ids = [self.tokenizer(s, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0) for s in stop]
            self.stop_ids = stop_ids
            if len(additional_stop_ids) > 0:
                self.stop_ids.extend([torch.tensor(x) for x in additional_stop_ids])
            # add eos id
            self.stop_ids.append(torch.tensor([self.tokenizer.eos_token_id]))
            self.stop_ids = [x.to(device) for x in self.stop_ids]
            self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaOnTokens(self.stop_ids)])
    
    def get_tokens_before_stop(self, tokens: torch.LongTensor):
        for i in range(len(tokens)):
            for stop_id in self.stop_ids:
                sub_tokens = tokens[:i+1]
                # if len(stop_id) <= i+1:
                #     print('compare', stop_id, sub_tokens[-len(stop_id): ], tokens)
                if len(stop_id) <= i+1 and torch.equal(stop_id, sub_tokens[-len(stop_id):]):
                    # stop
                    return sub_tokens[:-len(stop_id)]
        return tokens
    
    def sample_answer(self, question, seed, batch_size=1, max_tokens=100, temperature=1):
        # fix the randomness
        set_seed(seed)

        inputs = self.tokenizer([question], return_tensors='pt')
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask

        with torch.no_grad():
            outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_tokens,
                                    do_sample=True, top_k=0, top_p=1, temperature=temperature,
                                    stopping_criteria=self.stopping_criteria,
                                    return_dict_in_generate=True, output_scores=True,
                                    num_return_sequences=batch_size, pad_token_id = self.tokenizer.eos_token_id)
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits = True
        )
        input_length = 1 if self.model.config.is_encoder_decoder else input_ids.shape[1]
        generated_tokens = outputs.sequences[:, input_length:]

        results = []
        for tokens, probs in zip(generated_tokens, transition_scores):
            # filter based on stop words.
            valid_tokens = self.get_tokens_before_stop(tokens)
            valid_probs = probs[:len(valid_tokens)]
            results.append({
                'response': self.tokenizer.decode(valid_tokens, skip_special_tokens=True),
                'tokens': valid_tokens.tolist(),
                'probs': valid_probs.tolist(),
            })
        return results
    
        # print(outputs.keys())
        # print(outputs['sequences'].shape)
        # print(len(outputs['scores']))
        # print(outputs['scores'][0].shape)
        # print(len(outputs['logits']))
        # print(outputs['logits'][0].shape)

        # transform back to answers.
        # get only the answer part:

        assert outputs['sequences'].shape[0] == 1, 'expecting one answer'

        ans_ids = outputs['sequences'].reshape(-1)[len(input_ids.reshape(-1)): ]
        
        probs = []
        total_prob = 1
        
        for _next_id, _next_scores in zip(ans_ids, outputs['scores']):
            # print('shape', _next_scores.shape)
            _next_probabilities = F.softmax(_next_scores, dim=-1)
            _next_prob = _next_probabilities[0][_next_id].item()
            probs.append(_next_prob)
            total_prob *= _next_prob

def run_and_save(model_name, prompt_template: str, ds_splits: DatasetDict, 
                 run_config: Dict[str, Any], dataset_name: str, run_id: str, resume:bool = True):
    '''
    n: number of results obtained from each question.
    '''
    torch.manual_seed(run_config['seed'])
    
    client = LlamaHFClient(model_name)
    if model_name == 'meta-llama/Meta-Llama-3-8B':
        client.set_stop_tokens(additional_stop_ids=[], stop=['.', '\n', '\n\n', ','])
    else:
        client.set_stop_tokens()

    dest_folder = Path('data_out') / dataset_name / run_id
    dest_folder.mkdir(parents=True, exist_ok=True)

    n = run_config['generation_num']
    base_seed = run_config['seed']
    batch_size = run_config['batch_size']

    def _runsave(_dss: Dataset, _dest_file: Path, resume: bool):
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
                prompt = prompt_template.format(Question = q)
                batch_num = math.ceil(n / batch_size)
                for i in range(batch_num):
                    sequence_num = batch_size if i+1 < batch_num else n - batch_size * i
                    rets.extend(client.sample_answer(prompt, seed=base_seed + i, batch_size= sequence_num))
                res['pred'] = rets
                _json_s = json.dumps(res)
                wf.write('{}\n'.format(_json_s))
                # _res_json.append(res)
                # write this
                # print('------ {} -------'.format(_idx))
                # for i in rets:
                #    print(i)
                # if _idx >10:
                #     exit()
    _runsave(ds_splits, dest_folder / 'all_raw.jsonl', resume)
    # _runsave(ds_splits['cal'], dest_folder / 'calibration_raw.jsonl')
    # _runsave(ds_splits['val'], dest_folder / 'validation_raw.jsonl')
    # _runsave(ds_splits['test'], dest_folder / 'test_raw.jsonl')
