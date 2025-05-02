from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset, DatasetDict, Dataset
import logging
import os
from langchain.llms.utils import enforce_stop_tokens
import re
from cllm.utils import setup_logging, args_to_dict
from pathlib import Path
import cllm.constants as constants
from tqdm import tqdm
import json
import math
from typing import Dict, Any
import pyarrow as pa
from cllm.io import load_data

import argparse
        

def clean_data(dataset: Dataset):
    def filter_func(item):
    #    return len(item['answers']) == 1
    #    return not re.match(r'[.\n,]', item['answers'])
        for a in item['answers']:
            if re.match(r'[.\n,]', a): return False
        return True
    return dataset.filter(filter_func)


def load_web_questions(split):

    dataset = load_dataset("web_questions", split=split)
    # split into calibration, validation, & test.
    logging.info('total # items [%s]', len(dataset))

    dataset = clean_data(dataset)
    # logging.info('# of items after cleaning: [%s]', len(dataset))
    # # sample.
    # shuffled_dataset = dataset.shuffle(seed=seed).select(range(sample_num))
    # logging.info('# of items after sampling: [%s]', len(shuffled_dataset))
    
    # cal_res_datasets = shuffled_dataset.train_test_split(test_size=0.5, shuffle=False, seed=seed)
    # val_test_datasets = cal_res_datasets['test'].train_test_split(test_size=0.5, shuffle=False, seed=seed)

    # ds_splits = DatasetDict({
    #     'cal': cal_res_datasets['train'],
    #     'val': val_test_datasets['train'],
    #     'test': val_test_datasets['test']
    # })
    return dataset

def load_triviaqa(split):

    dataset = load_dataset('trivia_qa', 'rc', split=split)
    # train: 138384, validation: 17944, test: 17210
    # we only keep the following items: 'question', 'question_id', 'answer.aliases'
    dataset = dataset.map(lambda x: {
        'question': x['question'],
        'question_id': x['question_id'],
        'answers': x['answer']['aliases']
    }, 
    remove_columns=['answer', 'question_source', 'entity_pages', 'search_results', 'answer']
    )
    logging.info('total # items [%s]', len(dataset))

    dataset = clean_data(dataset)
    logging.info('# of items after cleaning: [%s]', len(dataset))
    # # sample.
    # shuffled_dataset = dataset.shuffle(seed=seed).select(range(sample_num))
    # logging.info('# of items after sampling: [%s]', len(shuffled_dataset))
    
    # cal_res_datasets = shuffled_dataset.train_test_split(test_size=0.5, shuffle=False, seed=seed)
    # val_test_datasets = cal_res_datasets['test'].train_test_split(test_size=0.5, shuffle=False, seed=seed)

    # ds_splits = DatasetDict({
    #     'cal': cal_res_datasets['train'],
    #     'val': val_test_datasets['train'],
    #     'test': val_test_datasets['test']
    # })
    return dataset

def load_usstock(folder, split):
    data = []
    with open('./datasets/{}/{}.jsonl'.format(folder, split)) as wf:
        for line in wf:
            data.append(json.loads(line))
    dataset = Dataset(pa.Table.from_pylist(data))
    # we get 
    logging.info('total # items [%s]', len(dataset))

    dataset = clean_data(dataset)
    logging.info('# of items after cleaning: [%s]', len(dataset))
    return dataset

def load_usstock_mon():
    data = []
    with open('./datasets/usstock_llm_mon/all.jsonl') as wf:
        for line in wf:
            data.append(json.loads(line))
    dataset = Dataset(pa.Table.from_pylist(data))
    # we get 
    logging.info('total # items [%s]', len(dataset))

    dataset = clean_data(dataset)
    logging.info('# of items after cleaning: [%s]', len(dataset))
    return dataset

def load_tde(file_name):
    data = load_data(Path('./datasets/TDE/dataset') / file_name)
    from cllm.dataset.tde.s8_rag_gen_prompt import build_question
    for obj in data:
        obj['question'] = build_question(obj, True)
        del obj['choices']
        del obj['seq_a']
        del obj['seq_b']
        del obj['desc']
    dataset = Dataset(pa.Table.from_pylist(data))
    logging.info('total # items [%s]', len(dataset))
    # transform questions.

    return dataset

def load_std(file_name):
    data = load_data(Path('./datasets/STD/dataset') / file_name)
    from cllm.dataset.tde.s8_rag_gen_prompt import build_question
    for obj in data:
        obj['question'] = build_question(obj, True)
        del obj['choices']
        del obj['seq_a']
        del obj['seq_b']
        del obj['desc']
    dataset = Dataset(pa.Table.from_pylist(data))
    logging.info('total # items [%s]', len(dataset))
    # transform questions.

    return dataset

def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config')
    parser.add_argument('--dataset_config')
    parser.add_argument('--run_config', default='config/runconfig/default.yaml')
    parser.add_argument('--output')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    return args_to_dict(args)


def execute_run(model_conf, dataset_conf, run_conf, output, resume):

    if dataset_conf['name'] == 'web_questions':
        dataset = load_web_questions(dataset_conf['split'])
    elif dataset_conf['name'] == 'triviaqa':
        dataset = load_triviaqa(dataset_conf['split'])
    elif dataset_conf['name'] == 'usstock':
        dataset = load_usstock('usstock_llm', dataset_conf['split'])
    elif dataset_conf['name'] == 'usstock2':
        dataset = load_usstock('usstock_llm2', dataset_conf['split'])
    elif dataset_conf['name'] == 'usstock_mon':
        dataset = load_usstock_mon()
    elif dataset_conf['name'] == 'tde':
        dataset = load_tde(dataset_conf['split'])
    elif dataset_conf['name'] == 'std':
        dataset = load_std(dataset_conf['split'])
    else:
        raise NotImplementedError()

    with open('prompts/{}'.format(dataset_conf['prompt'])) as rf:
        prompt_template = '\n'.join(rf.readlines())
            
    if model_conf['type'] == 'openai':
        from cllm.aux.open_ai import run_and_save
        run_and_save(model_conf['model'], prompt_template, dataset, run_conf,
                dataset_conf['name'], output, resume)
    elif model_conf['type'] == 'huggingface':
        from cllm.aux.huggingface import run_and_save
        run_and_save(model_conf['model'], prompt_template, dataset, run_conf,
                dataset_conf['name'], output, resume)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    setup_logging(logging.INFO, to_file=True)
    args = prepare_args()
    execute_run(args['model'], args['dataset'], args['run'], args['output'], args['resume'])
