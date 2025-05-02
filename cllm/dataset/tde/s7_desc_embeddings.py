from openai import OpenAI
from pathlib import Path
import os
import pickle
from tqdm import tqdm
from cllm.io import load_data
from cllm.dataset.tde.common import REVISED_MORE_INPUT_IDS
import logging
from cllm.utils import setup_logging

from dotenv import load_dotenv

load_dotenv()

def get_embeddings(input_folder, output_path, model='text-embedding-3-small'):
    client = OpenAI()
    res = {}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(input_folder).is_file():
        # read jsonl.
        for obj in tqdm(load_data(input_folder)):
            key = str(obj['id'])
            embed = client.embeddings.create(input=[obj['desc']], model=model).data[0].embedding
            res[key] = embed
    else:
        for file_name in tqdm(os.listdir(input_folder)):
            key = file_name.split('.')[0]
            with open(input_folder / file_name, 'r') as rf:
                content = rf.read()
            embed = client.embeddings.create(input=[content], model=model).data[0].embedding
            res[key] = embed
    # dump it.
    with open(output_path, 'wb') as wf:
        pickle.dump(res, wf)

def get_embeddings_fix_mode(origin_output_path, input_folder, output_path, model='text-embedding-3-small'):
    with open(origin_output_path, 'rb') as rf:
        origin_dict = pickle.load(rf)
    retained_dict = dict()
    for k, v in origin_dict.items():
        if int(k.split('-')[0]) not in REVISED_MORE_INPUT_IDS:
            retained_dict[k] = v

    client = OpenAI()
    res = {}
    if Path(input_folder).is_file():
        # read jsonl.
        for obj in tqdm(load_data(input_folder)):
            key = obj['id']
            if key in retained_dict:
                res[key] = retained_dict[key]
            else:
                embed = client.embeddings.create(input=[obj['desc']], model=model).data[0].embedding
                res[key] = embed
    else:
        for file_name in tqdm(os.listdir(input_folder)):
            key = file_name.split('.')[0]
            if key in retained_dict:
                res[key] = retained_dict[key]
            else:
                with open(input_folder / file_name, 'r') as rf:
                    content = rf.read()
                embed = client.embeddings.create(input=[content], model=model).data[0].embedding
                res[key] = embed
    # dump it.
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as wf:
        pickle.dump(res, wf)


if __name__ == '__main__':
    # set logging level to debug.
    setup_logging(level=logging.DEBUG, to_file=False)

    input_folder = Path('./datasets/TDE/')
    output_folder = input_folder / 'embedding'

    # desc_folder = input_folder / 'gpt_answers_desc_3.5'
    # get_embeddings(desc_folder, output_folder / 'embedding_answers_desc_3.5.pkl')

    # desc_folder = input_folder / 'gpt_seqdesc_desc_3.5'
    # get_embeddings(desc_folder, output_folder / 'embedding_seq_desc_3.5.pkl')

    # desc_folder = input_folder / 'gpt_more_outputs_desc_3.5'
    # get_embeddings(desc_folder, output_folder / 'more_embedding_seq_desc_3.5.pkl')


    # desc_folder = input_folder / 'gpt_more_outputs_desc_llama2-7b.jsonl'
    # get_embeddings(desc_folder, output_folder / 'more_embedding_seq_desc_llama2-7b.pkl')

    # desc_folder = input_folder / 'gpt_seqdesc_desc_llama2-13b.jsonl'
    # get_embeddings(desc_folder, output_folder / 'embedding_seq_desc_llama2-13b.pkl')

    # desc_folder = input_folder / 'gpt_seqdesc_desc_llama2-7b.jsonl'
    # get_embeddings(desc_folder, output_folder / 'embedding_seq_desc_llama2-7b.pkl')


    # desc_folder = input_folder / 'struct_gpt_origin_desc_3.5.jsonl'
    # get_embeddings(desc_folder, output_folder / 'embedding_struct_gpt_origin_desc_3.5.pkl')

    # desc_folder = input_folder / 'struct_gpt_more_outputs_desc_20_3.5_fix.jsonl'
    # get_embeddings(desc_folder, output_folder / 'embedding_struct_gpt_more_outputs_desc_20_3.5_fix.pkl')


    # desc_folder = input_folder / 'gpt_more_outputs_20_num1_desc_3.5.jsonl'
    # get_embeddings(desc_folder, output_folder / 'more_embedding_seq_desc_20_num1_3.5.pkl')
    # name = 3
    # desc_folder = input_folder / 'gpt_more_outputs_20_num{}_desc_3.5_fix.jsonl'.format(name)
    # origin_path = output_folder / 'more_embedding_seq_desc_20_num{}_3.5.pkl'.format(name)
    # get_embeddings_fix_mode(origin_path, desc_folder, output_folder / 'more_embedding_seq_desc_20_num{}_3.5_fix.pkl'.format(name))
    # desc_folder = input_folder / 'gpt_more_outputs_20_num2_desc_3.5.jsonl'
    # get_embeddings(desc_folder, output_folder / 'more_embedding_seq_desc_20_num2_3.5.pkl')
    # desc_folder = input_folder / 'gpt_more_outputs_20_num3_desc_3.5.jsonl'
    # get_embeddings(desc_folder, output_folder / 'more_embedding_seq_desc_20_num3_3.5.pkl')

    input_folder = Path('./datasets/TDE2/')
    output_folder = input_folder / 'embedding'
    # desc_folder = input_folder / 'struct_more_outputs_gpt_desc_3.5'
    # get_embeddings(desc_folder, output_folder / 'struct_more_embedding_desc_3.5.pkl')

    desc_folder = input_folder / 'function_answers_desc_3.5'
    get_embeddings(desc_folder, output_folder / 'embedding_function_desc_3.5.pkl')
