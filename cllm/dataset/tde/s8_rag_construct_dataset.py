'''
Construct QA with RAG.
'''
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

QUESTION_PROMPT = ''

def prepare_qa():
    pass

def load_jsonl_as_dict(jsonl_path):
    id_obj_dict = {}
    with open(jsonl_path, 'r') as rf:
        for line in rf:
            obj = json.loads(line)
            id_obj_dict[obj['id']] = obj
    return id_obj_dict

def load_embedding_as_dict(embedding_path):
    with open(embedding_path, 'rb') as rf:
        # id, embedding.
        return pickle.load(rf)
    
class NamedEmbeddingSearch():

    def __init__(self, db_dict, q_dict) -> None:
        self.db_dict = db_dict
        self.q_dict = q_dict
        # compute id - pos mapping.
        db_data = [(k, v)for k,v in db_dict.items()]
        db_data.sort(key=lambda x:x[0])
        self.data = pd.DataFrame(db_data, columns=['id', 'embed'])
        self.db_arr = np.array(self.data['embed'].to_list())
    
    def search_q(self, q_id, k):
        q = np.array(self.q_dict[q_id])
        distances = np.linalg.norm(self.db_arr - q, axis=1)
        neighbors = np.argpartition(distances, range(0, k))[:k]
        # convert back to ids.
        neighbor_ids = [self.data.iloc[i]['id'] for i in neighbors]
        return neighbor_ids
    

def construct_dataset_w_embedding(source_path, source_desc_path,
                                   func_desc_path, source_embed_path, func_embed_path,
                                   k, output_path):
    source_dict = load_jsonl_as_dict(source_path)
    source_desc_dict = load_jsonl_as_dict(source_desc_path)
    func_desc_dict = load_jsonl_as_dict(func_desc_path)

    source_embeddings = load_embedding_as_dict(source_embed_path)
    func_embeddings = load_embedding_as_dict(func_embed_path)

    search = NamedEmbeddingSearch(func_embeddings, source_embeddings)
    
    with open(output_path, 'w') as wf:
        for _id, _obj in tqdm(source_dict.items()):
            try:
                func_ids = search.search_q(_id, k)
            except: 
                func_ids = search.search_q(str(_id), k)
            # ground truth id
            if type(_id) == int:
                gt_id = _id
            elif type(_id) == str:
                gt_id = _obj['oq_id']
            else:
                print('unknown gt id for _id: ', _id)
                raise NotImplementedError()
            # generate the question.
            choices = []
            for i, fid in enumerate(func_ids):
                letter = chr(ord('A')+i)
                f_desc = func_desc_dict[fid]['desc']
                choices.append('{}. {}'.format(letter, f_desc))
            gt_index = [*func_ids, gt_id].index(gt_id)
            choices.append('{}. {}'.format(chr(ord('A')+len(func_ids)), 'None of the above'))
            question = '\n'.join(choices)
            answers = [chr(ord('A')+gt_index)]
            wf.write('{}\n'.format(json.dumps({
                'question_id': _id,
                'desc': source_desc_dict[_id]['desc'],
                'seq_a': _obj['seq_a'],
                'seq_b': _obj['seq_b'],
                'tip': _obj['tip'],
                'choices': question,
                'answers': answers
            })))
        # get function desc.

def generate_for_gpt_35():
    dataset_folder = Path('./datasets/TDE/')
    
    func_desc_path = dataset_folder / 'gpt_gt_func_desc.3.5.jsonl'

    func_desc_embeddings = dataset_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'

    # source_desc_path = dataset_folder / 'gpt_more_outputs_desc_3.5.jsonl'
    # source_path = dataset_folder / 'gpt_more_outputs_3.5' / 'all.jsonl'
    # source_desc_embeddings = dataset_folder / 'embedding' / 'more_embedding_seq_desc_3.5.pkl'
    # output_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_3.5.jsonl'

    source_desc_path = dataset_folder / 'gpt_seqdesc_desc_3.5.jsonl'
    source_path = dataset_folder / 'raw_datasets_transformed.jsonl'
    source_desc_embeddings = dataset_folder / 'embedding' / 'embedding_seq_desc_3.5.pkl'
    output_path = dataset_folder / 'dataset' / 'origin_w_embedding_k10_3.5.jsonl'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    k = 10

    construct_dataset_w_embedding(source_path, source_desc_path, func_desc_path, 
                                  source_desc_embeddings, func_desc_embeddings, k, output_path)

def generate_for_llama7b():
    dataset_folder = Path('./datasets/TDE/')
    
    func_desc_path = dataset_folder / 'gpt_gt_func_desc.3.5.jsonl'

    func_desc_embeddings = dataset_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'

    source_desc_path = dataset_folder / 'gpt_more_outputs_desc_llama2-7b.jsonl'
    source_path = dataset_folder / 'gpt_more_outputs_3.5' / 'all.jsonl'
    source_desc_embeddings = dataset_folder / 'embedding' / 'more_embedding_seq_desc_llama2-7b.pkl'
    output_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_llama2-7b.jsonl'


    # source_desc_path = dataset_folder / 'gpt_seqdesc_desc_llama2-7b.jsonl'
    # source_path = dataset_folder / 'raw_datasets_transformed.jsonl'
    # source_desc_embeddings = dataset_folder / 'embedding' / 'embedding_seq_desc_llama2-7b.pkl'
    # output_path = dataset_folder / 'dataset' / 'origin_w_embedding_k10_llama2-7b.jsonl'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    k = 10

    construct_dataset_w_embedding(source_path, source_desc_path, func_desc_path, 
                                  source_desc_embeddings, func_desc_embeddings, k, output_path)
def generate_for_llama13b():
    dataset_folder = Path('./datasets/TDE/')
    
    func_desc_path = dataset_folder / 'gpt_gt_func_desc.3.5.jsonl'

    func_desc_embeddings = dataset_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'

    # source_desc_path = dataset_folder / 'gpt_more_outputs_desc_llama2-13b.jsonl'
    # source_path = dataset_folder / 'gpt_more_outputs_3.5' / 'all.jsonl'
    # source_desc_embeddings = dataset_folder / 'embedding' / 'more_embedding_seq_desc_llama2-13b.pkl'
    # output_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_llama2-13b.jsonl'


    source_desc_path = dataset_folder / 'gpt_seqdesc_desc_llama2-13b.jsonl'
    source_path = dataset_folder / 'raw_datasets_transformed.jsonl'
    source_desc_embeddings = dataset_folder / 'embedding' / 'embedding_seq_desc_llama2-13b.pkl'
    output_path = dataset_folder / 'dataset' / 'origin_w_embedding_k10_llama2-13b.jsonl'

    output_path.parent.mkdir(parents=True, exist_ok=True)
    k = 10

    construct_dataset_w_embedding(source_path, source_desc_path, func_desc_path, 
                                  source_desc_embeddings, func_desc_embeddings, k, output_path)


if __name__ == "__main__":
    # generate_for_gpt_35()
    # generate_for_llama7b()
    generate_for_llama13b()
    pass