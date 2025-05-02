#  Analysis embeddings.
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from cllm.dataset.tde.s8_rag_construct_dataset import *

def knn_search(db, q, k):
    distances = np.linalg.norm(db - q, axis=1)
    neighbors = np.argpartition(distances, range(0, k))[:k]
    return neighbors

def load_embedding_as_df(file_path):
    with open(file_path, 'rb') as rf:
        key_embed = pickle.load(rf)
        data = []
        for k, item in key_embed.items():
            data.append((k, item))
        data.sort(key=lambda x: x[0])
        data = pd.DataFrame(data, columns=['id', 'embed'])
        def to_int(x):
            if type(x) == int: return x
            if '-' in x:
                # origin_qid - index.
                return int(x.split('-')[0])
            return int(x)
        data['id'] = data['id'].apply(to_int)
        return data

# evaluate 

def evaluate(func_path, ques_path, k):
    func_df = load_embedding_as_df(func_path)
    question_df = load_embedding_as_df(ques_path)

    # search.
    db = np.array(func_df['embed'].to_list())
    qs = np.array(question_df['embed'].to_list())
    # obtain mapping.

    contains = 0
    for i in tqdm(range(qs.shape[0])):
        q = qs[i]
        neighbors = knn_search(db, q, k)
        # convert neighbors to function id.
        q_id = question_df.iloc[i]['id']
        if q_id in func_df.iloc[neighbors]['id'].to_list(): 
            contains += 1
        
    rate = contains / qs.shape[0]
    print('k:', k, 'rate:', rate)
    return rate


def evaluate2(func_path, ques_path, k):
    func_dict = load_embedding_as_dict(func_path)
    question_dict = load_embedding_as_dict(ques_path)

    search = NamedEmbeddingSearch(func_dict, question_dict)

    contains = 0
    for _id in question_dict:
        res = search.search_q(_id, k)
        gt = _id
        if type(gt) == str:
            gt = int(_id.split('-')[0])
        # convert neighbors to function id.
        if gt in res:
            contains += 1
        
    rate = contains / len(question_dict)
    print('k:', k, 'rate:', rate)
    return rate

def analysis_origin_qs():
    input_folder = Path('./datasets/TDE/')
    func_path = input_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'
    # ques_path = input_folder / 'embedding_seq_desc_3.5.pkl'
    # output_name = 'embedding_recall_k.csv'
    ques_path = input_folder / 'embedding' / 'embedding_seq_desc_llama2-7b.pkl'
    output_name = 'llama2-7b_embedding_recall_k.csv'
    save_path = input_folder / 'analysis'
    save_path.mkdir(parents=True, exist_ok=True)

    data = []
    for k in ks:
        r = evaluate(func_path, ques_path, k)
        data.append((k, r))
    pd.DataFrame(data, columns=['k', 'rate']).to_csv(save_path / output_name)


def analysis_agumented_data():
    input_folder = Path('./datasets/TDE/')
    func_path = input_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'
    # ques_path = input_folder / 'more_embedding_seq_desc_3.5.pkl'
    # output_name = 'embedding_recall_k_aug.csv'
    ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_llama2-7b.pkl'
    output_name = 'llama2-7b_embedding_recall_k_aug.csv'
    save_path = input_folder / 'analysis'
    save_path.mkdir(parents=True, exist_ok=True)

    data = []
    for k in ks:
        r = evaluate(func_path, ques_path, k)
        data.append((k, r))
    pd.DataFrame(data, columns=['k', 'rate']).to_csv(save_path / output_name)

if __name__ == '__main__':
    ks = [1, 3, 5, 10, 15, 20, 30, 40, 60, 80, 100]
    analysis_agumented_data()
    analysis_origin_qs()
    