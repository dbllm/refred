import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from numpy.typing import ArrayLike

import pickle
from pathlib import Path
import pandas as pd

def write_tb_projector(seq_embeddings: ArrayLike, func_embeddings: ArrayLike, name):
    # Set up a logs directory, so Tensorboard knows where to look for files.
    # log_dir='../../logs/dist_analysis/{}'.format(name)
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    writer = SummaryWriter(name)
    writer.add_embedding(np.concatenate([seq_embeddings, func_embeddings]), 
                         metadata=
                         [*['query'] * len(seq_embeddings), *['func'] * len(func_embeddings)]
                         )
    # writer.add_embedding(func_embeddings, metadata=['func'] * len(func_embeddings))
    writer.add_text('test', 'some placeholder')
    writer.close()

def load_pickle_values(input_path):
    with open(input_path, 'rb') as rf:
        pk_dict = pickle.load(rf)
        return np.array(pd.Series(pk_dict.values()).to_list())
    
def write_embedding_desc_func():
    
    base_path = Path('./datasets/TDE/')
    embedding_path = base_path / 'embedding'
    target_path = Path('./data_out/tde/profile_result')

    # write_tb_projector(
    #     load_pickle_values(embedding_path / 'embedding_seq_desc_3.5.pkl'), 
    #     load_pickle_values(embedding_path / 'embedding_answers_desc_3.5.pkl'),
    #     target_path / 'embedding_origin'
    # )
    write_tb_projector(
        load_pickle_values(embedding_path / 'more_embedding_seq_desc_3.5.pkl'), 
        load_pickle_values(embedding_path / 'embedding_answers_desc_3.5.pkl'),
        target_path / 'embedding_aug'
    )

def write_specific_embedding():
    
    base_path = Path('./datasets/TDE/')
    embedding_path = base_path / 'embedding'
    target_path = Path('./data_out/tde/profile_result')

    # write_tb_projector(
    #     load_pickle_values(embedding_path / 'embedding_seq_desc_3.5.pkl'), 
    #     load_pickle_values(embedding_path / 'embedding_answers_desc_3.5.pkl'),
    #     target_path / 'embedding_origin'
    # )
    seq_embeddings = load_pickle_values(embedding_path / 'more_embedding_seq_desc_3.5.pkl')
    func_embeddings = load_pickle_values(embedding_path / 'embedding_answers_desc_3.5.pkl')

    writer = SummaryWriter(target_path / 'inspect_aug_0')
    writer.add_embedding(np.concatenate([seq_embeddings, func_embeddings]), 
                         metadata=
                         [*['query'] * len(seq_embeddings), *['func'] * len(func_embeddings)]
                         )
    # get index.
    writer.add_text('test', 'some placeholder')
    writer.close()

if __name__ == '__main__':
    write_embedding_desc_func()