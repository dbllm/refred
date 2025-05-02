from cllm.dataset.tde.s8_rag_construct_dataset import *
from pathlib import Path
import pandas as pd

def cal_rank(func_path, ques_path, res_path):
    func_dict = load_embedding_as_dict(func_path)
    question_dict = load_embedding_as_dict(ques_path)

    search = NamedEmbeddingSearch(func_dict, question_dict)

    data = []
    total_value = len(func_dict)
    for _id in tqdm(question_dict):
        res = search.search_q(_id, total_value)
        gt = _id
        if type(gt) == str:
            gt = int(_id.split('-')[0])
        # convert neighbors to function id.
        _rank = res.index(gt)
        _dist = np.linalg.norm(np.array(func_dict[gt]) - np.array(question_dict[_id]), axis=0)
        data.append((_id, gt, _rank, _dist))
    
    Path(res_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=['id', 'gt', 'rank', 'dist']).to_csv(res_path)

if __name__ == '__main__':
    input_folder = Path('./datasets/TDE/')
    func_path = input_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'

    ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_llama2-7b.pkl'
    output_name = 'llama2-7b_embedding_rank_k_aug.csv'

    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_3.5.pkl'
    # output_name = 'gpt3.5_embedding_rank_k_aug.csv'

    # ques_path = input_folder / 'embedding' / 'embedding_seq_desc_3.5.pkl'
    # output_name = 'gpt3.5_embedding_rank_k.csv'

    # ques_path = input_folder / 'embedding' / 'embedding_seq_desc_llama2-7b.pkl'
    # output_name = 'llama2-7b_embedding_rank_k.csv'

    save_path = input_folder / 'analysis'
    cal_rank(
        func_path, ques_path, save_path / output_name
    )