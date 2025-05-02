#
from pathlib import Path
from cllm.dataset.tde.s8_rag_construct_dataset import *

def compute_distance_recall(func_path, ques_path, dist_threshold):
    func_dict = load_embedding_as_dict(func_path)
    question_dict = load_embedding_as_dict(ques_path)

    func_arr = np.array(list(func_dict.values()))
    contains = 0
    retrival_num = 0
    for _id, embed in tqdm(question_dict.items()):
        gt = _id
        if type(gt) == str:
            gt = int(_id.split('-')[0])
        _dist = np.linalg.norm(np.array(func_dict[gt]) - np.array(question_dict[_id]), axis=0)
        if _dist <= dist_threshold:
            contains += 1
        # compute all.
        distances = np.linalg.norm(func_arr - np.array(question_dict[_id]), axis=1)
        retrival_num += distances[distances <= dist_threshold].shape[0]
    return contains / len(question_dict), contains / retrival_num if retrival_num > 0 else 1

def compute_distance_recalls(func_path, ques_path, res_path, thresholds):
    data = []
    for thres in thresholds:
        recall, precision = compute_distance_recall(func_path, ques_path, thres)
        data.append((thres, recall, precision))
    
    Path(res_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=['dist', 'recall', 'precision']).to_csv(res_path)

if __name__ == '__main__':
    input_folder = Path('./datasets/TDE/')
    func_path = input_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]

    # ques_path = input_folder / 'embedding' / 'embedding_seq_desc_llama2-7b.pkl'
    # output_name = 'embedding_distance_recall_llama2_7b.csv'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_llama2-7b.pkl'
    # output_name = 'embedding_distance_recall_aug_llama2_7b.csv'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_3.5.pkl'
    # output_name = 'embedding_distance_recall_aug_gpt3.5.csv'
    # ques_path = input_folder / 'embedding' / 'embedding_seq_desc_3.5.pkl'
    # output_name = 'embedding_distance_recall_gpt3.5.csv'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num2_3.5.pkl'
    # output_name = 'embedding_distance_20_num2_recall_gpt3.5.csv'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num3_3.5.pkl'
    # output_name = 'embedding_distance_20_num3_recall_gpt3.5.csv'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num1_3.5.pkl'
    # output_name = 'embedding_distance_20_num1_recall_gpt3.5.csv'
    ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num3_3.5_fix.pkl'
    output_name = 'embedding_distance_20_num3_recall_gpt3.5_fix.csv'

    save_path = input_folder / 'analysis'
    compute_distance_recalls(
        func_path, ques_path, save_path / output_name, thresholds
    )