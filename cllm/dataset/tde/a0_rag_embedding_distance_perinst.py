
from pathlib import Path
from cllm.dataset.tde.s8_rag_construct_dataset import *

def compute_gt_distances(func_path, ques_path):
    func_dict = load_embedding_as_dict(func_path)
    question_dict = load_embedding_as_dict(ques_path)

    # per questions.
    dist_res_dict = dict()
    for _id, embed in tqdm(question_dict.items()):
        gt = _id
        if type(gt) == str:
            gt = int(_id.split('-')[0])
        _dist = np.linalg.norm(np.array(func_dict[gt]) - np.array(question_dict[_id]), axis=0)
        dist_res_dict[_id] = _dist
    return dist_res_dict

def compute_distance_recalls(func_path, ques_path, res_path, thresholds):

    # 1. compute the gt distance values.
    id_dist_dict = compute_gt_distances(func_path, ques_path)
    raw_pdf = pd.DataFrame(list(id_dist_dict.items()), columns=['id', 'dist'])

    Path(res_path).mkdir(parents=True, exist_ok=True)

    for thres in thresholds:
        pdf = raw_pdf.copy()
        pdf['recall'] = pdf['dist'] <= thres
        pdf['thres'] = thres
        pdf.to_csv(res_path / 'thres_{}.csv'.format(thres))


if __name__ == '__main__':
    input_folder = Path('./datasets/TDE/')
    func_path = input_folder / 'embedding' / 'embedding_answers_desc_3.5.pkl'
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]

    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num2_3.5.pkl'
    # output_name = 'embedding_distance_20_num2_recall_gpt3.5_perinst'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num3_3.5.pkl'
    # output_name = 'embedding_distance_20_num3_recall_gpt3.5_perinst'
    # ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num1_3.5.pkl'
    # output_name = 'embedding_distance_20_num1_recall_gpt3.5_perinst'
    ques_path = input_folder / 'embedding' / 'more_embedding_seq_desc_20_num3_3.5_fix.pkl'
    output_name = 'embedding_distance_20_num3_recall_gpt3.5_perinst_fix'

    save_path = input_folder / 'analysis'
    compute_distance_recalls(
        func_path, ques_path, save_path / output_name, thresholds
    )