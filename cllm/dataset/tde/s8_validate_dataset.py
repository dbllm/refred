"""
Validate our generated datasets are correct.
"""
from pathlib import Path
from cllm.io import load_data
from cllm.dataset.tde.s8_rag_construct_dataset import NamedEmbeddingSearch, load_embedding_as_dict
from cllm.dataset.tde.s6_generate_outputs import *
from cllm.dataset.tde.common import *
from tqdm import tqdm
from cllm.utils import setup_logging

def validate_all(input_path, code_path, source_embed_path, func_embed_path, k):
    source_embeddings = load_embedding_as_dict(source_embed_path)
    func_embeddings = load_embedding_as_dict(func_embed_path)

    search = NamedEmbeddingSearch(func_embeddings, source_embeddings)

    obj_list = load_data(input_path)
    correct_count = 0
    skip_count = 0
    error_count = 0
    for obj in tqdm(obj_list):
        # get answer
        ans_id = ord(obj['answers'][0]) - ord('A')
        # get ans_id.
        res_list = search.search_q(obj['question_id'], k)
        if ans_id < len(res_list):
            func_id = res_list[ans_id]
            code_template = read_code(code_path / '{}.txt'.format(func_id))
            code = replace_input(code_template, obj['seq_a'])
            seq_b = execute_code(code)
            is_equal = seq_b.strip() == str(obj['seq_b']).strip()
            if is_equal:
                correct_count += 1
            else:
                error_count += 1
        else:
            skip_count += 1
            logging.info('skip since no function selected: %s, %s', obj['question_id'], ans_id)
        # retrieve function.
    logging.info('done. skipped: %s, correct: %s. error: %s', skip_count, correct_count, error_count)

if __name__ == '__main__':
    setup_logging(level=logging.INFO, to_file=False)
    dataset_folder = Path('./datasets/TDE/')
    code_path = dataset_folder / 'gpt_answers_func_manual'
    # source_path = dataset_folder / 'gpt_more_outputs_3.5' / 'all.jsonl'
    # source_desc_path = dataset_folder / 'gpt_more_outputs_desc_3.5.jsonl'
    # func_desc_path = dataset_folder / 'gpt_gt_func_desc.3.5.jsonl'

    func_desc_embeddings = dataset_folder / 'embedding_answers_desc_3.5.pkl'

    # source_desc_embeddings = dataset_folder / 'more_embedding_seq_desc_3.5.pkl'
    # input_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_3.5.jsonl'

    source_desc_embeddings = dataset_folder / 'embedding_seq_desc_3.5.pkl'
    input_path = dataset_folder / 'dataset' / 'origin_w_embedding_k10_3.5.jsonl'

    k = 10

    validate_all(input_path, code_path, source_desc_embeddings, func_desc_embeddings, k)
    # skipped: 265, correct: 1985. error: 0