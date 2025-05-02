from pathlib import Path
from cllm.reg.io import load_std_data, load_data, load_pickle
import numpy as np
import pandas as pd
import json
import re


QUESTION_DESC_TEMPLATE = """Which of the following function can be used to convert the following seq_a to seq_b?
seq_a: {}
seq_b: {}
Decscription: {}
"""



def prepare_pdf(csv_name, output_name):
    base_output_path = Path('./data_out/std/cp_predict/')
    source_path = base_output_path / '{}.csv'.format(csv_name)
    output_path = base_output_path / '{}.csv'.format(output_name)

    test_data_predict = pd.read_csv(source_path)
    # ask 
    full_pdf = load_std_data('aug')

    base_path = Path('./datasets/STD/')
    embedding_path = base_path / 'embedding'
    # gt_desc = load_data(base_path / 'function_answers_desc_3.5.jsonl')
    # gt_embedding = load_pickle(embedding_path / 'embedding_struct_funcs_desc_3.5.pkl')
    
    # get all gt ids.
    # gt_ids = sorted(gt_embedding.keys())
    # gt_embed_arr = np.array([gt_embedding[x] for x in gt_ids])
    
    # compute id
    all_gt_ids = sorted(full_pdf['gt_id'].unique())
    
    test_data = full_pdf[full_pdf['id'].isin(test_data_predict['id'])]
    test_data['pred'] = test_data['id'].apply(
        lambda x: test_data_predict[test_data_predict['id'] == x]['pred'].item()
    )
    # for each dist_arr, get elements within the range.

    test_data['s_gts'] = None
    test_data['s_gts'] = test_data['s_gts'].astype(object)
    for _i, _row in test_data.iterrows():
        _pred = eval(_row['pred'])
        included_gts = []
        for _dist, _gtid in zip(_row['dist_arr'], all_gt_ids):
            if _dist >= _pred[0] and _dist <= _pred[1]:
                included_gts.append(_gtid)
        test_data.at[_i, 's_gts'] = included_gts
    test_data = test_data.drop(columns=['dist_arr', 'emb', 'gt_emb'])
    
    test_data['c_gts'] = test_data['s_gts'].apply(lambda x: len(x))
    # print(test_data[test_data['c_gts'] > 1])

    # build choices.

    test_data.to_csv(output_path)

def build_text_multichoice_question(pdf_name, dest_path):
    base_output_path = Path('./data_out/std/cp_predict/')
    source_path = base_output_path / '{}.csv'.format(pdf_name)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    gt_desc = load_data(Path('./datasets/STD/') / 'function_answers_desc_3.5.jsonl')

    # convert to dict.
    gt_desc_dict = {}
    for _row in gt_desc:
        gt_desc_dict[_row['id']] = _row['desc']
    
    # print(gt_desc_dict)

    test_pdf = pd.read_csv(source_path)
    # get embedding questions.

    base_path = Path('./datasets/STD/')
    gt_desc = load_data(base_path / 'function_answers_desc_3.5.jsonl')
    trans_desc = load_data(base_path / 'struct_more_outputs_gpt_desc_3.5.jsonl')
    gt_desc_dict = {}
    for _row in gt_desc: gt_desc_dict[_row['id']] = _row['desc']

    trans_desc_dict = {}
    for _row in trans_desc: trans_desc_dict[_row['id']] = _row

    # load origin jsonl.
    trans_jsonl = load_data(base_path / 'outputs' / 'all.jsonl')
    trans_jsonl_dict = {}
    for _row in trans_jsonl: trans_jsonl_dict[_row['id']] = _row

    with open(dest_path, 'w') as wf:
        for _, _row in test_pdf.iterrows():
            _id = _row['id']
            gts = eval(_row['s_gts'])
            gt_id = _row['gt_id']

            # build choices.
            choices = []
            for i, fid in enumerate(gts):
                letter = chr(ord('A')+i)
                f_desc = gt_desc_dict[fid]
                choices.append('{}. {}'.format(letter, f_desc))
            gt_index = [*gts, gt_id].index(gt_id)
            choices.append('{}. {}'.format(chr(ord('A')+len(gts)), 'None of the above'))

            question = '\n'.join(choices)
            answers = [chr(ord('A')+gt_index)]
            wf.write('{}\n'.format(json.dumps({
                'question_id': _id,
                'desc': trans_desc_dict[_id]['desc'],
                'seq_a': trans_jsonl_dict[_id]['seq_a'],
                'seq_b': trans_jsonl_dict[_id]['seq_b'],
                'tip': None,
                'choices': question,
                'answers': answers
            })))


def build_code_multichoice_question(pdf_name, dest_path):
    base_output_path = Path('./data_out/std/cp_predict/')
    source_path = base_output_path / '{}.csv'.format(pdf_name)
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)


    test_pdf = pd.read_csv(source_path)
    # get embedding questions.

    base_path = Path('./datasets/STD/')
    func_jsonl = load_data(base_path / 'all_funcs.jsonl')
    func_dict = {}
    for funcobj in func_jsonl:
        func_dict[funcobj['id']] = extract_func(funcobj['name'])
    
    # gt_desc = load_data(base_path / 'function_answers_desc_3.5.jsonl')
    trans_desc = load_data(base_path / 'struct_more_outputs_gpt_desc_3.5.jsonl')
    # gt_desc_dict = {}
    # for _row in gt_desc: gt_desc_dict[_row['id']] = _row['desc']

    trans_desc_dict = {}
    for _row in trans_desc: trans_desc_dict[_row['id']] = _row

    # load origin jsonl.
    trans_jsonl = load_data(base_path / 'outputs' / 'all.jsonl')
    trans_jsonl_dict = {}
    for _row in trans_jsonl: trans_jsonl_dict[_row['id']] = _row

    with open(dest_path, 'w') as wf:
        for _, _row in test_pdf.iterrows():
            _id = _row['id']
            gts = eval(_row['s_gts'])
            gt_id = _row['gt_id']

            # build choices.
            choices = []
            for i, fid in enumerate(gts):
                letter = chr(ord('A')+i)
                f_desc = func_dict[fid]
                choices.append('{}. {}'.format(letter, f_desc))
            gt_index = [*gts, gt_id].index(gt_id)
            choices.append('{}. {}'.format(chr(ord('A')+len(gts)), 'None of the above'))

            question = '\n'.join(choices)
            answers = [chr(ord('A')+gt_index)]
            wf.write('{}\n'.format(json.dumps({
                'question_id': _id,
                'desc': trans_desc_dict[_id]['desc'],
                'seq_a': trans_jsonl_dict[_id]['seq_a'],
                'seq_b': trans_jsonl_dict[_id]['seq_b'],
                'tip': None,
                'choices': question,
                'answers': answers
            })))

def extract_func(name, base_path='./datasets/STD/functions'):
    content = []
    with open(Path(base_path) / '{}.py'.format(name)) as rf:
        for line in rf.readlines():
            if '# Example usage'.lower() in line.lower():
                break
            else:
                content.append(line)
    res = ''.join(content)
    return res

def generate_prompt(origin_path, output_path, prompt_path):
    jsonl = load_data(origin_path)
    state = np.random.RandomState(42)
    samples = state.choice(len(jsonl), 2)
    prompt_samples = []
    with open(output_path, 'w') as wf:
        for _id, _row in enumerate(jsonl):
            if _id in samples:
                prompt_samples.append(_row)
            else:
                wf.write('{}\n'.format(json.dumps(_row)))
    # generate sample.
    from cllm.dataset.tde.s8_rag_gen_prompt import build_question
    content = []
    for obj in prompt_samples:
        content.append("Q: {}\nA:{}\n".format(build_question(obj, True), obj['answers'][0]))
    with open(prompt_path, 'w') as wf:
        wf.write(''.join(content))
        # add template:
        wf.write("Q: {Question}\nA:")
    

if __name__ == '__main__':
    # prepare_pdf('testrun_1', 'testrun_1_sgts')
    # build_text_multichoice_question('testrun_1_sgts', './datasets/STD/dataset/run1_test.jsonl')
    # build_code_multichoice_question('testrun_1_sgts', './datasets/STD/dataset/run1_test.jsonl')
    
    # extract_func('nn_fibonacii')
    # build_code_multichoice_question('testrun_1_sgts', './datasets/STD/dataset/run1_test_code.jsonl')
    generate_prompt(
        './datasets/STD/dataset/run1_test_code.jsonl', 
        './datasets/STD/dataset/run1_test_code_test.jsonl',
        './prompts/std_code_2_desc.txt'
        )