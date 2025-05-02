import os
import json
from pathlib import Path
from cllm.io import load_data

def combine_from_folder(all_func_jsonl, transform_pair_jsonl, input_folder, output_path):
    
    all_funcs = load_data(all_func_jsonl)
    func_dict = {}
    for func_obj in all_funcs:
        func_dict[func_obj['id']] = func_obj

    all_transform_seqs = load_data(transform_pair_jsonl)
    remaining_seqs = []
    for obj in all_transform_seqs:
        if len(obj['ans']) == 1:
            remaining_seqs.append(obj)
    print('remaining:', len(remaining_seqs))

    with open(output_path, 'w') as wf:
        for trans_obj in remaining_seqs:
            name = trans_obj['id']
            # if not name.endswith('.txt'): continue
            with open(input_folder / f'{name}.txt', 'r') as rf:
                content = rf.read()
                _id = trans_obj['id']
                oq_id = trans_obj['ans'][0]

                wf.write('{}\n'.format(json.dumps({
                    'id': _id,
                    'oq_id': oq_id,
                    'desc': content,
                    'type': func_dict[oq_id]['type']
                })))
            


if __name__ == '__main__':
    base_path = Path('./datasets/STD')
    combine_from_folder(
        base_path / 'all_funcs.jsonl',
        base_path / 'all_funcs_match' / 'all.jsonl',
        base_path / 'struct_more_outputs_gpt_desc_3.5',
        base_path / 'struct_more_outputs_gpt_desc_3.5_onegt.jsonl'
    )

    # combine_from_folder_funcs(
    #     base_path / 'all_funcs.jsonl',
    #     base_path / 'function_answers_desc_3.5',
    #     base_path / 'function_answers_desc_3.5.jsonl'
    # )
