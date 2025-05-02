import os
import json
from pathlib import Path
from cllm.io import load_data

def combine_from_folder(all_func_jsonl, input_folder, output_path):
    all_funcs = load_data(all_func_jsonl)
    func_dict = {}
    for func_obj in all_funcs:
        func_dict[func_obj['id']] = func_obj

    with open(output_path, 'w') as wf:
        all_files = list(os.listdir(input_folder))
        # sort them.
        all_files.sort(key=lambda x: int(x.split('-')[0]))
        for name in all_files:
            if not name.endswith('.txt'): continue
            with open(input_folder / name, 'r') as rf:
                content = rf.read()
                _id = name.split('.')[0]
                oq_id = int(_id.split('-')[0])

                wf.write('{}\n'.format(json.dumps({
                    'id': _id,
                    'oq_id': oq_id,
                    'desc': content,
                    'type': func_dict[oq_id]['type']
                })))
            

def combine_from_folder_funcs(all_func_jsonl, input_folder, output_path):
    all_funcs = load_data(all_func_jsonl)
    with open(output_path, 'w') as wf:
        for obj in all_funcs:
            with open(input_folder / '{}.txt'.format(obj['id']), 'r') as rf:
                content = rf.read()
                _id = obj['id']
                wf.write('{}\n'.format(json.dumps({
                    'id': _id,
                    'desc': content
                })))


if __name__ == '__main__':
    base_path = Path('./datasets/STD')
    combine_from_folder(
        base_path / 'all_funcs.jsonl',
        base_path / 'struct_more_outputs_gpt_desc_3.5', 
        base_path / 'struct_more_outputs_gpt_desc_3.5.jsonl'
    )

    # combine_from_folder_funcs(
    #     base_path / 'all_funcs.jsonl',
    #     base_path / 'function_answers_desc_3.5',
    #     base_path / 'function_answers_desc_3.5.jsonl'
    # )
