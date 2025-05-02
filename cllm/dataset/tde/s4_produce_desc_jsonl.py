import os
import json
from pathlib import Path

def combine_from_folder(input_folder, output_path, mode='origin'):
    assert mode in ['origin', 'more']
    with open(output_path, 'w') as wf:
        all_files = list(os.listdir(input_folder))
        # sort them.
        if mode == 'origin':
            all_files.sort(key=lambda x: int(x.split('.')[0]))
        elif mode == 'more':
            all_files.sort(key=lambda x: int(x.split('-')[0]))
        else:
            raise NotImplementedError()
        for name in all_files:
            if not name.endswith('.txt'): continue
            with open(input_folder / name, 'r') as rf:
                content = rf.read()

            if mode == 'origin':
                _id = int(name.split('.')[0])
                wf.write('{}\n'.format(json.dumps({
                    'id': _id,
                    'desc': content
                })))
            elif mode == 'more':
                _id = name.split('.')[0]
                oq_id = int(_id.split('-')[0])

                wf.write('{}\n'.format(json.dumps({
                    'id': _id,
                    'oq_id': oq_id,
                    'desc': content
                })))
            else:
                raise NotImplementedError()

if __name__ == '__main__':
    # base_path = Path('./datasets/TDE')
    # combine_from_folder(base_path / 'gpt_seqdesc_desc_3.5', base_path / 'gpt_seqdesc_desc_3.5.jsonl', mode='origin')
    # combine_from_folder(base_path / 'gpt_more_outputs_desc_3.5', base_path / 'gpt_more_outputs_desc_3.5.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_answers_desc_3.5', base_path / 'gpt_gt_func_desc.3.5.jsonl', mode='origin')

    # combine_from_folder(base_path / 'gpt_more_outputs_desc_llama2-7b', base_path / 'gpt_more_outputs_desc_llama2-7b.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_seqdesc_desc_llama2-7b', base_path / 'gpt_seqdesc_desc_llama2-7b.jsonl', mode='origin')

    # combine_from_folder(base_path / 'gpt_seqdesc_desc_llama2-13b', base_path / 'gpt_seqdesc_desc_llama2-13b.jsonl', mode='origin')
    # combine_from_folder(base_path / 'gpt_more_outputs_20_num3_desc_3.5', base_path / 'gpt_more_outputs_20_num3_desc_3.5.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_more_outputs_20_num2_desc_3.5', base_path / 'gpt_more_outputs_20_num2_desc_3.5.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_more_outputs_20_num1_desc_3.5', base_path / 'gpt_more_outputs_20_num1_desc_3.5.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_more_outputs_20_num1_desc_3.5_fix', base_path / 'gpt_more_outputs_20_num1_desc_3.5_fix.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_more_outputs_20_num2_desc_3.5_fix', base_path / 'gpt_more_outputs_20_num2_desc_3.5_fix.jsonl', mode='more')
    # combine_from_folder(base_path / 'gpt_more_outputs_20_num3_desc_3.5_fix', base_path / 'gpt_more_outputs_20_num3_desc_3.5_fix.jsonl', mode='more')

    # combine_from_folder(base_path / 'struct_gpt_more_outputs_desc_20_3.5', base_path / 'struct_gpt_more_outputs_desc_20_3.5.jsonl', mode='more')
    # combine_from_folder(base_path / 'struct_gpt_more_outputs_desc_20_3.5_fix', base_path / 'struct_gpt_more_outputs_desc_20_3.5_fix.jsonl', mode='more')
    # combine_from_folder(base_path / 'struct_gpt_origin_desc_3.5', base_path / 'struct_gpt_origin_desc_3.5.jsonl', mode='origin')

    base_path = Path('./datasets/TDE2')
    combine_from_folder(base_path / 'function_answers_desc_3.5', base_path / 'function_answers_desc_3.5.jsonl', mode='origin')
    combine_from_folder(base_path / 'struct_more_outputs_gpt_desc_3.5', base_path / 'struct_more_outputs_gpt_desc_3.5.jsonl', mode='more')
