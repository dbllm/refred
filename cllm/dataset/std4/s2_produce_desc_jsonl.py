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
    remaining_seqs = all_transform_seqs

    with open(output_path, 'w') as wf:
        for trans_obj in remaining_seqs:
            name = trans_obj['id']
            # if not name.endswith('.txt'): continue
            with open(input_folder / f'{name}.txt', 'r') as rf:
                content = rf.read()
                _id = trans_obj['id']
                oq_ids = trans_obj['ans']
                
                expect_oq_iq = trans_obj['oq_id']
                good_ids = []
                if len(oq_ids) > 1:
                    # only keep the one with the same data types.
                    for _i_id in oq_ids:
                        if func_dict[_i_id]['type'] == func_dict[expect_oq_iq]['type']:
                            good_ids.append(_i_id)
                else:
                    good_ids = oq_ids

                # if len(oq_ids) > 1:
                #     print(trans_obj)
                #     print(trans_obj['id'])
                #     print(content)
                #     print(oq_ids)
                #     for _id in oq_ids[1:]:
                #         assert func_dict[_id]['type'] == func_dict[oq_ids[0]]['type'], \
                #             f"type {func_dict[_id]['type']} and {func_dict[oq_ids[0]]['type']} mismatch!"

                wf.write('{}\n'.format(json.dumps({
                    'id': _id,
                    'oq_ids': good_ids,
                    'desc': content,
                    # oq id.
                    'type': func_dict[trans_obj['oq_id']]['type']
                })))
            


if __name__ == '__main__':
    base_path = Path('./datasets/STD')
    combine_from_folder(
        base_path / 'all_funcs.jsonl',
        base_path / 'all_funcs_match' / 'all.jsonl',
        base_path / 'struct_more_outputs_gpt_desc_3.5',
        base_path / 'struct_more_outputs_gpt_desc_3.5_mgt.jsonl'
    )

    # combine_from_folder_funcs(
    #     base_path / 'all_funcs.jsonl',
    #     base_path / 'function_answers_desc_3.5',
    #     base_path / 'function_answers_desc_3.5.jsonl'
    # )
