import json
from cllm.aux.open_ai import OpenAIClient
import pandas as pd
from tqdm import tqdm
from time import sleep
prompt_input_template = """
Below is the description of a function. You are expected to understand the given description, and rewrite it in the following format.
Input: <describe the possible input values of the `test_input`, but do not mention the word `test_input`>
Output: <describe the possible output values of the `result`, but do not mention the word `result`>
Functionality: <describe the functionality of the function>
Below is the original description:
{description}
Your rewritten description starts here. 
"""

def prompt_input():
    # origin_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc.jsonl'
    origin_json_path = './datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_part1.jsonl'
    # output_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_2.jsonl'
    output_json_path = './datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_part2.jsonl'

    df = pd.read_json(origin_json_path, lines=True, orient='records')
    df['has_error'] = df['io_desc'].apply(lambda x: 'error' in x)
    df_no_error = df[df['has_error'] == False]


    df_no_error['origin_pid'] = df_no_error['metadata'].apply(lambda x: x['library_problem_id'])
    df_no_error['perturbation_pid'] = df_no_error['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df_no_error['is_extended'] = df_no_error['origin_pid'] != df_no_error['perturbation_pid']

    client = OpenAIClient(model_name="gpt-3.5-turbo-0125")
    # load as pandas
    df = pd.read_json(origin_json_path, lines=True)
    with open(output_json_path, 'w') as wf:
        for _idx, row in tqdm(df_no_error.iterrows(), total=len(df_no_error)):
            # if _idx < 6648: continue
            if row['uid'] < 14990: continue
            origin_desc = row['io_desc']
            prompt = prompt_input_template.format(description=origin_desc)
            result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=1000, batch_size=1, stop=[])
            wf.write(json.dumps({
                'uid': row['uid'],
                'problem_id': row['metadata']['problem_id'],
                'is_extended': row['is_extended'],
                'input_desc': result[0]['response']
                }))
            wf.write('\n')
        sleep(.1)    

def combine_jsonl():
    origin_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc.jsonl'
    rewritten_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten.jsonl'
    rewritten2_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_2.jsonl'
    output_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_combined.jsonl'
    df1 = pd.read_json(origin_json_path, lines=True)
    df1['has_error'] = df1['io_desc'].apply(lambda x: 'error' in x)
    df1_no_error = df1[df1['has_error'] == False]


    df1_no_error['origin_pid'] = df1_no_error['metadata'].apply(lambda x: x['library_problem_id'])
    df1_no_error['perturbation_pid'] = df1_no_error['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df1_no_error['is_extended'] = df1_no_error['origin_pid'] != df1_no_error['perturbation_pid']
    

    df2 = pd.read_json(rewritten_json_path, lines=True)
    assert len(df1_no_error) == len(df2)
    print(len(df1_no_error), len(df2))
    df1_no_error['input_desc'] = df2['input_desc']

    # use rewritten 2 if the uid is in the rewritten 2
    rewritten2_uid_mapping = {}
    for _idx, _rw in pd.read_json(rewritten2_json_path, lines=True).iterrows():
        rewritten2_uid_mapping[_rw['uid']] = _rw['input_desc']
    rewritten_values = []
    for _idx, _row in df1_no_error.iterrows():
        if _row['uid'] in rewritten2_uid_mapping:
            rewritten_values.append(rewritten2_uid_mapping[_row['uid']])
        else:
            rewritten_values.append(_row['input_desc'])
    df1_no_error['input_desc'] = rewritten_values

    df1_no_error.to_json(output_json_path, lines=True, orient='records')

def combine_jsonl_v2():
    origin_json_path = './datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_part1.jsonl'
    # rewritten_json_path = './datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten.jsonl'
    rewritten_json_path = './datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_part1.jsonl'
    rewritten2_json_path = './datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_part2.jsonl'
    output_json_path = './datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_combined.jsonl'
    df1 = pd.read_json(origin_json_path, lines=True)
    df1['has_error'] = df1['io_desc'].apply(lambda x: 'error' in x)
    df1_no_error = df1[df1['has_error'] == False]


    df1_no_error['origin_pid'] = df1_no_error['metadata'].apply(lambda x: x['library_problem_id'])
    df1_no_error['perturbation_pid'] = df1_no_error['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df1_no_error['is_extended'] = df1_no_error['origin_pid'] != df1_no_error['perturbation_pid']
    

    df2 = pd.read_json(rewritten_json_path, lines=True)
    assert len(df1_no_error) == len(df2)
    print(len(df1_no_error), len(df2))
    df1_no_error['input_desc'] = df2['input_desc']

    
    # use rewritten 2 if the uid is in the rewritten 2
    rewritten2_uid_mapping = {}
    for _idx, _rw in pd.read_json(rewritten2_json_path, lines=True).iterrows():
        rewritten2_uid_mapping[_rw['uid']] = _rw['input_desc']
    rewritten_values = []
    for _idx, _row in df1_no_error.iterrows():
        if _row['uid'] in rewritten2_uid_mapping:
            rewritten_values.append(rewritten2_uid_mapping[_row['uid']])
        else:
            rewritten_values.append(_row['input_desc'])
    df1_no_error['input_desc'] = rewritten_values

    df1_no_error.to_json(output_json_path, lines=True, orient='records')

if __name__ == "__main__":
    # prompt_input()
    combine_jsonl_v2()