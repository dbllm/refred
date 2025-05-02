import pandas as pd
from cllm.aux.open_ai import OpenAIClient
from tqdm import tqdm
import json
def generate_new_description(prompt_template, input_code, output_str, model="gpt-3.5-turbo-0125", max_tokens=3000):
    prompt = f"""
You are a helpful assistant. You are given a prompt sample, input code, and input/output string.
You need to generate a new description for the operation with the input code and input/output string folowing the same format as the prompt sample. Please also include the exact value of the input and output in your description.
A prompt sample is as follows:
{prompt_template}
Input code:
{input_code}
Input/output string:
{output_str}
Your description:
"""
    client = OpenAIClient(model_name=model)
    result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=max_tokens, batch_size=1, stop=[])
    client.close()
    return result[0]['response']

def gen_io_desc():

    # df = pd.read_json('datasets/DS-1000/test_input_code.jsonl', lines=True, orient='records')
    # print(df.head())
    # pass

    # df = pd.read_json('datasets/DS-1000/test_gt_function.jsonl', lines = True, orient='records')
    # df['problem_id'] = df['metadata'].apply(lambda x: x['problem_id'])
    # print(df.head())

    df = pd.read_json('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled.jsonl', lines=True, orient='records')
    # find those without error
    df['has_error'] = df['output'].apply(lambda x: 'Error' in x)
    df_no_error = df[df['has_error'] == False]
    print(df_no_error.head())
    print(df_no_error.shape)

    print(df_no_error.columns)

    result_path = 'datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_part1.jsonl'
    with open(result_path, 'w') as f:
        for idx, row in tqdm(df_no_error.iterrows(), total=df_no_error.shape[0]):
            # if row['uid'] <= 3443: continue
            # if row['uid'] <= 14030: continue
            input_code = row['input']
            output_str = row['output']
            prompt_template = row['prompt']
            
            try:    
                if 'Exit code' not in output_str:
                    new_desc = generate_new_description(prompt_template, input_code, output_str)
                    # print(new_desc)
                    # break
                    f.write(json.dumps({**row, 'io_desc': new_desc}) + '\n')
                else:
                    f.write(json.dumps({**row, 'io_desc': 'error'}) + '\n')
            except Exception as e:
                print(e)
                f.write(json.dumps({**row, 'io_desc': 'error'}) + '\n')
                continue

def count_io_desc():
    df = pd.read_json('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_part1.jsonl', lines=True, orient='records')
    df['has_error'] = df['io_desc'].apply(lambda x: 'error' in x)
    df_no_error = df[df['has_error'] == False]
    print(df_no_error.shape)

    # test metadata 
    print(df_no_error['metadata'][5100])
    
    df_no_error['origin_pid'] = df_no_error['metadata'].apply(lambda x: x['library_problem_id'])
    df_no_error['perturbation_pid'] = df_no_error['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df_no_error['extended'] = df_no_error['origin_pid'] != df_no_error['perturbation_pid']
    print(df_no_error['extended'].value_counts())


if __name__ == '__main__':
    # gen_io_desc() 
    count_io_desc()