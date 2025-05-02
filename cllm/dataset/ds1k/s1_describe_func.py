# describe the function in the gt_function column
from cllm.aux.open_ai import OpenAIClient
import pandas as pd
from tqdm import tqdm

def describe_func_llm(gt_function, model="gpt-3.5-turbo-0125", max_tokens=1000):
    prompt = f"""
Please summarize the functionality of the following function.The input is given as `test_input` and the output will be stored in `result`.
Describe the function use the following format:
Input: <describe the possible input values of the `test_input`, but do not mention the word `test_input`>
Output: <describe the possible output values of the `result`, but do not mention the word `result`>
Functionality: <describe the functionality of the function>
Below is the given function: 
{gt_function}
Answer:
"""
    client = OpenAIClient(model_name=model)
    result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=max_tokens, batch_size=1, stop=[])
    return result[0]['response']

def describe_func():
    df = pd.read_json('./datasets/DS-1000/test_gt_function.jsonl', lines=True)
    data = []
    for i in tqdm(range(len(df)), total=len(df)):
        # if i > 3:
        #     break
        # get problem id
        problem_id = df.iloc[i]['metadata']['problem_id']
        gt_function = df.iloc[i]['gt_function']
        description = describe_func_llm(gt_function)
        data.append({'problem_id': problem_id, 'description': description})
    df = pd.DataFrame(data)
    df.to_json('./datasets/DS-1000/test_gt_function_description_gpt_v1.jsonl', orient='records', lines=True)

def read_gt_function_description():
    df = pd.read_json('./datasets/DS-1000/test_gt_function_description_gpt_v1.jsonl', lines=True)
    for i in tqdm(range(len(df)), total=len(df)):
        print(df.iloc[i]['problem_id'], df.iloc[i]['description'])
        print('-'*100)

if __name__ == '__main__':
    describe_func()
    # read_gt_function_description()