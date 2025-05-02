import pandas as pd
from tqdm import tqdm
from cllm.aux.open_ai import OpenAIClient
from pathlib import Path
import json

def more_input_func_llm(gt_function, input_example, model="gpt-3.5-turbo-0125", max_tokens=2000):
    prompt = f"""
Please generate more input values for the following function. The input is given as `test_input` and the output will be stored in `result`.
The function is given as follows: 
{gt_function}
Below is one example of input:
{input_example}
Please generate 10 more input values for the function.
Answer:
Input 1: 
"""
    client = OpenAIClient(model_name=model)
    result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=max_tokens, batch_size=1, stop=[])
    return result[0]['response']

def get_more_input():
    df = pd.read_json('datasets/DS-1000/test_gt_function.jsonl', lines=True)
    output_path = Path('datasets/DS-10k/test_gt_function_more_input.jsonl')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # print(df.head())
    # print(df.columns)
    with open(output_path, 'w') as f:
        for i in tqdm(range(len(df)), total=len(df)):
            if i > 3:
                break
            problem_id = df.iloc[i]['metadata']['problem_id']
            gt_function = df.iloc[i]['gt_function']
            input_example = df.iloc[i]['prompt']
            more_input = more_input_func_llm(gt_function, input_example)
            f.write(json.dumps({'problem_id': problem_id, 'more_input': more_input}) + '\n', flush=True)
    print(f"Saved to {output_path}")

def read_input_sample():
    df = pd.read_json('datasets/DS-10k/test_gt_function_more_input.jsonl', lines=True)
    print(df.head())

    # print the first one.
    print(df.iloc[0]['more_input'])


if __name__ == "__main__":
    # get_more_input()
    read_input_sample()