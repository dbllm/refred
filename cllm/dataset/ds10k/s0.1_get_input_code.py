import pandas as pd
from tqdm import tqdm
import re
from cllm.aux.open_ai import OpenAIClient
import json
import itertools
def parse_output(output):
    # finds '>>>' and '<<<' and return the text between them.   
    parts = output.split('<<<')
    test_case = []
    for part in parts:
        if part.strip() == '': continue
        # replace '>>>*\n' with '' using regex
        part = re.sub(r'>>>.*?\n', '', part)
        test_case.append(part)
    return test_case

def generate_input_code(code, test_case, model="gpt-3.5-turbo-0125", max_tokens=2000):
    prompt = f"""Please provide the code that assigns the `test_input` variable that could produce the following output:
Output:
{test_case}
For your information, the exact code to run the function is:
{code}
Please write the code that assigns the `test_input` variable that could produce the above output.
For example, if the input consists of a single value of 1, then the code should be like `test_input = 1`; 
if the input consits of two values, then the code should be like `value1 = ...\nvalue2= ...\ntest_input = (value1, value2)`.
Answer:
"""
    client = OpenAIClient(model_name=model)
    result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=max_tokens, batch_size=1, stop=[])
    client.close()
    return result[0]['response']

def get_input_code():
    with open('datasets/DS-1000/test_exec_code_output.jsonl', 'r') as f:
        output_df = pd.read_json(f, lines=True)    

    with open('datasets/DS-1000/test_exec_code.jsonl', 'r') as f:
        gt_df = pd.read_json(f, lines=True)  
    # with open('datasets/DS-1000/test_gt_function.jsonl', 'r') as f:
    #     gt_df = pd.read_json(f, lines=True)  
    # gt_df['problem_id'] = gt_df['metadata'].apply(lambda x: x['problem_id'])

    # merge two df on problem_id.
    merged_df = pd.merge(output_df, gt_df, on='problem_id', how='inner')

    uid_generator = itertools.count(0)

    # with open('datasets/DS-1000/test_input_code.jsonl', 'w') as f:
    with open('datasets/DS-1000/test_input_code_v2.jsonl', 'w') as f:
        for _id, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
            # if _id < 317: continue
            problem_id = row['problem_id']
            # code = row['gt_function']
            code = row['code_context']
            output = row['output']
            # print(f"Problem ID: {problem_id}")
            # print(f"Code: {code}")
            # print(f"Output: {output}")
            # print("-" * 100)
            test_cases = parse_output(output)
            for tc_id, test_case in enumerate(test_cases):
                # print(test_case)
                try:
                    input_code = generate_input_code(code, test_case)
                    f.write(json.dumps({'problem_id': problem_id, 'test_case_id': tc_id, 'uid': next(uid_generator), 'test_case_string': test_case, 'test_case_code': input_code}) + '\n')
                except Exception as e:
                    print(e)
                    f.write(json.dumps({'problem_id': problem_id, 'test_case_id': tc_id, 'uid': next(uid_generator), 'test_case_string': test_case, 'test_case_code': 'error'}) + '\n')

            # break
            
    # print(merged_df.head())


if __name__ == "__main__":
    get_input_code()