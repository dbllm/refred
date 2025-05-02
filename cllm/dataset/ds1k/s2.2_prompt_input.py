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
    origin_json_path = './datasets/DS-1000/test_gt_function.jsonl'
    output_json_path = './datasets/DS-1000/test_rewritten_desc.jsonl'
    client = OpenAIClient(model_name="gpt-3.5-turbo-0125")
    # load as pandas
    df = pd.read_json(origin_json_path, lines=True)
    with open(output_json_path, 'w') as wf:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            origin_desc = row['prompt']
            prompt = prompt_input_template.format(description=origin_desc)
            result = client.complete(messages=[{"role": "user", "content": prompt}], seed=42, max_tokens=1000, batch_size=1, stop=[])
            wf.write(json.dumps({
                'problem_id': row['metadata']['problem_id'],
                'input_desc': result[0]['response']}))
            wf.write('\n')
        sleep(.1)    

if __name__ == "__main__":
    prompt_input()