import pandas as pd
from tqdm import tqdm
import re
def synthesize_exec_code():
    with open('./datasets/DS-1000/test_gt_function_combined_edit.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    print(df.head())

    executable_code = []    
    problem_id = []
    for i in tqdm(range(len(df)), total=len(df)):
        # origin code context.
        code_context = df.iloc[i]['code_context']
        # find: exec(code, test_env)
        code_context = code_context.replace('exec(code, test_env)', 
                                            "print('>>>test case ', i, ':')\n        print('input:', test_input)")
        # remove assert 
        code_context = code_context.replace('assert exec_test(test_env["result"], expected_result)', 
                                            "print('output:',expected_result)\n        print('<<<')\n\ntest_execution('')\nexit()")
        # print(code_context)
        # get problem id. 
        executable_code.append(code_context)
        problem_id.append(df.iloc[i]['metadata']['problem_id'])

    # create a new pandas dataframe.
    df_exec = pd.DataFrame({'code_context': executable_code, 'problem_id': problem_id})
    df_exec.to_json('./datasets/DS-1000/test_exec_code.jsonl', lines=True, orient='records')

if __name__ == '__main__':
    synthesize_exec_code()