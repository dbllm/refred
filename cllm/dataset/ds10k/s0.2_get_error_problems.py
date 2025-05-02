import pandas as pd

def get_error_problems():
    with open('datasets/DS-1000/test_input_code_v2.jsonl', 'r') as f:
        df = pd.read_json(f, lines=True)

    with open('datasets/DS-1000/test_gt_function.jsonl', 'r') as f:
        origin_df = pd.read_json(f, lines=True)
    origin_df['problem_id'] = origin_df['metadata'].apply(lambda x: x['problem_id'])

    with open('datasets/DS-1000/test_exec_code_output_null.jsonl', 'r') as f:
        exec_df = pd.read_json(f, lines=True)

    filtered_df = df[df['test_case_code'] == 'error']
    print(filtered_df['uid'].unique())

    # merge all df on problem_id.
    merged_df = pd.merge(df, origin_df, on='problem_id', how='inner')
    merged_df = pd.merge(merged_df, exec_df, on='problem_id', how='inner')

    merged_df.to_json('datasets/DS-1000/test_input_code_error_problems.jsonl', lines=True, orient='records')

if __name__ == '__main__':
    get_error_problems()