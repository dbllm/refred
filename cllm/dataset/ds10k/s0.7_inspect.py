import pandas as pd

def read_ds2k():
    df = pd.read_json('datasets/DS-1000/test_gt_function_more_inputs_outputs_refined_filled_io_desc_rewritten_combined.jsonl', lines=True, orient='records')
    df['has_error'] = df['io_desc'].apply(lambda x: 'error' in x)
    df_no_error = df[df['has_error'] == False]
    df_no_error['origin_pid'] = df_no_error['metadata'].apply(lambda x: x['library_problem_id'])
    df_no_error['perturbation_pid'] = df_no_error['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df_no_error['extended'] = df_no_error['origin_pid'] != df_no_error['perturbation_pid']
    print(df_no_error.shape)
    print(df_no_error['extended'].value_counts())

    print(df_no_error.columns)
    # print(df_no_error['input_desc'])
    print(df_no_error.iloc[6648:]['input_desc'])

    # mapping = read_problem_id_gt_id_mapping()
    # df_no_error['gt_id'] = df_no_error['problem_id'].apply(lambda x: mapping[x])


def read_problem_id_gt_id_mapping():
    # read original ds1k
    df_1k = pd.read_json('datasets/DS-1000/test_gt_function_combined_edit.jsonl', lines=True, orient='records')
    print(df_1k.columns)
    mapping = {}
    # get problem_id, gt_id mapping.
    for _idx, _row in df_1k.iterrows():
        assert _row['problem_id'] not in mapping
        mapping[_row['problem_id']] = _row['gt_id']
    return mapping


def read_ds10k():
    df = pd.read_json('datasets/DS-1000/test_gt_function_more_inputs_outputs_refined_filled_io_desc_rewritten_combined.jsonl', lines=True, orient='records')
    print(df.columns)
    print(len(df))
    df['perturbation_pid'] = df['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df['extended'] = df['origin_pid'] != df['perturbation_pid']
    print(df['extended'].value_counts())

def read_ds10k_v2():
    df = pd.read_json('datasets/DS-10k/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_combined.jsonl', lines=True, orient='records')
    print(df.columns)
    print(len(df))
    df['perturbation_pid'] = df['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df['extended'] = df['origin_pid'] != df['perturbation_pid']
    print(df['extended'].value_counts())

    # get None in input_desc
    print(df[df['input_desc'].isna()]['uid'])

def read_ds10k_v2_deepseek():
    df = pd.read_json('datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_combined.jsonl', lines=True, orient='records')
    print(df.columns)
    print(len(df))
    df['perturbation_pid'] = df['metadata'].apply(lambda x: x['perturbation_origin_id'])
    df['extended'] = df['origin_pid'] != df['perturbation_pid']
    print(df['extended'].value_counts())

    # get None in input_desc
    print(df[df['input_desc'].isna()]['uid'])

    # get None in io_desc
    print(df[df['io_desc'].isna()]['uid'])

if __name__ == '__main__':
    # read_ds2k()
    # read_ds10k()
    # read_ds10k_v2()
    read_ds10k_v2_deepseek()