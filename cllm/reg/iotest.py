from cllm.reg.io import load_std_mgt_as_one_data


if __name__ == '__main__':
    data = load_std_mgt_as_one_data('aug_mgtas1')
    print(data.columns)