from cllm.dataset.std.s1_generate_outputs import *
from cllm.io import load_data
from pathlib import Path

def execute_code_w_error(code_block, i):
    # create a separate file.
    with open('validate_tmp.py', 'w') as wf:
        wf.write(code_block)
    try:
        proc = subprocess.check_output(['python', 'validate_tmp.py'])
        output = proc.decode()
        logging.info('line: [%s], output: %s', i, output)
    except subprocess.CalledProcessError as e:
        logging.error('error: %s', str(e))
        return None
    # f = StringIO()
    # with redirect_stdout(f):
    #     try:
    #         exec(code_block)
    #     except Exception as e: 
    #         logging.warning('exception %s', e)
    #         pass
    # output = f.getvalue()
    return output.strip()

def match_all(offset=0):
    
    input_folder = Path('./datasets/TDE2/')
    code_path = input_folder / 'functions'
    transform_data_path = input_folder / 'outputs' / 'all.jsonl'
    transform_data = load_data(transform_data_path)

    output_path = input_folder / 'all_funcs_match'
    output_path.mkdir(parents=True, exist_ok=True)

    # load all funcs.
    all_func_ids = sorted(map(lambda x: int(x.split('.')[0]), os.listdir(code_path)))

    # load code
    code_dict = {}
    for func_id in all_func_ids:
        code = read_code(code_path / '{}.txt'.format(func_id))
        code_dict[func_id] = code

    # get all results.

    with open(output_path / 'all.jsonl', 'w') as wf:
        for trans_obj in transform_data:
            match_ids = []
            for func_id in all_func_ids:
                complete_code = replace_input(code_dict[func_id], trans_obj['seq_a'])
                seq_b = execute_code_w_error(complete_code, func_id)
                if seq_b == trans_obj['seq_b'].strip():
                    match_ids.append(func_id)
            # write this one.
            trans_obj['ans'] = match_ids
            wf.write('{}\n'.format(
                json.dumps(trans_obj)
            ))
                #     # only write when match.
                #     wf.write('{}\n'.format(json.dumps({
                #         'id': func_obj['id'],
                #         'match': True,
                #     })))
            

if __name__ == '__main__':
    match_all()