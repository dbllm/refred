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
    
    input_folder = Path('./datasets/STD/')
    code_path = input_folder / 'functions'
    transform_data_path = input_folder / 'outputs' / 'all.jsonl'
    transform_data = load_data(transform_data_path)
    func_data = load_data(input_folder / 'all_funcs.jsonl')

    output_path = input_folder / 'all_funcs_match'
    output_path.mkdir(parents=True, exist_ok=True)

    # load code
    code_dict = {}
    for func_obj in func_data:
        code = read_code(code_path / '{}.py'.format(func_obj['name']))
        code_dict[func_obj['id']] = code

    # get all results.

    with open(output_path / 'all.jsonl', 'w') as wf:
        for trans_obj in transform_data:
            match_ids = []
            for func_obj in func_data:
                complete_code = replace_input(code_dict[func_obj['id']], trans_obj['seq_a'])
                seq_b = execute_code_w_error(complete_code, func_obj['id'])
                print(seq_b)
                if seq_b == trans_obj['seq_b']:
                    match_ids.append(func_obj['id'])
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