from cllm.dataset.tde.common import *
from pathlib import Path
from cllm.utils import *
import logging
import subprocess
import pandas as pd

def validate_answer(code_block, expected_output):
    # create a separate file.
    with open('validate_tmp.py', 'w') as wf:
        wf.write(code_block)
    proc = subprocess.Popen(['python', 'validate_tmp.py'], stdout=subprocess.PIPE)
    output = proc.communicate()[0].decode()
    # f = StringIO()
    # with redirect_stdout(f):
    #     try:
    #         exec(code_block)
    #     except Exception as e: 
    #         logging.warning('exception %s', e)
    #         pass
    # output = f.getvalue()
    logging.info('output: %s', output)
    logging.info('expect: %s', expected_output)
    ans = output.strip() == str(expected_output).strip()
    logging.info('check result: %s', ans)
    return ans

def validate_correct_func(stop_when_error=True, offset=0):
    setup_logging(level=logging.INFO, to_file=False)
    input_path = Path('./datasets/TDE')
    dataset_path = input_path / 'raw_datasets.jsonl'
    correct_output = input_path / 'gpt_answers_func_manual'
    correct_output.mkdir(parents=True, exist_ok=True)
    manual_source_input = input_path / 'gpt_answers_func_3.5'

    correct_num = 0
    total_num = 0
    id_questions = read_dataset(dataset_path)
    for _id, _q_obj in tqdm(id_questions.items()):
        if _id in SKIP_IDS: continue
        if _id < offset: continue
        total_num += 1
        # read it.
        file_path = manual_source_input / '{}.txt'.format(_id)
        logging.info('checking file: %s', file_path)
        is_correct = False
        # check.
        pdf = pd.DataFrame(_q_obj['table'], columns=('seq_a', 'seq_b'))
        target_file = correct_output / '{}.txt'.format(_id)
        if not target_file.exists():
            # copy file.
            with open(target_file, 'w') as wf:
                with open(file_path, 'r') as rf:
                    wf.writelines(rf.readlines())

        # load python code.
        code_block = read_code(target_file)
        is_correct = validate_answer(code_block, pdf['seq_b'].to_list())

        if is_correct:
            correct_num += 1
        else:
            logging.info('error, will create a manual template')
            logging.info('output path: %s', target_file)
            with open(target_file, 'w') as wf:
                wf.write('// modified manually\n')
                wf.write('// seq_a = {}\n'.format(pdf['seq_a'].to_list()))
                wf.write('// seq_b = {}\n'.format(pdf['seq_b'].to_list()))
                with open(file_path, 'r') as rf:
                    wf.writelines(rf.readlines())
            if stop_when_error:
                exit()
    logging.info('done, total num: %s, correct num: %s', total_num, correct_num)

def check_seq_a(stop_when_error=True, offset=0):
    
    setup_logging(level=logging.INFO, to_file=False)
    input_path = Path('./datasets/TDE')
    dataset_path = input_path / 'raw_datasets.jsonl'
    correct_output = input_path / 'gpt_answers_func_manual'

    correct_num = 0
    total_num = 0
    id_questions = read_dataset(dataset_path)
    for _id, _q_obj in tqdm(id_questions.items()):
        if _id in SKIP_IDS: continue
        if _id < offset: continue
        total_num += 1
        is_correct = False
        # check.
        target_file = correct_output / '{}.txt'.format(_id)
        logging.info('checking file: %s', target_file)

        # load python code.
        code_block = read_code(target_file)
        # print(code_block)
        is_correct = re.findall(r'seq_a\s*=', code_block)

        logging.info('contains %s', is_correct)
        if is_correct:
            correct_num += 1
        else:
            if stop_when_error:
                exit()
    logging.info('done, total num: %s, correct num: %s', total_num, correct_num)


if __name__ == '__main__':
    # validate_correct_func(True, offset=223)
    # validate_correct_func()
    check_seq_a(offset=62)