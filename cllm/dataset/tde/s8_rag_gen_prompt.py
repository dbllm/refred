from pathlib import Path
import json
import random

QUESTION_TEMPLATE="""Which of the following function can be used to convert the following seq_a to seq_b?
seq_a: {}
seq_b: {}
"""

QUESTION_DESC_TEMPLATE = """Which of the following function can be used to convert the following seq_a to seq_b?
seq_a: {}
seq_b: {}
Decscription: {}
"""

def build_question(obj, use_desc=False):
    if use_desc:
        base_question = QUESTION_DESC_TEMPLATE.format(str(obj['seq_a']).strip(), str(obj['seq_b']).strip(), obj['desc'])
    else: 
        base_question = QUESTION_TEMPLATE.format(str(obj['seq_a']).strip(), str(obj['seq_b']).strip())
    if obj['tip']:
        base_question += '{}\n'.format(obj['tip'])
    return base_question + obj['choices']

def load_jsonl(input_path):
    jsons = []
    with open(input_path, 'r') as rf:
        for line in rf:
            jsons.append(json.loads(line))
    return jsons

def build_prompt(input_path, output_path, num_samples, use_desc=False):
    all_json = load_jsonl(input_path)
    content = []
    for obj in all_json[:num_samples]:
        content.append("Q: {}\nA:{}\n".format(build_question(obj, use_desc), obj['answers'][0]))
    # get first two
    with open(output_path, 'w') as wf:
        wf.write(''.join(content))
        # add template:
        wf.write("Q: {Question}\nA:")

def split_jsonl(input_path, output_path1, output_path2, num_left, seed=42):
    jsons = load_jsonl(input_path)
    # 
    random.seed(seed)
    random.shuffle(jsons)

    def dump_jsonl_sorted(output_path, obj_list):
        obj_list.sort(key=lambda x: tuple(map(int, x['question_id'].split('-'))))
        with open(output_path, 'w') as wf:
            for obj in obj_list:
                wf.write('{}\n'.format(json.dumps(obj)))
    
    dump_jsonl_sorted(output_path1, jsons[:len(jsons) - num_left])
    dump_jsonl_sorted(output_path2, jsons[len(jsons) - num_left: ])
            

if __name__ == '__main__':
    dataset_folder = Path('./datasets/TDE/')
    # input_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_3.5.jsonl'
    # part1_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_3.5_part1.jsonl'
    # part2_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_3.5_part2.jsonl'

    input_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_llama2-7b.jsonl'
    part1_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_llama2-7b_part1.jsonl'
    part2_path = dataset_folder / 'dataset' / 'all_w_embedding_k10_llama2-7b_part2.jsonl'

    num_samples = 2
    use_desc=False
    prompt_path = dataset_folder / 'dataset' / 'tde_7b_{}{}.txt'.format(num_samples, '_desc' if use_desc else '')

    split_jsonl(input_path, part1_path, part2_path, 10)
    build_prompt(part2_path, prompt_path, num_samples, use_desc=use_desc)