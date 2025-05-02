from pathlib import Path
import os
import json

def generate_all_funcs_json():
    input_folder = Path('./datasets/STD/')
    code_path = input_folder / 'functions'
    
    with open(input_folder / 'all_funcs.jsonl', 'w') as wf:
        for _id, name in enumerate(sorted(os.listdir(code_path))):
            _type = name.split('_')[0]
            _code = name.split('.')[0]
            wf.write('{}\n'.format(json.dumps({
                'id': _id,
                'type': _type,
                'name': _code,
            })))

if __name__ == '__main__':
    generate_all_funcs_json()