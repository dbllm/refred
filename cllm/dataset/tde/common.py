import re
from tqdm import tqdm
import json

SKIP_IDS = [
    61, 81, 94, 105, 124, 134, 143, 180, 211, 225, 238,
    73, # excluded since this require complex conversion.
]

# Transformation ids that we have modified for more inputs.
REVISED_MORE_INPUT_IDS = [0, 9, 20, 24, 60, 93, 156, 183, 184, 188, 189, 190, 192, 195, 196, 198, 201, 202, 206, 222, 223, 227, 228, 229, 234, 237]

def read_dataset(input_path):
    _dict = {}
    with open(input_path, 'r') as rf:
        for line in tqdm(rf.readlines()):
            obj = json.loads(line)
            _dict[obj['id']] = obj
    return _dict


def read_code(file_path):
    with open(file_path, 'r') as rf:
        file_content = rf.read()
    _match = re.search(r'```python\s+(.*?)```', file_content, re.DOTALL)
    if _match is not None:
        code_block = _match.group()[9:-3]
    else:
        # assume the whole file is the code.
        code_block = file_content
    return code_block