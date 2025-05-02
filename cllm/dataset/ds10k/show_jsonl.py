import argparse
from pathlib import Path
import json
def show_jsonl(file, n):
    # count lines.
    with open(file, 'r') as f:
        n_lines = sum(1 for _ in f)
    print(f'Total lines: {n_lines}')
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            json_obj = json.loads(line)
            print(json_obj['input_desc'])
            print('---'*10)

if __name__ == '__main__':
    base_folder = Path('./datasets/DS-10k_deepseek/test_gt_function_more_inputs_v2_outputs_refined_filled_io_desc_rewritten_combined.jsonl')

    show_jsonl(base_folder, 10)