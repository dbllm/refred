from pathlib import Path
import json
import pandas as pd

def get_gt_count():
    # print all unique groud-truth function ids.
    file_path = Path("datasets/DS-1000/test_gt_function_combined_edit.jsonl")
    with open(file_path, "r") as f:
        df = pd.read_json(f, lines=True)
    print(df["gt_id"].nunique())

if __name__ == "__main__":
    get_gt_count()