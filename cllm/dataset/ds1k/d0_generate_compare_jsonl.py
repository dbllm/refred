import pandas as pd
from pathlib import Path
def generate_compare_jsonl():
    origin_file = Path("datasets/DS-1000/test_gt_function.jsonl")
    desc_file = Path("datasets/DS-1000/test_rewritten_desc.jsonl")
    input_desc = Path("datasets/DS-1000/test_gt_function_description_gpt_v1.jsonl")

    # read them all.
    with open(origin_file, "r") as f:
        origin_pd = pd.read_json(f, lines=True)
    origin_pd['problem_id'] = origin_pd["metadata"].apply(lambda x: x["problem_id"])
    origin_pd['gt_id'] = origin_pd['metadata'].apply(lambda x: x['problem_id'])
    with open(desc_file, "r") as f:
        desc_pd = pd.read_json(f, lines=True)
    with open(input_desc, "r") as f:
        input_desc_pd = pd.read_json(f, lines=True)

    # merge them.
    merged_pd = pd.merge(origin_pd, desc_pd, on="problem_id")
    merged_pd = pd.merge(merged_pd, input_desc_pd, on="problem_id")

    # save them.
    output_file = Path("datasets/DS-1000/test_gt_function_combined.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_pd.to_json(output_file, lines=True, orient="records")
    
if __name__ == "__main__":
    generate_compare_jsonl()
