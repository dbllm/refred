import dotenv
dotenv.load_dotenv()

import argparse
from pathlib import Path
from tqdm import tqdm
import json
import torch
import os
import numpy as np
import pickle
from openai import OpenAI, RateLimitError
import backoff
from cllm.aux.open_ai import convert_qa_to_openai_template, append_message

class OpenAIEmbeddingClient:

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.client = OpenAI(api_key = os.environ.get('OPENAI_API_KEY'))

    # see https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    @backoff.on_exception(backoff.expo, RateLimitError)
    def get_embedding(self, text):
        response = self.client.embeddings.create(
            input=[text], model=self.model_name
        )
        embedding = response.data[0].embedding
        return np.array(embedding)

def compute_openai_embedding(input_file, output_file, prompt_file, model_name):
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)

    client = OpenAIEmbeddingClient(model_name)

    if prompt_file is not None:
        with open('prompts/{}'.format(prompt_file), 'r') as rf:
            prompt_template = '\n'.join(rf.readlines())
            # convert to messages
            prompt_messages = convert_qa_to_openai_template(prompt_template)
        

    with open(input_file, 'r') as rf:
        arr = []
        for i, line in enumerate(tqdm(rf)):
            item = json.loads(line)
            q = item['question']
            qm = append_message(prompt_messages, q)
            e1 = client.get_embedding('\n'.join([x['content'] for x in qm]))
            arr.append(e1)
        np.save(output_file, np.array(arr))

# get all answers.

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--model_name', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parser_args()
    compute_openai_embedding(args.input, args.output, args.prompt, args.model_name)