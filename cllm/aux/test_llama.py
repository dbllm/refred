from cllm.aux.huggingface import LlamaHFClient

def test_llama():
    client = LlamaHFClient('meta-llama/Meta-Llama-3-8B')
    # client.set_stop_tokens(additional_stop_ids=[], stop=['.', '\n', '\n\n', ','])
    result = client.sample_answer(question="What is the capital of France?", max_tokens=1000, seed=42, batch_size=1)
    print(result)

if __name__ == "__main__":
    test_llama()
