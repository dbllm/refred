from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence Transformer model
# model_name = 'all-mpnet-base-v2'  # Replace with the model of your choice
# model = SentenceTransformer(model_name)

# Example: A long description
long_description = """
Sentence Transformers allow you to compute dense vector representations (embeddings) for sentences, paragraphs, or documents.
These embeddings can be used for clustering, semantic search, or other NLP tasks.
It supports several pre-trained models optimized for various tasks and domains.
"""

# # Generate the embedding
# embedding = model.encode(long_description)

# # Print the embedding (a dense vector)
# print("Embedding:", embedding)
# print("Embedding shape:", embedding.shape)

class SentenceTransformerEmbeddingClient:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.model.encode(text)

if __name__ == '__main__':
    client = SentenceTransformerEmbeddingClient()
    embedding = client.get_embedding('hello world')
    print(type(embedding))
    print(embedding.shape)