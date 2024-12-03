from typing import List

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import StaticEmbedding

# Initialize a StaticEmbedding module
static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
model = SentenceTransformer(modules=[static_embedding])


def get_embeddings(texts: List[str]) -> List[List[float]]:
    return [embedding.tolist() for embedding in model.encode(texts)]


def get_sentence_embedding_dimensions() -> int:
    return model.get_sentence_embedding_dimension()
