"""
models/embeddings.py
Embeddings generator for RAG (Retrieval-Augmented Generation).
Uses SentenceTransformer (MiniLM) for lightweight, local vector embeddings.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize the SentenceTransformer model once (cached globally)
_sentence_model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(texts):
    """
    Generate vector embeddings for a list of text chunks.
    Uses SentenceTransformer by default.
    """
    if not isinstance(texts, list):
        texts = [texts]

    try:
        vectors = _sentence_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(vectors)
    except Exception as e:
        raise RuntimeError(f"Embedding generation failed: {str(e)}")
