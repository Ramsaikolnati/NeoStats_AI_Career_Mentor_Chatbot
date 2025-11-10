"""
utils/rag_utils.py — Handles document retrieval from FAISS index.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config.config import FAISS_INDEX_PATH, KB_DIR

# Load same model used for embedding
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_index(index_path=FAISS_INDEX_PATH):
    """Load FAISS index and metadata (document names)."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    index = faiss.read_index(index_path)
    with open(index_path + ".meta", "rb") as f:
        doc_names = pickle.load(f)
    return index, doc_names

def query_faiss(query: str, top_k: int = 3):
    """Retrieve top_k most similar documents to the query."""
    index, doc_names = load_faiss_index()
    query_vec = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(doc_names):
            doc_path = os.path.join(KB_DIR, doc_names[idx])
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
            results.append({
                "document": doc_names[idx],
                "content": text,
                "score": float(distances[0][i])
            })
    return results
