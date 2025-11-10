"""
build_index.py — creates FAISS vector index from local KB docs.

Usage:
    python build_index.py
"""

import os
import faiss
import numpy as np
import pickle
from tqdm import tqdm
from config.config import OPENAI_API_KEY, OPENAI_EMBED_MODEL, KB_DIR, FAISS_INDEX_PATH
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> list:
    """Create embeddings locally using sentence-transformers."""
    return model.encode(text).tolist()


def load_kb_documents(kb_dir: str) -> list:
    """Load all .txt and .md documents from KB directory."""
    docs = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt") or filename.endswith(".md"):
            path = os.path.join(kb_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append((filename, content))
    return docs

def build_faiss_index(docs: list, index_path: str):
    """Embed documents and build FAISS index."""
    print(f"Building FAISS index for {len(docs)} documents...")
    embeddings = []
    doc_names = []

    for name, text in tqdm(docs, desc="Embedding documents"):
        try:
            emb = embed_text(text)
            embeddings.append(emb)
            doc_names.append(name)
        except Exception as e:
            print(f"[ERROR] Failed to embed {name}: {e}")

    # Convert to numpy float32 array
    vectors = np.array(embeddings).astype("float32")
    dimension = len(vectors[0])

    # Initialize FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    # Save FAISS index and metadata
    faiss.write_index(index, index_path)
    with open(index_path + ".meta", "wb") as f:
        pickle.dump(doc_names, f)

    print(f"\n✅ FAISS index built successfully: {index_path}")
    print(f"📁 Metadata saved: {index_path}.meta")

def main():
    if not os.path.exists(KB_DIR):
        print(f"[ERROR] KB directory '{KB_DIR}' not found.")
        return

    docs = load_kb_documents(KB_DIR)
    if not docs:
        print(f"[WARN] No text files found in '{KB_DIR}'.")
        return

    build_faiss_index(docs, FAISS_INDEX_PATH)

if __name__ == "__main__":
    main()
