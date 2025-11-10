from utils.rag_utils import query_faiss

if __name__ == "__main__":
    query = "What is Artificial Intelligence?"
    results = query_faiss(query)
    for r in results:
        print(f"\n📄 {r['document']} (score={r['score']:.4f})")
        print(f"Snippet: {r['content'][:150]}...")
