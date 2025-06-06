from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path

INDEX_PATH = Path("vector_store/faiss.index")
META_PATH = Path("vector_store/metadata.pkl")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

index = faiss.read_index(str(INDEX_PATH))

with META_PATH.open("rb") as f:
    metadata = pickle.load(f)

def search(query: str, top_k: int = 5) -> list[dict]:
    query_vec = model.encode(["query: " + query]).astype("float32")
    distances, ids = index.search(query_vec, top_k)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        meta = metadata[idx]
        results.append({
            "id": idx,
            "distance": float(dist),
            "text": meta.get("text", ""),
            "source": meta.get("source", ""),
            "page": meta.get("page", "?")
        })
    return results

if __name__ == "__main__":
    query = input("Введите запрос: ")
    results = search(query)

    print(f"\n🔍 Вопрос: {query}\n")
    for i, item in enumerate(results):
        print(f"[{i+1}] {item['source']} стр. {item['page']} | id={item['id']} | dist={item['distance']:.4f}")
        text = item['text'].strip().replace("\n", " ")
        print("→", text[:300], "..." if len(text) > 300 else "")
        print("-" * 80)
