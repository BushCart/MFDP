from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from pathlib import Path
from ollama import Client
import tiktoken

# Параметры
INDEX_PATH = Path("vector_store/faiss.index")
META_PATH = Path("vector_store/metadata.pkl")
MAX_TOKENS = 1500
ollama = Client(host="http://localhost:11434")


# Загрузка модели и индекса
embedder = SentenceTransformer('intfloat/multilingual-e5-base')
index = faiss.read_index(str(INDEX_PATH))
enc = tiktoken.get_encoding("cl100k_base")

# Загрузка метаданных
with META_PATH.open("rb") as f:
    metadata = pickle.load(f)

# Подсчет токенов для ограничения контекста
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

# Оценка уверенности по минимальному расстоянию
def get_confidence(filtered: list[dict]) -> str:
    min_dist = min(r["distance"] for r in filtered)
    if min_dist < 0.35:
        return "🟢 Уверенность: высокая"
    elif min_dist < 0.55:
        return "🟡 Уверенность: средняя"
    else:
        return "🔴 Уверенность: низкая"

# Построение контекста с ограничением по токенам
def build_context(filtered: list[dict]) -> tuple[str, list[dict]]:
    context_chunks = []
    token_total = 0
    used = []

    for r in filtered:
        tokens = count_tokens(r["text"])
        if token_total + tokens > MAX_TOKENS:
            break
        context_chunks.append(r["text"])
        token_total += tokens
        used.append({
            "page": r["page"],
            "id": r["id"],
            "source": r["source"]
        })


    return "\n---\n".join(context_chunks), used


# Поиск релевантных чанков
def search(query: str, top_k: int = 5) -> list[dict]:
    query_vec = embedder.encode(["query: " + query]).astype("float32")
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

# Основной пайплайн: поиск + генерация
def search_with_llm(query: str, top_k: int = 5):
    results = search(query, top_k)
    filtered = [r for r in results if r["distance"] < 0.5]

    if not filtered:
        print("🔴 Подходящих фрагментов не найдено. Ответ не сформирован.")
        return

    confidence = get_confidence(filtered)
    context, used_chunks = build_context(filtered)

    system = "Ты — технический ассистент. Отвечай по делу, кратко, без выдумок. Используй только приведённый контекст. Ты отвечаешь строго на русском языке."
    prompt = f"Контекст:\n{context}\n\nВопрос: {query}\n\nОтвет:"

    try:
        response = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
        print(f"\n🔍 Ответ ({confidence}):\n" + response['message']['content'])
        print("\n📚 Использованные источники:")
        for s in used_chunks:
            print(f"- {s['source']} стр. {s['page']} (id={s['id']})")

    except Exception as e:
        print("[✗] Ошибка при обращении к Ollama:")
        print(repr(e))
        print("\n[✎] Последний prompt был:")
        print(prompt[:500])

if __name__ == "__main__":
    query = input("Введите вопрос: ")
    search_with_llm(query)
