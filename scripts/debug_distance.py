from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

EMBEDDINGS_PATH = Path("embeddings/embeddings.jsonl")
TARGET_TEXT_FRAGMENT = "частотных диапазонах"

# 1. Найдём нужный чанк
with EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        chunk = json.loads(line)
        if TARGET_TEXT_FRAGMENT in chunk["text"].lower():
            print(f"[✓] Найден чанк id={i}")
            print(chunk["text"])
            chunk_vec = np.array(chunk["embedding"]).astype("float32")
            break
    else:
        print("Чанк не найден")
        exit()

# 2. Получим вектор запроса
query = "определенные частотные диапозоны"
query_vec = model.encode(query)

# 3. Считаем косинусную близость
cos_sim = util.cos_sim(query_vec, chunk_vec)[0][0].item()
print(f"\n[🔍] Косинусная близость запроса к нужному чанку: {cos_sim:.4f}")
