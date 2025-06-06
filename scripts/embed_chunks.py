from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

INPUT_PATH = Path("parsed/parsed.jsonl")
OUTPUT_PATH = Path("embeddings/embeddings.jsonl")
BATCH_SIZE = 32

model = SentenceTransformer('intfloat/multilingual-e5-base')

buffer = []

with INPUT_PATH.open("r", encoding="utf-8") as infile, OUTPUT_PATH.open("w", encoding="utf-8") as outfile:
    for line in infile:
        chunk = json.loads(line)
        buffer.append(chunk)

        if len(buffer) == BATCH_SIZE:
            texts = ["passage: " + item["text"] for item in buffer]
            embeddings = model.encode(texts)

            for chunk, emb in zip(buffer, embeddings):
                chunk["embedding"] = emb.tolist()
                outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            buffer = []

    if buffer:
        texts = [item["text"] for item in buffer]
        embeddings = model.encode(texts)
        for chunk, emb in zip(buffer, embeddings):
            chunk["embedding"] = emb.tolist()
            outfile.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print(f"[✓] Готово: эмбеддинги сохранены -> {OUTPUT_PATH}")