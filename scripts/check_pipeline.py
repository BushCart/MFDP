import os
import subprocess

print("\n[✓] Проверка структуры проекта...")

required_files = [
    "scripts/parse_docs.py",
    "scripts/embed_chunks.py",
    "scripts/build_faiss_index.py",
    "scripts/query_engine.py",
]
for path in required_files:
    if not os.path.exists(path):
        print(f"[✗] Отсутствует: {path}")
        exit(1)

print("[✓] Структура в порядке")

print("\n[✓] Прогон сквозного пайплайна...")
steps = [
    "python -m scripts.parse_docs",
    "python -m scripts.embed_chunks",
    "python -m scripts.build_faiss_index",
]

for cmd in steps:
    print(f"\n[*] Выполняется: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[✗] Ошибка выполнения: {cmd}")
        exit(1)

print("\n[✓] Сквозной пайплайн отработал успешно.")
