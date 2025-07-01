from src.parsing.docx_parser import parse_docx
from src.parsing.pdf_parser import parse_pdf
from src.parsing.pptx_parser import parse_pptx
from src.chunking.simple_splitter import text_splitter
from pathlib import Path
import json

INPUT_DIR = Path("data/raw")
OUTPUT_PATH = Path("parsed/parsed.jsonl")

parsers = {
    ".docx": parse_docx,
    ".pdf": parse_pdf,
    ".pptx": parse_pptx,
}

results = []

for file in INPUT_DIR.iterdir():
    ext = file.suffix.lower()

    if ext == ".doc":
        print(f"[!] Пропущен файл '{file.name}': формат .doc не поддерживается. Сконвертируйте в .docx.")
        continue

    parser = parsers.get(ext)
    if parser:
        print(f"[*] Обработка: {file.name}")
        row_blocks = parser(str(file))
        for row in row_blocks:
            buffer_chunk = text_splitter(row["text"])
            for ch in buffer_chunk:
                ch.update({
                    "source": row["source"],
                    "page": row["page"],
                })
                results.append(ch)
    else:
        print(f"[!] Пропущен файл '{file.name}': неизвестное расширение.")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"[✓] Готово: сохранено {len(results)} чанков -> {OUTPUT_PATH}")
