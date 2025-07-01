import fitz
from pathlib import Path


def parse_pdf(path: str) -> list[dict]:
    doc = fitz.open(path)
    result = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()
        if text:
            result.append({
                "text": text,
                "source": Path(path).name,
                "page": page_num + 1

            })
    return result


if __name__ == "__main__":
    from pprint import pprint

    result = parse_pdf("data/raw/test_1.pdf")
    pprint(result[:2])
