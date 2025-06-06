from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pathlib import Path

def iter_block_items(parent):
    for child in parent.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, parent)
        elif child.tag.endswith("}tbl"):
            yield Table(child, parent)

def parse_docx(path: str) -> list[dict]:
    doc = Document(path)
    blocks = []

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if text:
                blocks.append(text)
        elif isinstance(block, Table):
            rows = []
            for row in block.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            blocks.append("[Таблица]\n" + "\n".join(rows))

    if not blocks:
        return []

    return [{
        "text": "\n\n".join(blocks),
        "source": Path(path).name,
        "page": None
    }]
