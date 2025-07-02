from src.chunking.simple_splitter import text_splitter


def test_returns_list_of_dicts():
    text = "Катя побежала. За спиной ее грохнул залп. Одна из пуль просвистела возле виска, другая вырвала клок из рукава полушубка."
    result = text_splitter(text, max_length=50, overlap=10)
    assert isinstance(result, list)
    assert all(isinstance(chunk, dict) for chunk in result)
    assert all("text" in chunk and "chunk_id" in chunk for chunk in result)


def test_no_negative_overlap():
    text = "Hello World. " * 20
    result = text_splitter(text, max_length=50, overlap=10)
    for chunk in result:
        assert len(chunk["text"]) > 0


def test_short_text_returns_one_chunk():
    text = "Короткий текст."
    result = text_splitter(text, max_length=100)
    assert len(result) == 1
    assert result[0]["text"] == text


def test_empty_text_returns_empty_list():
    text = ""
    result = text_splitter(text)
    assert result == []
