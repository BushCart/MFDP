from scripts.query_engine_llm import search

def test_search_returns_results():
    query = "тестовый запрос"
    results = search(query, top_k=3)
    assert isinstance(results, list)
    assert all("text" in r and "source" in r and "page" in r for r in results)
