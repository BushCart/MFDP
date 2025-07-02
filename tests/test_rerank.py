import pytest
pytest.skip("Пропускаем rerank-тесты в предзащите", allow_module_level=True)

from scripts.query_engine_llm import rerank

def test_rerank_outputs_scores():
    dummy_candidates = [
        {"text": "Просто тестовый текст.", "distance": 0.1}
    ]
    result = rerank("Какой-то вопрос", dummy_candidates, top_k=1)
    assert "score" in result[0]
