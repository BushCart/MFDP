# Установка и запуск

🔹 Требования:

* Python 3.10+
* Git
* [Ollama](https://ollama.com/download)
* Желательно: виртуальная среда (venv / conda)

---

## 1. Клонируй репозиторий

```bash
git clone https://github.com/BushCart/MFDP.git
cd MFDP
```

## 2. Активируй виртуальную среду

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate        # Windows
```

## 3. Установи зависимости

```bash
pip install -r requirements.txt
```

## 4. Скачай и запусти Ollama

```bash
ollama run mistral
```

## 5. Обработай документы

```bash
python -m scripts.parse_docs
python -m scripts.embed_chunks
python -m scripts.build_faiss_index
```

## 6. Задай вопрос

```bash
python -m scripts.query_engine_llm
```
