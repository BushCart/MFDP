import pandas as pd
from pathlib import Path
from tests.gold_standard import GOLD_STANDARD
from scripts.query_engine_llm import search 

def evaluate_all(k: int = 5) -> pd.DataFrame:
    records = []
    from pprint import pprint

    for i, (question, relevant_chunks) in enumerate(GOLD_STANDARD.items()):
        if i > 0:  # протестируем только первый вопрос
            break

        found_chunks = search(question, top_k=5)
        found_set    = set((Path(r["source"]).name, str(r["page"])) for r in found_chunks)
        relevant_set = set((Path(r["source"]).name, str(r["page"])) for r in relevant_chunks)

        print("Вопрос:", question)
        print("Найдено (source, page):")
        pprint(found_set)
        print("Эталон (source, page):")
        pprint(relevant_set)
        print("Пересечение:", found_set & relevant_set)
        break

    for question, relevant in GOLD_STANDARD.items():
        found = search(question, top_k=k)
        found_set     = set((Path(r["source"]).name, str(r["page"])) for r in found_chunks)
        relevant_set  = set((Path(r["source"]).name, str(r["page"])) for r in relevant_chunks)
        p = len(found_set & relevant_set) / k
        rr = 0.0
        for idx, r in enumerate(found, 1):
            if (r["source"], str(r["page"])) in relevant_set:
                rr = 1.0 / idx
                break
        records.append({
            "Вопрос": question,
            f"P@{k}": round(p, 2),
            "RR": round(rr, 2)
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = evaluate_all(k=5)
    print(df.to_markdown(index=False))
    df.to_csv("results.csv", index=False)
    print(f"\nСохранено в {Path('results.csv').resolve()}")
