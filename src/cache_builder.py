"""Build and save Bing search cache for all FreshQA questions."""
import time
from src.bing_client import search
from src.utils import load_json, save_json


def build_cache() -> None:
    questions = load_json("data/freshqa_questions.json")
    cache = {}

    for i, q in enumerate(questions):
        qid = q["question_id"]
        query = q["question"]
        print(f"[{i+1}/{len(questions)}] Searching: {query[:80]}...")

        try:
            results = search(query)
            cache[qid] = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": results,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            cache[qid] = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "results": [],
            }

        time.sleep(0.5)

    save_json(cache, "data/search_cache.json")
    total = sum(1 for v in cache.values() if v["results"])
    empty = len(cache) - total
    print(f"Cache built: {total} with results, {empty} empty")


if __name__ == "__main__":
    build_cache()
