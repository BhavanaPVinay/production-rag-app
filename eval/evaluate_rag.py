import json
import os
from pathlib import Path
from typing import Any, Dict, List

from app.rag_pipeline import RAGPipeline


def run_evaluation() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent
    queries_file = project_root / "eval" / "eval_queries.json"
    docs_file = project_root / "data" / "sample_docs.txt"

    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found at: {queries_file}")
    if not docs_file.exists():
        raise FileNotFoundError(f"Sample docs file not found at: {docs_file}")

    with open(queries_file, "r", encoding="utf-8") as f:
        queries: List[Dict[str, Any]] = json.load(f)

    # Initialize pipeline with a dedicated evaluation collection
    persist_dir = str(project_root / "chroma_db")
    pipeline = RAGPipeline(
        model_name="all-MiniLM-L6-v2",
        persist_directory=persist_dir,
        collection_name="eval_collection",
    )

    # Index sample documents
    indexed_count = pipeline.index_file(str(docs_file))

    def evaluate_method(method: str) -> Dict[str, Any]:
        hits_at_2 = 0
        reciprocal_ranks: List[float] = []
        details: List[Dict[str, Any]] = []

        for q_item in queries:
            query_str = q_item["query"]
            keyword = q_item["expected_keyword"].lower()

            if method == "dense":
                retrieved = pipeline.retrieve(query=query_str, top_k=5)
            else:
                retrieved = pipeline.hybrid_retrieve(query=query_str, top_k=5)

            match_rank = None
            for rank, item in enumerate(retrieved, start=1):
                doc_text = item.get("document", "").lower()
                if keyword in doc_text:
                    match_rank = rank
                    break

            hit_2 = 1 if (match_rank is not None and match_rank <= 2) else 0
            rr = (1.0 / match_rank) if match_rank is not None else 0.0

            hits_at_2 += hit_2
            reciprocal_ranks.append(rr)

            details.append({
                "id": q_item.get("id"),
                "query": query_str,
                "expected_keyword": q_item["expected_keyword"],
                "match_rank": match_rank,
                "hit_at_2": bool(hit_2),
                "reciprocal_rank": round(rr, 4),
            })

        total = len(queries)
        hit_rate = round(hits_at_2 / total, 4) if total > 0 else 0.0
        mrr = round(sum(reciprocal_ranks) / total, 4) if total > 0 else 0.0

        return {
            "hit_rate_at_2": hit_rate,
            "mrr": mrr,
            "hits_at_2": hits_at_2,
            "total_queries": total,
            "query_details": details,
        }

    dense_results = evaluate_method("dense")
    hybrid_results = evaluate_method("hybrid")

    report = {
        "status": "success",
        "indexed_chunks": indexed_count,
        "total_queries": len(queries),
        "metrics": {
            "dense": {
                "hit_rate_at_2": dense_results["hit_rate_at_2"],
                "mrr": dense_results["mrr"],
            },
            "hybrid": {
                "hit_rate_at_2": hybrid_results["hit_rate_at_2"],
                "mrr": hybrid_results["mrr"],
            },
        },
        "comparison": {
            "hit_rate_at_2_gain": round(
                hybrid_results["hit_rate_at_2"] - dense_results["hit_rate_at_2"], 4
            ),
            "mrr_gain": round(hybrid_results["mrr"] - dense_results["mrr"], 4),
        },
        "dense_breakdown": dense_results["query_details"],
        "hybrid_breakdown": hybrid_results["query_details"],
    }

    return report


if __name__ == "__main__":
    report = run_evaluation()
    print(json.dumps(report, indent=2))
