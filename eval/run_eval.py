"""Offline evaluation over the golden dataset using deepeval/ragas."""
import argparse
import asyncio
import json
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from agents.graph import agent_graph
from eval.metrics.retrieval_metrics import mrr_at_k, recall_at_k

RESULTS_PATH = Path("eval/results/latest.json")


async def run_single(item: dict) -> dict:
    query = item["question"]
    state = {"query": query, "user_groups": ["public"]}
    
    # Run the graph
    final_state = await agent_graph.ainvoke(state)
    
    ans = final_state.get("answer", "")
    context = final_state.get("context", "")
    retrieved_docs = final_state.get("documents", [])
    retrieved_ids = [d.get("doc_id") for d in retrieved_docs]
    gt_ids = item.get("source_doc_ids", [])

    # Retrieval metrics
    r_at_10 = recall_at_k(retrieved_ids, gt_ids, 10)
    mrr = mrr_at_k(retrieved_ids, gt_ids, 10)

    res = {
        "query": query,
        "recall@10": r_at_10,
        "mrr@10": mrr,
        "answer": ans,
    }

    # If it's adversarial, we just check refusal
    if item.get("type") in ("adversarial", "acl_probe", "pii_probe", "doc_injection"):
        res["refusal_accuracy"] = 1.0 if final_state.get("needs_escalation") else 0.0
        return res

    # Generation metrics (Faithfulness / Relevancy) via DeepEval
    tc = LLMTestCase(
        input=query,
        actual_output=ans,
        retrieval_context=[context] if context else [],
        expected_output=item.get("answer")
    )
    faith = FaithfulnessMetric(threshold=0.7)
    rel = AnswerRelevancyMetric(threshold=0.7)

    try:
        faith.measure(tc)
        res["faithfulness"] = faith.score
    except Exception:
        res["faithfulness"] = 0.0

    try:
        rel.measure(tc)
        res["relevancy"] = rel.score
    except Exception:
        res["relevancy"] = 0.0

    return res


async def main(dataset_path: str):
    lines = Path(dataset_path).read_text().strip().split("\n")
    dataset = [json.loads(l) for l in lines if l.strip()]

    print(f"Running eval on {len(dataset)} cases...")
    tasks = [run_single(item) for item in dataset]
    results = await asyncio.gather(*tasks)

    # Aggregate
    agg = {}
    for key in ["recall@10", "mrr@10", "faithfulness", "relevancy", "refusal_accuracy"]:
        vals = [r[key] for r in results if key in r]
        if vals:
            agg[key] = sum(vals) / len(vals)

    print("\n--- Aggregate Results ---")
    for k, v in agg.items():
        print(f"{k}: {v:.4f}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(agg, indent=2))
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eval/golden_dataset/questions.jsonl")
    args = parser.parse_args()
    asyncio.run(main(args.dataset))
