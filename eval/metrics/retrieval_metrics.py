def recall_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    retrieved = retrieved_ids[:k]
    hits = sum(1 for gt in ground_truth_ids if gt in retrieved)
    return hits / len(ground_truth_ids) if ground_truth_ids else 0.0

def mrr_at_k(retrieved_ids: list[str], ground_truth_ids: list[str], k: int) -> float:
    retrieved = retrieved_ids[:k]
    for rank, rid in enumerate(retrieved, start=1):
        if rid in ground_truth_ids:
            return 1.0 / rank
    return 0.0
