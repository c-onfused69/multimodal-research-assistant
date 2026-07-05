"""Exports high-confidence RAG traces from Langfuse into JSONL for finetuning."""
import json
from pathlib import Path

from langfuse import Langfuse

# NOTE: Requires langfuse SDK setup in environment
lf = Langfuse()


def export_positive_traces(output_path: str = "data/finetune/dataset.jsonl"):
    print("Fetching traces with user feedback score > 0 or high agent confidence...")
    # Pseudo-query: In reality, you use lf.client.trace.list(tags=["positive"])
    # This is a stub showing the data shape expected by the finetuner.
    dataset = [
        {
            "messages": [
                {"role": "user", "content": "How does the cache work?"},
                {"role": "assistant", "content": "The system uses Redis for semantic caching... [1]"}
            ]
        }
    ]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")
    print(f"Exported {len(dataset)} traces to {output_path}")


if __name__ == "__main__":
    export_positive_traces()
