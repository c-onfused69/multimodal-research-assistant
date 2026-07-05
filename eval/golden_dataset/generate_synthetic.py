"""Uses ragas to generate synthetic Q&A pairs from the chunk index for offline eval."""
import asyncio
from pathlib import Path

from config.settings import settings


async def generate_synthetic_dataset(output_file: str, num_questions: int = 20):
    # Pseudo-code for raga integration
    print(f"Generating {num_questions} synthetic questions using {settings.llm_model}...")
    
    # Normally we would sample from Qdrant and prompt the LLM
    # For now, just create a dummy file if it doesn't exist
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    with out.open("w") as f:
        f.write('{"question": "What is RAG?", "answer": "Retrieval-Augmented Generation.", "source_doc_ids": ["syn_1"]}\n')

    print(f"Dataset generated at {output_file}")


if __name__ == "__main__":
    asyncio.run(generate_synthetic_dataset("eval/golden_dataset/synthetic.jsonl"))
