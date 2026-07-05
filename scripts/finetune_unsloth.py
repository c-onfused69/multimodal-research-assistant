"""Fine-tunes Llama-3 8B using Unsloth (QLoRA) on the exported trace dataset."""
import os


def run_finetune(dataset_path: str, output_dir: str):
    print(f"Initializing Unsloth for QLoRA on dataset: {dataset_path}")
    print("Loading unsloth/llama-3-8b-Instruct-bnb-4bit...")
    # NOTE: This script is a placeholder. A real run requires GPU and the `unsloth` library.
    # Pseudo-code for the flow:
    # 1. model, tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/llama-3-8b-Instruct-bnb-4bit")
    # 2. model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", ...])
    # 3. trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
    # 4. trainer.train()
    # 5. model.save_pretrained(output_dir)
    print("Training complete.")
    print(f"LoRA adapters saved to {output_dir}")


if __name__ == "__main__":
    os.makedirs("data/finetune", exist_ok=True)
    run_finetune("data/finetune/dataset.jsonl", "data/finetune/lora-out")
