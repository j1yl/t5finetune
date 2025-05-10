from dataset import tokenizer, tokenized_ds
import time
from transformers import (
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
from adapters import T5AdapterModel, AdapterConfig
import torch
import argparse
from typing import Dict, Any
import json


def get_model_and_config(fine_tune_type: str, model_name: str = "t5-small") -> tuple:
    """Get model and configuration based on fine-tuning type."""
    base_model = T5ForConditionalGeneration.from_pretrained(model_name)

    if fine_tune_type == "full":
        return base_model, {}

    elif fine_tune_type == "adapter":
        model = T5AdapterModel.from_pretrained(model_name)
        adapter_config = AdapterConfig.load("pfeiffer", reduction_factor=16)
        model.add_adapter("summarization", config=adapter_config)
        model.train_adapter("summarization")
        return model, {"adapter_name": "summarization"}

    elif fine_tune_type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,  # LoRA attention dimension
            lora_alpha=32,  # LoRA alpha parameter
            lora_dropout=0.1,
            target_modules=["q", "v"],  # Target attention modules
        )
        model = get_peft_model(base_model, peft_config)
        return model, {"peft_config": peft_config}

    else:
        raise ValueError(f"Unknown fine-tuning type: {fine_tune_type}")


def train_and_evaluate(
    fine_tune_type: str,
    model_name: str = "t5-small",
    output_dir: str = "./out",
    learning_rate: float = 3e-4,
    batch_size: int = 8,
    num_epochs: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """Train and evaluate the model with timing and metrics."""
    start_time = time.time()

    # Get model and config
    model, config = get_model_and_config(fine_tune_type, model_name)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"{output_dir}/{fine_tune_type}",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir=f"./logs/{fine_tune_type}",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        report_to="none",
        gradient_accumulation_steps=4,
        warmup_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
    )

    # Train
    train_result = trainer.train()

    # Evaluate
    eval_result = trainer.evaluate()

    # Calculate total time
    total_time = time.time() - start_time

    # Save model
    trainer.save_model(f"{output_dir}/{fine_tune_type}/final_model")
    tokenizer.save_pretrained(f"{output_dir}/{fine_tune_type}/final_model")

    # Return results
    return {
        "fine_tune_type": fine_tune_type,
        "training_time": total_time,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_result["eval_loss"],
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune T5 model using different methods"
    )
    parser.add_argument("--model", default="t5-small", help="Base model to use")
    parser.add_argument("--output_dir", default="./out", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    # Run all fine-tuning methods
    results = []
    for method in ["full", "adapter", "lora"]:
        print(f"\nTraining with {method} fine-tuning...")
        result = train_and_evaluate(
            fine_tune_type=method,
            model_name=args.model,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
        )
        results.append(result)

        # Print results
        print(f"\nResults for {method} fine-tuning:")
        print(f"Training time: {result['training_time']:.2f} seconds")
        print(f"Training loss: {result['train_loss']:.4f}")
        print(f"Evaluation loss: {result['eval_loss']:.4f}")

    # Save all results
    with open(f"{args.output_dir}/fine_tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
