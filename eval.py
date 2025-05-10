import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_metric
import torch
from tqdm import tqdm


def load_results(results_path: str) -> List[Dict]:
    """Load training results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def create_comparison_plots(results: List[Dict], output_dir: str):
    """Create comparison plots for different metrics."""
    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Set style
    plt.style.use("seaborn")
    sns.set_palette("husl")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Training Time Comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="fine_tune_type", y="training_time")
    plt.title("Training Time Comparison")
    plt.xlabel("Fine-tuning Method")
    plt.ylabel("Time (seconds)")
    plt.savefig(f"{output_dir}/training_time_comparison.png")
    plt.close()

    # 2. Loss Comparison
    plt.figure(figsize=(10, 6))
    df_melted = pd.melt(
        df,
        id_vars=["fine_tune_type"],
        value_vars=["train_loss", "eval_loss"],
        var_name="Loss Type",
        value_name="Loss",
    )
    sns.barplot(data=df_melted, x="fine_tune_type", y="Loss", hue="Loss Type")
    plt.title("Training vs Evaluation Loss")
    plt.xlabel("Fine-tuning Method")
    plt.ylabel("Loss")
    plt.legend(title="Loss Type")
    plt.savefig(f"{output_dir}/loss_comparison.png")
    plt.close()


def analyze_lora_hyperparameters(results: List[Dict], output_dir: str):
    """Analyze LoRA hyperparameters if available."""
    lora_results = [r for r in results if r["fine_tune_type"] == "lora"]
    if not lora_results:
        return

    # Extract LoRA configurations
    ranks = []
    alphas = []
    losses = []
    times = []

    for result in lora_results:
        if "config" in result and "peft_config" in result["config"]:
            config = result["config"]["peft_config"]
            ranks.append(config.get("r", 0))
            alphas.append(config.get("lora_alpha", 0))
            losses.append(result["eval_loss"])
            times.append(result["training_time"])

    if ranks:
        # Rank vs Loss
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, losses)
        plt.title("LoRA Rank vs Evaluation Loss")
        plt.xlabel("LoRA Rank")
        plt.ylabel("Evaluation Loss")
        plt.savefig(f"{output_dir}/lora_rank_vs_loss.png")
        plt.close()

        # Rank vs Training Time
        plt.figure(figsize=(10, 6))
        plt.scatter(ranks, times)
        plt.title("LoRA Rank vs Training Time")
        plt.xlabel("LoRA Rank")
        plt.ylabel("Training Time (seconds)")
        plt.savefig(f"{output_dir}/lora_rank_vs_time.png")
        plt.close()


def evaluate_model_performance(model_path: str, test_dataset, tokenizer):
    """Evaluate model performance on test dataset."""
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    # Load metrics
    rouge = load_metric("rouge")
    bleu = load_metric("sacrebleu")

    predictions = []
    references = []

    for example in tqdm(test_dataset, desc="Evaluating"):
        inputs = tokenizer(
            example["text"], return_tensors="pt", truncation=True, max_length=512
        )
        outputs = model.generate(**inputs, max_length=64)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        predictions.append(prediction)
        references.append([example["summary"]])  # BLEU expects list of references

    # Calculate ROUGE scores
    rouge_scores = rouge.compute(predictions=predictions, references=references)

    # Calculate BLEU score
    bleu_score = bleu.compute(predictions=predictions, references=references)

    return {
        "rouge1": rouge_scores["rouge1"].mid.fmeasure,
        "rouge2": rouge_scores["rouge2"].mid.fmeasure,
        "rougeL": rouge_scores["rougeL"].mid.fmeasure,
        "bleu": bleu_score["score"],
    }


def generate_comparison_report(results: List[Dict], output_dir: str):
    """Generate a comprehensive comparison report."""
    report = []
    report.append("# Fine-tuning Methods Comparison Report\n")

    # Overall comparison
    report.append("## Overall Comparison\n")
    df = pd.DataFrame(results)
    report.append("### Training Time and Loss\n")
    report.append(
        df[["fine_tune_type", "training_time", "train_loss", "eval_loss"]].to_markdown()
    )

    # Method-specific analysis
    report.append("\n## Method-specific Analysis\n")
    for method in ["full", "adapter", "lora"]:
        method_results = [r for r in results if r["fine_tune_type"] == method]
        if method_results:
            report.append(f"\n### {method.upper()} Fine-tuning\n")
            report.append(
                f"- Training Time: {method_results[0]['training_time']:.2f} seconds"
            )
            report.append(f"- Training Loss: {method_results[0]['train_loss']:.4f}")
            report.append(f"- Evaluation Loss: {method_results[0]['eval_loss']:.4f}")

            if method == "lora":
                config = method_results[0]["config"].get("peft_config", {})
                report.append("\nLoRA Configuration:")
                report.append(f"- Rank: {config.get('r', 'N/A')}")
                report.append(f"- Alpha: {config.get('lora_alpha', 'N/A')}")
                report.append(f"- Dropout: {config.get('lora_dropout', 'N/A')}")

    # Save report
    with open(f"{output_dir}/comparison_report.md", "w") as f:
        f.write("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuning results")
    parser.add_argument(
        "--results_path",
        default="./out/fine_tuning_results.json",
        help="Path to training results JSON file",
    )
    parser.add_argument(
        "--output_dir",
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--model_dir", default="./out", help="Directory containing fine-tuned models"
    )
    args = parser.parse_args()

    # Load results
    results = load_results(args.results_path)

    # Create comparison plots
    create_comparison_plots(results, args.output_dir)

    # Analyze LoRA hyperparameters
    analyze_lora_hyperparameters(results, args.output_dir)

    # Generate comparison report
    generate_comparison_report(results, args.output_dir)

    print(f"Evaluation complete. Results saved in {args.output_dir}")


if __name__ == "__main__":
    main()
