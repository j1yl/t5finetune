import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import argparse
from transformers import T5ForConditionalGeneration
import evaluate
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
    plt.style.use("default")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Training Time Comparison
    plt.figure()
    plt.bar(df["fine_tune_type"], df["training_time"])
    plt.title("Training Time Comparison")
    plt.xlabel("Fine-tuning Method")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_time_comparison.png")
    plt.close()

    # 2. Loss Comparison
    plt.figure()
    x = np.arange(len(df["fine_tune_type"]))
    width = 0.35

    plt.bar(x - width/2, df["train_loss"], width, label='Training Loss')
    plt.bar(x + width/2, df["eval_loss"], width, label='Evaluation Loss')
    plt.title("Training vs Evaluation Loss")
    plt.xlabel("Fine-tuning Method")
    plt.ylabel("Loss")
    plt.xticks(x, df["fine_tune_type"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/loss_comparison.png")
    plt.close()

    # 3. ROUGE and BLEU Scores
    plt.figure(figsize=(12, 6))
    metrics = ['rouge1', 'rouge2', 'rougeL', 'bleu']
    x = np.arange(len(df["fine_tune_type"]))
    width = 0.2

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df['metrics'].apply(lambda x: x[metric]), width, label=metric.upper())

    plt.title("ROUGE and BLEU Scores Comparison")
    plt.xlabel("Fine-tuning Method")
    plt.ylabel("Score")
    plt.xticks(x + width*1.5, df["fine_tune_type"], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png")
    plt.close()


def analyze_lora_hyperparameters(results: List[Dict], output_dir: str):
    """Analyze LoRA hyperparameters if available."""
    lora_results = [r for r in results if r["fine_tune_type"] == "lora"]
    if not lora_results:
        return

    # Extract LoRA configurations
    ranks = []
    losses = []
    times = []
    metrics = {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'bleu': []
    }

    for result in lora_results:
        if "config" in result and "peft_config" in result["config"]:
            config = result["config"]["peft_config"]
            ranks.append(config.get("r", 0))
            losses.append(result["eval_loss"])
            times.append(result["training_time"])
            for metric in metrics:
                metrics[metric].append(result["metrics"][metric])

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

        # Rank vs Metrics
        plt.figure(figsize=(12, 6))
        for metric, values in metrics.items():
            plt.plot(ranks, values, marker='o', label=metric.upper())
        plt.title("LoRA Rank vs Metrics")
        plt.xlabel("LoRA Rank")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig(f"{output_dir}/lora_rank_vs_metrics.png")
        plt.close()


def evaluate_model_performance(model_path: str, test_dataset, tokenizer):
    """Evaluate model performance on test dataset."""
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()

    # Load metrics
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

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
        "rouge1": rouge_scores["rouge1"],
        "rouge2": rouge_scores["rouge2"],
        "rougeL": rouge_scores["rougeL"],
        "bleu": bleu_score["score"],
    }


def generate_comparison_report(results: List[Dict], output_dir: str):
    """Generate a comprehensive comparison report."""
    report = []
    report.append("# Fine-tuning Methods Comparison Report\n")

    # Overall comparison
    report.append("## Overall Comparison\n")
    df = pd.DataFrame(results)
    
    # Create metrics table
    metrics_df = pd.DataFrame([
        {
            'fine_tune_type': r['fine_tune_type'],
            'training_time': r['training_time'],
            'train_loss': r['train_loss'],
            'eval_loss': r['eval_loss'],
            'rouge1': r['metrics']['rouge1'],
            'rouge2': r['metrics']['rouge2'],
            'rougeL': r['metrics']['rougeL'],
            'bleu': r['metrics']['bleu']
        }
        for r in results
    ])
    
    report.append("### Training Time, Loss, and Metrics\n")
    report.append(metrics_df.to_markdown(index=False))

    # Method-specific analysis
    report.append("\n## Method-specific Analysis\n")
    for method in ["base", "full", "adapter", "lora"]:
        method_results = [r for r in results if r["fine_tune_type"] == method]
        if method_results:
            report.append(f"\n### {method.upper()} Fine-tuning\n")
            if method != "base":
                report.append(f"- Training Time: {method_results[0]['training_time']:.2f} seconds")
                report.append(f"- Training Loss: {method_results[0]['train_loss']:.4f}")
            report.append(f"- Evaluation Loss: {method_results[0]['eval_loss']:.4f}")
            report.append("\nMetrics:")
            for metric, value in method_results[0]['metrics'].items():
                report.append(f"- {metric.upper()}: {value:.4f}")

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
        default="./results",
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
