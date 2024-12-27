import asyncio
import os
import pandas as pd
import csv
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics._string import NonLLMStringSimilarity, DistanceMeasure
from ragas.metrics import RougeScore

# Initialize metrics
metrics = {
    "NonLLM_StringSimilarity_Levenshtein": NonLLMStringSimilarity(distance_measure=DistanceMeasure.LEVENSHTEIN),
    "NonLLM_StringSimilarity_Hamming": NonLLMStringSimilarity(distance_measure=DistanceMeasure.HAMMING),
    "ROUGE_Score": RougeScore(),
}

from sentence_transformers import SentenceTransformer

async def evaluate_sample(sample):
    sample_results = {}
    for metric_name, scorer in metrics.items():
        try:
            score = await scorer.single_turn_ascore(sample)
            sample_results[metric_name] = score
        except Exception as e:
            sample_results[metric_name] = f"Error: {e}"
    return sample_results

async def evaluate_all_datasets():
    thresholds = [0.25, 0.50, 0.75]
    results_folder = "results"  # Folder containing multiple CSV files
    all_files = [os.path.join(results_folder, f) for f in os.listdir(results_folder) if f.endswith('.csv')]
    combined_results = []
    combined_accuracies = []

    for file in all_files:
        data = pd.read_csv(file)
        results = []
        metric_accuracies = {threshold: {metric: 0 for metric in metrics.keys()} for threshold in thresholds}
        total_samples = 0

        for index, row in data.iterrows():
            sample = SingleTurnSample(
                response=row["openai_answer"],
                reference=row["actual_answer"]
            )
            sample_results = {
                "file": file,
                "index": index,
                "link": row.get("link", ""),
                "question": row.get("question", "")
            }
            metric_scores = await evaluate_sample(sample)
            sample_results.update(metric_scores)

            # Count accuracy based on thresholds
            for threshold in thresholds:
                for metric_name, score in metric_scores.items():
                    if isinstance(score, (int, float)) and score >= threshold:
                        metric_accuracies[threshold][metric_name] += 1

            total_samples += 1
            results.append(sample_results)

        # Calculate accuracy for each metric at each threshold for this file
        accuracy_results = {
            metric: {
                threshold: (metric_accuracies[threshold][metric] / total_samples) * 100
                for threshold in thresholds
            } for metric in metrics.keys()
        }

        # Append accuracy results for this file to combined accuracies
        for metric, thresholds_acc in accuracy_results.items():
            combined_accuracies.append({
                "file": os.path.basename(file),
                "Metric": metric,
                "Accuracy at 0.25": thresholds_acc[0.25],
                "Accuracy at 0.5": thresholds_acc[0.50],
                "Accuracy at 0.75": thresholds_acc[0.75],
            })

        # Save per-file accuracies to a CSV
        accuracy_file = f"eval/{os.path.basename(file).replace('.csv', '')}_metric_accuracies.csv"
        with open(accuracy_file, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Metric", "Accuracy at 0.25", "Accuracy at 0.5", "Accuracy at 0.75"])
            for metric, thresholds_acc in accuracy_results.items():
                writer.writerow([metric] + [thresholds_acc[threshold] for threshold in thresholds])

        print(f"Accuracy results for {file} saved to {accuracy_file}")

        combined_results.extend(results)

    # Save combined results for all files
    combined_output_file = "eval/combined_evaluation_results.csv"
    fieldnames = ["file", "index", "link", "question"] + list(metrics.keys())

    with open(combined_output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined_results)

    print(f"Combined evaluation results saved to {combined_output_file}")

    # Save combined accuracies for all files
    combined_accuracy_file = "eval/combined_metric_accuracies.csv"
    with open(combined_accuracy_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["file", "Metric", "Accuracy at 0.25", "Accuracy at 0.5", "Accuracy at 0.75"])
        writer.writeheader()
        writer.writerows(combined_accuracies)

    print(f"Combined accuracy results saved to {combined_accuracy_file}")

asyncio.run(evaluate_all_datasets())