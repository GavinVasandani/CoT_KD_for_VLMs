# integration_test_pipeline.py

import random
import os
import sys
import argparse
import subprocess
import json
from statistics import mean, stdev

sys.path.append("hypo_dataset/")
from contextvqa_split import run_contextvqa_split

def run_pipeline_once(dataset_path, run_index, seed):
    # Fix split ratios but allow internal reshuffle of scenes
    random.seed(seed)
    while True:
        train_ratio = random.uniform(0.6, 0.8)
        val_ratio = random.uniform(0.1, 0.3)
        test_ratio = 1.0 - train_ratio - val_ratio
        if 0.1 <= test_ratio <= 0.3:
            break

    output_dir = f"hypo_dataset/cot_contextvqa_split/run_{run_index}"
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_contextvqa.json")
    val_path = os.path.join(output_dir, "val_contextvqa.json")
    test_path = os.path.join(output_dir, "test_contextvqa.json")

    # Perform the dataset split
    run_contextvqa_split(
        json_path=dataset_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        output_train_path=train_path,
        output_val_path=val_path,
        output_test_path=test_path
    )

    # Set environment variables
    os.environ["TRAIN_CONTEXTVQA_PATH"] = train_path
    os.environ["VAL_CONTEXTVQA_PATH"] = val_path
    os.environ["TEST_CONTEXTVQA_PATH"] = test_path

    print(f"\n✅ Run {run_index}: Splits complete. Running pipeline...")

    # Call the pipeline script (cot_inference_pipeline.sh)
    result = subprocess.run(["bash", "cot_inference_pipeline.sh"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Pipeline failed at run {run_index}: {result.stderr}")
        return None

    # Find the latest result file written by model_with_lora_evaluate.py
    result_dir = "results/"
    result_files = sorted(
        [f for f in os.listdir(result_dir) if f.endswith(".txt")],
        key=lambda f: os.path.getmtime(os.path.join(result_dir, f)),
        reverse=True
    )

    if not result_files:
        print(f"❌ No results found after run {run_index}.")
        return None

    latest_result = os.path.join(result_dir, result_files[0])
    with open(latest_result, "r") as f:
        lines = f.readlines()
        metrics = {}
        for line in lines:
            if ":" in line:
                key, val = line.strip().split(":")
                metrics[key.strip()] = float(val.strip().replace("%", ""))  # remove % if present

    print(f"✅ Run {run_index} complete. Metrics from {latest_result}: {metrics}")
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--dataset", type=str, default="hypo_dataset/contextvqa_full.json",
                        help="Path to benchmark dataset to test integration across multiple randomized runs")
    parser.add_argument("-n", "--num_runs", type=int, default=3,
                        help="Number of integration test runs to perform")
    args = parser.parse_args()

    seed = 42
    all_metrics = []

    for i in range(args.num_runs):
        metrics = run_pipeline_once(args.dataset, run_index=i, seed=seed)
        if metrics:
            all_metrics.append(metrics)

    print("\n📊 Aggregated Results Across Runs:")
    if all_metrics:
        keys = all_metrics[0].keys()
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                print(f"{key}: Mean = {mean(values):.2f}, Std Dev = {stdev(values):.2f}")
    else:
        print("⚠️ No successful runs to report.")

if __name__ == "__main__":
    main()
