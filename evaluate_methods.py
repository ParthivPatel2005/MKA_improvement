"""Benchmark different alpha-initialisation strategies for MKA layer merging."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline import format_example, gen_prompt
from train_learnable_alpha import LearnableAlphaConfig, LearnableAlphaMKA


def evaluate_model(
    fused_model_dir: str,
    data_dir: str,
    device: str,
    ntrain: int = 5,
    max_subjects: int | None = None,
    max_samples: int | None = None,
) -> Tuple[float, float]:
    tokenizer = AutoTokenizer.from_pretrained(
        fused_model_dir,
        use_fast=True,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        fused_model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()

    subjects = sorted(
        {
            file.split("_test.csv")[0]
            for file in os.listdir(os.path.join(data_dir, "test"))
            if file.endswith("_test.csv")
        }
    )
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    accs: Dict[str, float] = {}
    ppls: Dict[str, float] = {}

    with torch.no_grad():
        for subject in subjects:
            dev_path = os.path.join(data_dir, "dev", f"{subject}_dev.csv")
            test_path = os.path.join(data_dir, "test", f"{subject}_test.csv")
            dev_df = pd.read_csv(dev_path, header=None)
            test_df = pd.read_csv(test_path, header=None)

            cors = []
            losses = []
            sample_range = range(len(test_df))
            if max_samples is not None:
                sample_range = list(sample_range)[:max_samples]

            for idx in tqdm(sample_range, desc=f"Evaluating {subject}"):
                prompt_end = format_example(test_df, idx, include_answer=False)
                train_prompt = gen_prompt(dev_df, subject, ntrain)
                prompt = train_prompt + prompt_end
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                labels = inputs["input_ids"].clone()
                labels[:, :-len(tokenizer(prompt_end).input_ids)] = -100
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                losses.append(loss.item())
                logits = outputs.logits[:, -1, :]
                choices_ids = [tokenizer(c).input_ids[-1] for c in ["A", "B", "C", "D"]]
                probs = torch.softmax(logits[:, choices_ids], dim=-1)
                choice_idx = int(torch.argmax(probs, dim=-1))
                prediction = ["A", "B", "C", "D"][choice_idx]
                answer = test_df.iloc[idx, test_df.shape[1] - 1]
                cors.append(prediction == answer)

            acc = float(np.mean(cors)) if cors else 0.0
            ppl = float(np.exp(np.mean(losses))) if losses else float("inf")
            accs[subject] = acc
            ppls[subject] = ppl

    overall_acc = float(np.mean(list(accs.values()))) if accs else 0.0
    overall_ppl = float(np.mean(list(ppls.values()))) if ppls else float("inf")
    torch.cuda.empty_cache()
    return overall_acc, overall_ppl


def run_method(
    name: str,
    alpha_strategy: str,
    train_steps: int,
    args: argparse.Namespace,
    results_dir: str,
) -> Dict[str, float | None]:
    output_dir = os.path.join(results_dir, name)
    config = LearnableAlphaConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=output_dir,
        target_compression_ratio=args.target_compression,
        device=args.device,
        alpha_init_strategy=alpha_strategy,
        similarity_matrix_path=args.similarity_matrix,
        merge_pairs_path=args.merge_pairs,
        num_calibration_samples=args.num_calibration_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_training_steps=train_steps,
        max_seq_length=args.max_seq_length,
        random_seed=args.seed,
    )

    runner = LearnableAlphaMKA(config)
    runner.identify_merge_pairs()
    runner.replace_with_mergeable_layers()
    if train_steps > 0:
        dataloader = runner.prepare_calibration_dataloader()
        runner.train_alpha_parameters(dataloader)
    else:
        runner.learned_alphas = runner._extract_learned_alphas()

    fused_dir = os.path.join(output_dir, "models")
    plots_dir = os.path.join(output_dir, "plots")
    runner.bake_alphas_and_fuse(fused_dir)
    correlation = runner.plot_alpha_analysis(os.path.join(plots_dir, "alpha_analysis.png"))

    accuracy, perplexity = evaluate_model(
        fused_model_dir=fused_dir,
        data_dir=args.data_dir,
        device=args.device,
        ntrain=args.ntrain,
        max_subjects=args.max_subjects,
        max_samples=args.max_eval_samples,
    )

    metrics: Dict[str, float | None] = {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "alpha_similarity_correlation": correlation,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alpha strategies on MMLU")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--similarity_matrix", type=str, required=True)
    parser.add_argument("--merge_pairs", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--target_compression", type=float, default=0.4)
    parser.add_argument("--num_calibration_samples", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_training_steps", type=int, default=500)
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results: Dict[str, Dict[str, float | None]] = {}
    results["mka_similarity"] = run_method(
        name="baseline_similarity",
        alpha_strategy="similarity",
        train_steps=0,
        args=args,
        results_dir=args.output_dir,
    )
    results["fixed_05"] = run_method(
        name="baseline_fixed_05",
        alpha_strategy="uniform",
        train_steps=0,
        args=args,
        results_dir=args.output_dir,
    )
    results["fixed_07"] = run_method(
        name="baseline_fixed_07",
        alpha_strategy="fixed_07",
        train_steps=0,
        args=args,
        results_dir=args.output_dir,
    )
    results["learned"] = run_method(
        name="learned_alpha",
        alpha_strategy="uniform",
        train_steps=args.num_training_steps,
        args=args,
        results_dir=args.output_dir,
    )

    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
