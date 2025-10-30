"""Plot utilities for learnable-alpha experiments."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_accuracy_bar(results: Dict[str, Dict[str, float]], save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    labels = list(results.keys())
    values = [results[label].get("accuracy", 0.0) for label in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=values, palette="viridis", ax=ax)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("MMLU accuracy across strategies")
    for index, value in enumerate(values):
        ax.text(index, value + 0.01, f"{value * 100:.2f}%", ha="center")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_alpha_histogram(alpha_data: Dict[str, float], save_path: str) -> None:
    if not alpha_data:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    alphas = list(alpha_data.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(alphas, bins=20, kde=True, color="steelblue", ax=ax)
    ax.set_xlabel("Alpha value")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of learned alpha values")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_alpha_vs_similarity(alpha_data: Dict[str, float], similarity_scores: Dict[str, float], save_path: str) -> None:
    if not alpha_data or not similarity_scores:
        return
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    alphas = []
    similarities = []
    for key, alpha in alpha_data.items():
        if key in similarity_scores:
            similarities.append(similarity_scores[key])
            alphas.append(alpha)
    if not alphas:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=similarities, y=alphas, ax=ax, color="darkgreen")
    ax.set_xlabel("Similarity score")
    ax.set_ylabel("Learned alpha")
    ax.set_title("Alpha vs similarity")
    if len(alphas) > 1:
        corr = np.corrcoef(similarities, alphas)[0, 1]
        ax.text(0.05, 0.95, f"corr={corr:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot learnable-alpha experiment results")
    parser.add_argument("--results_file", type=str, required=True)
    parser.add_argument("--learned_alphas", type=str, default=None)
    parser.add_argument("--similarity_scores", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = load_json(args.results_file)
    plot_accuracy_bar(results, os.path.join(args.output_dir, "accuracy_comparison.png"))

    alpha_data: Dict[str, float] = {}
    similarity_scores: Dict[str, float] = {}
    if args.learned_alphas and os.path.exists(args.learned_alphas):
        alpha_payload = load_json(args.learned_alphas)
        alpha_data = alpha_payload.get("learned_alphas", alpha_payload)
        similarity_scores = alpha_payload.get("similarity_scores", {})
    if args.similarity_scores and os.path.exists(args.similarity_scores):
        similarity_scores = load_json(args.similarity_scores)

    plot_alpha_histogram(alpha_data, os.path.join(args.output_dir, "alpha_distribution.png"))
    plot_alpha_vs_similarity(
        alpha_data,
        similarity_scores,
        os.path.join(args.output_dir, "alpha_vs_similarity.png"),
    )


if __name__ == "__main__":
    main()
