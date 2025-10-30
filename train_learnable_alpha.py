"""Training utilities for learnable-alpha MKA experiments.

This script orchestrates the full workflow required to learn per-merge mixing
coefficients after the original MKA similarity computation phase has produced
candidate layer pairs. The pipeline is intentionally modular so that it can be
called programmatically or executed as a standalone CLI entry point.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from mergeable_layer import MergeableLayer, create_mergeable_layer

CHOICES = ["A", "B", "C", "D"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_example(row: Sequence[str], include_answer: bool = True) -> str:
    prompt = row[0]
    num_choices = len(row) - 2
    for idx in range(num_choices):
        prompt += f"\n{CHOICES[idx]}. {row[idx + 1]}"
    prompt += "\nAnswer:"
    if include_answer:
        prompt += f" {row[num_choices + 1]}\n"
    return prompt


class MMLUCalibrationDataset(Dataset):
    """Lightweight dataset that yields formatted calibration prompts."""

    def __init__(
        self,
        data_dir: str,
        subjects: Optional[Iterable[str]] = None,
        num_samples: int = 1_000,
        seed: int = 7,
    ) -> None:
        self.prompts: List[str] = []
        rng = random.Random(seed)
        dev_dir = os.path.join(data_dir, "dev")
        if not os.path.isdir(dev_dir):
            raise FileNotFoundError(f"Missing dev split directory: {dev_dir}")

        subject_files = sorted(f for f in os.listdir(dev_dir) if f.endswith("_dev.csv"))
        if subjects is not None:
            subject_whitelist = set(subjects)
            subject_files = [
                f for f in subject_files if f.rsplit("_dev.csv", 1)[0] in subject_whitelist
            ]

        for file_name in subject_files:
            subject_path = os.path.join(dev_dir, file_name)
            df = pd.read_csv(subject_path, header=None)
            indices = list(range(len(df)))
            rng.shuffle(indices)
            for idx in indices[: min(len(indices), max(1, num_samples // len(subject_files) or 1))]:
                row = df.iloc[idx].tolist()
                prompt = format_example(row, include_answer=True)
                self.prompts.append(prompt)
                if len(self.prompts) >= num_samples:
                    return

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]


@dataclass
class LearnableAlphaConfig:
    model_path: str
    data_dir: str
    output_dir: str
    target_compression_ratio: float = 0.4
    device: str = "cuda"
    alpha_init_strategy: str = "uniform"
    similarity_matrix_path: Optional[str] = None
    merge_pairs_path: Optional[str] = None
    num_calibration_samples: int = 1_000
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_training_steps: int = 500
    max_seq_length: int = 512
    random_seed: int = 7


class LearnableAlphaMKA:
    """Coordinator that manages learnable-alpha layer merging."""

    def __init__(self, config: LearnableAlphaConfig) -> None:
        self.config = config
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.logger = logging.getLogger("learnable_alpha")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.logger.addHandler(handler)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
            device_map="auto" if self.config.device == "auto" else None,
        ).to(self.config.device)

        self.transformer_layers, self._layer_parent_attr = self._resolve_transformer_layers()
        self.merge_pairs: List[Tuple[int, int, float]] = []
        self.similarity_scores: Dict[str, float] = {}
        self.learned_alphas: Dict[str, float] = {}

    def _resolve_transformer_layers(self) -> Tuple[nn.ModuleList, str]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers, "model"
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h, "transformer"
        raise ValueError("Unsupported model architecture: cannot find layer stack")

    def load_similarity_matrix(self, path: str) -> np.ndarray:
        if path.endswith(".npy"):
            matrix = np.load(path)
        else:
            with open(path, "rb") as handle:
                matrix = pickle.load(handle)
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Similarity matrix must be a square array")
        return matrix

    def identify_merge_pairs(self) -> List[Tuple[int, int, float]]:
        matrix_path = self.config.similarity_matrix_path
        if matrix_path is None:
            raise ValueError("similarity_matrix_path must be provided to identify pairs")
        matrix = self.load_similarity_matrix(matrix_path)

        if self.config.merge_pairs_path and os.path.exists(self.config.merge_pairs_path):
            with open(self.config.merge_pairs_path, "r", encoding="utf-8") as handle:
                raw_pairs = json.load(handle)
            merge_pairs: List[Tuple[int, int, float]] = []
            for entry in raw_pairs:
                if isinstance(entry, dict):
                    merge_pairs.append(
                        (int(entry["layer_l"]), int(entry["layer_m"]), float(entry["score"]))
                    )
                else:
                    layer_l, layer_m, score = entry
                    merge_pairs.append((int(layer_l), int(layer_m), float(score)))
        else:
            merge_pairs = self._build_pairs_from_similarity(matrix)

        self.merge_pairs = merge_pairs
        self.similarity_scores = {
            f"{l}_{m}": score for l, m, score in self.merge_pairs
        }
        self.logger.info("Identified %d merge pairs", len(self.merge_pairs))
        return self.merge_pairs

    def _build_pairs_from_similarity(self, matrix: np.ndarray) -> List[Tuple[int, int, float]]:
        num_layers = matrix.shape[0]
        target_merges = max(1, int(round(num_layers * self.config.target_compression_ratio)))
        candidates: List[Tuple[float, int, int]] = []
        for i in range(num_layers):
            for j in range(i + 1, num_layers):
                candidates.append((float(matrix[i, j]), i, j))
        candidates.sort(reverse=True)

        selected: List[Tuple[int, int, float]] = []
        used_layers = set()
        for score, layer_i, layer_j in candidates:
            if layer_i in used_layers or layer_j in used_layers:
                continue
            selected.append((layer_i, layer_j, score))
            used_layers.update({layer_i, layer_j})
            if len(selected) >= target_merges:
                break
        if not selected:
            raise RuntimeError("Unable to derive merge pairs from similarity matrix")
        return selected

    def replace_with_mergeable_layers(self) -> None:
        if not self.merge_pairs:
            raise RuntimeError("Call identify_merge_pairs() before replacing layers")

        pairs = sorted(self.merge_pairs, key=lambda p: p[0], reverse=True)
        for layer_l_idx, layer_m_idx, score in pairs:
            if layer_l_idx >= len(self.transformer_layers) or layer_m_idx >= len(self.transformer_layers):
                raise IndexError("Merge pair indices exceed available layers")
            if layer_m_idx == layer_l_idx:
                continue

            layer_l = self.transformer_layers[layer_l_idx]
            layer_m = self.transformer_layers[layer_m_idx]

            alpha_init = self._determine_alpha_init(score)
            mergeable = create_mergeable_layer(layer_l, layer_m, alpha_init=alpha_init)
            setattr(mergeable, "layer_pair", (layer_l_idx, layer_m_idx))
            self.transformer_layers[layer_l_idx] = mergeable
            self.transformer_layers[layer_m_idx] = nn.Identity()
            self.logger.info(
                "Inserted MergeableLayer for (%d, %d) with alpha_init=%.4f",
                layer_l_idx,
                layer_m_idx,
                alpha_init,
            )

    def _determine_alpha_init(self, score: float) -> float:
        strategy = self.config.alpha_init_strategy
        if strategy == "similarity":
            return float(score)
        if strategy == "uniform":
            return 0.5
        if strategy == "fixed_07":
            return 0.7
        raise ValueError(f"Unknown alpha initialisation strategy: {strategy}")

    def prepare_calibration_dataloader(self) -> DataLoader:
        dataset = MMLUCalibrationDataset(
            data_dir=self.config.data_dir,
            num_samples=self.config.num_calibration_samples,
            seed=self.config.random_seed,
        )

        def collate(batch: List[str]) -> Dict[str, torch.Tensor]:
            tokenised = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
            )
            labels = tokenised["input_ids"].clone()
            tokenised["labels"] = labels
            return {k: v.to(self.config.device) for k, v in tokenised.items()}

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate,
        )

    def train_alpha_parameters(self, dataloader: DataLoader) -> None:
        alpha_params: List[torch.nn.Parameter] = []
        for param in self.model.parameters():
            param.requires_grad = False
        for module in self.model.modules():
            if isinstance(module, MergeableLayer):
                for param in module.parameters():
                    if param.requires_grad:
                        alpha_params.append(param)
                        param.requires_grad = True
        if not alpha_params:
            self.logger.warning("No alpha parameters found; skipping training")
            return

        optimizer = Adam(alpha_params, lr=self.config.learning_rate)
        self.model.train()
        losses: List[float] = []
        dataloader_iter = iter(dataloader)

        for step in tqdm(range(self.config.num_training_steps), desc="Training alpha"):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            outputs = self.model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if (step + 1) % 10 == 0:
                avg_loss = float(np.mean(losses[-10:]))
                tqdm.write(f"Step {step + 1}: loss={avg_loss:.4f}")

        if losses:
            self.logger.info("Final average loss: %.4f", float(np.mean(losses[-10:])))

        self.learned_alphas = self._extract_learned_alphas()

    def _extract_learned_alphas(self) -> Dict[str, float]:
        learned: Dict[str, float] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, MergeableLayer):
                pair = getattr(module, "layer_pair", None)
                if pair is not None:
                    key = f"{pair[0]}_{pair[1]}"
                else:
                    key = name
                learned[key] = float(module.alpha.detach().cpu().item())
        return learned

    def bake_alphas_and_fuse(self, output_path: str) -> None:
        os.makedirs(output_path, exist_ok=True)
        layers = self.transformer_layers
        fused_layers: List[nn.Module] = []
        for layer in layers:
            if isinstance(layer, MergeableLayer):
                alpha = float(layer.alpha.detach().cpu().item())
                fused_layer = self._fuse_layer_weights(layer.layer_l, layer.layer_m, alpha)
                fused_layers.append(fused_layer)
            elif isinstance(layer, nn.Identity):
                continue
            else:
                fused_layers.append(layer)

        module_list = nn.ModuleList(fused_layers)
        if self._layer_parent_attr == "model":
            self.model.model.layers = module_list
        else:
            self.model.transformer.h = module_list

        self.model.config.num_hidden_layers = len(fused_layers)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        self.logger.info("Saved fused model with %d layers to %s", len(fused_layers), output_path)

    def _fuse_layer_weights(
        self, layer_l: nn.Module, layer_m: nn.Module, alpha: float
    ) -> nn.Module:
        import copy

        fused_layer = copy.deepcopy(layer_l)
        state_dict = fused_layer.state_dict()
        l_state = layer_l.state_dict()
        m_state = layer_m.state_dict()
        for key in state_dict.keys():
            state_dict[key] = alpha * l_state[key] + (1.0 - alpha) * m_state[key]
        fused_layer.load_state_dict(state_dict)
        return fused_layer

    def plot_alpha_analysis(self, save_path: str) -> Optional[float]:
        if not self.learned_alphas:
            self.logger.warning("No learned alpha values available for plotting")
            return None

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        layer_labels = list(self.learned_alphas.keys())
        alpha_values = [self.learned_alphas[label] for label in layer_labels]
        sim_values = [
            self.similarity_scores.get(label, np.nan) for label in layer_labels
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(range(len(alpha_values)), alpha_values, color="steelblue")
        axes[0].set_xlabel("Merge pair index")
        axes[0].set_ylabel("Learned alpha")
        axes[0].set_title("Distribution of learned alpha values")
        axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1)
        axes[0].axhline(0.7, color="orange", linestyle="--", linewidth=1)

        valid_mask = [not np.isnan(v) for v in sim_values]
        correlation = None
        if any(valid_mask):
            filtered_sim = np.array([v for v, keep in zip(sim_values, valid_mask) if keep])
            filtered_alpha = np.array([a for a, keep in zip(alpha_values, valid_mask) if keep])
            axes[1].scatter(filtered_sim, filtered_alpha, color="darkgreen", alpha=0.7)
            axes[1].set_xlabel("Similarity score")
            axes[1].set_ylabel("Learned alpha")
            axes[1].set_title("Alpha vs similarity")
            if len(filtered_sim) > 1:
                correlation = float(np.corrcoef(filtered_sim, filtered_alpha)[0, 1])
                axes[1].text(
                    0.05,
                    0.95,
                    f"corr={correlation:.3f}",
                    transform=axes[1].transAxes,
                    verticalalignment="top",
                )
        else:
            axes[1].set_visible(False)

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        self.logger.info("Saved alpha analysis plot to %s", save_path)
        return correlation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learnable alpha parameters")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./experiments/learnable_alpha")
    parser.add_argument("--similarity_matrix", type=str, required=True)
    parser.add_argument("--merge_pairs", type=str, default=None)
    parser.add_argument("--target_compression", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--alpha_init_strategy",
        type=str,
        default="uniform",
        choices=["similarity", "uniform", "fixed_07"],
    )
    parser.add_argument("--num_calibration_samples", type=int, default=1_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_training_steps", type=int, default=500)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = LearnableAlphaConfig(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        target_compression_ratio=args.target_compression,
        device=args.device,
        alpha_init_strategy=args.alpha_init_strategy,
        similarity_matrix_path=args.similarity_matrix,
        merge_pairs_path=args.merge_pairs,
        num_calibration_samples=args.num_calibration_samples,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_training_steps=args.num_training_steps,
        max_seq_length=args.max_seq_length,
        random_seed=args.seed,
    )

    runner = LearnableAlphaMKA(config)
    runner.identify_merge_pairs()
    runner.replace_with_mergeable_layers()
    dataloader = runner.prepare_calibration_dataloader()
    runner.train_alpha_parameters(dataloader)

    models_dir = os.path.join(config.output_dir, "models")
    plots_dir = os.path.join(config.output_dir, "plots")
    runner.bake_alphas_and_fuse(models_dir)
    correlation = runner.plot_alpha_analysis(os.path.join(plots_dir, "alpha_analysis.png"))

    results = {
        "merge_pairs": runner.merge_pairs,
        "learned_alphas": runner.learned_alphas,
        "alpha_similarity_correlation": correlation,
    }
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "learnable_alpha_metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
