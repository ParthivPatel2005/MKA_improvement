# Repository Change Log: Learnable Alpha Integration

This document summarises the modifications introduced to extend the original MKA
pipeline with learnable alpha capabilities and accompanying experiment tooling.

## Added Modules and Scripts

- `mergeable_layer.py`: Implements `MergeableLayer` and `MLPMergeableLayer` wrappers
  that freeze two transformer blocks and expose a trainable mixing coefficient.
- `train_learnable_alpha.py`: Orchestrates similarity loading, layer replacement,
  alpha calibration, and fused model export. Provides a reusable
  `LearnableAlphaMKA` class plus CLI entry point.
- `evaluate_methods.py`: Benchmarks four strategies (similarity heuristic, fixed 0.5,
  fixed 0.7, learnable alpha) end-to-end on MMLU.
- `plot_results.py`: Generates accuracy bar plots, learned-alpha histogram, and
  alpha-versus-similarity scatter visualisations.
- `run_experiment.sh`: Shell harness to execute the evaluation suite and plotting
  pipeline with configurable hyperparameters.

## Core Pipeline Enhancements (`pipeline.py`)

- Imported `mergeable_layer` utilities; added learnable-alpha CLI flags
  (`--use_learnable_alpha`, `--use_mlp_merge`, calibration controls, alpha hyperparameters).
- Added calibration dataset/dataloader helpers, mergeable-layer replacement logic,
  alpha training routine, learned-alpha extraction, and permanent fusion helpers.
- Branch main execution to support both original similarity-based merging and the
  new learnable-alpha path; persists learned coefficients to
  `merged_weights/learned_alphas.json`.
- **Two distinct modes**: scalar α (trainable parameter) and MLP α (dynamic prediction).

### New/Updated Helper Functions

- `_CalibrationDataset`: lightweight dataset class that samples calibration prompts
  from MMLU dev CSVs for alpha training.
- `prepare_calibration_dataloader`: wraps `_CalibrationDataset` with tokenizer
  collate function, yielding batches for fine-tuning `alpha` parameters.
- `replace_layers_with_mergeable`: swaps selected transformer blocks with
  `MergeableLayer` or `MLPMergeableLayer` wrappers based on `use_mlp` flag.
- `train_alpha_parameters`: trains only the mergeable-layer coefficients using the
  calibration dataloader (works for both scalar and MLP modes).
- `extract_learned_alphas`: collects the learned alpha values for reporting (marks
  MLP layers with -1.0 indicator).
- `_fuse_layers`: blends two frozen blocks into a single fused block using the
  learned coefficient.
- `fuse_mergeable_layers`: replaces mergeable wrappers with permanently fused
  layers and updates the model configuration (MLP layers fused with 0.5 ratio).

## Documentation & Dependencies

- Expanded `README.md` overview with learnable-alpha workflow description and
  noted scaffolded directories for model checkpoints, data, and similarity matrix.
- Introduced `requirements.txt` capturing new plotting/analysis dependencies
  (`matplotlib`, `seaborn`, `datasets`, `scipy`, `jupyter`, `tensorboard`) and
  version pins for `transformers`/`huggingface-hub` compatibility.
- Added `changes.md` (this file) to track repository evolution.

## Repository Structure & Scaffolding

- Created local directories under `MKA_improvement/` for
  `models/meta-llama/Meta-Llama-3-8B/`, `experiments/`, and placeholder
  `similarity_matrix.pkl` to simplify CLI usage.

## Authentication Adjustments

- Ensured Hugging Face hub credentials are handled externally (removed inline
  token) and documented login steps for accessing gated models.

## Operational Notes

- Evaluation and training scripts assume MMLU CSVs reside in `data/dev` and
  `data/test`, a valid similarity matrix is provided, and sufficient GPU memory is
  available for Meta-Llama checkpoints.
