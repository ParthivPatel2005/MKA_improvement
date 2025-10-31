# Pruning-via-Merging-Compressing-LLMs-via-Manifold-Alignment-Based-Layer-Merging

## Overview

This project provides a comprehensive pipeline for **fusing layers in pre-trained causal language models** using manifold learning techniques. Layer fusion aims to reduce the number of layers in a model by intelligently combining similar layers, potentially leading to more efficient models without significant loss in performance.

### Learnable Alpha Extension

This project extends the original MKA method with **two novel approaches**:

#### 1. Learnable Scalar Alpha (Trainable Parameter)
- **What**: Treats α as a single trainable scalar per layer pair
- **How**: Freezes both layers, trains only α via gradient descent
- **Usage**: `--use_learnable_alpha` flag
- **Command**:
  ```bash
  python pipeline.py --model_path "meta-llama/Meta-Llama-3-8B" \
    --num_layer 14 --data_dir "./data" --use_learnable_alpha \
    --alpha_training_steps 500 --alpha_learning_rate 1e-4
  ```

#### 2. MLP-Based Dynamic Merging
- **What**: Small MLP predicts α dynamically based on activation statistics
- **How**: MLP takes (mean, std, min, max) and outputs α ∈ [0,1]
- **Usage**: Add `--use_mlp_merge` flag
- **Command**:
  ```bash
  python pipeline.py --model_path "meta-llama/Meta-Llama-3-8B" \
    --num_layer 14 --data_dir "./data" --use_learnable_alpha --use_mlp_merge \
    --alpha_training_steps 500 --alpha_learning_rate 1e-4
  ```

#### Comparison Suite
- **Original MKA** (α = S_lm), **Fixed α = 0.5**, **Fixed α = 0.7**
- **Learned Scalar α**, **Learned MLP α** (use `--include_mlp` in evaluate_methods.py)

- `mergeable_layer.py` implements both variants
- `train_learnable_alpha.py` automates training pipeline
- `evaluate_methods.py` benchmarks all methods
- `plot_results.py` visualises results

The pipeline involves:

1. **Model Evaluation:** Assessing the model's performance on multiple-choice datasets.
2. **Activation Extraction:** Capturing activations from each model layer for analysis.
3. **Manifold Learning:** Applying diffusion kernels to embed high-dimensional activations into lower-dimensional spaces.
4. **Similarity Computation:** Calculating similarities between layers based on mutual information and entropy estimates.
5. **Layer Fusion:** Combining layers with high similarity to create a more compact model.
6. **Saving the Fused Model:** Exporting the modified model for future use.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/layer-fusion](https://github.com/SempraETY/Pruning-via-Merging-Compressing-LLMs-via-Manifold-Alignment-Based-Layer-Merging.git
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   conda create -n maniflod_compression python=3.9 -y
   conda activate maniflod_compression
   ```

3. **Install Dependencies:**

   ```bash
   pip install torch transformers numpy pandas scikit-learn tqdm
   ```

## Data Format

The pipeline expects MMLU data

## Usage

Run the main script with appropriate command-line arguments to perform layer fusion.

### Command-Line Arguments:

- `--ntrain`, `-k`: **(int, default=5)** Number of training examples to include in prompts.
- `--ngpu`, `-g`: **(int, default=4)** Number of GPUs to use.
- `--model_path`: **(str, default="/data/yangzhao/point/baichuan/Meta-Llama-3-70B")** Path to the pre-trained model.
- `--num_tasks`, `-n`: **(int, default=57)** Number of MMLU tasks to process.
- `--num_samples`, `-m`: **(int, default=1)** Number of samples per task.
- `--data_dir`, `-d`: **(str, default="data")** Directory containing the data.
- `--num_layer`, `-i`: **(int, default=1)** Number of layers to fuse.

### Example Command:

```bash
python pipeline.py --ntrain 10 --ngpu 2 --model_path "/path/to/model" --num_tasks 50 --num_samples 5 --data_dir "./data" --num_layer 2
```

## Process Workflow

1. **Initialization:**
   - Parses command-line arguments.
   - Sets up directories for embeddings, fusion information, and merged weights.
   - Configures logging and sets random seeds for reproducibility.

2. **Model and Tokenizer Loading:**
   - Loads the tokenizer and pre-trained causal language model from the specified `model_path`.

3. **Data Processing:**
   - Iterates through each subject's test data.
   - For each selected sample, formats the prompt and extracts activations from all model layers.

4. **Manifold Learning:**
   - Applies a diffusion kernel to the stacked activations of each layer to obtain lower-dimensional embeddings.

5. **Similarity Computation:**
   - Calculates a similarity matrix based on normalized pointwise information bottleneck (NPIB) between layers.

6. **Layer Fusion:**
   - Identifies pairs of layers with high similarity.
   - Fuses specified layers by blending their weights based on computed fusion ratios.
   - Updates the model configuration and removes fused layers.

7. **Saving the Fused Model:**
   - Saves the modified model's state dictionary and configuration to the designated directory.

## Outputs

- **Embeddings Directory:** Contains pickled files of embedded activations for each layer.
- **Fusion Info Directory:** Stores logs detailing the fusion process.
- **Merged Weights Directory:** Holds the fused model's configuration and the `pytorch_model.bin` file.
- Note: To obtain the final weight, you also need to update the weight index configuration file. If you do not update the configuration file, the weights that do not exist in the index will be randomly initialized, causing problems

**Example Structure:**

```
EMNLP2024/
└── layer_fus/
    └── al/
        └── Meta-Llama-3-70B/
            └── fused_2_layers/
                └── iteration/
                    ├── embeddings/
                    │   ├── layer_0_embedded.pkl
                    │   ├── layer_1_embedded.pkl
                    │   └── ...
                    ├── fusion_info/
                    │   └── experiment.log
                    └── merged_weights/
                        ├── config.json
                        └── pytorch_model.bin
```
