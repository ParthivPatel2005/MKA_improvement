# Learnable Alpha MKA Implementation Guide

## Overview

This implementation extends the MKA (Manifold-Based Knowledge Alignment) method with **learnable alpha parameters** to find optimal layer merging ratios, rather than using heuristic similarity scores.

## Key Insight

The paper shows that using α = S_lm (similarity score) outperforms fixed values like 0.5 or 0.7. However, **they never proved this is optimal**. Our approach treats α as a **learnable parameter** to discover the true optimal merging ratio through gradient descent.

---

## Implementation Structure

### 1. MergeableLayer Module (`mergeable_layer.py`)

```python
import torch
import torch.nn as nn

class MergeableLayer(nn.Module):
    """
    A module that combines two transformer layers with a learnable mixing coefficient.
    
    Forward pass: output = alpha * layer_l(x) + (1 - alpha) * layer_m(x)
    
    Args:
        layer_l: First transformer layer (will be frozen)
        layer_m: Second transformer layer (will be frozen)
        alpha_init: Initial value for alpha (default: 0.5, or use S_lm from MKA)
    """
    def __init__(self, layer_l, layer_m, alpha_init=0.5):
        super().__init__()
        
        # Store the two layers (frozen)
        self.layer_l = layer_l
        self.layer_m = layer_m
        
        # Freeze all parameters in both layers
        for param in self.layer_l.parameters():
            param.requires_grad = False
        for param in self.layer_m.parameters():
            param.requires_grad = False
        
        # Learnable alpha parameter (initialized to alpha_init)
        # Use logit parameterization to ensure alpha stays in [0, 1]
        logit_init = torch.logit(torch.tensor(alpha_init))
        self.alpha_logit = nn.Parameter(logit_init)
    
    @property
    def alpha(self):
        """Get alpha value through sigmoid to ensure [0, 1] range"""
        return torch.sigmoid(self.alpha_logit)
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """
        Forward pass that blends outputs from both layers
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask for the layers
            **kwargs: Additional arguments passed to the layers
        
        Returns:
            Blended output: alpha * output_l + (1 - alpha) * output_m
        """
        # Get alpha value
        alpha = self.alpha
        
        # Forward through both layers
        output_l = self.layer_l(hidden_states, attention_mask=attention_mask, **kwargs)
        output_m = self.layer_m(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Handle tuple outputs (some layers return (hidden_states, attention_weights))
        if isinstance(output_l, tuple):
            hidden_l = output_l[0]
            hidden_m = output_m[0]
            blended_hidden = alpha * hidden_l + (1 - alpha) * hidden_m
            return (blended_hidden,) + output_l[1:]  # Keep other outputs from layer_l
        else:
            return alpha * output_l + (1 - alpha) * output_m


class MLPMergeableLayer(nn.Module):
    """
    Advanced version: Uses a small MLP to compute merged weights
    instead of linear interpolation
    
    Forward: output = MLP([features_l, features_m])
    where MLP learns complex merging patterns
    """
    def __init__(self, layer_l, layer_m, hidden_dim=64):
        super().__init__()
        
        self.layer_l = layer_l
        self.layer_m = layer_m
        
        # Freeze original layers
        for param in self.layer_l.parameters():
            param.requires_grad = False
        for param in self.layer_m.parameters():
            param.requires_grad = False
        
        # Small MLP to compute merging weights
        # Input: concatenated statistics from both outputs
        self.merge_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output alpha in [0, 1]
        )
    
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Forward through both layers
        output_l = self.layer_l(hidden_states, attention_mask=attention_mask, **kwargs)
        output_m = self.layer_m(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Extract hidden states
        if isinstance(output_l, tuple):
            hidden_l = output_l[0]
            hidden_m = output_m[0]
        else:
            hidden_l = output_l
            hidden_m = output_m
        
        # Compute statistics for MLP input
        stats_l = hidden_l.mean()
        stats_m = hidden_m.mean()
        mlp_input = torch.stack([stats_l, stats_m])
        
        # Get alpha from MLP
        alpha = self.merge_mlp(mlp_input)
        
        # Blend outputs
        blended_hidden = alpha * hidden_l + (1 - alpha) * hidden_m
        
        if isinstance(output_l, tuple):
            return (blended_hidden,) + output_l[1:]
        else:
            return blended_hidden
```

---

### 2. Modified Pipeline (`train_learnable_alpha.py`)

```python
import torch
import torch.nn as nn
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import json
import pickle

from mergeable_layer import MergeableLayer


class LearnableAlphaMKA:
    """
    Enhanced MKA with learnable alpha parameters
    
    Pipeline:
    1. Run original MKA to identify layer pairs
    2. Replace pairs with MergeableLayer modules
    3. Train only alpha parameters
    4. Bake learned alphas and fuse layers
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        target_compression_ratio: float = 0.40
    ):
        self.model_path = model_path
        self.device = device
        self.target_compression_ratio = target_compression_ratio
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Storage for merge pairs and learned alphas
        self.merge_pairs = []
        self.learned_alphas = {}
        self.similarity_scores = {}
    
    def identify_merge_pairs(self, calibration_data) -> List[Tuple[int, int, float]]:
        """
        Step 1: Run MKA algorithm to identify which layers to merge
        
        Returns:
            List of (layer_l_idx, layer_m_idx, similarity_score)
        """
        print("Step 1: Running MKA to identify merge pairs...")
        
        # This calls the original MKA pipeline
        # Extract activations, compute manifolds, calculate similarities
        from original_mka_pipeline import run_mka_analysis
        
        merge_pairs = run_mka_analysis(
            model=self.model,
            data=calibration_data,
            target_ratio=self.target_compression_ratio
        )
        
        self.merge_pairs = merge_pairs
        
        for layer_l, layer_m, sim_score in merge_pairs:
            self.similarity_scores[f"{layer_l}_{layer_m}"] = sim_score
        
        print(f"Identified {len(merge_pairs)} layer pairs to merge")
        return merge_pairs
    
    def replace_with_mergeable_layers(
        self,
        alpha_init_strategy: str = "similarity"  # "similarity", "uniform", or "fixed"
    ):
        """
        Step 2: Replace identified layer pairs with MergeableLayer modules
        
        Args:
            alpha_init_strategy: How to initialize alpha
                - "similarity": Use S_lm from MKA (paper's heuristic)
                - "uniform": Use 0.5
                - "fixed": Use a fixed value like 0.7
        """
        print("Step 2: Replacing layer pairs with MergeableLayer modules...")
        
        # Access the transformer layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
        
        # Sort merge pairs in reverse order to avoid index issues
        sorted_pairs = sorted(self.merge_pairs, key=lambda x: x[0], reverse=True)
        
        for layer_l_idx, layer_m_idx, sim_score in sorted_pairs:
            # Determine alpha initialization
            if alpha_init_strategy == "similarity":
                alpha_init = sim_score
            elif alpha_init_strategy == "uniform":
                alpha_init = 0.5
            elif alpha_init_strategy == "fixed":
                alpha_init = 0.7
            else:
                raise ValueError(f"Unknown strategy: {alpha_init_strategy}")
            
            # Create MergeableLayer
            layer_l = layers[layer_l_idx]
            layer_m = layers[layer_m_idx]
            
            mergeable = MergeableLayer(layer_l, layer_m, alpha_init=alpha_init)
            
            # Replace layer_l with mergeable layer
            layers[layer_l_idx] = mergeable
            
            # Remove layer_m (will be handled by merging later)
            # For now, just mark it
            layers[layer_m_idx] = nn.Identity()
        
        print(f"Replaced {len(sorted_pairs)} layer pairs with MergeableLayer modules")
    
    def train_alpha_parameters(
        self,
        calibration_dataset,
        num_steps: int = 500,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        log_interval: int = 50
    ):
        """
        Step 3: Train only the alpha parameters on calibration data
        
        Args:
            calibration_dataset: Dataset for training (small subset)
            num_steps: Number of training steps
            batch_size: Batch size for training
            learning_rate: Learning rate for Adam optimizer
        """
        print("Step 3: Training alpha parameters...")
        
        # Collect only alpha parameters
        alpha_params = []
        for name, param in self.model.named_parameters():
            if 'alpha_logit' in name:
                param.requires_grad = True
                alpha_params.append(param)
            else:
                param.requires_grad = False
        
        print(f"Training {len(alpha_params)} alpha parameters")
        
        # Setup optimizer
        optimizer = Adam(alpha_params, lr=learning_rate)
        
        # Training loop
        self.model.train()
        losses = []
        
        for step in tqdm(range(num_steps), desc="Training alpha"):
            # Sample batch
            batch = calibration_dataset.sample(batch_size)
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Forward pass (language modeling)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            # Logging
            if (step + 1) % log_interval == 0:
                avg_loss = np.mean(losses[-log_interval:])
                print(f"Step {step+1}/{num_steps}, Loss: {avg_loss:.4f}")
        
        print("Training completed!")
        
        # Extract learned alpha values
        self._extract_learned_alphas()
    
    def _extract_learned_alphas(self):
        """Extract learned alpha values from MergeableLayer modules"""
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
        
        for idx, layer in enumerate(layers):
            if isinstance(layer, MergeableLayer):
                alpha_value = layer.alpha.item()
                self.learned_alphas[f"layer_{idx}"] = alpha_value
                print(f"Layer {idx}: Learned alpha = {alpha_value:.4f}")
    
    def bake_alphas_and_fuse(self, output_path: str):
        """
        Step 4: Use learned alphas to permanently fuse layers
        
        Args:
            output_path: Path to save the fused model
        """
        print("Step 4: Baking learned alphas and fusing layers...")
        
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        else:
            layers = self.model.layers
        
        new_layers = []
        skip_next = set()
        
        for idx, layer in enumerate(layers):
            if idx in skip_next:
                continue
            
            if isinstance(layer, MergeableLayer):
                # Get learned alpha
                alpha = layer.alpha.item()
                
                # Fuse the two layers' weights
                fused_layer = self._fuse_layer_weights(
                    layer.layer_l,
                    layer.layer_m,
                    alpha
                )
                
                new_layers.append(fused_layer)
                
                print(f"Fused layers with alpha={alpha:.4f}")
            elif not isinstance(layer, nn.Identity):
                new_layers.append(layer)
        
        # Replace layers in model
        if hasattr(self.model, 'model'):
            self.model.model.layers = nn.ModuleList(new_layers)
        else:
            self.model.layers = nn.ModuleList(new_layers)
        
        # Update config
        self.model.config.num_hidden_layers = len(new_layers)
        
        # Save model
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        print(f"Saved fused model to {output_path}")
        print(f"Original layers: {len(layers)}, Fused layers: {len(new_layers)}")
    
    def _fuse_layer_weights(self, layer_l, layer_m, alpha):
        """
        Fuse two layers using learned alpha
        
        θ_fused = alpha * θ_l + (1 - alpha) * θ_m
        """
        import copy
        fused_layer = copy.deepcopy(layer_l)
        
        # Fuse all parameters
        for (name_l, param_l), (name_m, param_m) in zip(
            layer_l.named_parameters(),
            layer_m.named_parameters()
        ):
            # Get corresponding parameter in fused layer
            fused_param = dict(fused_layer.named_parameters())[name_l]
            
            # Compute weighted combination
            fused_param.data = alpha * param_l.data + (1 - alpha) * param_m.data
        
        return fused_layer
    
    def plot_alpha_analysis(self, save_path: str = "alpha_analysis.png"):
        """
        Plot learned alpha values vs MKA similarity scores
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract data
        layer_names = list(self.learned_alphas.keys())
        alpha_values = [self.learned_alphas[k] for k in layer_names]
        
        # Plot 1: Learned alpha values
        axes[0].bar(range(len(alpha_values)), alpha_values, color='steelblue')
        axes[0].axhline(y=0.5, color='red', linestyle='--', label='α=0.5')
        axes[0].axhline(y=0.7, color='orange', linestyle='--', label='α=0.7')
        axes[0].set_xlabel('Layer Pair Index')
        axes[0].set_ylabel('Learned Alpha Value')
        axes[0].set_title('Learned Alpha Values for Each Layer Pair')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Correlation with similarity scores
        sim_values = [self.similarity_scores.get(k.replace('layer_', ''), 0.5) 
                     for k in layer_names]
        
        axes[1].scatter(sim_values, alpha_values, color='darkgreen', s=100, alpha=0.6)
        axes[1].plot([0, 1], [0, 1], 'r--', label='α = S_lm (MKA heuristic)')
        axes[1].set_xlabel('MKA Similarity Score (S_lm)')
        axes[1].set_ylabel('Learned Alpha Value')
        axes[1].set_title('Learned Alpha vs MKA Similarity Score')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Compute correlation
        correlation = np.corrcoef(sim_values, alpha_values)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                    transform=axes[1].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved alpha analysis plot to {save_path}")
        
        return correlation
```

---

### 3. Evaluation Script (`evaluate_methods.py`)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import json

from train_learnable_alpha import LearnableAlphaMKA


def evaluate_mmlu(model, tokenizer, num_samples=None):
    """
    Evaluate model on MMLU benchmark
    
    Returns:
        Average accuracy across all subjects
    """
    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all", split="test")
    
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for example in tqdm(dataset, desc="Evaluating MMLU"):
            # Format question
            question = example['question']
            choices = example['choices']
            answer = example['answer']
            
            # Create prompt
            prompt = f"Question: {question}\\n"
            for i, choice in enumerate(choices):
                prompt += f"{chr(65+i)}. {choice}\\n"
            prompt += "Answer:"
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Get logits for A, B, C, D tokens
            outputs = model(**inputs)
            logits = outputs.logits[0, -1]
            
            # Get probabilities for answer tokens
            token_ids = [tokenizer.encode(chr(65+i), add_special_tokens=False)[0] 
                        for i in range(len(choices))]
            probs = torch.softmax(logits[token_ids], dim=0)
            
            # Predict
            predicted = torch.argmax(probs).item()
            
            if predicted == answer:
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    return accuracy


def run_baseline_experiments(model_path: str, calibration_data, test_data):
    """
    Run all baseline experiments and our method
    
    Baselines:
    1. Original MKA with α = S_lm
    2. Fixed α = 0.5
    3. Fixed α = 0.7
    4. Learned α (our method)
    """
    results = {}
    
    # Baseline 1: Original MKA (α = S_lm)
    print("\\n" + "="*60)
    print("Baseline 1: Original MKA (α = S_lm)")
    print("="*60)
    
    mka_original = LearnableAlphaMKA(model_path)
    mka_original.identify_merge_pairs(calibration_data)
    mka_original.replace_with_mergeable_layers(alpha_init_strategy="similarity")
    mka_original.bake_alphas_and_fuse("./models/mka_original")
    
    model_orig = AutoModelForCausalLM.from_pretrained("./models/mka_original")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    acc_orig = evaluate_mmlu(model_orig, tokenizer, test_data)
    results['mka_original'] = acc_orig
    print(f"Original MKA Accuracy: {acc_orig:.2f}%")
    
    # Baseline 2: Fixed α = 0.5
    print("\\n" + "="*60)
    print("Baseline 2: Fixed α = 0.5")
    print("="*60)
    
    mka_fixed_05 = LearnableAlphaMKA(model_path)
    mka_fixed_05.identify_merge_pairs(calibration_data)
    mka_fixed_05.replace_with_mergeable_layers(alpha_init_strategy="uniform")
    mka_fixed_05.bake_alphas_and_fuse("./models/mka_fixed_05")
    
    model_05 = AutoModelForCausalLM.from_pretrained("./models/mka_fixed_05")
    acc_05 = evaluate_mmlu(model_05, tokenizer, test_data)
    results['fixed_05'] = acc_05
    print(f"Fixed α=0.5 Accuracy: {acc_05:.2f}%")
    
    # Baseline 3: Fixed α = 0.7
    print("\\n" + "="*60)
    print("Baseline 3: Fixed α = 0.7")
    print("="*60)
    
    mka_fixed_07 = LearnableAlphaMKA(model_path)
    mka_fixed_07.identify_merge_pairs(calibration_data)
    mka_fixed_07.replace_with_mergeable_layers(alpha_init_strategy="fixed")
    mka_fixed_07.bake_alphas_and_fuse("./models/mka_fixed_07")
    
    model_07 = AutoModelForCausalLM.from_pretrained("./models/mka_fixed_07")
    acc_07 = evaluate_mmlu(model_07, tokenizer, test_data)
    results['fixed_07'] = acc_07
    print(f"Fixed α=0.7 Accuracy: {acc_07:.2f}%")
    
    # Our Method: Learned α
    print("\\n" + "="*60)
    print("Our Method: Learned α")
    print("="*60)
    
    mka_learned = LearnableAlphaMKA(model_path)
    mka_learned.identify_merge_pairs(calibration_data)
    mka_learned.replace_with_mergeable_layers(alpha_init_strategy="uniform")
    mka_learned.train_alpha_parameters(
        calibration_dataset=calibration_data,
        num_steps=500,
        batch_size=4,
        learning_rate=1e-4
    )
    mka_learned.bake_alphas_and_fuse("./models/mka_learned")
    
    model_learned = AutoModelForCausalLM.from_pretrained("./models/mka_learned")
    acc_learned = evaluate_mmlu(model_learned, tokenizer, test_data)
    results['learned'] = acc_learned
    print(f"Learned α Accuracy: {acc_learned:.2f}%")
    
    # Plot alpha analysis
    correlation = mka_learned.plot_alpha_analysis("alpha_correlation.png")
    results['alpha_correlation'] = correlation
    
    # Save results
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for method, acc in results.items():
        if method != 'alpha_correlation':
            print(f"{method:20s}: {acc:.2f}%")
    print(f"\\nAlpha-Similarity Correlation: {correlation:.3f}")
    
    return results


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "meta-llama/Llama-3-8B"
    TARGET_COMPRESSION = 0.40
    
    # Prepare data
    # Calibration: small subset for training alpha
    # Test: MMLU test set
    calibration_data = load_calibration_data(num_samples=1000)
    test_data = load_mmlu_test()
    
    # Run experiments
    results = run_baseline_experiments(
        model_path=MODEL_PATH,
        calibration_data=calibration_data,
        test_data=test_data
    )
```

---

### 4. Main Execution Script (`run_experiment.sh`)

```bash
#!/bin/bash

# Setup environment
conda activate manifold_compression

# Set paths
MODEL_PATH="meta-llama/Meta-Llama-3-8B"
DATA_DIR="./data/mmlu"
OUTPUT_DIR="./experiments/learnable_alpha"
TARGET_COMPRESSION=0.40

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/models
mkdir -p $OUTPUT_DIR/plots

# Step 1: Run baseline experiments
echo "Running baseline experiments..."
python evaluate_methods.py \\
    --model_path $MODEL_PATH \\
    --data_dir $DATA_DIR \\
    --output_dir $OUTPUT_DIR \\
    --target_compression $TARGET_COMPRESSION \\
    --num_calibration_samples 1000 \\
    --num_training_steps 500 \\
    --learning_rate 1e-4 \\
    --batch_size 4

# Step 2: Generate comparison plots
echo "Generating comparison plots..."
python plot_results.py \\
    --results_file $OUTPUT_DIR/results.json \\
    --output_dir $OUTPUT_DIR/plots

# Step 3: Run ablation studies
echo "Running ablation studies..."
# Test different numbers of training steps
for steps in 100 200 500 1000; do
    echo "Testing with $steps training steps..."
    python evaluate_methods.py \\
        --model_path $MODEL_PATH \\
        --data_dir $DATA_DIR \\
        --output_dir $OUTPUT_DIR/ablation_steps_$steps \\
        --num_training_steps $steps \\
        --tag "steps_$steps"
done

# Test different learning rates
for lr in 1e-5 5e-5 1e-4 5e-4; do
    echo "Testing with learning rate $lr..."
    python evaluate_methods.py \\
        --model_path $MODEL_PATH \\
        --data_dir $DATA_DIR \\
        --output_dir $OUTPUT_DIR/ablation_lr_$lr \\
        --learning_rate $lr \\
        --tag "lr_$lr"
done

echo "Experiments completed! Results saved to $OUTPUT_DIR"
```

---

## Expected Results

Based on the paper's findings, we expect:

1. **Original MKA (α = S_lm)**: ~63.47% MMLU accuracy at 40% compression
2. **Fixed α = 0.5**: ~61.45% MMLU accuracy 
3. **Fixed α = 0.7**: ~61.84% MMLU accuracy
4. **Learned α (Our Method)**: **~64-65% MMLU accuracy** (hypothesis: should outperform all baselines)

### Key Insights to Validate:

- **Do learned α values correlate with S_lm scores?** 
  - If yes: MKA heuristic is a good proxy
  - If no: Learning reveals better merging patterns

- **Do learned α values cluster near specific values?**
  - Might reveal optimal fixed α that works across layers

- **Does training converge quickly?**
  - If yes: Few gradient steps sufficient
  - If no: May need more sophisticated training

---

## Modifications to Original Repository

### Files to Add:

1. `mergeable_layer.py` - New MergeableLayer module
2. `train_learnable_alpha.py` - Training pipeline for alpha
3. `evaluate_methods.py` - Comprehensive evaluation script
4. `plot_results.py` - Visualization utilities

### Files to Modify:

1. `pipeline.py` - Add option to use learnable alpha mode
2. `requirements.txt` - Add dependencies (matplotlib, datasets)

### Integration with Existing Code:

The existing MKA pipeline (`pipeline.py`) handles:
- Activation extraction
- Manifold learning (diffusion kernels)
- Similarity computation

Our modifications **extend** this by adding a training stage after similarity computation, before final fusion.

---

## Usage Example

```bash
# 1. Clone and setup
git clone https://github.com/SempraETY/Pruning-via-Merging.git
cd Pruning-via-Merging

# 2. Add new files
cp /path/to/mergeable_layer.py .
cp /path/to/train_learnable_alpha.py .
cp /path/to/evaluate_methods.py .

# 3. Install dependencies
pip install datasets matplotlib scipy

# 4. Run experiments
bash run_experiment.sh
```

---

## Advanced Extension: MLP-based Merging

For even better performance, replace linear interpolation with an MLP:

```python
# In train_learnable_alpha.py, replace:
mergeable = MergeableLayer(layer_l, layer_m, alpha_init)

# With:
mergeable = MLPMergeableLayer(layer_l, layer_m, hidden_dim=64)
```

This allows the model to learn **non-linear merging patterns** beyond simple weighted averages.

---

## Citation

If you use this code, please cite both the original MKA paper and acknowledge this extension:

```bibtex
@article{liu2024pruning,
  title={Pruning via Merging: Compressing LLMs via Manifold Alignment Based Layer Merging},
  author={Liu, Deyuan and Qin, Zhanyue and Wang, Hairu and others},
  journal={arXiv preprint arXiv:2406.16330},
  year={2024}
}
```

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size or use gradient checkpointing

```python
model.gradient_checkpointing_enable()
```

### Issue: Alpha values not converging
**Solution**: Try different learning rates or initialization strategies

### Issue: Learned alpha stuck at 0.5
**Solution**: Check if gradients are flowing properly; may need to adjust logit initialization

---

## Contact

For questions or issues with this implementation, please open an issue on GitHub or contact the authors of the original MKA paper.
