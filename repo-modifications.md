"""
Repository Structure Modifications for Learnable Alpha MKA

This file describes the exact changes needed to the existing repository:
https://github.com/SempraETY/Pruning-via-Merging
"""
```
ORIGINAL_REPOSITORY_STRUCTURE = """
Pruning-via-Merging/
├── pipeline.py              # Main MKA pipeline
├── README.md                # Original documentation
├── requirements.txt         # Dependencies
└── data/                    # MMLU data directory
"""
```

```
MODIFIED_REPOSITORY_STRUCTURE = """
Pruning-via-Merging/
├── pipeline.py              # [MODIFY] Add learnable alpha mode
├── mergeable_layer.py       # [NEW] MergeableLayer implementation
├── train_learnable_alpha.py # [NEW] Training pipeline
├── evaluate_methods.py      # [NEW] Evaluation script
├── plot_results.py          # [NEW] Plotting utilities
├── run_experiment.sh        # [NEW] Bash script to run all experiments
├── README.md                # [MODIFY] Update with new features
├── requirements.txt         # [MODIFY] Add new dependencies
├── data/                    # MMLU data
└── experiments/             # [NEW] Output directory
    ├── models/              # Saved models
    ├── plots/               # Generated plots
    └── results/             # JSON results
"""
```

# =============================================================================
# FILE 1: mergeable_layer.py (NEW FILE)
# =============================================================================

MERGEABLE_LAYER_CODE = '''
"""
Mergeable Layer Module for Learnable Alpha MKA

This module implements layer merging with learnable alpha parameters.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MergeableLayer(nn.Module):
    """
    Combines two transformer layers with a learnable mixing coefficient.
    
    Forward: output = alpha * layer_l(x) + (1 - alpha) * layer_m(x)
    
    Args:
        layer_l: First transformer layer (frozen)
        layer_m: Second transformer layer (frozen)
        alpha_init: Initial alpha value (0 to 1)
        use_logit_param: If True, uses logit parameterization for alpha
    """
    
    def __init__(
        self, 
        layer_l: nn.Module, 
        layer_m: nn.Module, 
        alpha_init: float = 0.5,
        use_logit_param: bool = True
    ):
        super().__init__()
        
        self.layer_l = layer_l
        self.layer_m = layer_m
        self.use_logit_param = use_logit_param
        
        # Freeze both layers
        for param in self.layer_l.parameters():
            param.requires_grad = False
        for param in self.layer_m.parameters():
            param.requires_grad = False
        
        # Initialize alpha parameter
        if use_logit_param:
            # Logit parameterization: alpha = sigmoid(logit)
            # This ensures alpha stays in (0, 1)
            logit_init = torch.logit(torch.tensor(alpha_init, dtype=torch.float32))
            self.alpha_logit = nn.Parameter(logit_init)
        else:
            # Direct parameterization with clamping
            self.alpha_raw = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
    
    @property
    def alpha(self) -> torch.Tensor:
        """Get current alpha value"""
        if self.use_logit_param:
            return torch.sigmoid(self.alpha_logit)
        else:
            return torch.clamp(self.alpha_raw, 0.0, 1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ):
        """
        Forward pass through mergeable layer
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim]
            attention_mask: Attention mask
            Other args: Passed to both layers
            
        Returns:
            Weighted combination of both layer outputs
        """
        # Get current alpha
        alpha = self.alpha
        
        # Forward through layer_l
        output_l = self.layer_l(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        # Forward through layer_m
        output_m = self.layer_m(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs
        )
        
        # Handle different output formats
        if isinstance(output_l, tuple):
            # Output format: (hidden_states, present_key_value, attention_weights, ...)
            hidden_l = output_l[0]
            hidden_m = output_m[0]
            
            # Blend hidden states
            blended_hidden = alpha * hidden_l + (1 - alpha) * hidden_m
            
            # Keep other outputs from layer_l
            return (blended_hidden,) + output_l[1:]
        else:
            # Simple tensor output
            return alpha * output_l + (1 - alpha) * output_m
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return f'alpha={self.alpha.item():.4f}'


class MLPMergeableLayer(nn.Module):
    """
    Advanced mergeable layer using MLP to compute mixing weights
    
    Instead of learning a single alpha, learns a small MLP that computes
    alpha based on input statistics.
    """
    
    def __init__(
        self,
        layer_l: nn.Module,
        layer_m: nn.Module,
        hidden_dim: int = 64,
        input_features: int = 4  # mean, std, min, max
    ):
        super().__init__()
        
        self.layer_l = layer_l
        self.layer_m = layer_m
        
        # Freeze layers
        for param in self.layer_l.parameters():
            param.requires_grad = False
        for param in self.layer_m.parameters():
            param.requires_grad = False
        
        # MLP to compute alpha from input statistics
        self.alpha_mlp = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def compute_statistics(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute input statistics for MLP"""
        # Flatten to [batch * seq_len, hidden_dim]
        flat = hidden_states.view(-1, hidden_states.size(-1))
        
        # Compute statistics
        mean = flat.mean(dim=0).mean()
        std = flat.std(dim=0).mean()
        min_val = flat.min()
        max_val = flat.max()
        
        # Stack into feature vector
        features = torch.stack([mean, std, min_val, max_val])
        return features
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass with MLP-computed alpha"""
        
        # Compute input statistics
        input_features = self.compute_statistics(hidden_states)
        
        # Compute alpha from MLP
        alpha = self.alpha_mlp(input_features)
        
        # Forward through both layers
        output_l = self.layer_l(hidden_states, attention_mask=attention_mask, **kwargs)
        output_m = self.layer_m(hidden_states, attention_mask=attention_mask, **kwargs)
        
        # Blend outputs
        if isinstance(output_l, tuple):
            hidden_l = output_l[0]
            hidden_m = output_m[0]
            blended_hidden = alpha * hidden_l + (1 - alpha) * hidden_m
            return (blended_hidden,) + output_l[1:]
        else:
            return alpha * output_l + (1 - alpha) * output_m


def create_mergeable_layer(
    layer_l: nn.Module,
    layer_m: nn.Module,
    alpha_init: float = 0.5,
    mode: str = "simple"
) -> nn.Module:
    """
    Factory function to create mergeable layers
    
    Args:
        layer_l: First layer
        layer_m: Second layer
        alpha_init: Initial alpha value
        mode: "simple" or "mlp"
        
    Returns:
        MergeableLayer or MLPMergeableLayer
    """
    if mode == "simple":
        return MergeableLayer(layer_l, layer_m, alpha_init)
    elif mode == "mlp":
        return MLPMergeableLayer(layer_l, layer_m)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Test code
    print("Testing MergeableLayer...")
    
    # Create dummy layers
    layer1 = nn.Linear(512, 512)
    layer2 = nn.Linear(512, 512)
    
    # Create mergeable layer
    mergeable = MergeableLayer(layer1, layer2, alpha_init=0.7)
    
    # Test forward pass
    x = torch.randn(2, 10, 512)
    output = mergeable(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Alpha value: {mergeable.alpha.item():.4f}")
    print(f"Number of trainable params: {sum(p.numel() for p in mergeable.parameters() if p.requires_grad)}")
    
    print("\\nTest passed!")
'''

# =============================================================================
# FILE 2: Modifications to pipeline.py
# =============================================================================

PIPELINE_MODIFICATIONS = '''
"""
Modifications to add to the existing pipeline.py file

Add these functions and modify the main execution flow
"""

# Add this import at the top
from mergeable_layer import MergeableLayer, create_mergeable_layer

# Add this new function
def replace_layers_with_mergeable(
    model,
    merge_pairs,
    similarity_scores,
    alpha_init_strategy="similarity"
):
    """
    Replace layer pairs with MergeableLayer modules
    
    Args:
        model: The transformer model
        merge_pairs: List of (layer_l_idx, layer_m_idx, sim_score)
        similarity_scores: Dictionary of similarity scores
        alpha_init_strategy: How to initialize alpha
            - "similarity": Use S_lm score
            - "uniform": Use 0.5
            - "fixed_07": Use 0.7
    
    Returns:
        Modified model with MergeableLayer modules
    """
    # Get layer list
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers
    
    # Sort in reverse to avoid index issues
    sorted_pairs = sorted(merge_pairs, key=lambda x: x[0], reverse=True)
    
    for layer_l_idx, layer_m_idx, sim_score in sorted_pairs:
        # Determine alpha initialization
        if alpha_init_strategy == "similarity":
            alpha_init = sim_score
        elif alpha_init_strategy == "uniform":
            alpha_init = 0.5
        elif alpha_init_strategy == "fixed_07":
            alpha_init = 0.7
        else:
            raise ValueError(f"Unknown strategy: {alpha_init_strategy}")
        
        # Create MergeableLayer
        layer_l = layers[layer_l_idx]
        layer_m = layers[layer_m_idx]
        
        mergeable = MergeableLayer(layer_l, layer_m, alpha_init=alpha_init)
        
        # Replace layer_l
        layers[layer_l_idx] = mergeable
        
        # Mark layer_m for removal
        layers[layer_m_idx] = nn.Identity()
        
        print(f"Created MergeableLayer for layers {layer_l_idx} and {layer_m_idx} with alpha_init={alpha_init:.4f}")
    
    return model


# Add this new function
def train_alpha_parameters(
    model,
    calibration_dataloader,
    num_steps=500,
    learning_rate=1e-4,
    device="cuda"
):
    """
    Train only the alpha parameters in MergeableLayer modules
    
    Args:
        model: Model with MergeableLayer modules
        calibration_dataloader: DataLoader for calibration data
        num_steps: Number of training steps
        learning_rate: Learning rate for optimizer
        device: Device to train on
    
    Returns:
        Trained model with learned alpha values
    """
    from torch.optim import Adam
    from tqdm import tqdm
    
    # Collect only alpha parameters
    alpha_params = []
    for name, param in model.named_parameters():
        if 'alpha_logit' in name or 'alpha_raw' in name:
            param.requires_grad = True
            alpha_params.append(param)
        else:
            param.requires_grad = False
    
    print(f"Training {len(alpha_params)} alpha parameters")
    
    if len(alpha_params) == 0:
        print("No alpha parameters found! Skipping training.")
        return model
    
    # Setup optimizer
    optimizer = Adam(alpha_params, lr=learning_rate)
    
    # Training loop
    model.train()
    losses = []
    
    pbar = tqdm(range(num_steps), desc="Training alpha")
    
    data_iter = iter(calibration_dataloader)
    
    for step in pbar:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            # Restart iterator
            data_iter = iter(calibration_dataloader)
            batch = next(data_iter)
        
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Update progress bar
        if (step + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
    
    print(f"Training completed. Final loss: {np.mean(losses[-10:]):.4f}")
    
    return model


# Add this to the main function argument parser
parser.add_argument(
    '--use_learnable_alpha',
    action='store_true',
    help='Use learnable alpha instead of fixed similarity-based alpha'
)

parser.add_argument(
    '--alpha_training_steps',
    type=int,
    default=500,
    help='Number of steps to train alpha parameters'
)

parser.add_argument(
    '--alpha_learning_rate',
    type=float,
    default=1e-4,
    help='Learning rate for alpha parameter training'
)

# Modify the main execution to include learnable alpha mode
if args.use_learnable_alpha:
    print("\\n" + "="*60)
    print("LEARNABLE ALPHA MODE")
    print("="*60)
    
    # Replace layers with MergeableLayer
    model = replace_layers_with_mergeable(
        model, 
        merge_pairs, 
        similarity_matrix,
        alpha_init_strategy="uniform"  # Start from 0.5
    )
    
    # Prepare calibration dataloader
    calibration_dataloader = prepare_calibration_dataloader(
        data_dir=args.data_dir,
        batch_size=4,
        num_samples=1000
    )
    
    # Train alpha parameters
    model = train_alpha_parameters(
        model,
        calibration_dataloader,
        num_steps=args.alpha_training_steps,
        learning_rate=args.alpha_learning_rate,
        device=device
    )
    
    # Extract learned alphas
    learned_alphas = {}
    for name, module in model.named_modules():
        if isinstance(module, MergeableLayer):
            learned_alphas[name] = module.alpha.item()
            print(f"{name}: alpha = {module.alpha.item():.4f}")
    
    # Save learned alphas
    import json
    with open(os.path.join(args.output_dir, 'learned_alphas.json'), 'w') as f:
        json.dump(learned_alphas, f, indent=2)
'''

# =============================================================================
# FILE 3: requirements.txt additions
# =============================================================================

REQUIREMENTS_ADDITIONS = '''
# Add these to the existing requirements.txt

matplotlib>=3.5.0
seaborn>=0.11.0
datasets>=2.0.0
scipy>=1.7.0
jupyter>=1.0.0
tensorboard>=2.8.0
'''

# =============================================================================
# FILE 4: Quick start guide
# =============================================================================

QUICK_START = '''
# Quick Start Guide for Learnable Alpha MKA

## 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/SempraETY/Pruning-via-Merging.git
cd Pruning-via-Merging

# Create environment
conda create -n learnable_mka python=3.9 -y
conda activate learnable_mka

# Install dependencies
pip install torch transformers numpy pandas scikit-learn tqdm
pip install matplotlib seaborn datasets scipy
```

## 2. Add New Files

Copy the following files to the repository:
- mergeable_layer.py
- train_learnable_alpha.py
- evaluate_methods.py
- plot_results.py

## 3. Run Experiments

### Baseline: Original MKA (alpha = S_lm)
```bash
python pipeline.py \\
    --model_path "meta-llama/Meta-Llama-3-8B" \\
    --data_dir "./data" \\
    --num_layer 14 \\
    --ngpu 4
```

### Our Method: Learnable Alpha
```bash
python pipeline.py \\
    --model_path "meta-llama/Meta-Llama-3-8B" \\
    --data_dir "./data" \\
    --num_layer 14 \\
    --ngpu 4 \\
    --use_learnable_alpha \\
    --alpha_training_steps 500 \\
    --alpha_learning_rate 1e-4
```

## 4. Evaluate on MMLU

```bash
python evaluate_methods.py \\
    --model_path "./merged_weights/Meta-Llama-3-8B-fused" \\
    --data_dir "./data/mmlu" \\
    --output_file "results.json"
```

## 5. Visualize Results

```bash
python plot_results.py \\
    --results_file "results.json" \\
    --learned_alphas "learned_alphas.json" \\
    --similarity_scores "similarity_matrix.pkl" \\
    --output_dir "./plots"
```

This will generate:
- alpha_distribution.png: Histogram of learned alpha values
- alpha_vs_similarity.png: Scatter plot comparing learned α with S_lm
- accuracy_comparison.png: Bar chart of MMLU accuracy across methods
'''

print("Repository modification guide generated successfully!")
print("\\nNext steps:")
print("1. Add the new files to the repository")
print("2. Modify existing files as indicated")
print("3. Update requirements.txt")
print("4. Run experiments following the quick start guide")
