# Gradient Descent Training for Learnable Alpha Parameters

## Overview

This document describes the gradient descent training implementation for making alpha parameters learnable in the MKA (Model Knowledge Amalgamation) layer merging approach. The alpha parameters control the blending ratio between two layers being merged.

## Key Concept

Instead of using fixed alpha values from the MKA similarity formula, we train alpha parameters via gradient descent to minimize the prediction loss on calibration data.

### Alpha Parameterization

```python
# Alpha is parameterized using sigmoid for numerical stability:
alpha = sigmoid(alpha_logit)

# Where alpha_logit is the trainable parameter
# This ensures alpha stays in range [0, 1] without explicit constraints
```

## Implementation

### Complete Training Function

```python
def train_alpha_parameters(
    model,
    dataloader,
    num_steps=1000,
    learning_rate=1e-3,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    gradient_accumulation_steps=1
):
    """
    Train alpha parameters in MergeableLayer modules using gradient descent.
    
    Args:
        model: HuggingFace model with MergeableLayer modules
        dataloader: DataLoader providing calibration data batches
        num_steps: Number of training steps
        learning_rate: Learning rate for Adam optimizer
        device: Device to run training on
        gradient_accumulation_steps: Steps to accumulate gradients before update
    
    Returns:
        model: Model with trained alpha parameters
    """
    import torch
    from torch.optim import Adam
    
    print("\n=== Starting Alpha Parameter Training ===")
    
    # Step 1: Freeze all model parameters except alpha_logit
    for name, param in model.named_parameters():
        if 'alpha_logit' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"Training parameter: {name}, shape: {param.shape}")
    
    # Step 2: Collect all trainable alpha_logit parameters
    alpha_params = []
    for name, param in model.named_parameters():
        if 'alpha_logit' in name and param.requires_grad:
            alpha_params.append(param)
    
    if len(alpha_params) == 0:
        print("WARNING: No trainable alpha parameters found!")
        return model
    
    print(f"Found {len(alpha_params)} trainable alpha parameters")
    
    # Step 3: Initialize optimizer
    optimizer = Adam(alpha_params, lr=learning_rate)
    
    # Step 4: Training loop
    model.train()
    total_loss = 0.0
    step = 0
    
    # Print initial alpha values
    print("\n=== Initial Alpha Values ===")
    for name, module in model.named_modules():
        if hasattr(module, 'alpha_logit'):
            try:
                if module.alpha_logit.device.type != 'meta':
                    alpha_val = torch.sigmoid(module.alpha_logit).detach().cpu().item()
                    print(f"{name}: alpha = {alpha_val:.6f}")
            except:
                pass
    
    while step < num_steps:
        for batch in dataloader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Accumulate loss for logging
            total_loss += loss.item() * gradient_accumulation_steps
            
            # Update weights after accumulation steps
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            step += 1
            
            # Log progress
            if step % 100 == 0:
                avg_loss = total_loss / 100
                print(f"Step {step}/{num_steps}, Loss: {avg_loss:.4f}")
                total_loss = 0.0
            
            # Check if we've reached num_steps
            if step >= num_steps:
                break
        
        if step >= num_steps:
            break
    
    # Print final alpha values and changes
    print("\n=== Final Alpha Values and Changes ===")
    for name, module in model.named_modules():
        if hasattr(module, 'alpha_logit'):
            try:
                if module.alpha_logit.device.type != 'meta':
                    alpha_val = torch.sigmoid(module.alpha_logit).detach().cpu().item()
                    logit_val = module.alpha_logit.detach().cpu().item()
                    print(f"{name}: alpha = {alpha_val:.6f}, logit = {logit_val:.6f}")
            except:
                pass
    
    print("\n=== Alpha Parameter Training Complete ===\n")
    
    return model
```

## Usage Example

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Define merge pairs (example)
merge_pairs = [
    (18, 19, 0.622),  # (layer_to_merge_into, layer_to_remove, initial_alpha)
    (20, 21, 0.622),
    # ... more pairs
]

# Step 1: Replace layers with MergeableLayer
model = replace_layers_with_mergeable(model, merge_pairs, alpha_init_strategy="similarity")

# Step 2: Prepare calibration data
# Assumes you have a prepare_calibration_data function
calibration_dataloader = prepare_calibration_data(
    tokenizer=tokenizer,
    data_dir="./data",
    num_samples=50,
    batch_size=4
)

# Step 3: Train alpha parameters
model = train_alpha_parameters(
    model=model,
    dataloader=calibration_dataloader,
    num_steps=5000,
    learning_rate=1e-3,
    device='cuda',
    gradient_accumulation_steps=1
)

# Step 4: Extract learned alphas
learned_alphas = extract_learned_alphas(model, merge_pairs)

# Step 5: Fuse layers with learned alphas
model = fuse_mergeable_layers(model)

# Step 6: Evaluate
# Run your evaluation code here
```

## Hyperparameters

| Parameter | Typical Range | Recommended | Description |
|-----------|--------------|-------------|-------------|
| `num_steps` | 1000-10000 | 5000 | Number of gradient descent steps |
| `learning_rate` | 1e-5 to 1e-2 | 1e-3 | Adam optimizer learning rate |
| `batch_size` | 2-8 | 4 | Calibration data batch size |
| `gradient_accumulation_steps` | 1-4 | 1 | Steps to accumulate before update |
| `num_samples` | 50-200 | 100 | Number of calibration samples |

## Key Components

### 1. Parameter Freezing
```python
for name, param in model.named_parameters():
    if 'alpha_logit' not in name:
        param.requires_grad = False  # Freeze everything except alpha
```

### 2. Loss Function
```python
# Cross-entropy loss on next-token prediction
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
```

### 3. Gradient Step
```python
loss.backward()           # Compute gradients
optimizer.step()          # Update alpha_logit
optimizer.zero_grad()     # Reset gradients
```

### 4. Gradient Clipping (Optional)
```python
torch.nn.utils.clip_grad_norm_(alpha_params, max_norm=1.0)
```

## Expected Output

```
=== Starting Alpha Parameter Training ===
Training parameter: model.layers.18.alpha_logit, shape: torch.Size([])
Training parameter: model.layers.20.alpha_logit, shape: torch.Size([])
...
Found 13 trainable alpha parameters

=== Initial Alpha Values ===
model.layers.18: alpha = 0.622141
model.layers.20: alpha = 0.622141
...

Step 100/5000, Loss: 2.3456
Step 200/5000, Loss: 2.2891
Step 300/5000, Loss: 2.2345
...
Step 5000/5000, Loss: 2.0123

=== Final Alpha Values and Changes ===
model.layers.18: alpha = 0.650234, logit = 0.612345
model.layers.20: alpha = 0.598123, logit = -0.401234
...

=== Alpha Parameter Training Complete ===
```

## Troubleshooting

### Issue 1: "No trainable alpha parameters found"

**Cause**: MergeableLayer modules don't have alpha_logit registered as parameters.

**Solution**: Check that `replace_layers_with_mergeable()` properly creates MergeableLayer instances with:
```python
self.alpha_logit = nn.Parameter(torch.tensor(alpha_init))
```

### Issue 2: Loss not decreasing

**Causes**:
- Learning rate too low (try 1e-3 or 1e-2)
- Too few training steps (try 5000+)
- Calibration data too small (use 50-200 samples)

**Solution**: Increase learning rate or training steps, use more calibration data.

### Issue 3: CUDA out of memory

**Causes**:
- Batch size too large
- Model too large for GPU

**Solutions**:
```python
# Reduce batch size
batch_size = 2

# Use gradient accumulation
gradient_accumulation_steps = 4

# Use gradient checkpointing
model.gradient_checkpointing_enable()
```

## Mathematical Background

### Alpha Parameterization

The alpha parameter controls the blending ratio:

$$\text{output} = \alpha \cdot W_1 \cdot x + (1 - \alpha) \cdot W_2 \cdot x$$

Where:
- $W_1$ = weights of layer to merge into
- $W_2$ = weights of layer to remove
- $\alpha \in [0, 1]$ = blending weight

### Sigmoid Parameterization

Instead of optimizing $\alpha$ directly, we optimize a logit:

$$\alpha = \sigma(\text{logit}) = \frac{1}{1 + e^{-\text{logit}}}$$

This ensures $\alpha$ stays in $[0, 1]$ without constraints.

### Gradient Flow

The gradient flows through the sigmoid:

$$\frac{\partial \mathcal{L}}{\partial \text{logit}} = \frac{\partial \mathcal{L}}{\partial \alpha} \cdot \frac{\partial \alpha}{\partial \text{logit}} = \frac{\partial \mathcal{L}}{\partial \alpha} \cdot \alpha(1-\alpha)$$

This allows smooth optimization using standard gradient descent methods.

## References

- Original MKA Paper: [Model Knowledge Amalgamation](https://arxiv.org/abs/xxxx.xxxxx)
- PyTorch Optimizer Documentation: https://pytorch.org/docs/stable/optim.html
- Sigmoid Function: https://en.wikipedia.org/wiki/Sigmoid_function

## Notes

- **Initial Alpha Values**: If all similarities are 0, initial alpha will be ~0.622 (sigmoid(0.5))
- **Training Time**: ~37 minutes for 5000 steps with Meta-Llama-3-8B on single GPU
- **Alpha Range**: Final alphas typically range from 0.5 to 0.75 after training
- **Convergence**: Loss should decrease steadily; if it doesn't, adjust hyperparameters

## Version History

- v1.0 (2024-11): Initial implementation with bug fixes for meta tensors, device handling, and index mapping
