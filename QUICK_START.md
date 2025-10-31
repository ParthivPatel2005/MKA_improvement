# Quick Start Guide
## Learnable Alpha MKA - Two Experimental Approaches

---

## 📚 Two Notebooks Available

### 1. Scalar Alpha: `train_scalar_alpha.ipynb`
**Purpose:** Train a single scalar α parameter per layer pair  
**Use When:** Testing if S_lm heuristic is optimal  
**Output:** Static α values (one per layer pair)

### 2. MLP Alpha: `train_mlp_alpha.ipynb`
**Purpose:** Train MLP to predict α dynamically from input  
**Use When:** Testing if input-dependent merging improves accuracy  
**Output:** Dynamic α (varies per input sample)

---

## 🚀 Quick Commands

### Scalar Alpha (Static Coefficients)
```bash
python pipeline.py \
  --model_path "meta-llama/Meta-Llama-3-8B" \
  --num_layer 14 \
  --data_dir "./data" \
  --use_learnable_alpha \
  --alpha_training_steps 500 \
  --alpha_learning_rate 1e-4
```

### MLP Alpha (Dynamic Coefficients)
```bash
python pipeline.py \
  --model_path "meta-llama/Meta-Llama-3-8B" \
  --num_layer 14 \
  --data_dir "./data" \
  --use_learnable_alpha \
  --use_mlp_merge \
  --alpha_training_steps 500 \
  --alpha_learning_rate 1e-4
```

### Compare All Methods (5 Experiments)
```bash
python evaluate_methods.py \
  --model_path "meta-llama/Meta-Llama-3-8B" \
  --data_dir "./data" \
  --similarity_matrix "similarity_matrix.pkl" \
  --output_dir "./experiments" \
  --include_mlp
```

---

## 📁 File Reference

| File | Purpose |
|------|---------|
| `train_scalar_alpha.ipynb` | Interactive notebook for scalar α training |
| `train_mlp_alpha.ipynb` | Interactive notebook for MLP α training |
| `pipeline.py` | Main training script (CLI) |
| `evaluate_methods.py` | Benchmark all 5 methods |
| `mergeable_layer.py` | Core wrapper classes |
| `train_learnable_alpha.py` | Training coordinator |
| `plot_results.py` | Visualization utilities |

---

## 🎯 Key Differences

| Aspect | Scalar Alpha | MLP Alpha |
|--------|--------------|-----------|
| **Command Flag** | `--use_learnable_alpha` | `--use_learnable_alpha --use_mlp_merge` |
| **Layer Type** | `MergeableLayer` | `MLPMergeableLayer` |
| **Trainable Params** | 1 scalar per layer pair | ~4K params per MLP |
| **Alpha Behavior** | Static (fixed per layer) | Dynamic (varies per input) |
| **Learned Output** | Single α value | -1.0 indicator (dynamic) |
| **Fusion Strategy** | Use learned α | Use fixed 0.5 |
| **Evaluation Flag** | Automatic | `--include_mlp` required |

---

## 📊 5 Experimental Methods

When running `evaluate_methods.py --include_mlp`:

1. **mka_similarity**: Original paper's S_lm heuristic
2. **fixed_05**: Uniform α = 0.5 for all layers
3. **fixed_07**: Fixed α = 0.7 for all layers
4. **learned**: Learned scalar α (one per layer)
5. **learned_mlp**: Learned MLP α (dynamic per input)

---

## 🔧 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha_training_steps` | 500 | Number of training iterations |
| `--alpha_learning_rate` | 1e-4 | Adam optimizer learning rate |
| `--calibration_samples` | 1000 | Samples from MMLU dev set |
| `--calibration_batch_size` | 4 | Batch size for training |
| `--alpha_init_strategy` | uniform | Initial α value (uniform/similarity/fixed_07) |

---

## 📈 Expected Results Structure

```
experiments/
├── mka_similarity/
│   ├── models/           # Fused model checkpoint
│   ├── plots/            # Alpha analysis plots
│   └── results.json      # Accuracy metrics
├── fixed_05/
│   └── ...
├── fixed_07/
│   └── ...
├── learned/
│   └── ...
├── learned_mlp/          # Only if --include_mlp used
│   └── ...
└── results.json          # Combined comparison
```

---

## ✅ Pre-flight Checklist

- [ ] Model checkpoint available (or HuggingFace access)
- [ ] MMLU data in `./data/dev/` and `./data/test/`
- [ ] Python environment with PyTorch + Transformers
- [ ] Sufficient GPU memory (8B model needs ~16GB)
- [ ] Set `EXECUTE_COMMANDS = True` in notebooks

---

## 🎓 Research Questions

### Scalar Alpha Experiments
- Is the S_lm similarity heuristic near-optimal?
- How much do learned α values deviate from similarity scores?
- What's the correlation between learned α and similarity?

### MLP Alpha Experiments
- Does dynamic merging outperform static coefficients?
- What activation patterns predict high/low α values?
- Do different MMLU subjects require different strategies?

---

## 🐛 Troubleshooting

### Issue: Model not found
**Solution:** Check HuggingFace authentication: `huggingface-cli login`

### Issue: CUDA out of memory
**Solution:** Reduce `--calibration_batch_size` to 2 or 1

### Issue: No learned_mlp results
**Solution:** Add `--include_mlp` flag to evaluate_methods.py

### Issue: Import error for mergeable_layer
**Solution:** Verify `mergeable_layer.py` is in same directory as `pipeline.py`

---

## 📞 Quick Reference

**View learned alphas:**
```python
import json
with open("merged_weights/learned_alphas.json") as f:
    data = json.load(f)
    print(data["learned_alphas"])
```

**Check MLP layer:**
```python
# -1.0 indicates MLP layer (dynamic alpha)
# 0.0-1.0 indicates scalar alpha (static)
```

**Plot results:**
```bash
python plot_results.py --results_dir ./experiments
```

---

**Last Updated:** November 1, 2025  
**Version:** 1.0  
**Status:** Production Ready ✅
