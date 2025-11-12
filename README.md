# 🧠 Implementing GPT from scratch

> A clean, modular re-implementation and extension of Andrej Karpathy’s **"Let’s Build GPT"** tutorial.  
> This project walks step-by-step through building a **decoder-only Transformer** (GPT-style language model) **from scratch** using PyTorch.

---

## 🚀 Overview

This repository contains a progressive implementation of the core ideas behind modern large language models, built incrementally to support full understanding of:

- Tokenization and embeddings  
- Self-attention and multi-head attention  
- Transformer blocks (attention + feed-forward + residual connections)  
- Language modeling and text generation  

The goal is to **understand every component** of a GPT-like model by building it manually, not just importing from `torch.nn.Transformer`.

---

## 📂 Project Structure

### 🧾 Notebook
- **`bigram.ipynb`** — A complete **step-by-step walkthrough** of the model’s development — from tokenization and dataset creation to training, loss visualization, and text generation. Includes extensive comments and explanations for educational purposes.

### ⚙️ Scripts
| File | Description |
|------|-------------|
| **`bigram.py`** | Implements the simplest possible language model (a **bigram model**) that predicts the next token based only on the current one. Serves as the conceptual starting point. |
| **`self-attention.py`** | Introduces the **self-attention mechanism**, the core idea that allows each token to attend to previous tokens. Implements causal masking so tokens only attend to the past. |
| **`multihead-attention.py`** | Expands to **multi-head attention**, where several attention heads run in parallel to capture different relationships in the sequence. Adds feed-forward layers, residual connections, and layer normalization — forming a full **Transformer block**. |

---

## 🧮 Model Architecture

The final pipeline implements a **decoder-only Transformer** similar in structure to GPT:

```
Input → Token & Positional Embeddings
      → [ (Self-Attention + Feed-Forward + Residual + LayerNorm) × N Blocks ]
      → LayerNorm
      → Linear Head → Softmax → Next-Token Prediction
```

### Default hyperparameters
| Parameter | Value | Description |
|---|---:|---|
| `n_embd` | 384 | Embedding size |
| `n_head` | 6 | Attention heads per block |
| `n_layer` | 6 | Transformer blocks |
| `block_size` | 256 | Context length (max tokens attended) |
| `dropout` | 0.2 | Regularization |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 3e-4 | Optimizer learning rate |

> You can tweak these inside each script to explore scaling laws and capacity vs. compute trade-offs.

---

## 🔧 Setup

### 1) Create environment (recommended)
```bash
# Using conda (Windows/macOS/Linux)
conda create -n gpt-karpathy python=3.10 -y
conda activate gpt-karpathy
```

### 2) Install PyTorch
**CPU only**
```bash
pip install torch torchvision torchaudio
```

**GPU (CUDA 12.1 wheel)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**GPU via conda (recommended on Windows)**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 📥 Data

The examples use **Tiny Shakespeare**. It is available in the following link:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

---

## ⚡ Training & Generation

### Train a full Transformer block model
```bash
python multihead-attention.py
```

### Train simpler baselines
```bash
python bigram.py
python self-attention.py
```

### Generate text programmatically
```python
import torch
from multihead_attention import BigramLanguageModel, decode  # adjust import as needed

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BigramLanguageModel().to(device).eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=500)[0].tolist()
print(decode(out))
```

> Ensure your script exposes `decode` or duplicate the `encode/decode` helpers as in the examples.

---

## 🧪 Evaluation (quick loss estimate)

`estimate_loss()` (already provided in the scripts) evaluates train/val losses every `eval_interval` steps over `eval_iters` mini-batches. Reduce `eval_iters` to speed up training without changing convergence:
```python
eval_iters = 50   # (default 200)
eval_interval = 1000  # (default 500)
```

---

## 📊 Model Size & Rough Compute (default config)

- **Parameters**: ~**10.7M**  
- **FLOPs / iteration (forward+backward, batch=64, seq=256)**: ~**1.3×10¹²**  
- **Time on RTX 4050**: ~**30 min** for 5k iters without optimizations (could reduce by using `torch.compile` for ~30% speedup, and other tricks)

> Numbers are approximate; actual speed depends on drivers, kernels, and I/O.

---

## 🧩 References

- 🎥 **Let’s Build GPT from Scratch** — Andrej Karpathy: https://www.youtube.com/watch?v=kCc8FmEb1nY  
- 🧾 **nanoGPT** — Minimal GPT training code: https://github.com/karpathy/nanoGPT  
- 📘 **Attention Is All You Need** (Vaswani et al., 2017): https://arxiv.org/abs/1706.03762

---

## 👤 Author

Built and extended by **Jon Ortega**, following and expanding Andrej Karpathy’s educational GPT tutorial.  
Includes further explanations, efficiency improvements, GPU optimization, and a clear modular code structure.

---

## 📝 License

This project is for educational and research purposes.  
If you use or extend it, consider linking back to this repository and the original Karpathy tutorial.