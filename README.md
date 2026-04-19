# Mora V3

A minimal byte-level language model built on a **Recursive Dilated Causal Convolutional** architecture in TensorFlow/Keras.

The core idea: a single shared 1D causal convolution is applied recursively with exponentially increasing dilations (1, W, W², ...) until the receptive field covers the full sequence length. This gives the model a large effective context window with very few parameters.

## Features

- **Byte-level tokenization** — operates directly on raw UTF-8 bytes (256 classes), no external tokenizer needed
- **Recursive weight sharing** — one convolutional block reused across all dilation levels
- **Length extrapolation** — architecture generalizes to sequence lengths longer than those seen during training
- **Cached autoregressive generation** — efficient inference with warmup cache and single-step updates
- **Interactive REPL** — generate text interactively from the command line

## Project Structure

```
Mora_V3/
├── simplified.py          # Model definition (RecursiveConvLM), dataset builder, generation logic
├── train_simplified.py    # Training on Tiny Shakespeare, extrapolation eval, interactive generation
└── text_cleaned.txt       # Dialogue corpus (not currently wired into training)
```

## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install tensorflow numpy
```

## Usage

### Quick sanity check

Run the model file directly to verify shapes and parameter count:

```bash
python simplified.py
```

### Train on Tiny Shakespeare

```bash
python train_simplified.py
```

This will:
1. Download the Tiny Shakespeare dataset
2. Train the model for the configured number of epochs
3. Save weights to `mora_shakespeare_finetuned.weights.h5`
4. Run an extrapolation test at increasing sequence lengths
5. Start an interactive generation prompt (`>>`)

### Hyperparameters

All hyperparameters are configured at the top of `train_simplified.py`:

| Parameter | Description |
|-----------|-------------|
| `W` | Convolution kernel width / dilation base |
| `DIM` | Model hidden dimension |
| `SEQ_LEN` | Training sequence length |
| `DROPOUT` | Dropout rate |
| `LR` | Learning rate |
| `EPOCHS` | Number of training epochs |
| `BATCH` | Batch size |

## Extrapolation Results

Trained on `seq_len=128`, the model maintains stable perplexity when evaluated on sequences up to **780x longer** than training:

| Context Length | Perplexity | Loss   | Passes | Time  |
|----------------|-----------|--------|--------|-------|
| 256 tokens     | 3.61      | 1.2836 | 6      | 0.1s  |
| 1,000 tokens   | 4.03      | 1.3935 | 7      | 0.1s  |
| 10,000 tokens  | 3.84      | 1.3451 | 9      | 0.2s  |
| 100,000 tokens | 4.65      | 1.5372 | 11     | 1.0s  |

The recursive dilation scheme naturally extends to longer sequences by simply adding more passes — no retraining or positional encoding adjustments needed.

## Architecture Overview

```
Input bytes → Embedding(256, DIM)
    ↓
┌─────────────────────────────┐
│  Causal Conv1D (dilation=d) │  ← shared weights
│  GELU + Dropout             │
│  Residual + LayerNorm       │
└─────────────────────────────┘
    ↓ repeat with d = 1, W, W², ... until d ≥ seq_len
    ↓
Dense(256) → logits
```

## License

This project is for research and educational purposes.
