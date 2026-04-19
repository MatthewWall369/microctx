"""
Train RecursiveConvLM
"""

import os, math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from simplified import build_model, bytes_to_dataset

# ── config ────────────────────────────────────────────────────────────────────
W        = 3
DIM      = 256
SEQ_LEN  = 128
DROPOUT  = 0.1
LR       = 5e-3
EPOCHS   = 1
BATCH    = 64

TRAIN_STRIDE = SEQ_LEN
VAL_STRIDE   = SEQ_LEN
DATA_PATH    = None   # set to a file path to train on your own dataset

# ── data ──────────────────────────────────────────────────────────────────────
if DATA_PATH and os.path.isfile(DATA_PATH):
    raw = np.memmap(DATA_PATH, dtype=np.uint8, mode='r')
    split = int(len(raw) * 0.9)
    train_src, val_src = raw[:split], raw[split:]
    val_text = bytes(val_src).decode('utf-8', errors='replace')
else:
    path = keras.utils.get_file(
        'shakespeare.txt',
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    )
    text = open(path).read()
    split = int(len(text) * 0.9)
    train_src, val_text = text[:split], text[split:]
    val_src = val_text

train_ds, num_windows = bytes_to_dataset(
    train_src, seq_len=SEQ_LEN, stride=TRAIN_STRIDE, batch_size=BATCH
)
print(f'Seq len: {SEQ_LEN} | W={W} | Dim={DIM} | Windows: {num_windows:,}')

num_passes = 0
d = 1
while d < SEQ_LEN:
    num_passes += 1
    d *= W
print(f'Recursive passes: {num_passes}')
# ── model ─────────────────────────────────────────────────────────────────────
model = build_model(dim=DIM, W=W, dropout=DROPOUT)
model.optimizer.learning_rate = LR

_ = model(tf.zeros((1, SEQ_LEN), dtype=tf.int32))
total = sum(np.prod(v.shape) for v in model.trainable_variables)
print(f'Params: {total:,}')

history = model.fit(train_ds, epochs=EPOCHS)

SAVE_PATH = 'mora_shakespeare_finetuned.weights.h5'
model.save_weights(SAVE_PATH)
print(f'Saved weights to {SAVE_PATH}')

# ── extrapolation test ────────────────────────────────────────────────────────
import time

print('\n\n-- Extrapolation Test --')
print(f'Trained on seq_len={SEQ_LEN}\n')

if isinstance(val_src, np.ndarray):
    raw = np.array(val_src, dtype=np.uint8)
else:
    raw = np.frombuffer(val_text.encode('utf-8'), dtype=np.uint8)

for test_len in [256, 1_000, 10_000, 100_000, 500_000]:
    if test_len + 1 > len(raw):
        print(f'  {test_len:>7,} tokens: skipped (need {test_len+1}, have {len(raw):,})')
        continue

    d, passes = 1, 0
    while d < test_len:
        passes += 1; d *= W

    xs = tf.constant(raw[:test_len][None].astype(np.int32))
    ys = raw[1:test_len + 1].astype(np.int32)

    t0 = time.time()
    logits = model(xs, training=False)
    elapsed = time.time() - t0

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ys, logits=logits[0])
    avg_loss = float(tf.reduce_mean(loss))
    ppl = float(tf.exp(avg_loss))

    tag = '(train len)' if test_len == SEQ_LEN else '<-- extrapolating!'
    print(f'  {test_len:>7,} tokens: ppl={ppl:8.2f}  loss={avg_loss:.4f}  passes={passes}  ({elapsed:.1f}s)  {tag}')

print('\nDone.')


while True:
    prompt = input('>> ')
    out = model.generate(prompt, max_new=1024, temperature=0.8, top_k=3, stream=True)