"""
Recursive Dilated Causal Conv LM — the simplest version.

One conv layer. Apply it recursively with dilation 1, W, W², ...
until dilation >= T. That's the whole model.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class RecursiveConvLM(keras.Model):
    def __init__(self, dim=256, W=3, vocab=256, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.dim   = dim
        self.W     = W
        self.vocab = vocab

        self.embed = layers.Embedding(vocab, dim)
        self.norm  = layers.LayerNormalization()

        # THE one conv — shared across all dilations
        self.conv_kernel = self.add_weight(name='k', shape=(W, dim, dim), initializer='glorot_uniform')
        self.conv_bias   = self.add_weight(name='b', shape=(dim,),        initializer='zeros')

        self.out_norm = layers.LayerNormalization()
        self.head     = layers.Dense(vocab)
        self.drop     = layers.Dropout(dropout)

    def call(self, x, training=False):
        T = x.shape[1]  # static shape — graph-safe
        h = self.embed(x)
        #h = tf.one_hot(x, depth=self.vocab)

        dil = 1
        while dil < T:
            residual = h
            h = self.norm(h)
            pad = dil * (self.W - 1)
            h = tf.pad(h, [[0,0],[pad,0],[0,0]])
            h = tf.nn.conv1d(h, self.conv_kernel, stride=1, padding='VALID', dilations=dil)
            h = h + self.conv_bias
            h = tf.nn.gelu(h)
            h = self.drop(h, training=training)
            h = residual + h

            dil *= self.W

        h = self.out_norm(h)
        return self.head(h)

    @staticmethod
    def _tensor_to_list(t):
        """(1, T, dim) tensor -> list of T (dim,) tensors."""
        return [t[0, i, :] for i in range(t.shape[1])]

    def _warmup_cache(self, ids, num_levels):
        """Full forward on prompt, returns list-based cache."""
        x = tf.constant([ids], dtype=tf.int32)
        h = self.embed(x)
        cache = [self._tensor_to_list(h)]
        dil = 1
        for _ in range(num_levels):
            residual = h
            h = self.norm(h)
            pad = dil * (self.W - 1)
            h = tf.pad(h, [[0, 0], [pad, 0], [0, 0]])
            h = tf.nn.conv1d(h, self.conv_kernel, stride=1, padding='VALID', dilations=dil)
            h = h + self.conv_bias
            h = tf.nn.gelu(h)
            h = residual + h
            cache.append(self._tensor_to_list(h))
            dil *= self.W
        return cache

    def _cached_step(self, token_id, cache, num_levels):
        """One token forward using list-based cache. O(1) append, no copies."""
        h_vec = self.embed(tf.constant([[token_id]], dtype=tf.int32))[0, 0, :]  # (dim,)
        cache[0].append(h_vec)
        zero = tf.zeros((self.dim,))

        dil = 1
        for level in range(num_levels):
            T = len(cache[level])
            residual = cache[level][-1]  # (dim,)

            taps = []
            for w in range(self.W):
                idx = T - 1 - dil * (self.W - 1 - w)
                taps.append(cache[level][idx] if idx >= 0 else zero)
            taps = self.norm(tf.stack(taps)[None, :, :])  # (1, W, dim)

            conv_out = tf.zeros((self.dim,))
            for w in range(self.W):
                conv_out = conv_out + tf.linalg.matvec(
                    tf.transpose(self.conv_kernel[w]), taps[0, w, :]
                )
            conv_out = tf.nn.gelu(conv_out + self.conv_bias)

            h_vec = residual + conv_out
            cache[level + 1].append(h_vec)
            dil *= self.W

        return h_vec  # last level output for this position

    def _logits_from_cache(self, cache):
        """Get logits for the last position in the cache."""
        h = cache[-1][-1][None, None, :]  # (1, 1, dim)
        return self.head(self.out_norm(h))[0, 0, :]  # (vocab,)

    def generate(self, prompt, max_new=200, temperature=1.0, top_k=None, stream=False):
        if isinstance(prompt, str):
            ids = list(prompt.encode('utf-8', errors='replace'))
        else:
            ids = list(prompt)
        if not ids:
            ids = [0]

        max_T = len(ids) + max_new
        num_levels = 0
        d = 1
        while d < max_T:
            num_levels += 1
            d *= self.W

        cache = self._warmup_cache(ids, num_levels)

        for _ in range(max_new):
            logits = self._logits_from_cache(cache)
            logits = logits / temperature
            if top_k is not None:
                top_vals, _ = tf.math.top_k(logits, k=top_k)
                logits = tf.where(logits < top_vals[-1], tf.fill(logits.shape, -1e9), logits)
            next_id = tf.random.categorical(logits[None], 1)[0, 0].numpy()
            ids.append(int(next_id))

            if stream:
                c = bytes([next_id]).decode('utf-8', errors='replace')
                print(c, end='', flush=True)

            self._cached_step(next_id, cache, num_levels)

        if stream:
            print()
        return bytes(ids).decode('utf-8', errors='replace')


def build_model(dim=256, W=3, dropout=0.1):
    model = RecursiveConvLM(dim=dim, W=W, dropout=dropout)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def bytes_to_dataset(text, seq_len=256, stride=None, batch_size=32):
    if stride is None:
        stride = seq_len
    raw = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
    xs, ys = [], []
    for i in range(0, len(raw) - seq_len, stride):
        xs.append(raw[i:i + seq_len])
        ys.append(raw[i + 1:i + seq_len + 1])
    xs = np.array(xs, dtype=np.int32)
    ys = np.array(ys, dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((xs, ys))
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    W   = 3
    DIM = 256
    SEQ = 256

    model = build_model(dim=DIM, W=W)
    x     = tf.random.uniform((2, SEQ), minval=0, maxval=256, dtype=tf.int32)
    out   = model(x, training=False)
    total = sum(np.prod(v.shape) for v in model.trainable_variables)

    num_passes = 0
    d = 1
    while d < SEQ:
        num_passes += 1
        d *= W

    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    print(f'Params: {total:,}')
    print(f'Passes: {num_passes} (dil = 1, {W}, {W}², ... until >= {SEQ})')
    print(f'W={W}  dim={DIM}  seq={SEQ}')
