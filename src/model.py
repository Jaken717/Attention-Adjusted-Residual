import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

# Shape notation used in the comments below:
#   B = batch, T = sequence length of the current input (dst), P = cached past length,
#   S = P + T = source length (S = T when there is no past), H = n_head, D = n_embd, V = n_vocab.
#   For the 124M model: H = 12, D = 768, D/H = 64, 4D = 3072, V = 50257.

def default_hparams():
    return HParams(
        n_vocab=0,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
    )

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)  # max: [..., 1] along axis; x keeps its shape, [B, H, T, S] for attention
    ex = tf.exp(x)  # same shape as x
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)  # sum: [..., 1] along axis; result same shape as x

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))  # same shape as x, [B, T, 4D] in mlp

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with tf.variable_scope(scope):
        n_state = x.shape[-1].value
        g = tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1))  # [D]
        b = tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0))  # [D]
        u = tf.reduce_mean(x, axis=axis, keepdims=True)  # [B, T, 1]
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)  # [B, T, 1]
        x = (x - u) * tf.rsqrt(s + epsilon)  # [B, T, D]
        x = x*g + b  # [B, T, D]
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])  # [..., m] -> [..., n, m/n]

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])  # [..., a, b] -> [..., a*b]

def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev))  # [1, nx, nf]
        b = tf.get_variable('b', [nf], initializer=tf.constant_initializer(0))  # [nf]
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])  # [B*T, nx] @ [nx, nf] -> [B*T, nf], reshaped to [B, T, nf]
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]  # [nd, 1]
    j = tf.range(ns)  # [ns]
    m = i >= j - ns + nd  # [nd, ns], broadcast
    return tf.cast(m, dtype)  # [nd, ns]


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])  # [B, T, D] -> [B, T, H, D/H] -> [B, H, T, D/H]

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))  # [B, H, T, D/H] -> [B, T, H, D/H] -> [B, T, D]

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)  # [T, S]
        b = tf.reshape(b, [1, 1, nd, ns])  # [1, 1, T, S]
        w = w*b - tf.cast(1e10, w.dtype)*(1-b)  # [B, H, T, S]
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)  # [B, H, T, D/H] @ [B, H, D/H, S] -> [B, H, T, S]
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))  # [B, H, T, S]

        w = mask_attn_weights(w)  # [B, H, T, S]
        w = softmax(w)             # post-softmax attention weights (what the model actually uses); [B, H, T, S]
        a = tf.matmul(w, v)  # [B, H, T, S] @ [B, H, S, D/H] -> [B, H, T, D/H]
        return a

    with tf.variable_scope(scope):
        c = conv1d(x, 'c_attn', n_state*3)  # [B, T, D] -> [B, T, 3D]
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))  # three [B, T, D], each split to [B, H, T, D/H]
        present = tf.stack([k, v], axis=1)  # [B, 2, H, T, D/H]
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)  # each [B, H, P, D/H]
            k = tf.concat([pk, k], axis=-2)  # [B, H, S, D/H]
            v = tf.concat([pv, v], axis=-2)  # [B, H, S, D/H]
        a = multihead_attn(q, k, v)
        a = merge_heads(a)  # [B, T, D]
        a = conv1d(a, 'c_proj', n_state)  # [B, T, D]
        return a, present


def mlp(x, scope, n_state, *, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state))  # [B, T, D] -> [B, T, 4D]
        h2 = conv1d(h, 'c_proj', nx)  # [B, T, 4D] -> [B, T, D]
        return h2


def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1'), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2'), 'mlp', nx*4, hparams=hparams)  # norm: [B, T, D]; m: [B, T, D]
        x = x + m  # [B, T, D]
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)  # [...] -> [size, ...]

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)  # [T] -> [B, T]


def model(hparams, X, past=None, scope='model', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)

        wpe = tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01))  # [n_ctx, D]
        wte = tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02))  # [V, D]
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))  # [B, T, D] + [B, T, D] -> [B, T, D]

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)  # [B, n_layer, 2, H, T, D/H]
        h = norm(h, 'ln_f')  # [B, T, D]

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])  # [B*T, D]
        logits = tf.matmul(h_flat, wte, transpose_b=True)  # [B*T, D] @ [D, V] -> [B*T, V]
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])  # [B, T, V]
        results['logits'] = logits
        return results
