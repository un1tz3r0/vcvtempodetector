"""Neural beat activations: numpy inference of RTNeural JSON models, the reference for the C++.

The file format is RTNeural's (Keras layout): `in_shape`, then `layers`, each with `type`,
`shape`, `weights` and an optional `activation`. Supported: dense, conv1d (causal, dilated,
grouped), gru, lstm, batchnorm, prelu and activation layers, the set RTNeural's json_parser
builds at run time. A `beatref` block carries what RTNeural ignores: the feature front end, the
meaning of the outputs and the activation latency.

Outputs: 1 = beat; 2 = beat and downbeat (each sigmoid); 3 = softmax over (none, beat, downbeat),
madmom's downbeat layout, where beat = P(beat) + P(downbeat).
"""
import json
from pathlib import Path

import numpy as np

from .features import Features, Frontend

ACTIVATIONS = {
    'tanh': np.tanh,
    'relu': lambda x: np.maximum(x, 0),
    'sigmoid': lambda x: 1 / (1 + np.exp(-x)),
    'elu': lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0))),
    'softmax': lambda x: (e := np.exp(x - x.max(-1, keepdims=True))) / e.sum(-1, keepdims=True),
    '': lambda x: x,
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def conv1d(x, w, b, dilation, groups):
    """Causal: output t sees inputs t, t - d, ..., t - (k - 1) d. w is [k][in/groups][out], and its
    last tap multiplies the current input (Keras order, as RTNeural reads it)."""
    k, n = len(w), len(x)
    pad = np.concatenate([np.zeros(((k - 1) * dilation, x.shape[1])), x])
    cin, cout = w.shape[1], w.shape[2]
    y = np.tile(b, (n, 1))
    for g in range(groups):
        xi, yo = slice(g * cin, (g + 1) * cin), slice(g * cout // groups, (g + 1) * cout // groups)
        for tap in range(k):
            y[:, yo] += pad[tap * dilation:tap * dilation + n, xi] @ w[tap][:, yo]
    return y


def gru(x, W, U, b):
    """Keras GRU with reset_after (PyTorch's formulation): gates z, r, h."""
    n = U.shape[0]
    h, xw = np.zeros(n), x @ W + b[0]
    out = np.empty((len(x), n))
    for t, xt in enumerate(xw):
        hu = h @ U + b[1]
        z, r = sigmoid(xt[:n] + hu[:n]), sigmoid(xt[n:2 * n] + hu[n:2 * n])
        h = (1 - z) * np.tanh(xt[2 * n:] + r * hu[2 * n:]) + z * h
        out[t] = h
    return out


def lstm(x, W, U, b):
    """Keras LSTM: gates i, f, c, o."""
    n = U.shape[0]
    h, c = np.zeros(n), np.zeros(n)
    xw = x @ W + b
    out = np.empty((len(x), n))
    for t, xt in enumerate(xw):
        g = xt + h @ U
        i, f, o = sigmoid(g[:n]), sigmoid(g[n:2 * n]), sigmoid(g[3 * n:])
        c = f * c + i * np.tanh(g[2 * n:3 * n])
        h = o * np.tanh(c)
        out[t] = h
    return out


class Network:
    def __init__(self, spec):
        self.spec = spec
        meta = spec.get('beatref', {})
        self.frontend = Frontend.from_dict(meta.get('frontend'))
        self.latency = float(meta.get('latency', 0.015))
        self.inputs = int(spec['in_shape'][-1])
        self.layers = spec['layers']
        self.outputs = int(self.layers[-1]['shape'][-1])
        if self.inputs != self.frontend.size:
            raise ValueError(f'model takes {self.inputs} inputs but its front end makes {self.frontend.size}')
        if self.outputs not in (1, 2, 3):
            raise ValueError(f'model has {self.outputs} outputs; expected 1 (beat), 2 (beat, downbeat) or 3 (softmax)')

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def forward(self, x):
        for layer in self.layers:
            kind, w = layer['type'], [np.asarray(a, float) for a in layer['weights']]
            if kind in ('dense', 'time-distributed-dense'):
                x = x @ w[0] + (w[1] if len(w) > 1 else 0)
            elif kind == 'conv1d':
                x = conv1d(x, w[0], w[1], np.ravel(layer['dilation'])[-1], layer.get('groups', 1))
            elif kind == 'gru':
                x = gru(x, *w)
            elif kind == 'lstm':
                x = lstm(x, *w)
            elif kind == 'batchnorm':
                gamma, beta, mean, var = w if len(w) == 4 else (1, 0, *w)
                x = (x - mean) / np.sqrt(var + layer['epsilon']) * gamma + beta
            elif kind == 'prelu':
                x = np.where(x > 0, x, w[0][0] * x)
            elif kind != 'activation':
                raise ValueError(f'unsupported layer type {kind!r}')
            x = ACTIVATIONS[layer.get('activation', '')](x)
        return x

    def activations(self, x, sr, fps=100.0):
        """(beat, downbeat) activation per frame for mono signal x; downbeat is None if the model
        has no downbeat output."""
        y = self.forward(Features(sr, self.frontend, fps)(x))
        if self.outputs == 3:
            return y[:, 1] + y[:, 2], y[:, 2]
        return y[:, 0], (y[:, 1] if self.outputs == 2 else None)
