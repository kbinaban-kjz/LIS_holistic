import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _relu(x):
    return np.maximum(0.0, x)


def _lstm(x, W, U, b, return_sequences):
    """Cella LSTM di Keras. Ordine dei gate: i, f, c, o.
    Attivazione tanh e recurrent_activation sigmoid (default di Keras)."""
    unita = U.shape[0]
    h = np.zeros(unita)
    c = np.zeros(unita)
    uscite = []
    for t in range(x.shape[0]):
        z = x[t] @ W + h @ U + b
        i, f, g, o = np.split(z, 4)
        i, f, o = _sigmoid(i), _sigmoid(f), _sigmoid(o)
        c = f * c + i * np.tanh(g)
        h = o * np.tanh(c)
        uscite.append(h)
    return np.array(uscite) if return_sequences else h


class ModelloLIS:
    """Stessa architettura di allena_modello.py:
    LSTM(64) -> LSTM(32) -> Dense(32, relu) -> Dense(n_parole, softmax).
    Il Dropout serve solo in allenamento, quindi qui non c'è."""

    def __init__(self, percorso_npz):
        dati = np.load(percorso_npz)
        self.p = {k: dati[k] for k in dati.files}

    def predict(self, sequenza):
        """sequenza: array (30, 63). Restituisce le probabilità per ogni parola."""
        p = self.p
        h = _lstm(sequenza, p['l1_W'], p['l1_U'], p['l1_b'], True)
        h = _lstm(h, p['l2_W'], p['l2_U'], p['l2_b'], False)
        h = _relu(h @ p['d1_W'] + p['d1_b'])
        z = h @ p['d2_W'] + p['d2_b']
        e = np.exp(z - z.max())
        return e / e.sum()
