import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, MultiHeadAttention, LayerNormalization, 
    GlobalAveragePooling1D, Dropout, Embedding, Add
)
from tensorflow.keras.utils import to_categorical

tf.keras.utils.set_random_seed(0)

# --- Configurazione ---
DATASET_DIR = 'dataset_sequenze'
MODEL_PATH = 'modello_lis_transformer.keras'

parole = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
print("Parole trovate:", parole)

X, y = [], []
for indice_parola, parola in enumerate(parole):
    cartella = os.path.join(DATASET_DIR, parola)
    for nome_file in os.listdir(cartella):
        if nome_file.endswith('.npy'):
            sequenza = np.load(os.path.join(cartella, nome_file))
            X.append(sequenza)
            y.append(indice_parola)

X = np.array(X, dtype=np.float32)  # Shape: (n_campioni, 30, 201)
y = np.array(y, dtype=np.int32)

print(f"Totale campioni: {len(X)}, forma di X: {X.shape}")
y_categorico = to_categorical(y, num_classes=len(parole))

# --- Split stratificato e mescolato ---
def split_stratificato(etichette, frazione, rng):
    idx_a, idx_b = [], []
    for classe in np.unique(etichette):
        idx = np.where(etichette == classe)[0]
        rng.shuffle(idx)
        n_b = max(1, int(round(len(idx) * frazione)))
        idx_b.extend(idx[:n_b])
        idx_a.extend(idx[n_b:])
    idx_a, idx_b = np.array(idx_a), np.array(idx_b)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    return idx_a, idx_b

rng = np.random.default_rng(42)

idx_trainval, idx_test = split_stratificato(y, 0.2, rng)

# Dentro Train+Val: Train (80%) / Validation (20%)
pos_train, pos_val = split_stratificato(y[idx_trainval], 0.2, rng)
idx_train = idx_trainval[pos_train]
idx_val = idx_trainval[pos_val]

X_train, y_train = X[idx_train], y_categorico[idx_train]
X_val, y_val = X[idx_val], y_categorico[idx_val]
X_test, y_test = X[idx_test], y_categorico[idx_test]

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

# --- Modulo Positional Encoding ---
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.d_model = d_model
        self.pos_emb = Embedding(input_dim=sequence_len, output_dim=d_model)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        return x + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_len": self.sequence_len, "d_model": self.d_model})
        return config

# --- Blocco Transformer con Pre-LayerNormalization ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, n_teste, dim_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_teste = n_teste
        self.dim_ff = dim_ff
        self.dropout_rate = dropout

        self.norm1 = LayerNormalization()
        self.attn = MultiHeadAttention(num_heads=n_teste, key_dim=d_model // n_teste)
        self.drop1 = Dropout(dropout)

        self.norm2 = LayerNormalization()
        self.ffn = tf.keras.Sequential([
            Dense(dim_ff, activation='gelu'),
            Dropout(dropout),
            Dense(d_model),
            Dropout(dropout)
        ])

    def call(self, inputs, training=False):
        # Sub-layer 1: Self Attention (Pre-LN)
        x_norm = self.norm1(inputs)
        attn_out = self.attn(x_norm, x_norm, training=training)
        x = inputs + self.drop1(attn_out, training=training)

        # Sub-layer 2: Feed Forward (Pre-LN)
        x_norm2 = self.norm2(x)
        ffn_out = self.ffn(x_norm2, training=training)
        return x + ffn_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "n_teste": self.n_teste,
            "dim_ff": self.dim_ff,
            "dropout": self.dropout_rate
        })
        return config

# --- Costruzione del Modello ---
seq_len = X.shape[1]      # 30
n_features = X.shape[2]   # 201
D_MODEL = 64              # Aumentato leggermente con adeguata regolarizzazione
N_BLOCKS = 2              # 2 blocchi transformer per catturare dinamiche temporali migliori

ingresso = Input(shape=(seq_len, n_features))

# Input Projection + Normalizzazione
x = Dense(D_MODEL, activation='gelu')(ingresso)
x = LayerNormalization()(x)
x = PositionalEncoding(sequence_len=seq_len, d_model=D_MODEL)(x)

# Blocchi Transformer
for _ in range(N_BLOCKS):
    x = TransformerBlock(d_model=D_MODEL, n_teste=4, dim_ff=128, dropout=0.15)(x)

# Pooling e Classificazione
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
uscita = Dense(len(parole), activation='softmax')(x)

modello = tf.keras.Model(inputs=ingresso, outputs=uscita)

# Ottimizzatore AdamW con Weight Decay
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

modello.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

modello.summary()

# --- Callback ---
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
        verbose=1
    )
]


modello.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=16,  # Un batch size leggermente più grande riduce il rumore sul gradiente
    callbacks=callbacks
)

# --- Valutazione e Salvataggio ---
perdita, accuratezza = modello.evaluate(X_test, y_test)
print(f"\nAccuratezza sul test set: {accuratezza:.2%}")

modello.save(MODEL_PATH)
print(f"Modello salvato in: {MODEL_PATH}")

with open('parole.txt', 'w') as f:
    for parola in parole:
        f.write(parola + '\n')