import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D, Dropout, Embedding
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold, cross_val_score
'''
allena2 mette le parole che vede una dietro all'altro
'''
tf.keras.utils.set_random_seed(0)

# --- Configurazione ---
DATASET_DIR = 'dataset_sequenze'
MODEL_PATH = 'modello_lis_transformer.keras'

parole = sorted(os.listdir(DATASET_DIR))
print("Parole trovate:", parole)

X, y = [], []
for indice_parola, parola in enumerate(parole):
    cartella = os.path.join(DATASET_DIR, parola)
    for nome_file in os.listdir(cartella):
        sequenza = np.load(os.path.join(cartella, nome_file))
        X.append(sequenza)
        y.append(indice_parola)

X = np.array(X)  # shape (n_campioni, 30, 201)
y = np.array(y)

print(f"Totale campioni: {len(X)}, forma di X: {X.shape}")

y_categorico = to_categorical(y, num_classes=len(parole))


# --- Split stratificato e mescolato ---
def split_stratificato(etichette, frazione, rng):
    """Divide gli indici in due gruppi (a, b) mantenendo tutte le classi in entrambi.
    Il gruppo b contiene circa `frazione` dei campioni di ogni classe (minimo 1).
    Gli indici restituiti sono mescolati, così non restano ordinati per classe."""
    idx_a, idx_b = [], []
    for classe in np.unique(etichette):
        idx = np.where(etichette == classe)[0]
        rng.shuffle(idx)
        n_b = max(1, int(round(len(idx) * frazione)))
        idx_b.extend(idx[:n_b])
        idx_a.extend(idx[n_b:])
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    rng.shuffle(idx_a)
    rng.shuffle(idx_b)
    return idx_a, idx_b


rng = np.random.default_rng(42)

# 1) train+val (80%) / test (20%)
idx_trainval, idx_test = split_stratificato(y, 0.3, rng) # ho aumentato i dati del test per fare un po di tuning 

# k_folds = KFold(n_splits = 5)

# scores = cross_val_score(clf, X, y, cv = k_folds)

# print("Cross Validation Scores: ", scores)
# print("Average CV Score: ", scores.mean())
# print("Number of CV Scores used in Average: ", len(scores))

# 2) dentro train+val: train (80%) / validation (20%), sempre stratificato
pos_train, pos_val = split_stratificato(y[idx_trainval], 0.2, rng)
idx_train = idx_trainval[pos_train]
idx_val = idx_trainval[pos_val]

X_train, y_train = X[idx_train], y_categorico[idx_train]
X_val, y_val = X[idx_val], y_categorico[idx_val]
X_test, y_test = X[idx_test], y_categorico[idx_test]

print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")


# --- Definizione Architettura Mini-Transformer ---

class PositionalEncoding(tf.keras.layers.Layer):
    """Aggiunge informazioni sull'ordine temporale delle 30 frame"""
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
        # Necessario per salvare/ricaricare il modello con load_model
        config = super().get_config()
        config.update({"sequence_len": self.sequence_len, "d_model": self.d_model})
        return config


def blocco_transformer(x, n_teste=4, dim_ff=64, dropout=0.1):
    d_model = x.shape[-1]
    key_dim = d_model // n_teste

    # Self-Attention
    attn = MultiHeadAttention(num_heads=n_teste, key_dim=key_dim)(x, x)
    attn = Dropout(dropout)(attn)
    x = LayerNormalization()(x + attn)

    # Feed Forward Network
    ff = Dense(dim_ff, activation='relu')(x)
    ff = Dense(d_model)(ff)
    ff = Dropout(dropout)(ff)
    return LayerNormalization()(x + ff)


# Costruzione del modello
seq_len = X.shape[1]      # 30
n_features = X.shape[2]   # 201
D_MODEL= 32

ingresso = Input(shape=(seq_len, n_features))
x = Dense(D_MODEL)(ingresso)                                       # prima il dense era 64, l'ho abbassato per aver un transformer più piccolo
x = PositionalEncoding(sequence_len=seq_len, d_model=D_MODEL)(x)   # Positional Encoding
x = blocco_transformer(x, n_teste=4, dim_ff=64, dropout=0.1) # prima la diff era 128
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
uscita = Dense(len(parole), activation='softmax')(x)

modello = tf.keras.Model(inputs=ingresso, outputs=uscita)
modello.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modello.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,  # più alto: con batch piccoli e pochi dati la val_loss è rumorosa
    restore_best_weights=True,
)

# 6. Training (validation set esplicito e stratificato)
modello.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=8,
    callbacks=[early_stopping],
)

# 7. Valutazione e salvataggio
perdita, accuratezza = modello.evaluate(X_test, y_test)
print(f"\nAccuratezza sul test set: {accuratezza:.2%}")

modello.save(MODEL_PATH)
print(f"Modello salvato in: {MODEL_PATH}")

with open('parole.txt', 'w') as f:
    for parola in parole:
        f.write(parola + '\n')

