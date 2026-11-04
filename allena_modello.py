import os
import numpy as np
import tensorflow as tf
tf.keras.utils.set_random_seed(0)  # risultati ripetibili
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

# --- Configurazione ---
DATASET_DIR = 'dataset_sequenze'
MODEL_PATH = 'modello_lis.h5'

# 1. Trova automaticamente le parole (= nomi delle sottocartelle)
parole = sorted(os.listdir(DATASET_DIR))
print("Parole trovate:", parole)

# 2. Carica tutte le sequenze e le etichette
X = []  # sequenze di landmark
y = []  # etichetta numerica corrispondente

for indice_parola, parola in enumerate(parole):
    cartella = os.path.join(DATASET_DIR, parola)
    for nome_file in os.listdir(cartella):
        sequenza = np.load(os.path.join(cartella, nome_file))  # shape (30, N_FEATURES)
        X.append(sequenza)
        y.append(indice_parola)

X = np.array(X)  # shape (n_campioni, 30, N_FEATURES)
y = np.array(y)

print(f"Totale campioni: {len(X)}, forma di X: {X.shape}")

# 3. One-hot encoding delle etichette (es. 0 -> [1,0], 1 -> [0,1])
y_categorico = to_categorical(y, num_classes=len(parole))

# 4. Split train/test (80/20, stratificato: stessa proporzione di parole in entrambi)
rng = np.random.default_rng(42)
idx_train, idx_test = [], []
for classe in np.unique(y):
    idx = np.where(y == classe)[0]
    rng.shuffle(idx)
    n_test = max(1, int(round(len(idx) * 0.2)))
    idx_test.extend(idx[:n_test])
    idx_train.extend(idx[n_test:])
X_train, y_train = X[idx_train], y_categorico[idx_train]
X_test, y_test = X[idx_test], y_categorico[idx_test]
print(f"Train: {len(X_train)} campioni, test: {len(X_test)} campioni")

# 5. Costruisci il modello LSTM
# modello = Sequential([
#     LSTM(64, return_sequences=True, activation='tanh', input_shape=X.shape[1:]),
#     Dropout(0.2),
#     LSTM(32, return_sequences=False, activation='tanh'),
#     Dropout(0.2),
#     Dense(32, activation='relu'),
#     Dense(len(parole), activation='softmax')  # una probabilità per ogni parola
# ])

def blocco_transformer(x, n_teste=4, dim_ff=64, dropout=0.1):
    attn = MultiHeadAttention(num_heads=n_teste, key_dim=x.shape[-1] // n_teste)(x, x)
    x = LayerNormalization()(x + attn)
    ff = Dense(dim_ff, activation='relu')(x)
    ff = Dense(x.shape[-1])(ff)
    return LayerNormalization()(x + Dropout(dropout)(ff))

ingresso = Input(shape=X.shape[1:])          # (30, N_FEATURES)
x = Dense(64)(ingresso)                       # proietta le feature in una dimensione comune
x = blocco_transformer(x)
x = GlobalAveragePooling1D()(x)
uscita = Dense(len(parole), activation='softmax')(x)
modello = tf.keras.Model(ingresso, uscita)

modello.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
modello.summary()

# 6. Allena il modello
modello.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8
)

# 7. Valuta e salva
perdita, accuratezza = modello.evaluate(X_test, y_test)
print(f"\nAccuratezza sul test set: {accuratezza:.2%}")

modello.save(MODEL_PATH)
print(f"Modello salvato in: {MODEL_PATH}")

# Salvo anche l'elenco delle parole, servirà per l'inferenza (per tradurre indice -> parola)
with open('parole.txt', 'w') as f:
    for parola in parole:
        f.write(parola + '\n')
