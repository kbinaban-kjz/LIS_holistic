import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.layers import (
    Dense, MultiHeadAttention, LayerNormalization, Dropout, Embedding
)
from estrai_features import crea_detector, rileva, estrai_features, disegna

# --- Configurazione ---
MODEL_PATH = 'modello_lis_transformer.keras'   # oppure modello_v2.keras / modello_v3.keras
LUNGHEZZA_SEQUENZA = 30
SOGLIA_CONFIDENZA = 0.5
MOSTRA_DEBUG = True

FRAME_STABILI = 10   # frame consecutivi per confermare una parola
FRAME_PAUSA = 15     # frame senza segno (o "neutro") per poter ripetere la stessa parola



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
        config.update({
            "sequence_len": self.sequence_len,
            "d_model": self.d_model,
        })
        return config


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


# --- Caricamento modello e parole ---
modello = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'PositionalEncoding': PositionalEncoding,
        'TransformerBlock': TransformerBlock,
    }
)
modello.summary()

with open('parole.txt', 'r') as f:
    parole = [riga.strip() for riga in f.readlines()]
print("Parole riconoscibili:", parole)

# --- Rilevatore e webcam ---
detector = crea_detector()
cap = cv2.VideoCapture(0)
print("Webcam avviata! 'q' per uscire.")

# Creazione finestra ridimensionabile
NOME_FINESTRA = 'Riconoscimento LIS'
cv2.namedWindow(NOME_FINESTRA, cv2.WINDOW_NORMAL)

buffer_sequenza = deque(maxlen=LUNGHEZZA_SEQUENZA)
parola_riconosciuta = ""
confidenza_corrente = 0.0

frase = []
candidata = None
conteggio = 0
frame_pausa = 0
ultima_aggiunta = None

def disegna_testo_con_sfondo(img, testo, pos, font_scale=0.6, color=(0,0,0), thickness=1, bg_color=(255, 255, 255)):
    """Funzione di supporto per renderizzare il testo visibile su qualsiasi sfondo."""
    x, y = pos
    text_size, _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + 5), bg_color, -1)
    cv2.putText(img, testo, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Impossibile accedere alla webcam.")
        break

    frame = cv2.flip(frame, 1)

    result = rileva(detector, frame)
    disegna(frame, result)
    features_frame = estrai_features(result)

    buffer_sequenza.append(features_frame)

    if len(buffer_sequenza) == LUNGHEZZA_SEQUENZA:
        sequenza_array = np.expand_dims(np.array(buffer_sequenza, dtype=np.float32), axis=0)
        predizione = modello(sequenza_array, training=False).numpy()[0]

        indice_predetto = int(np.argmax(predizione))
        confidenza_corrente = float(predizione[indice_predetto])
        parola_predetta = parole[indice_predetto]

        if MOSTRA_DEBUG:
            print("  ".join(f"{p}: {prob:.0%}" for p, prob in zip(parole, predizione)))

        # Una parola è "valida" se la confidenza supera la soglia e non è 'neutro'
        if confidenza_corrente >= SOGLIA_CONFIDENZA and parola_predetta != 'neutro':
            parola_riconosciuta = parola_predetta
            frame_pausa = 0  # Azzera il contatore di pausa

            # Se la parola è uguale alla candidata corrente, incrementa il contatore
            if parola_riconosciuta == candidata:
                conteggio += 1
            else:
                # È cambiata la parola rilevata
                candidata = parola_riconosciuta
                conteggio = 1

            # Aggiungi alla frase SOLO SE:
            # 1. Raggiunge i frame stabili richiesti
            # 2. È diversa dall'ultima parola effettivamente inserita nella frase
            if conteggio >= FRAME_STABILI and candidata != ultima_aggiunta:
                frase.append(candidata)
                ultima_aggiunta = candidata  # Salva l'ultima parola inserita
        else:
            # Nessun segno valido o classe 'neutro'
            parola_riconosciuta = "..."
            frame_pausa += 1

            # Solo se c'è una PAUSA VERA e prolungata permettiamo di ripetere la stessa parola
            if frame_pausa >= FRAME_PAUSA:
                candidata = None
                conteggio = 0
                ultima_aggiunta = None  # Ora si può ripetere la stessa parola (es. "grazie grazie")


    BG = (50, 50, 50)

    testo_stato = f"Parola: {parola_riconosciuta} ({confidenza_corrente:.0%})"
    disegna_testo_con_sfondo(frame, testo_stato, (10, 30), font_scale=0.8, color=(255, 255, 255), thickness=2, bg_color=BG)

# 2. Frase Composta
    testo_frase = f"Frase: {' '.join(frase)}"
    disegna_testo_con_sfondo(frame, testo_frase, (10, 65), font_scale=0.8, color=(255, 255, 255), thickness=2, bg_color=BG)

# 3. Istruzioni Utente
    istruzioni = "Comandi: [Q] Esci | [C] Cancella | [U] Elimina Ultima"
    disegna_testo_con_sfondo(frame, istruzioni, (10, 100), font_scale=0.6, color=(255, 255, 255), thickness=1, bg_color=BG)


    cv2.imshow(NOME_FINESTRA, frame)

    tasto = cv2.waitKey(1) & 0xFF
    if tasto == ord('q'):          # esci
        break
    elif tasto == ord('c'):        # cancella la frase
        frase.clear()
        ultima_aggiunta = None
    elif tasto == ord('u'):        # elimina l'ultima parola
        if frase:
            frase.pop()

cap.release()
cv2.destroyAllWindows()