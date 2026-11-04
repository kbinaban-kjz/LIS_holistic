import cv2
import numpy as np
from estrai_features import crea_detector, rileva, estrai_features, disegna
from lstm_numpy import ModelloLIS

MODEL_PATH = 'pesi_lis.npz'
LUNGHEZZA_SEQUENZA = 30
SOGLIA_CONFIDENZA = 0.9 
MOSTRA_DEBUG = True      # Stampa in console le probabilità per ogni parola

# 1. Carica modello e parole
modello = ModelloLIS(MODEL_PATH)
with open('parole.txt', 'r') as f:
    parole = [riga.strip() for riga in f.readlines()]
print("Parole riconoscibili:", parole)

# 2. Configura il rilevatore Holistic (201 feature)
detector = crea_detector()

# 3. Avvia la webcam
cap = cv2.VideoCapture(0)
print("Webcam avviata! 'q' per uscire.")

buffer_sequenza = []
parola_riconosciuta = ""
confidenza_corrente = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Impossibile accedere alla webcam.")
        break

    frame = cv2.flip(frame, 1)

    # Rilevamento ed estrazione feature (201 valori)
    result = rileva(detector, frame)
    disegna(frame, result)
    features_frame = estrai_features(result)

    # Aggiorna il buffer scorrevole dei 30 frame
    buffer_sequenza.append(features_frame)
    if len(buffer_sequenza) > LUNGHEZZA_SEQUENZA:
        buffer_sequenza.pop(0)

    # Predizione quando il buffer è pieno
    if len(buffer_sequenza) == LUNGHEZZA_SEQUENZA:
        sequenza_array = np.array(buffer_sequenza)  # shape (30, 201)
        predizione = modello.predict(sequenza_array)

        indice_predetto = np.argmax(predizione)
        confidenza_corrente = predizione[indice_predetto]

        if MOSTRA_DEBUG:
            probabilita_str = "  ".join(
                f"{p}: {prob:.0%}" for p, prob in zip(parole, predizione)
            )
            print(probabilita_str)

        if confidenza_corrente >= SOGLIA_CONFIDENZA:
            parola_riconosciuta = parole[indice_predetto]
        else:
            parola_riconosciuta = "..."

    # Mostra a schermo
    testo = f"{parola_riconosciuta} ({confidenza_corrente:.0%})"
    cv2.putText(frame, testo, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    cv2.imshow('Riconoscimento LIS', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()