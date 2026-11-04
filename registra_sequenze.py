import cv2
import os
import numpy as np
from estrai_features import crea_detector, rileva, estrai_features, disegna


DATASET_DIR = 'dataset_sequenze'
PAROLA_CORRENTE = 'no'   # cambia questa prima di registrare una nuova parola
LUNGHEZZA_SEQUENZA = 30   


detector = crea_detector()

#Prepara la cartella 
cartella_parola = os.path.join(DATASET_DIR, PAROLA_CORRENTE)
os.makedirs(cartella_parola, exist_ok=True)
# conta quante sequenze esistono già, per non sovrascriverle
indice_sequenza = len(os.listdir(cartella_parola))

# webcam 
cap = cv2.VideoCapture(0)
print("Webcam avviata!")
print("'r' per iniziare a registrare, 'q' per uscire.")
print(f"Parola corrente: {PAROLA_CORRENTE}")

registrando = False
buffer_sequenza = []  # frame di feature accumulati durante la registrazione

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Impossibile accedere la webcam.")
        break

    frame = cv2.flip(frame, 1)
    result = rileva(detector, frame)
    disegna(frame, result)

    # feature del frame (zeri per le parti non rilevate)
    features_frame = estrai_features(result)

    # stato del rilevamento
    stato = (f"corpo:{'ok' if result.pose_landmarks else '--'} "
             f"mano sx:{'ok' if result.left_hand_landmarks else '--'} "
             f"mano dx:{'ok' if result.right_hand_landmarks else '--'}")
    cv2.putText(frame, stato, (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Logica di registrazione ---
    if registrando:
        buffer_sequenza.append(features_frame)
        cv2.putText(frame, f"REC {len(buffer_sequenza)}/{LUNGHEZZA_SEQUENZA}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if len(buffer_sequenza) == LUNGHEZZA_SEQUENZA:
            sequenza_array = np.array(buffer_sequenza)  # shape (30, N_FEATURES)
            percorso_file = os.path.join(cartella_parola, f'{indice_sequenza}.npy')
            np.save(percorso_file, sequenza_array)
            print(f"Sequenza salvata: {percorso_file}")

            indice_sequenza += 1
            buffer_sequenza = []
            registrando = False
    else:
        cv2.putText(frame, f"Parola: {PAROLA_CORRENTE}  |  'r' registra  'q' esci",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow('Registrazione LIS con holistic', frame)

    tasto = cv2.waitKey(1) & 0xFF

    if tasto == ord('q'):
        break
    elif tasto == ord('r') and not registrando:
        registrando = True
        buffer_sequenza = []
        print("Registrazione avviata...")

cap.release()
cv2.destroyAllWindows()
