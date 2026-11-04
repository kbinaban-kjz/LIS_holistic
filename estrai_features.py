"""Codice condiviso da registra_sequenze.py e riconosci_webcam.py:
crea il rilevatore Holistic ed estrae il vettore di feature di ogni frame.
Usarlo in entrambi garantisce che registrazione e riconoscimento
calcolino le feature esattamente nello stesso modo."""
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODELLO_HOLISTIC = 'holistic_landmarker.task'

# --- Cosa entra nel vettore di feature ---
# Punti della posa: naso(0), spalle(11,12), gomiti(13,14), polsi(15,16)
POSA_IDX = [0, 7, 8, 11, 12, 13, 14, 15, 16]
SPALLA_A, SPALLA_B = 11, 12   # servono per centrare e scalare tutto
# Viso (opzionale): punta del naso, labbra interne, angoli bocca, sopracciglia
USA_VISO = True
VISO_IDX = [1, 0, 13, 14, 17, 61, 291, 105, 334, 66, 296, 159, 145, 386, 374, 152]

N_FEATURES = (len(POSA_IDX) * 3 + 2 * 21 * 3
              + (len(VISO_IDX) * 3 if USA_VISO else 0))

_ultimo_timestamp = 0


def crea_detector():
    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODELLO_HOLISTIC),
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def rileva(detector, frame_bgr):
    """Esegue Holistic su un frame BGR (già specchiato) e restituisce il risultato."""
    global _ultimo_timestamp
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # in modalità VIDEO i timestamp devono essere strettamente crescenti
    ts = max(int(time.monotonic() * 1000), _ultimo_timestamp + 1)
    _ultimo_timestamp = ts
    return detector.detect_for_video(mp_image, ts)


def _punti(landmarks, indici=None):
    if not landmarks:
        return None
    # Se landmarks ha l'attributo .landmarks (tipico dei risultati MediaPipe Tasks)
    lms = landmarks.landmarks if hasattr(landmarks, 'landmarks') else landmarks
    
    if indici is not None:
        lms = [lms[i] for i in indici]
        
    return np.array([[l.x, l.y, l.z] for l in lms], dtype=np.float32)


def _normalizza(punti, quanti, centro, scala):
    if punti is None:
        return np.zeros(quanti * 3, dtype=np.float32)
    p = punti.copy()
    p[:, :2] -= centro[:2]   # origine = centro delle spalle
    p /= scala               # unità = larghezza delle spalle
    return p.flatten()


def estrai_features(result):
    """Vettore di N_FEATURES valori per un frame.
    Le coordinate sono relative alle spalle (posizione e distanza dalla
    webcam non contano). Parti non rilevate = zeri. Se le spalle non si
    vedono, l'intero frame è zero."""
    posa = _punti(result.pose_landmarks)
    if posa is None:
        return np.zeros(N_FEATURES, dtype=np.float32)

    a, b = posa[SPALLA_A], posa[SPALLA_B]
    centro = (a + b) / 2
    scala = float(np.linalg.norm((a - b)[:2]))
    if scala < 1e-3:
        return np.zeros(N_FEATURES, dtype=np.float32)

    parti = [
        _normalizza(posa[POSA_IDX], len(POSA_IDX), centro, scala),
        _normalizza(_punti(result.left_hand_landmarks), 21, centro, scala),
        _normalizza(_punti(result.right_hand_landmarks), 21, centro, scala),
    ]
    if USA_VISO:
        viso = _punti(result.face_landmarks, VISO_IDX)
        parti.append(_normalizza(viso, len(VISO_IDX), centro, scala))
    return np.concatenate(parti)


def disegna(frame, result):
    """Disegna i punti usati come feature sul frame."""
    h, w = frame.shape[:2]
    gruppi = [
        (result.pose_landmarks, POSA_IDX, (255, 200, 0)),
        (result.left_hand_landmarks, None, (0, 255, 0)),
        (result.right_hand_landmarks, None, (0, 200, 255)),
    ]
    if USA_VISO:
        gruppi.append((result.face_landmarks, VISO_IDX, (255, 0, 255)))
    for landmarks, idx, colore in gruppi:
        if not landmarks:
            continue
        lms = landmarks.landmarks if hasattr(landmarks, 'landmarks') else landmarks
        for i in (idx if idx is not None else range(len(lms))):
            cv2.circle(frame, (int(lms[i].x * w), int(lms[i].y * h)), 4, colore, -1)
