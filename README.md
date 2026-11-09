# 🤟 LIS_holistic: Riconoscimento della Lingua dei Segni Italiana

Un prototipo che applica tecniche avanzate di **Computer Vision** e **Deep Learning** per il riconoscimento automatico della Lingua dei Segni Italiana (LIS) in tempo reale.

Sfruttando **MediaPipe Holistic**, il sistema estrae i landmark chiave del corpo, delle mani e del viso da ciascun frame per alimentare modelli sequenziali capaci di interpretare la dinamica temporale dei gesti.

## 🛠️ Confronto Architetturale dei Modelli di Addestramento

All'interno del repository sono presenti diversi script di addestramento (`allena_modello*.py`). La differenza fondamentale tra essi risiede nell'architettura di rete utilizzata per elaborare la sequenza temporale dei landmark:

| File Script | Architettura | Descrizione e Dettagli Tecnologici | 
| ----- | ----- | ----- | 
| **`allena_modello.py`** | **Baseline / ML Classico** | Modello base di riferimento. Elabora i singoli frame o sequenze mediate tramite algoritmi di Machine Learning tradizionali (es. Random Forest / SVM) o reti Feed-Forward dense per stabilire una baseline di accuratezza. | 
| **`allena_modello2.py`** | **LSTM (Long Short-Term Memory)** | Utilizza una rete neurale ricorrente basata su **strati LSTM**. È progettata per gestire le dipendenze temporali sequenziali a breve e medio termine, analizzando il movimento sequenziale delle mani e del corpo frame dopo frame. | 
| **`allena_modello3.py`** | **Mini-Transformer** | Implementa un'architettura **Mini-Transformer** leggera basata su meccanismi di **Self-Attention** (`MultiHeadAttention`). Supera i limiti delle reti ricorrenti consentendo alla rete di focalizzarsi contestualmente sui frame e sui landmark più rilevanti dell'intero gesto LIS. | 

## 💻 Requisiti di Sistema

* **Python:** `3.11.x` *(Consigliato per la piena compatibilità con TensorFlow 2.x su Windows)*

* **Librerie principali:** `tensorflow`, `mediapipe`, `opencv-python`, `numpy`, `scikit-learn`

## 🔄 Flusso di Lavoro Operativo (Workflow)

Per utilizzare correttamente il sistema dal principio (dalla registrazione dei dati al test finale), segui questi tre passaggi in sequenza:

### 1. Preparazione dell'Ambiente

Apri il terminale PowerShell nella cartella del progetto ed esegui:

```
# Crea e attiva l'ambiente virtuale
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

# Installa le dipendenze
pip install -r requirements.txt

```

### Passo 1: Registrazione del Dataset (Crea/Aggiungi una parola)

Prima di allenare il modello, è necessario acquisire i dati di input tramite webcam per ogni parola o segno da riconoscere.

1. Apri lo script di registrazione (es. `registra_dati.py` o lo script dedicato al data collection).

2. Modifica nel codice la variabile contenente l'etichetta/parola che intendi registrare (es. `label = "GRAZIE"` oppure `LABEL = "CIAO"`).

3. Avvia lo script:

   ```
   python registra_dati.py
   
   ```

4. Esegui il gesto davanti alla webcam durante la registrazione. Lo script estrarrà i landmark tramite MediaPipe Holistic e li salverà sotto forma di numpy array (`.npy`).

5. **Ripeti la procedura** modificando la parola nel codice per tutte le classi/segni che desideri includere nel progetto.

### Passo 2: Addestramento del Modello

Una volta raccolti i dati per tutte le parole:

1. Seleziona lo script di addestramento desiderato in base all'architettura che vuoi utilizzare (`allena_modello.py`, `allena_modello2.py` per LSTM, o `allena_modello3.py` per Mini-Transformer).

2. Avvia l'addestramento dal terminale:

   ```
   # Esempio: addestramento tramite Mini-Transformer
   python allena_modello3.py
   
   ```

3. Al termine dell'esecuzione, lo script genererà e salverà il file del modello allenato (es. `.h5` o `.keras`).

### Passo 3: Esecuzione del Riconoscimento in Tempo Reale

Con il modello salvato, puoi avviare l'inferenza live da webcam:

1. Lancia lo script di riconoscimento:

   ```
   python riconosci_webcam2.py
   
   ```

2. Mettiti davanti alla webcam ed esegui i gesti registrati in precedenza. Il sistema mostrerà a schermo il tracciamento dei landmark e la parola riconosciuta in tempo reale.
