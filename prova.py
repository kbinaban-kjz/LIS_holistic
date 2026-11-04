import os
import numpy as np

DATASET_DIR = 'dataset_sequenze'
TARGET_SHAPE = (30, 201)
trovati_corrotti = 0

for root, _, files in os.walk(DATASET_DIR):
    for file in files:
        if file.endswith('.npy'):
            path = os.path.join(root, file)
            shape = np.load(path).shape
            if shape != TARGET_SHAPE:
                print(f"FILE DA ELIMINARE: {path} -> ha forma {shape} invece di {TARGET_SHAPE}")
                trovati_corrotti += 1

if trovati_corrotti == 0:
    print("\nTutti i file hanno la dimensione corretta (30, 201)!")
else:
    print(f"\nTrovati {trovati_corrotti} file con forma diversa da eliminare.")