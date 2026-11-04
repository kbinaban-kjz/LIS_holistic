import numpy as np
from tensorflow.keras.models import load_model
from lstm_numpy import ModelloLIS

# Da eseguire dove TensorFlow funziona (Colab o PC di casa), dopo allena_modello.py
modello = load_model('modello_lis.h5', compile=False)
lstm1, _, lstm2, _, dense1, dense2 = modello.layers

l1_W, l1_U, l1_b = lstm1.get_weights()
l2_W, l2_U, l2_b = lstm2.get_weights()
d1_W, d1_b = dense1.get_weights()
d2_W, d2_b = dense2.get_weights()

np.savez('pesi_lis.npz',
         l1_W=l1_W, l1_U=l1_U, l1_b=l1_b,
         l2_W=l2_W, l2_U=l2_U, l2_b=l2_b,
         d1_W=d1_W, d1_b=d1_b,
         d2_W=d2_W, d2_b=d2_b)
print("Pesi salvati in pesi_lis.npz")

# Verifica: la versione numpy deve dare le stesse probabilità di Keras
numpy_modello = ModelloLIS('pesi_lis.npz')
for _ in range(5):
    x = np.random.rand(30, modello.input_shape[-1]).astype('float32')
    atteso = modello.predict(x[None, ...], verbose=0)[0]
    ottenuto = numpy_modello.predict(x)
    print("Differenza massima:", np.abs(atteso - ottenuto).max())
print("Se le differenze sono ~1e-6 o meno, è tutto corretto.")
