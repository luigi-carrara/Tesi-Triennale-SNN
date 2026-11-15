# =====================================================================
# 04_snn_prova.py
# Prova della SNN con i pesi casuali, senza addestrarla.
# Serve a vedere due cose:
#   - la SNN non addestrata indovina a caso (circa il 10% con 10 classi);
#   - la CNN addestrata (se esiste cnn_eurosat.pt) è molto più brava.
# Così si capisce quanto vale davvero l'addestramento dello script 05.
# =====================================================================

import torch
from modelli import CNN, SNN
from dati import carica_dati, CLASSI
from pathlib import Path

torch.manual_seed(42)

_, test_loader = carica_dati()

# --- SNN con pesi casuali ---
snn_modello = SNN(beta=0.9, num_steps=25)
snn_modello.eval()

corretti = 0
spike_uscita = 0.0
n_immagini = 0
with torch.no_grad():
    for immagini, etichette in test_loader:
        spk, _ = snn_modello(immagini)               # [passi, batch, classi]
        conteggio = spk.sum(dim=0)                   # spike per classe
        corretti += (conteggio.argmax(dim=1) == etichette).sum().item()
        spike_uscita += spk.sum().item()
        n_immagini += immagini.size(0)

print(f"SNN non addestrata: accuratezza {corretti / n_immagini * 100:.2f}% "
      f"(a caso sarebbe {100 / CLASSI:.0f}%)")
print(f"Spike di uscita per immagine: {spike_uscita / n_immagini:.1f}")

# --- CNN addestrata, per confronto ---
if Path("cnn_eurosat.pt").exists():
    cnn = CNN()
    cnn.load_state_dict(torch.load("cnn_eurosat.pt"))
    cnn.eval()
    corretti = 0
    with torch.no_grad():
        for immagini, etichette in test_loader:
            corretti += (cnn(immagini).argmax(dim=1) == etichette).sum().item()
    print(f"CNN addestrata: accuratezza {corretti / n_immagini * 100:.2f}%")
else:
    print("CNN addestrata non trovata (cnn_eurosat.pt): esegui prima 02_cnn.py.")
