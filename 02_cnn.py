# =====================================================================
# 02_cnn.py
# Addestramento della CNN su EuroSAT.
# Impostazioni identiche a quelle della SNN (05_snn.py), per un confronto equo:
#   batch 128, Adam, learning rate 0,001, 5 epoche fissate a priori, seed 42.
# I pesi vengono salvati in cnn_eurosat.pt
# =====================================================================

import time
import json
from pathlib import Path
import torch
import torch.nn as nn
from modelli import CNN
from dati import carica_dati, LATO

# Seme fisso: stessi pesi iniziali e stesso ordine dei dati a ogni esecuzione
torch.manual_seed(42)

# --- Dati ---
# Nessuna normalizzazione a media zero: la SNN vuole pixel in [0, 1]
# e la CNN usa lo stesso preprocessing.
batch_size = 128
train_loader, test_loader = carica_dati(batch_size)

# --- Modello ---
model = CNN()
n_params = sum(p.numel() for p in model.parameters())
print("Parametri totali:", n_params)

# --- Addestramento ---
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5      # fissato a priori, non scelto guardando il test


def train_one_epoch():
    """Un passaggio sul training set. Restituisce la loss media."""
    model.train()
    total_loss = 0.0
    for images, labels in train_loader:
        out = model(images)
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loss.item() è la media del batch: la moltiplico per il numero di
        # immagini per avere poi la media corretta su tutto il dataset
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate():
    """Accuratezza sul test set."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
    return correct / len(test_loader.dataset)


# Salvo la storia dell'addestramento per i grafici di confronto
storia = {"modello": "CNN", "dataset": "EuroSAT", "lato": LATO, "parametri": n_params,
          "epoca": [], "loss": [], "accuratezza": [], "tempo_s": []}

for epoch in range(num_epochs):
    start = time.time()
    loss = train_one_epoch()
    acc = evaluate()
    durata = time.time() - start
    print(f"Epoca {epoch+1}/{num_epochs} - loss: {loss:.4f} - "
          f"accuratezza test: {acc*100:.2f}% - tempo: {durata:.1f} s")
    storia["epoca"].append(epoch + 1)
    storia["loss"].append(loss)
    storia["accuratezza"].append(acc)
    storia["tempo_s"].append(durata)

torch.save(model.state_dict(), "cnn_eurosat.pt")
print("Modello salvato in cnn_eurosat.pt")

Path("risultati").mkdir(exist_ok=True)
with open("risultati/cnn.json", "w") as f:
    json.dump(storia, f, indent=2)
print("Risultati salvati in risultati/cnn.json")
