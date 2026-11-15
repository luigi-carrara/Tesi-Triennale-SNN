# =====================================================================
# 05_snn.py
# Addestramento della SNN su EuroSAT.
# Impostazioni identiche alla CNN (02_cnn.py): batch 128, Adam,
# learning rate 0,001, 5 epoche fissate a priori, seed 42.
# Parametri della SNN: beta 0,9, 25 passi temporali, codifica diretta,
# gradiente surrogato fast sigmoid.
# I pesi vengono salvati in snn_eurosat.pt
# =====================================================================

import time
import json
from pathlib import Path
import torch
import snntorch.functional as SF
from modelli import SNN
from dati import carica_dati, LATO

torch.manual_seed(42)

# --- Dati (identici alla CNN) ---
batch_size = 128
train_loader, test_loader = carica_dati(batch_size)

# --- Modello ---
beta = 0.9
num_steps = 25
model = SNN(beta=beta, num_steps=num_steps)
n_params = sum(p.numel() for p in model.parameters())
print("Parametri totali:", n_params)    # deve essere uguale a quello della CNN

# --- Addestramento ---
# ce_rate_loss applica la cross-entropy agli spike d'uscita di ogni passo e li
# accumula: il neurone della classe giusta deve sparare molto, gli altri poco.
loss_fn = SF.ce_rate_loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        spk, mem = model(images)         # [passi, batch, classi]
        loss = loss_fn(spk, labels)
        optimizer.zero_grad()
        loss.backward()                  # backpropagation nel tempo, col gradiente surrogato
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        if (i + 1) % 20 == 0:
            print(f"  batch {i+1}/{len(train_loader)} - loss: {loss.item():.4f}")
    return total_loss / len(train_loader.dataset)


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            spk, _ = model(images)
            conteggi = spk.sum(dim=0)           # spike totali per ogni neurone d'uscita
            pred = conteggi.argmax(dim=1)       # vince chi ha sparato di più
            correct += (pred == labels).sum().item()
    return correct / len(test_loader.dataset)


storia = {"modello": "SNN", "dataset": "EuroSAT", "lato": LATO, "parametri": n_params,
          "beta": beta, "num_steps": num_steps,
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


torch.save(model.state_dict(), "snn_eurosat.pt")
print("Modello salvato in snn_eurosat.pt")

Path("risultati").mkdir(exist_ok=True)
with open("risultati/snn.json", "w") as f:
    json.dump(storia, f, indent=2)
print("Risultati salvati in risultati/snn.json")
