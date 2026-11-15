# =====================================================================
# 08_sweep_T.py
# Quanto contano i passi temporali? Addestra la SNN con valori diversi
# di T e per ciascuno salva accuratezza e conteggi delle operazioni.
# L'energia si calcola poi dai conteggi (pagina Streamlit, scheda "Sweep su T").
#
# ATTENZIONE: è lungo. Ogni T è un addestramento completo (5 epoche) e il
# costo cresce con T. Conviene lanciarlo sul PC, non sul Raspberry Pi.
# Impostazioni come 05_snn.py: batch 128, Adam, lr 0,001, seed 42.
# I risultati vengono salvati dopo ogni T, così se lo interrompi non perdi tutto.
# =====================================================================

import time
import json
from pathlib import Path
import torch
import snntorch.functional as SF
from modelli import SNN
from dati import carica_dati
from conteggi import conta_attivita

VALORI_T = [2, 4, 8, 16, 25]
NUM_EPOCHE = 5
BETA = 0.9

train_loader, test_loader = carica_dati(128)
loss_fn = SF.ce_rate_loss()

Path("risultati").mkdir(exist_ok=True)
risultati = []

for T in VALORI_T:
    print(f"\n=== T = {T} ===")
    torch.manual_seed(42)                    # stesso punto di partenza per ogni T
    model = SNN(beta=BETA, num_steps=T)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inizio = time.time()

    for epoca in range(NUM_EPOCHE):
        model.train()
        for immagini, etichette in train_loader:
            spk, _ = model(immagini)
            loss = loss_fn(spk, etichette)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"  epoca {epoca+1}/{NUM_EPOCHE} - ultima loss: {loss.item():.4f}")

    durata = time.time() - inizio
    r = conta_attivita(model, test_loader)       # accuratezza e conteggi sul test set
    r["tempo_addestramento_s"] = durata
    risultati.append(r)
    print(f"  accuratezza test: {r['accuratezza']*100:.2f}% - addestramento: {durata:.0f} s")

    with open("risultati/sweep_T.json", "w") as f:
        json.dump(risultati, f, indent=2)

print("\nRisultati salvati in risultati/sweep_T.json")
