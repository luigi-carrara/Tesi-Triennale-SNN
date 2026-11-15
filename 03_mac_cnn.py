# =====================================================================
# 03_mac_cnn.py
# Conteggio delle MAC (multiply-accumulate) della CNN e stima
# dell'energia per immagine (modello energetico del par. 5.3).

#
#IMPORTANTE:
#   - si contano solo convoluzioni e strato lineare;
#     pooling, ReLU e bias sono trascurati (costo minimo);
#   - la stima ignora gli accessi alla memoria.
# =====================================================================

import json
from pathlib import Path
import torch
import torch.nn as nn
from modelli import CNN
from dati import CANALI, LATO
import energia

# Non servono i pesi addestrati: le MAC dipendono solo dalla struttura.
model = CNN()
model.eval()

mac_per_layer = {}    # nome del layer -> numero di MAC


def conta_mac(nome):
    """Hook che calcola le MAC del layer."""
    def hook(modulo, ingresso, uscita):
        if isinstance(modulo, nn.Conv2d):
            # ogni valore in uscita costa (canali in ingresso x kernel x kernel) MAC;
            # uscita[0] è la prima immagine del batch, numel() = canali x altezza x larghezza
            mac_per_uscita = modulo.in_channels * modulo.kernel_size[0] * modulo.kernel_size[1]
            mac = uscita[0].numel() * mac_per_uscita
        else:  # nn.Linear
            mac = modulo.in_features * modulo.out_features
        mac_per_layer[nome] = mac
    return hook


for nome, modulo in model.named_modules():
    if isinstance(modulo, (nn.Conv2d, nn.Linear)):
        modulo.register_forward_hook(conta_mac(nome))

# Un'immagine qualsiasi: nella CNN il conteggio non dipende dal contenuto,
# perché ogni pixel viene moltiplicato anche se vale zero.
x = torch.zeros(1, CANALI, LATO, LATO)
with torch.no_grad():
    model(x)

totale = sum(mac_per_layer.values())
for nome, mac in mac_per_layer.items():
    print(f"{nome}: {mac:,} MAC")
print(f"Totale: {totale:,} MAC per immagine")
# Atteso con immagini 32x32: conv1 705.600 | conv2 1.920.000 | fc 16.000 | totale 2.641.600

# Energia = numero di MAC x energia di una MAC, nei due scenari
energia_uJ = {}
for scenario in energia.SCENARI:
    energia_uJ[scenario] = energia.energia_cnn(totale, scenario)["totale"]    # in microjoule
    print(f"Energia stimata ({scenario}): {energia_uJ[scenario]:.2f} µJ per immagine")

Path("risultati").mkdir(exist_ok=True)
with open("risultati/cnn_mac.json", "w") as f:
    json.dump({"mac_per_layer": mac_per_layer, "mac_totali": totale,
               "energia_uJ": energia_uJ}, f, indent=2)
print("Risultati salvati in risultati/cnn_mac.json")
