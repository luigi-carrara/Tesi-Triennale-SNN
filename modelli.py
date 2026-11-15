# =====================================================================
# modelli.py
# Le due reti del caso di studio (capitolo 5):
#   - CNN: rete convoluzionale classica (neuroni ReLU)
#   - SNN: stessa struttura, ma con neuroni LIF
# Hanno gli stessi parametri, così il confronto è equo.
# =====================================================================

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from dati import CANALI, LATO, CLASSI


def lato_finale(lato):
    """Lato delle mappe dopo conv 5x5 -> pool -> conv 5x5 -> pool."""
    lato = (lato - 4) // 2    # conv 5x5 toglie 4 pixel, il pool dimezza
    lato = (lato - 4) // 2
    return lato


# ---------------------------------------------------------------------
# CNN baseline
#   conv(12 filtri 5x5) -> pool -> ReLU
#   conv(64 filtri 5x5) -> pool -> ReLU -> lineare -> 10 punteggi
# ---------------------------------------------------------------------
class CNN(nn.Module):

    def __init__(self, canali=CANALI, lato=LATO, classi=CLASSI):
        super().__init__()
        self.conv1 = nn.Conv2d(canali, 12, kernel_size=5)
        self.conv2 = nn.Conv2d(12, 64, kernel_size=5)
        n = lato_finale(lato)
        self.fc = nn.Linear(64 * n * n, classi)

        self.pool = nn.MaxPool2d(2)    # senza pesi, lo riuso due volte
        self.relu = nn.ReLU()

    def forward(self, x):                        # x: [batch, canali, lato, lato]
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.flatten(1)                         # il batch resta separato
        return self.fc(x)                        # la softmax è dentro la loss


# ---------------------------------------------------------------------
# SNN con la stessa struttura
#   - al posto delle ReLU ci sono neuroni LIF
#   - la rete gira per num_steps passi temporali
#   - tra uno strato e l'altro passano solo spike (0 o 1)
# ---------------------------------------------------------------------
class SNN(nn.Module):

    # beta: quanto potenziale resta a ogni passo (0,9 = il 90%)
    def __init__(self, beta=0.9, num_steps=25, canali=CANALI, lato=LATO, classi=CLASSI):
        super().__init__()
        self.num_steps = num_steps

        # lo spike è un gradino e non si può derivare: nella backpropagation
        # lo sostituisco con la derivata di una fast sigmoid
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Un LIF per strato, perché ognuno ha i propri potenziali di membrana.
        # I LIF non hanno pesi, quindi i parametri sono quelli della CNN.
        # Soglia 1,0 e reset per sottrazione (valori predefiniti di snnTorch).
        self.conv1 = nn.Conv2d(canali, 12, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.conv2 = nn.Conv2d(12, 64, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        n = lato_finale(lato)
        self.fc = nn.Linear(64 * n * n, classi)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):                        # x: [batch, canali, lato, lato]
        # potenziali a zero a ogni immagine, così la rete non ricorda la precedente
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk_rec, mem_rec = [], []

        for t in range(self.num_steps):
            # strato 1: codifica diretta, riceve i pixel veri a ogni passo
            # (quindi conv1 fa moltiplicazioni, non solo somme)
            cur1 = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            # strato 2: riceve solo spike
            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            # uscita: un neurone per classe
            cur3 = self.fc(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            spk_rec.append(spk3)
            mem_rec.append(mem3)

        # [passi, batch, classi]. La predizione si ricava fuori: vince chi spara di più.
        return torch.stack(spk_rec), torch.stack(mem_rec)
