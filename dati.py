# =====================================================================
# dati.py
# Caricamento del dataset, usato da tutti gli script.
# Per cambiare dataset o dimensione delle immagini basta modificare qui.
#
# =====================================================================

from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CANALI = 3    # RGB
LATO = 32     # le immagini originali sono 64x64: le riduco per tenere piccola la rete
CLASSI = 10


def carica_set():
    """Restituisce (train_set, test_set, nomi_classi)."""
    trasformazione = transforms.Compose([
        transforms.Resize((LATO, LATO)),
        transforms.ToTensor(),             # pixel tra 0 e 1, come serve alla SNN
    ])

    # Se ho scaricato lo zip da Zenodo e l'ho scompattato in data/EuroSAT_RGB
    # uso quello, altrimenti lo scarica torchvision.
    cartella_zenodo = Path("data/EuroSAT_RGB")
    if cartella_zenodo.exists():
        tutto = datasets.ImageFolder(cartella_zenodo, transform=trasformazione)
    else:
        tutto = datasets.EuroSAT(root="data", download=True, transform=trasformazione)

    # EuroSAT non ha una divisione ufficiale training/test: la faccio io, 80/20.
    # Il seme fisso garantisce che tutti gli script vedano gli stessi dati di test.
    n_test = int(0.2 * len(tutto))
    generatore = torch.Generator().manual_seed(42)
    train_set, test_set = random_split(tutto, [len(tutto) - n_test, n_test], generator=generatore)
    return train_set, test_set, tutto.classes


def carica_dati(batch_size=128):
    """Restituisce (train_loader, test_loader)."""
    train_set, test_set, _ = carica_set()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
