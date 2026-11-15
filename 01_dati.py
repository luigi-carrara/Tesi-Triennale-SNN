# Controllo del dataset: quante immagini ci sono, che forma hanno
# e quante per classe.

import torch
from dati import carica_set

torch.manual_seed(42)

train_set, test_set, classi = carica_set()

print("Immagini di training:", len(train_set))
print("Immagini di test:", len(test_set))
print("Classi:", classi)

# Un'immagine: dovrebbe essere [3, 32, 32] = 3 canali (RGB), 32 righe, 32 colonne
img, etichetta = train_set[0]
print("Forma di un'immagine:", tuple(img.shape))
print("Valori min/max:", img.min().item(), img.max().item())
print("Etichetta del primo elemento:", etichetta, "->", classi[etichetta])

# Immagini per classe nel training: serve a capire se il dataset è bilanciato.
# train_set è un sottoinsieme di tutto il dataset: gli indici stanno in .indices
tutte = train_set.dataset.targets
per_classe = [0] * len(classi)
for i in train_set.indices:
    per_classe[tutte[i]] += 1
for nome, n in zip(classi, per_classe):
    print(f"  {nome}: {n}")
