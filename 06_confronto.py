# =====================================================================
# 06_confronto.py
# Grafici di confronto CNN - SNN a partire dai risultati salvati
# in risultati/*.json. Salva il grafico in risultati/confronto.png
# =====================================================================

import json
from pathlib import Path
import matplotlib.pyplot as plt

CARTELLA = Path("risultati")
COLORI = {"CNN": "#2B5C8A", "SNN": "#D9822B"}   # stessi colori in tutti i grafici

def carica(nome):
    """Legge risultati/<nome>.json; se non esiste avvisa e restituisce None."""
    percorso = CARTELLA / f"{nome}.json"
    if not percorso.exists():
        print(f"Attenzione: {percorso} non trovato (esegui prima l'addestramento)")
        return None
    with open(percorso) as f:
        return json.load(f)

# tiene solo i modelli per cui esistono risultati
risultati = [r for r in (carica("cnn"), carica("snn")) if r is not None]
if not risultati:
    raise SystemExit("Nessun risultato da mostrare.")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# --- Grafico 1: accuratezza sul test per epoca ---
for r in risultati:
    colore = COLORI[r["modello"]]
    acc = [a * 100 for a in r["accuratezza"]]            # da frazione a percentuale
    ax1.plot(r["epoca"], acc, marker="o", color=colore, label=r["modello"])
    # scrive il valore finale accanto all'ultimo punto
    ax1.annotate(f"{acc[-1]:.2f}%", (r["epoca"][-1], acc[-1]),
                 textcoords="offset points", xytext=(5, -12), color=colore)
ax1.set_title("Accuratezza sul test per epoca")
ax1.set_xlabel("Epoca")
ax1.set_ylabel("Accuratezza (%)")
ax1.set_xticks(risultati[0]["epoca"])
ax1.grid(alpha=0.3)
ax1.legend()

# --- Grafico 2: tempo medio per epoca ---
nomi = [r["modello"] for r in risultati]
tempi = [sum(r["tempo_s"]) / len(r["tempo_s"]) for r in risultati]
barre = ax2.bar(nomi, tempi, color=[COLORI[n] for n in nomi], width=0.5)
ax2.bar_label(barre, fmt="%.1f s")                        # valore sopra ogni barra
ax2.set_title("Tempo medio per epoca (simulazione su CPU)")
ax2.set_ylabel("Secondi")

fig.suptitle("Confronto CNN - SNN su MNIST")
fig.tight_layout()
fig.savefig(CARTELLA / "confronto.png", dpi=300)
print("Grafico salvato in risultati/confronto.png")
plt.show()                                                # apre la finestra del grafico