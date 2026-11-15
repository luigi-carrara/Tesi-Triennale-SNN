# =====================================================================
# 07_confronto.py
# Grafici di confronto CNN - SNN a partire dai risultati salvati
# in risultati/*.json:
#   - confronto.png: accuratezza e tempo di addestramento
#   - confronto_energia.png: energia stimata per immagine
# =====================================================================

import json
from pathlib import Path
import matplotlib.pyplot as plt

CARTELLA = Path("risultati")
COLORI = {"CNN": "#2B5C8A", "SNN": "#D9822B"}    # stessi colori in tutti i grafici


def carica(nome):
    """Legge risultati/<nome>.json; se non esiste avvisa e restituisce None."""
    percorso = CARTELLA / f"{nome}.json"
    if not percorso.exists():
        print(f"Attenzione: {percorso} non trovato (esegui prima lo script che lo crea)")
        return None
    with open(percorso) as f:
        return json.load(f)


# tengo solo i modelli per cui esistono risultati
risultati = [r for r in (carica("cnn"), carica("snn")) if r is not None]
if not risultati:
    raise SystemExit("Nessun risultato da mostrare.")
dataset = risultati[0].get("dataset", "MNIST")

# ---------------------------------------------------------------------
# Figura 1: accuratezza e tempo
# ---------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for r in risultati:
    colore = COLORI[r["modello"]]
    acc = [a * 100 for a in r["accuratezza"]]            # da frazione a percentuale
    ax1.plot(r["epoca"], acc, marker="o", color=colore, label=r["modello"])
    ax1.annotate(f"{acc[-1]:.2f}%", (r["epoca"][-1], acc[-1]),
                 textcoords="offset points", xytext=(5, -12), color=colore)
ax1.set_title("Accuratezza sul test per epoca")
ax1.set_xlabel("Epoca")
ax1.set_ylabel("Accuratezza (%)")
ax1.set_xticks(risultati[0]["epoca"])
ax1.grid(alpha=0.3)
ax1.legend()

nomi = [r["modello"] for r in risultati]
tempi = [sum(r["tempo_s"]) / len(r["tempo_s"]) for r in risultati]
barre = ax2.bar(nomi, tempi, color=[COLORI[n] for n in nomi], width=0.5)
ax2.bar_label(barre, fmt="%.1f s")
ax2.set_title("Tempo medio per epoca (simulazione su CPU)")
ax2.set_ylabel("Secondi")

fig.suptitle(f"Confronto CNN - SNN su {dataset}")
fig.tight_layout()
fig.savefig(CARTELLA / "confronto.png", dpi=300)
print("Grafico salvato in risultati/confronto.png")

# ---------------------------------------------------------------------
# Figura 2: energia stimata per immagine
# ---------------------------------------------------------------------
cnn_mac = carica("cnn_mac")
snn_en = carica("snn_energia")

if cnn_mac is not None and snn_en is not None:
    fig2, assi = plt.subplots(1, 2, figsize=(11, 4.8))
    sfumature = ["#D9822B", "#F0B97A", "#F7DDBA"]    # strato 1, somme sinaptiche, neuroni

    print("\nEnergia stimata per immagine (µJ):")
    for ax, scenario in zip(assi, ("fp32", "int8")):
        e_cnn = cnn_mac["energia_uJ"][scenario]
        e = snn_en["energia_uJ"][scenario]

        ax.bar(0, e_cnn, color=COLORI["CNN"], width=0.6)
        ax.text(0, e_cnn, f"{e_cnn:.2f}", ha="center", va="bottom")

        # SNN: barra impilata, per vedere da dove viene l'energia
        for pos, conv1 in ((1, e["conv1_una_volta"]), (2, e["conv1_ogni_passo"])):
            parti = [("strato 1 (MAC)", conv1),
                     ("somme sinaptiche", e["sop"]),
                     ("aggiornamento neuroni", e["neuroni"])]
            sotto = 0
            for (nome, valore), colore in zip(parti, sfumature):
                ax.bar(pos, valore, bottom=sotto, color=colore, width=0.6,
                       label=nome if pos == 1 else None)
                sotto += valore
            ax.text(pos, sotto, f"{sotto:.2f}", ha="center", va="bottom")

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["CNN", "SNN\n(strato 1\nuna volta)", "SNN\n(strato 1\na ogni passo)"])
        ax.set_title(f"Scenario {scenario}")
        ax.set_ylabel("µJ per immagine")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        print(f"  {scenario}: CNN {e_cnn:.2f} | SNN strato 1 una volta {e['totale_conv1_una_volta']:.2f}"
              f" | SNN strato 1 a ogni passo {e['totale_conv1_ogni_passo']:.2f}")

    assi[0].legend(loc="upper left", fontsize=8)
    fig2.suptitle(f"Energia stimata per immagine, CNN - SNN su {dataset} (modello di Horowitz, 45 nm)")
    fig2.tight_layout()
    fig2.savefig(CARTELLA / "confronto_energia.png", dpi=300)
    print("Grafico salvato in risultati/confronto_energia.png")

plt.show()
