# =====================================================================
# 06_energia_snn.py
# Stima dell'energia per immagine della SNN, partendo dagli spike reali
# emessi sul test set (serve la SNN addestrata: snn_eurosat.pt).
# Il conteggio degli spike è in conteggi.py, i costi sono in energia.py.
#
# Semplificazioni da dichiarare nella tesi:
#   - nella versione base gli accessi alla memoria non sono contati
#     (la pagina Streamlit permette di provare un modello semplice);
#   - l'aggiornamento di un neurone costa come una MAC (perdita + somma);
#   - i costi sono quelli di energia.py, non di un chip reale.
#
# Lo strato 1 ha due varianti: calcolato a ogni passo (come fa il codice
# di modelli.py) oppure una volta sola, perché l'ingresso è sempre lo
# stesso e il risultato si potrebbe memorizzare.
# =====================================================================

import json
from pathlib import Path
import torch
from modelli import SNN
from dati import carica_dati
from conteggi import conta_attivita
import energia

torch.manual_seed(42)

_, test_loader = carica_dati()

model = SNN(beta=0.9, num_steps=25)
model.load_state_dict(torch.load("snn_eurosat.pt"))
T = model.num_steps

r = conta_attivita(model, test_loader)

print(f"Accuratezza sul test: {r['accuratezza']*100:.2f}%")
print("(se è intorno al 10% i pesi caricati non sono quelli della SNN)")

print("\nSpike per immagine:")
for nome in ("lif1", "lif2", "lif3"):
    print(f"  {nome}: {r['spike_per_immagine']['spike_' + nome]:,.0f}  "
          f"(tasso di spike {r['tasso_di_spike'][nome]*100:.1f}%)")
print(f"SOP per immagine: {r['sop_per_immagine']:,.0f}")
print(f"MAC dello strato 1: {r['mac_conv1_per_passo']:,} per passo "
      f"({r['mac_conv1_per_passo'] * T:,} su {T} passi)")

# --- Energia nei due scenari, in microjoule per immagine (versione base) ---
energia_uJ = {}
for scenario in energia.SCENARI:
    e = energia.energia_snn(r["mac_conv1_per_passo"], T, r["sop_per_immagine"],
                            r["aggiornamenti_neuroni"], scenario)
    energia_uJ[scenario] = e
    print(f"\nEnergia stimata ({scenario}), µJ per immagine:")
    print(f"  strato 1 una volta / a ogni passo: {e['conv1_una_volta']:.2f} / {e['conv1_ogni_passo']:.2f}")
    print(f"  somme sinaptiche: {e['sop']:.2f}   aggiornamento neuroni: {e['neuroni']:.2f}")
    print(f"  totale (strato 1 una volta): {e['totale_conv1_una_volta']:.2f}")
    print(f"  totale (strato 1 a ogni passo): {e['totale_conv1_ogni_passo']:.2f}")

Path("risultati").mkdir(exist_ok=True)
with open("risultati/snn_energia.json", "w") as f:
    json.dump({"modello": "SNN", **r, "energia_uJ": energia_uJ}, f, indent=2)
print("\nRisultati salvati in risultati/snn_energia.json")
