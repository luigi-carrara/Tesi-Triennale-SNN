# =====================================================================
# energia.py
# Costo energetico delle operazioni, in joule.
# Fonte: Horowitz, "Computing's energy problem", ISSCC 2014
# (tecnologia CMOS a 45 nm). Sono valori di riferimento per confrontare
# le due reti, non l'energia reale di un chip.
# =====================================================================


#Per la rappresentazione sono convenzionalmente utilizzati o i numeri in virgola mobile a 32 bit o gli interi a 8
SCENARI = {
    "fp32": {"mult": 3.7e-12, "add": 0.9e-12},    # virgola mobile a 32 bit
    "int8": {"mult": 0.2e-12, "add": 0.03e-12},   # interi a 8 bit
}


def e_mac(scenario):
    """Energia di una MAC (moltiplicazione + somma)."""
    return SCENARI[scenario]["mult"] + SCENARI[scenario]["add"]


def e_add(scenario):
    """Energia di una sola somma (operazione sinaptica della SNN)."""
    return SCENARI[scenario]["add"]


# ---------------------------------------------------------------------
# Varianti di calcolo (aggiunte in base all'articolo di Höfflin e Wassner
#(https://www.sciencedirect.com/science/article/pii/S0094576526004455)
# ---------------------------------------------------------------------

# Costo dell'aggiornamento di un neurone LIF a ogni passo:
#  - "1 MAC": la mia ipotesi di base (perdita + somma)
#  - "2 AC + 2 MAC": ipotesi più prudente, usata nell'articolo citato
VARIANTI_AGGIORNAMENTO = ("1 MAC", "2 AC + 2 MAC")

# Rapporto tra l'energia di un accesso alla memoria e quella di una somma.
# 0 = accessi non contati (come nella versione base).
# Modello MOLTO semplificato, solo per vedere quanto pesano gli accessi:
# non è il modello dell'articolo né una misura.
RAPPORTI_MEMORIA = (0, 1, 50, 100)


def e_aggiornamento(scenario, variante="1 MAC"):
    """Energia dell'aggiornamento di un neurone per un passo."""
    if variante == "1 MAC":
        return e_mac(scenario)
    return 2 * e_add(scenario) + 2 * e_mac(scenario)


def e_accesso(scenario, rapporto_memoria):
    """Energia di un accesso alla memoria (peso o stato del neurone)."""
    return rapporto_memoria * e_add(scenario)


def energia_cnn(mac_totali, scenario, rapporto_memoria=0):
    """Energia della CNN per immagine, in microjoule.
    Si assume un accesso alla memoria (il peso) per ogni MAC, senza riuso."""
    calcolo = mac_totali * e_mac(scenario) * 1e6
    memoria = mac_totali * e_accesso(scenario, rapporto_memoria) * 1e6
    return {"calcolo": calcolo, "memoria": memoria, "totale": calcolo + memoria}


def energia_snn(mac_conv1_passo, passi, sop, aggiornamenti, scenario,
                variante="1 MAC", rapporto_memoria=0):
    """Energia della SNN per immagine, in microjoule.
    Accessi alla memoria: un peso per ogni MAC dello strato 1 e per ogni SOP,
    due accessi (lettura e scrittura del potenziale) per ogni aggiornamento."""
    e = {
        "conv1_una_volta": mac_conv1_passo * e_mac(scenario) * 1e6,
        "conv1_ogni_passo": mac_conv1_passo * passi * e_mac(scenario) * 1e6,
        "sop": sop * e_add(scenario) * 1e6,
        "neuroni": aggiornamenti * e_aggiornamento(scenario, variante) * 1e6,
    }
    accessi_comuni = sop + 2 * aggiornamenti
    e["memoria_una_volta"] = (mac_conv1_passo + accessi_comuni) * e_accesso(scenario, rapporto_memoria) * 1e6
    e["memoria_ogni_passo"] = (mac_conv1_passo * passi + accessi_comuni) * e_accesso(scenario, rapporto_memoria) * 1e6
    e["totale_conv1_una_volta"] = e["conv1_una_volta"] + e["sop"] + e["neuroni"] + e["memoria_una_volta"]
    e["totale_conv1_ogni_passo"] = e["conv1_ogni_passo"] + e["sop"] + e["neuroni"] + e["memoria_ogni_passo"]
    return e
