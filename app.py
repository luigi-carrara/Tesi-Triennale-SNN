# =====================================================================
# app.py
# Pagina Streamlit per mostrare i risultati del confronto CNN - SNN.
# Si avvia con:   streamlit run app.py
#
# Legge i risultati già salvati in risultati/*.json (scritti dagli script
# 02, 03, 05, 06) e i modelli addestrati (cnn_eurosat.pt, snn_eurosat.pt).
# Gli script vanno quindi eseguiti prima, nell'ordine 01 -> 07.
# =====================================================================

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

import energia
from dati import CANALI, LATO, carica_set
from modelli import CNN, SNN

CARTELLA = Path("risultati")

# Gli script che si possono lanciare dalla pagina (gli stessi di main.py)
SCRIPT = [
    ("1. Controllo dati EuroSAT", "01_dati.py"),
    ("2. Addestra CNN", "02_cnn.py"),
    ("3. Conta MAC della CNN", "03_mac_cnn.py"),
    ("4. Prova SNN (SNN non addestrata)", "04_snn_prova.py"),
    ("5. Addestra SNN", "05_snn.py"),
    ("6. Energia della SNN (spike reali)", "06_energia_snn.py"),
    ("7. Grafici di confronto", "07_confronto.py"),
    ("8. Sweep su T", "08_sweep_T.py"),
]

st.set_page_config(page_title="CNN contro SNN su EuroSAT", layout="wide")


# ---------------------------------------------------------------------
# Funzioni di caricamento (con cache, così non rileggo tutto a ogni click)
# ---------------------------------------------------------------------
def leggi_json(nome):
    percorso = CARTELLA / f"{nome}.json"
    if not percorso.exists():
        return None
    with open(percorso) as f:
        return json.load(f)


@st.cache_resource
def carica_test_set():
    _, test_set, nomi_classi = carica_set()
    return test_set, nomi_classi


@st.cache_resource
def carica_cnn():
    if not Path("cnn_eurosat.pt").exists():
        return None
    modello = CNN()
    modello.load_state_dict(torch.load("cnn_eurosat.pt"))
    modello.eval()
    return modello


@st.cache_resource
def carica_snn():
    if not Path("snn_eurosat.pt").exists():
        return None
    modello = SNN(beta=0.9, num_steps=25)
    modello.load_state_dict(torch.load("snn_eurosat.pt"))
    modello.eval()
    return modello


def immagine_per_grafico(tensore):
    """Da tensore [3, H, W] a array [H, W, 3] per matplotlib."""
    return tensore.permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------
# Intestazione
# ---------------------------------------------------------------------
st.title("Reti neurali a impulsi contro CNN")
st.caption("Classificazione di immagini satellitari (EuroSAT) e stima del costo energetico. ")

tab_dati, tab_acc, tab_energia, tab_sweep, tab_demo, tab_esegui = st.tabs(
    ["Dati", "Accuratezza e tempi", "Energia", "Sweep su T", "Demo", "Esegui script"])


# ---------------------------------------------------------------------
# Scheda 1: i dati
# ---------------------------------------------------------------------
with tab_dati:
    try:
        test_set, nomi_classi = carica_test_set()
    except Exception as errore:
        st.error(f"Non riesco a caricare il dataset: {errore}")
        st.stop()

    st.write(f"Immagini di test: **{len(test_set)}**, classi: **{len(nomi_classi)}**. "
             f"Le immagini originali sono 64x64; le riduco a {LATO}x{LATO} pixel.")

    # una immagine per classe, la prima che trovo nel test set
    esempi = {}
    for i in range(len(test_set)):
        immagine, etichetta = test_set[i]
        if etichetta not in esempi:
            esempi[etichetta] = immagine
        if len(esempi) == len(nomi_classi):
            break

    colonne = st.columns(5)
    for posizione, etichetta in enumerate(sorted(esempi)):
        with colonne[posizione % 5]:
            st.image(immagine_per_grafico(esempi[etichetta]),
                     caption=nomi_classi[etichetta], width=130, clamp=True)


# ---------------------------------------------------------------------
# Scheda 2: accuratezza e tempi
# ---------------------------------------------------------------------
with tab_acc:
    cnn = leggi_json("cnn")
    snn = leggi_json("snn")
    storie = [s for s in (cnn, snn) if s is not None]

    if not storie:
        st.info("Nessun risultato trovato. Esegui prima 02_cnn.py e 05_snn.py.")
    else:
        colonne = st.columns(len(storie))
        for colonna, storia in zip(colonne, storie):
            with colonna:
                st.metric(f"{storia['modello']}: accuratezza finale",
                          f"{storia['accuratezza'][-1] * 100:.1f}%")
                st.metric("Tempo medio per epoca",
                          f"{sum(storia['tempo_s']) / len(storia['tempo_s']):.1f} s")
                st.metric("Parametri", f"{storia['parametri']:,}")

        fig, ax = plt.subplots(figsize=(6, 3.5))
        for storia in storie:
            ax.plot(storia["epoca"], [a * 100 for a in storia["accuratezza"]],
                    marker="o", label=storia["modello"])
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Accuratezza sul test (%)")
        ax.set_xticks(storie[0]["epoca"])
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)


# ---------------------------------------------------------------------
# Selettori di energia (usati dalle schede Energia e Sweep su T)
# ---------------------------------------------------------------------
def selettori_energia(chiave):
    """Tre scelte: precisione, costo dell'aggiornamento dei neuroni, accessi alla memoria."""
    c1, c2, c3 = st.columns(3)
    scenario = c1.radio("Precisione dei calcoli", list(energia.SCENARI), horizontal=True,
                        key="scenario_" + chiave)
    variante = c2.radio("Aggiornamento di un neurone", energia.VARIANTI_AGGIORNAMENTO,
                        key="variante_" + chiave,
                        help="1 MAC è la mia ipotesi di base; 2 AC + 2 MAC è quella, più "
                             "prudente, dell'articolo di Höfflin e Wassner (2026).")
    rapporto = c3.radio("Costo accesso memoria / somma", energia.RAPPORTI_MEMORIA,
                        horizontal=True, key="memoria_" + chiave,
                        help="0 = accessi non contati. Modello molto semplice: un accesso "
                             "per ogni peso e per ogni potenziale. Non è una misura.")
    return scenario, variante, rapporto


# ---------------------------------------------------------------------
# Scheda 3: energia
# ---------------------------------------------------------------------
with tab_energia:
    mac_cnn = leggi_json("cnn_mac")
    en_snn = leggi_json("snn_energia")

    if mac_cnn is None or en_snn is None:
        st.info("Mancano i risultati di energia. Esegui 03_mac_cnn.py e 06_energia_snn.py.")
    else:
        st.write("Energia stimata per **una** immagine, contando le operazioni e "
                 "associando a ciascuna un costo (Horowitz, ISSCC 2014, 45 nm). "
                 "Non è una misura su hardware reale. Con i selettori puoi cambiare "
                 "le ipotesi e vedere quanto cambia la conclusione.")

        scenario, variante, rapporto = selettori_energia("energia")

        c = energia.energia_cnn(mac_cnn["mac_totali"], scenario, rapporto)
        e_snn = energia.energia_snn(en_snn["mac_conv1_per_passo"], en_snn["num_steps"],
                                    en_snn["sop_per_immagine"], en_snn["aggiornamenti_neuroni"],
                                    scenario, variante, rapporto)

        colonne = st.columns(3)
        colonne[0].metric("CNN", f"{c['totale']:.2f} µJ")
        colonne[1].metric("SNN (strato 1 una volta)", f"{e_snn['totale_conv1_una_volta']:.2f} µJ")
        colonne[2].metric("SNN (strato 1 a ogni passo)", f"{e_snn['totale_conv1_ogni_passo']:.2f} µJ")

        # barre impilate: CNN, SNN con strato 1 calcolato una volta, SNN con strato 1 a ogni passo
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        etichette = ["CNN", "SNN\n(strato 1 una volta)", "SNN\n(strato 1 a ogni passo)"]
        ax.bar(0, c["calcolo"], color="tab:blue", label="CNN: calcolo")
        ax.bar(0, c["memoria"], bottom=c["calcolo"], color="tab:purple", label="accessi alla memoria")
        for x, chiave, chiave_mem in ((1, "conv1_una_volta", "memoria_una_volta"),
                                      (2, "conv1_ogni_passo", "memoria_ogni_passo")):
            basso = 0
            for parte, colore, nome in ((chiave, "tab:orange", "SNN: strato 1 (MAC)"),
                                        ("sop", "tab:green", "SNN: somme sinaptiche"),
                                        ("neuroni", "tab:red", "SNN: aggiornamento neuroni"),
                                        (chiave_mem, "tab:purple", None)):
                ax.bar(x, e_snn[parte], bottom=basso, color=colore,
                       label=nome if x == 1 else None)
                basso += e_snn[parte]
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(etichette)
        ax.set_ylabel("Energia per immagine (µJ)")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

        st.write("**Attività della SNN** (media per immagine, su "
                 f"{en_snn['immagini_test']} immagini di test, {en_snn['num_steps']} passi):")
        for nome in ("lif1", "lif2", "lif3"):
            st.write(f"- strato {nome[-1]}: {en_snn['spike_per_immagine']['spike_' + nome]:,.0f} spike, "
                     f"tasso di spike {en_snn['tasso_di_spike'][nome] * 100:.1f}%")

        st.caption("Con la codifica diretta lo strato 1 riceve i pixel veri a ogni passo, "
                   "quindi è costoso: per questo mostro entrambe le varianti. Il modello della "
                   "memoria è grossolano (nessun riuso dei pesi nella CNN, un accesso per ogni "
                   "peso): serve a vedere la sensibilità, non a dare un valore.")


# ---------------------------------------------------------------------
# Scheda 3b: sweep sul numero di passi T
# ---------------------------------------------------------------------
with tab_sweep:
    sweep = leggi_json("sweep_T")
    mac_cnn = leggi_json("cnn_mac")
    storia_cnn = leggi_json("cnn")

    if sweep is None:
        st.info("Nessun risultato. Esegui 08_sweep_T.py (è lungo: un addestramento per ogni T).")
    else:
        st.write("Accuratezza ed energia della SNN al variare del numero di passi temporali T. "
                 "Più T è grande, più la SNN costa. L'articolo di Höfflin e Wassner (2026) "
                 "trova che il vantaggio energetico regge solo per T piccoli.")
        scenario, variante, rapporto = selettori_energia("sweep")

        valori_t = [r["num_steps"] for r in sweep]
        accuratezza = [r["accuratezza"] * 100 for r in sweep]
        una_volta, ogni_passo = [], []
        for r in sweep:
            e = energia.energia_snn(r["mac_conv1_per_passo"], r["num_steps"], r["sop_per_immagine"],
                                    r["aggiornamenti_neuroni"], scenario, variante, rapporto)
            una_volta.append(e["totale_conv1_una_volta"])
            ogni_passo.append(e["totale_conv1_ogni_passo"])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
        ax1.plot(valori_t, accuratezza, marker="o", color="tab:red", label="SNN")
        if storia_cnn is not None:
            ax1.axhline(storia_cnn["accuratezza"][-1] * 100, color="tab:blue", linestyle="--", label="CNN")
        ax1.set_xlabel("Passi temporali T")
        ax1.set_ylabel("Accuratezza sul test (%)")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(valori_t, una_volta, marker="o", color="tab:orange", label="SNN (strato 1 una volta)")
        ax2.plot(valori_t, ogni_passo, marker="s", color="tab:red", label="SNN (strato 1 a ogni passo)")
        if mac_cnn is not None:
            ax2.axhline(energia.energia_cnn(mac_cnn["mac_totali"], scenario, rapporto)["totale"],
                        color="tab:blue", linestyle="--", label="CNN")
        ax2.set_xlabel("Passi temporali T")
        ax2.set_ylabel("Energia per immagine (µJ)")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

        st.dataframe({"T": valori_t,
                      "Accuratezza (%)": [round(a, 2) for a in accuratezza],
                      "Energia, strato 1 una volta (µJ)": [round(x, 2) for x in una_volta],
                      "Energia, strato 1 a ogni passo (µJ)": [round(x, 2) for x in ogni_passo]})


# ---------------------------------------------------------------------
# Scheda 4: demo su una singola immagine
# ---------------------------------------------------------------------
with tab_demo:
    cnn_modello = carica_cnn()
    snn_modello = carica_snn()

    if cnn_modello is None or snn_modello is None:
        st.info("Servono i modelli addestrati (cnn_eurosat.pt e snn_eurosat.pt). "
                "Esegui prima 02_cnn.py e 05_snn.py.")
    else:
        test_set, nomi_classi = carica_test_set()

        sinistra, destra = st.columns([1, 3])
        with sinistra:
            if "indice" not in st.session_state:
                st.session_state["indice"] = 0
            if st.button("Immagine casuale"):
                st.session_state["indice"] = int(np.random.randint(len(test_set)))
            indice = st.number_input("Indice nel test set", 0, len(test_set) - 1,
                                     key="indice")
            passi = st.slider("Passi temporali della SNN", 5, 50, 25,
                              help="La SNN è addestrata con 25 passi. Cambiarli qui "
                                   "influisce solo sulla demo e può peggiorare la predizione.")

        immagine, etichetta = test_set[int(indice)]

        with sinistra:
            st.image(immagine_per_grafico(immagine), width=200, clamp=True)
            st.write(f"Classe vera: **{nomi_classi[etichetta]}**")

        # --- CNN: una passata, softmax sui punteggi ---
        with torch.no_grad():
            punteggi = cnn_modello(immagine.unsqueeze(0))
            prob_cnn = torch.softmax(punteggi, dim=1)[0]

        # --- SNN: registro gli spike di ogni strato a ogni passo con degli hook ---
        spike_strato = {"lif1": [], "lif2": []}
        h1 = snn_modello.lif1.register_forward_hook(
            lambda m, i, o: spike_strato["lif1"].append(o[0].sum().item()))
        h2 = snn_modello.lif2.register_forward_hook(
            lambda m, i, o: spike_strato["lif2"].append(o[0].sum().item()))
        snn_modello.num_steps = passi
        with torch.no_grad():
            spk, mem = snn_modello(immagine.unsqueeze(0))
        h1.remove()
        h2.remove()
        snn_modello.num_steps = 25                       # lo rimetto com'era

        spk = spk[:, 0, :]                               # [passi, classi]
        mem = mem[:, 0, :]
        conteggio = spk.sum(dim=0)                       # spike per classe

        pred_cnn = int(prob_cnn.argmax())
        pred_snn = int(conteggio.argmax())

        with destra:
            c1, c2 = st.columns(2)
            c1.metric("CNN predice", nomi_classi[pred_cnn],
                      "corretto" if pred_cnn == etichetta else "sbagliato",
                      delta_color="normal" if pred_cnn == etichetta else "inverse")
            c2.metric("SNN predice", nomi_classi[pred_snn],
                      "corretto" if pred_snn == etichetta else "sbagliato",
                      delta_color="normal" if pred_snn == etichetta else "inverse")

            fig, assi = plt.subplots(1, 3, figsize=(13, 3.8))

            # 1) probabilità della CNN
            assi[0].barh(nomi_classi, prob_cnn.numpy(), color="tab:blue")
            assi[0].set_title("CNN: probabilità per classe")
            assi[0].invert_yaxis()

            # 2) spike di uscita della SNN nel tempo (raster plot)
            tempi, classi = np.nonzero(spk.numpy())
            assi[1].scatter(tempi, classi, marker="|", s=120, color="tab:red")
            assi[1].set_yticks(range(len(nomi_classi)))
            assi[1].set_yticklabels(nomi_classi, fontsize=7)
            assi[1].set_xlim(-0.5, passi - 0.5)
            assi[1].set_xlabel("Passo temporale")
            assi[1].set_title("SNN: spike di uscita nel tempo")
            assi[1].invert_yaxis()

            # 3) spike totali per classe
            assi[2].barh(nomi_classi, conteggio.numpy(), color="tab:red")
            assi[2].set_title("SNN: spike totali per classe")
            assi[2].invert_yaxis()

            fig.tight_layout()
            st.pyplot(fig)

            # attività negli strati nascosti
            fig2, ax = plt.subplots(figsize=(6, 3))
            ax.plot(spike_strato["lif1"], label="strato 1")
            ax.plot(spike_strato["lif2"], label="strato 2")
            ax.set_xlabel("Passo temporale")
            ax.set_ylabel("Spike emessi")
            ax.set_title("Attività degli strati nascosti")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig2)

            st.caption("Nella SNN la classe prevista è quella che emette più spike. "
                       "I neuroni dello strato 1 ricevono i pixel a ogni passo, quindi "
                       "sono i più attivi.")


# ---------------------------------------------------------------------
# Scheda 5: lancio degli script dalla pagina
# ---------------------------------------------------------------------
with tab_esegui:
    st.write("Lancia gli script del progetto senza usare il terminale. "
             "L'output compare qui sotto mentre lo script gira.")
    st.warning("L'addestramento (script 2 e 5) può durare molti minuti, soprattutto su Raspberry Pi. "
               "Finché uno script gira la pagina è occupata: non cliccare altro. "
               "Per fermarlo davvero premi Ctrl+C nel terminale in cui hai avviato Streamlit.")

    scelta = None
    for testo, file in SCRIPT:
        if st.button(testo, key=file):
            scelta = file

    if scelta is not None:
        if not Path(scelta).exists():
            st.error(f"Il file {scelta} non è nella cartella del progetto.")
        else:
            st.write(f"**In esecuzione: {scelta}**")
            area = st.empty()
            righe = []

            # PYTHONUNBUFFERED fa comparire le righe man mano e non tutte alla fine
            ambiente = dict(os.environ, PYTHONUNBUFFERED="1")
            processo = subprocess.Popen([sys.executable, scelta],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, env=ambiente)
            for riga in processo.stdout:
                righe.append(riga.rstrip())
                area.code("\n".join(righe[-60:]))      # mostro le ultime 60 righe
            processo.wait()

            if processo.returncode == 0:
                st.success(f"{scelta} terminato.")
            else:
                st.error(f"{scelta} terminato con errore (codice {processo.returncode}). "
                         "Leggi le ultime righe qui sopra.")

            # i modelli e i risultati sono cambiati: svuoto la cache per rileggerli
            st.cache_resource.clear()

            # se lo script ha prodotto grafici, li mostro
            for nome in ("confronto.png", "confronto_energia.png"):
                if scelta == "07_confronto.py" and (CARTELLA / nome).exists():
                    st.image(str(CARTELLA / nome))
