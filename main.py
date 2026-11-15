# =====================================================================
# main.py
# Pannello di controllo della tesi: lancia gli script del caso di studio
# e ne mostra l'output in tempo reale, senza doverli avviare uno a uno.
# Gli script restano indipendenti: il pannello li esegue come processi separati.
# =====================================================================

import os
import sys
import queue
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext

# Cartella in cui si trova main.py: gli script vengono eseguiti da qui,
# così trovano la cartella "data" e i file .pt
CARTELLA = Path(__file__).resolve().parent

# Elenco degli script: (testo del pulsante, nome del file)
SCRIPT = [
    ("1. Controllo dati EuroSAT", "01_dati.py"),
    ("2. Addestra CNN", "02_cnn.py"),
    ("3. Conta MAC della CNN", "03_mac_cnn.py"),
    ("4. Prova SNN (confronto tra CNN addestrata e SNN NON addestrata)", "04_snn_prova.py"),
    ("5. Addestra SNN", "05_snn.py"),
    ("6. Energia della SNN (spike reali)", "06_energia_snn.py"),
    ("7. Grafici di confronto", "07_confronto.py"),
    ("8. Sweep su T (molto lungo)", "08_sweep_T.py"),
]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tesi SNN - Pannello di controllo")
        self.geometry("950x600")

        self.processo = None          # script in esecuzione (None se nessuno)
        self.coda = queue.Queue()     # righe di output in arrivo dallo script

        # --- Colonna di sinistra: pulsanti ---
        colonna = ttk.Frame(self, padding=10)
        colonna.pack(side="left", fill="y")

        ttk.Label(colonna, text="Esperimenti", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))

        self.pulsanti = []
        for testo, file in SCRIPT:
            # f=file "congela" il nome del file per ciascun pulsante
            b = ttk.Button(colonna, text=testo, width=28, command=lambda f=file: self.avvia(f))
            b.pack(fill="x", pady=3)
            self.pulsanti.append(b)

        self.btn_stop = ttk.Button(colonna, text="Interrompi", command=self.interrompi, state="disabled")
        self.btn_stop.pack(fill="x", pady=(20, 3))

        ttk.Button(colonna, text="Pulisci output", command=self.pulisci).pack(fill="x", pady=3)

        self.stato = ttk.Label(colonna, text="Pronto", foreground="gray")
        self.stato.pack(anchor="w", pady=(15, 0))

        # --- Area di destra: output degli script ---
        self.output = scrolledtext.ScrolledText(self, font=("Consolas", 10), state="disabled")
        self.output.pack(side="right", fill="both", expand=True, padx=(0, 10), pady=10)

        # controlla la coda ogni 100 ms e chiude correttamente la finestra
        self.after(100, self.aggiorna_output)
        self.protocol("WM_DELETE_WINDOW", self.chiudi)

    # ---------------- gestione dell'output ----------------
    def scrivi(self, testo):
        """Aggiunge testo in fondo all'area di output."""
        self.output.configure(state="normal")
        self.output.insert("end", testo)
        self.output.see("end")                 # scorre automaticamente in basso
        self.output.configure(state="disabled")

    def pulisci(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    # ---------------- esecuzione degli script ----------------
    def avvia(self, file):
        """Lancia uno script come processo separato."""
        if self.processo is not None:          # uno script alla volta
            return
        self.scrivi(f"\n===== Avvio {file} =====\n")

        # -u: output non bufferizzato, così le righe arrivano subito
        # PYTHONIOENCODING: caratteri come "µ" arrivano correttamente
        ambiente = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.processo = subprocess.Popen(
            [sys.executable, "-u", file],      # sys.executable = il Python del .venv
            cwd=CARTELLA,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,   # anche gli errori nell'output
            text=True, encoding="utf-8", errors="replace", env=ambiente)

        # un thread separato legge l'output, così la finestra non si blocca
        threading.Thread(target=self.leggi, args=(self.processo,), daemon=True).start()
        self.imposta_in_esecuzione(True, file)

    def leggi(self, processo):
        """Eseguito nel thread: legge le righe dello script e le mette in coda."""
        for riga in processo.stdout:
            self.coda.put(riga)
        processo.wait()
        self.coda.put(None)                    # None = segnale di fine

    def aggiorna_output(self):
        """Eseguito nella finestra ogni 100 ms: svuota la coda e mostra le righe.
        (tkinter non permette di modificare la finestra da un altro thread,
        per questo si passa dalla coda.)"""
        try:
            while True:
                riga = self.coda.get_nowait()
                if riga is None:
                    self.scrivi(f"===== Terminato (codice {self.processo.returncode}) =====\n")
                    self.processo = None
                    self.imposta_in_esecuzione(False)
                else:
                    self.scrivi(riga)
        except queue.Empty:
            pass
        self.after(100, self.aggiorna_output)

    def interrompi(self):
        if self.processo is not None:
            self.scrivi("\n[Interruzione richiesta: i risultati di questa esecuzione non verranno salvati]\n")
            self.processo.terminate()

    def imposta_in_esecuzione(self, attivo, file=""):
        """Disabilita i pulsanti durante l'esecuzione e aggiorna lo stato."""
        for b in self.pulsanti:
            b.configure(state="disabled" if attivo else "normal")
        self.btn_stop.configure(state="normal" if attivo else "disabled")
        self.stato.configure(text=f"In esecuzione: {file}" if attivo else "Pronto",
                             foreground="darkorange" if attivo else "gray")

    def chiudi(self):
        """Alla chiusura della finestra, ferma anche lo script in corso."""
        if self.processo is not None:
            self.processo.terminate()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()