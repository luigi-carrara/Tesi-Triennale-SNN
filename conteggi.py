# =====================================================================
# conteggi.py
# Conta gli spike di una SNN addestrata sul test set e ne ricava le
# operazioni (SOP, aggiornamenti dei neuroni, MAC dello strato 1).
# Usato da 06_energia_snn.py e da 08_sweep_T.py.
#
# Cosa si conta, per ogni immagine:
#   - strato 1 (codifica diretta): MAC, perché riceve i pixel veri;
#   - strati successivi: operazioni sinaptiche (SOP), cioè una somma per
#     ogni spike e per ogni neurone che lo riceve;
#   - aggiornamento dei potenziali di membrana a ogni passo.
# Il max-pooling lavora sulle correnti prima del LIF: non costa operazioni.
# =====================================================================

import torch
import torch.nn.functional as F
from dati import CANALI, LATO


def conta_attivita(model, test_loader):
    """Fa girare la SNN sul test set e restituisce i conteggi medi per immagine."""
    T = model.num_steps
    model.eval()

    # Quante somme provoca uno spike in ciascuna posizione dell'ingresso di una
    # convoluzione senza padding. Ai bordi un ingresso tocca meno posizioni di
    # uscita che al centro, quindi la mappa non è uniforme.
    mappe = {}

    def fan_out(conv, h, w):
        chiave = (id(conv), h, w)
        if chiave not in mappe:
            k = conv.kernel_size[0]
            uscita = torch.ones(1, 1, h - k + 1, w - k + 1)
            pesi = torch.ones(1, 1, k, k)
            posizioni = F.conv_transpose2d(uscita, pesi)       # [1, 1, h, w]
            mappe[chiave] = posizioni * conv.out_channels      # ogni posizione vale per tutti i filtri
        return mappe[chiave]

    conteggi = {"spike_lif1": 0.0, "spike_lif2": 0.0, "spike_lif3": 0.0,
                "sop_conv2": 0.0, "sop_fc": 0.0}
    neuroni = {}     # neuroni di ogni strato LIF, per immagine

    # Gli hook leggono gli spike di ogni strato LIF durante l'inferenza.
    # uscita = (spike, potenziale): a noi servono gli spike.
    def hook_lif1(modulo, ingresso, uscita):
        spk = uscita[0]                                  # [batch, 12, h, w]
        conteggi["spike_lif1"] += spk.sum().item()
        # questi spike entrano in conv2
        conteggi["sop_conv2"] += (spk * fan_out(model.conv2, spk.shape[-2], spk.shape[-1])).sum().item()
        neuroni["lif1"] = spk[0].numel()

    def hook_lif2(modulo, ingresso, uscita):
        spk = uscita[0]                                  # [batch, 64, h, w]
        conteggi["spike_lif2"] += spk.sum().item()
        # questi spike entrano nello strato lineare: ogni spike raggiunge tutte le uscite
        conteggi["sop_fc"] += spk.sum().item() * model.fc.out_features
        neuroni["lif2"] = spk[0].numel()

    def hook_lif3(modulo, ingresso, uscita):
        spk = uscita[0]                                  # [batch, classi]
        conteggi["spike_lif3"] += spk.sum().item()       # strato d'uscita: nessuno strato dopo
        neuroni["lif3"] = spk[0].numel()

    hook = [model.lif1.register_forward_hook(hook_lif1),
            model.lif2.register_forward_hook(hook_lif2),
            model.lif3.register_forward_hook(hook_lif3)]

    corretti = 0
    n_immagini = 0
    with torch.no_grad():
        for immagini, etichette in test_loader:
            spk, _ = model(immagini)
            corretti += (spk.sum(dim=0).argmax(dim=1) == etichette).sum().item()
            n_immagini += immagini.size(0)

    for h in hook:
        h.remove()       # così il modello si può riusare senza conteggiare due volte

    # MAC dello strato 1 per ogni passo temporale
    k1 = model.conv1.kernel_size[0]
    with torch.no_grad():
        uscite_conv1 = model.conv1(torch.zeros(1, CANALI, LATO, LATO)).numel()
    mac_conv1_passo = uscite_conv1 * model.conv1.in_channels * k1 * k1

    spike_per_immagine = {k: v / n_immagini for k, v in conteggi.items() if k.startswith("spike")}
    sop_per_immagine = (conteggi["sop_conv2"] + conteggi["sop_fc"]) / n_immagini
    aggiornamenti = sum(neuroni.values()) * T            # aggiornamenti dei potenziali per immagine

    # Tasso di spike: quota di (neurone, passo) in cui il neurone ha sparato
    tasso = {nome: spike_per_immagine["spike_" + nome] / (neuroni[nome] * T)
             for nome in ("lif1", "lif2", "lif3")}

    return {"num_steps": T, "immagini_test": n_immagini,
            "accuratezza": corretti / n_immagini,
            "spike_per_immagine": spike_per_immagine, "tasso_di_spike": tasso,
            "sop_per_immagine": sop_per_immagine,
            "mac_conv1_per_passo": mac_conv1_passo,
            "aggiornamenti_neuroni": aggiornamenti}
