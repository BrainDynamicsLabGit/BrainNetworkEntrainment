# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 05:58:46 2025

@author: elida
"""

import numpy as np 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from pathlib import Path
# Cargar archivo .npz
BASE_DIR = Path(__file__).resolve().parent
direct = BASE_DIR / "data.npz"
data = np.load(direct)
H = data["H"]       # (20, 40)
Pxx_all = data["Pxx"]

freqs = np.linspace(0, 80, 401)
K_all_values = np.arange(0, 10, 0.5)   # 20 valores
MD_all_values = np.arange(0, 41, 1)    # 41 valores

# --- Restricciones ---
k_mask = (K_all_values >= 3.5) & (K_all_values <= 6)
md_mask = (MD_all_values >= 17) & (MD_all_values <= 24)

K_sel = K_all_values[k_mask]
MD_sel = MD_all_values[md_mask]

fig1_spectrums = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(len(K_sel), len(MD_sel), wspace=0.05, hspace=0.05)

# Etiquetas solo para rango seleccionado
plt.xticks(np.arange(len(MD_sel)) + 0.5, MD_sel, fontsize=18)
plt.yticks(np.arange(len(K_sel)) + 0.5, [f"{val:.2f}" for val in K_sel], fontsize=18)

plt.xlabel("mean delay (ms)", fontsize=24)
plt.ylabel("global coupling", fontsize=24)

# --- Puntos a marcar con colores ---
puntos = {(4, 21): "yellow", (5.5, 22): "turquoise"}

for k_idx, K in enumerate(K_sel):
    for md_idx, MD in enumerate(MD_sel):
        ax = fig1_spectrums.add_subplot(gs[len(K_sel)-1-k_idx, md_idx])

        # Si este punto es especial → colorear fondo
        if (K, MD) in puntos:
            ax.set_facecolor(puntos[(K, MD)])
            ax.patch.set_alpha(0.8)  # transparencia (0=transparente, 1=opaco)

        # Graficar espectro en negro por encima
        ax.plot(freqs, Pxx_all[np.where(K_all_values == K)[0][0],
                               np.where(MD_all_values == MD)[0][0],
                               0:401], 'k', linewidth=2)

        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
