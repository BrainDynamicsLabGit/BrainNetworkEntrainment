# -*- coding: utf-8 -*-
"""Reproduce los heatmaps de activación a partir del Excel.

Coloque este archivo y Activacion_heatmaps_desde_figura.xlsx en la misma carpeta.
Del Excel se lee únicamente la hoja 'Activacion'.
"""

from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Activacion_heatmaps_desde_npy.xlsx"


NODES = [68, 1, 70]
NODE_NAMES = {
    68: "R Precuneus",
    1: "L Precentral",
    70: "R Paracentr Lob",
}
FREQUENCIES = [13.0, 29.4, 43.0]
COMBINATIONS = [
    (4.0, 21.0),
    (5.0, 19.0),
    (5.0, 20.0),
    (5.5, 22.0),
    (4.5, 19.0),
    (4.5, 18.0),
    (5.0, 21.0),
]
FREQUENCY_COLORS = {
    13.0: "orange",
    29.4: "red",
    43.0: "magenta",
}



activation = pd.read_excel(EXCEL_FILE, sheet_name="Activacion")
labels = (
    activation[["Nodo_AAL", "Region_AAL"]]
    .drop_duplicates()
    .sort_values("Nodo_AAL")
    ["Region_AAL"]
    .tolist()
)

figure = plt.figure(figsize=(18, 16))
grid = GridSpec(
    1,
    11,
    figure=figure,
    width_ratios=[1, 1, 1, 0.22, 1, 1, 1, 0.22, 1, 1, 1],
    wspace=0.15,
)

axes_positions = [[0, 1, 2], [4, 5, 6], [8, 9, 10]]

for node_index, node in enumerate(NODES):
    group_axes = []

    for frequency_index, frequency in enumerate(FREQUENCIES):
        axis = figure.add_subplot(grid[0, axes_positions[node_index][frequency_index]])
        group_axes.append(axis)

        selected = activation[
            (activation["Nodo_estimulado"] == node)
            & np.isclose(activation["Frecuencia_Hz"], frequency)
        ]

        matrix = np.zeros((90, len(COMBINATIONS)), dtype=int)

        for row in selected.itertuples(index=False):
            matrix[int(row.Nodo_AAL) - 1, int(row.Orden_combinacion) - 1] = int(row.Activado)

        cmap = colors.ListedColormap(["lightgray", FREQUENCY_COLORS[frequency]])

        sns.heatmap(
            matrix,
            cmap=cmap,
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            linewidths=0.7,
            linecolor="black",
            xticklabels=[f"K={k:g} MD={md:g}" for k, md in COMBINATIONS],
            yticklabels=labels if node_index == 0 and frequency_index == 0 else False,
            ax=axis,
        )

        axis.set_title(f"f={frequency:g} Hz", fontsize=12, pad=7)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", labelrotation=90, labelsize=7, length=0)

        if node_index == 0 and frequency_index == 0:
            axis.tick_params(axis="y", labelsize=7, length=0)
        else:
            axis.tick_params(axis="y", left=False, labelleft=False)


    first_position = group_axes[0].get_position()
    last_position = group_axes[-1].get_position()
    center = (first_position.x0 + last_position.x1) / 2
    figure.text(
        center,
        0.965,
        NODE_NAMES[node],
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )



