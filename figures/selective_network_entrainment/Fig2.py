# -*- coding: utf-8 -*-
"""Genera los glass brains y las vistas corticales leyendo un Excel.

Coloque en la misma carpeta que este código:
    - Datos_propagacion_AAL.xlsx
    - pial_both.gii
    - sulc_both.gii

Los archivos NPY y AAL_labels.mat ya no son necesarios para crear la figura.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import nilearn.datasets
import nilearn.image
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting, surface


BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Entreinment_data.xlsx"
MESH = BASE_DIR / "pial_both.gii"
SULC_MAP = BASE_DIR / "sulc_both.gii"
OUTPUT_DIR = BASE_DIR / "Resultados_figura"
VIEWS = ["lateral", "posterior", "dorsal","medial"]

# La configuración de la figura queda definida directamente en el código.
NODE_STIM = [68, 1, 70]

NODE_NAMES = {
    68: "R Precuneus",
    1: "L Precentral",
    70: "R Paracentr Lob",
}

NODE_HEMISPHERE = {
    68: "right",
    1: "left",
    70: "right",
}

FSTIM_LIST = [13.0, 29.4, 43.0]

FREQUENCY_COLORS = {
    13.0: ["#FCE6B5", "#E2A126"],      # naranja
    29.4: ["#FADBD8", "#E57368"],      # salmón
    43.0: ["#E5CCF2", "#913DB4"],      # morado
}


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Del Excel se cargan únicamente los nodos propagados y sus valores.
propagated = pd.read_excel(EXCEL_FILE, sheet_name="Entrenamiento")

# Cargar el atlas AAL utilizado para transformar nodos en regiones volumétricas.
atlas_data = nilearn.datasets.fetch_atlas_aal(version="SPM12")
aal_img = nilearn.image.load_img(atlas_data.maps)
aal_data = aal_img.get_fdata()

for node in NODE_STIM:
    node_name = NODE_NAMES[node]
    hemisphere = NODE_HEMISPHERE[node]

    node_output_dir = OUTPUT_DIR / f"Node_{node}"
    node_output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # GLASS BRAIN: UNA SOLA VEZ POR NODO
    # ========================================================

    stimulated_region = int(atlas_data["indices"][node - 1])
    stimulated_data = np.zeros_like(aal_data)
    stimulated_data[aal_data == stimulated_region] = 1

    stimulated_img = nilearn.image.new_img_like(
        aal_img,
        stimulated_data,
        affine=aal_img.affine,
        copy_header=True,
    )

    glass_display = plotting.plot_glass_brain(
        stimulated_img,
        title=node_name,
        threshold=0.5,
        cmap="gray_r",
        colorbar=False,
        black_bg=False,
        alpha=0.9,
    )
    glass_display.savefig(
        node_output_dir / f"Glass_brain_Node_{node}.png",
        dpi=300,
    )
    glass_display.close()

    # ========================================================
    # TRES FRECUENCIAS × TRES VISTAS
    # ========================================================

    for frequency in FSTIM_LIST:
        colors = FREQUENCY_COLORS[frequency]
        custom_cmap = LinearSegmentedColormap.from_list(
            f"cmap_{frequency:g}",
            colors,
        )

        selected_rows = propagated[
            (propagated["Nodo_estimulado"] == node)
            & np.isclose(propagated["Frecuencia_Hz"], frequency)
        ]

        propagated_data = np.zeros_like(aal_data)

        for propagated_row in selected_rows.itertuples(index=False):
            propagated_node = int(propagated_row.Nodo_AAL)
            propagated_region = int(
                atlas_data["indices"][propagated_node - 1]
            )
            propagated_data[
                aal_data == propagated_region
            ] = float(propagated_row.Valor_propagacion)

        propagated_img = nilearn.image.new_img_like(
            aal_img,
            propagated_data,
            affine=aal_img.affine,
            copy_header=True,
        )
        surface_texture = surface.vol_to_surf(
            propagated_img,
            str(MESH),
        )

        for view in VIEWS:
            figure = plotting.plot_surf(
                str(MESH),
                surface_texture,
                hemi=hemisphere,
                view=view,
                threshold=0.1,
                cmap=custom_cmap,
                bg_map=str(SULC_MAP),
                colorbar=False,
                title=view.capitalize(),
            )

            output_file = (
                node_output_dir
                / f"Node_{node}_Fstim_{frequency:g}_{view}.png"
            )
            figure.savefig(
                output_file,
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.05,
                transparent=True,
            )
            plt.close(figure)

  
