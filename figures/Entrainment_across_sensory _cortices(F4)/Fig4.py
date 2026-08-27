# -*- coding: utf-8 -*-


from pathlib import Path
import matplotlib.pyplot as plt
import nilearn.datasets
import nilearn.image
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting, surface

BASE_DIR = Path(__file__).resolve().parent
EXCEL_FILE = BASE_DIR / "Entrenamiento_data.xlsx"
MESH = BASE_DIR / "pial_both.gii"
SULC_MAP = BASE_DIR / "sulc_both.gii"
OUTPUT_DIR = BASE_DIR / "Resultados_figura"

NODES = [43, 79]
NODE_NAMES = {43: "L Calcarine", 79: "L Heschl"}
NODE_HEMISPHERE = {43: "left", 79: "left"}
FREQUENCIES = [13.0, 29.4, 43.0]
COMBINATIONS = [(4.0, 21.0), (5.5, 22.0)]
VIEWS = ["medial", "posterior", "dorsal"]
FREQUENCY_COLORS = {
    13.0: ["#FCE6B5", "#E2A126"],
    29.4: ["#FADBD8", "#E57368"],
    43.0: ["#E5CCF2", "#913DB4"],
}

for path in (EXCEL_FILE, MESH, SULC_MAP):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró: {path}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
propagated = pd.read_excel(EXCEL_FILE, sheet_name="Entrenamiento")
required = {"Nodo_estimulado", "Frecuencia_Hz", "K", "MD_ms", "Nodo_AAL", "Valor_propagacion", "Activado"}
missing = required - set(propagated.columns)
if missing:
    raise ValueError(f"Faltan columnas: {sorted(missing)}")

atlas = nilearn.datasets.fetch_atlas_aal(version="SPM12")
aal_img = nilearn.image.load_img(atlas.maps)
aal_data = aal_img.get_fdata()

for node in NODES:
    node_dir = OUTPUT_DIR / f"Node_{node}"
    node_dir.mkdir(parents=True, exist_ok=True)

    # Glass brain: una sola vez por nodo.
    stimulated_region = int(atlas["indices"][node - 1])
    stimulated_data = np.zeros_like(aal_data)
    stimulated_data[aal_data == stimulated_region] = 1
    stimulated_img = nilearn.image.new_img_like(aal_img, stimulated_data, affine=aal_img.affine, copy_header=True)
    display = plotting.plot_glass_brain(stimulated_img, title=NODE_NAMES[node], threshold=0.5, cmap="gray_r", colorbar=False, black_bg=False, alpha=0.9)
    display.savefig(node_dir / f"Glass_brain_Node_{node}.png", dpi=300)
    display.close()

    for k_value, md_value in COMBINATIONS:
        combination_dir = node_dir / f"K_{k_value:g}_MD_{md_value:g}"
        combination_dir.mkdir(parents=True, exist_ok=True)

        for frequency in FREQUENCIES:
            rows = propagated[
                (propagated["Nodo_estimulado"] == node)
                & np.isclose(propagated["Frecuencia_Hz"], frequency)
                & np.isclose(propagated["K"], k_value)
                & np.isclose(propagated["MD_ms"], md_value)
                & (propagated["Activado"] == 1)
            ]
            if rows.empty:
                raise ValueError(f"Sin datos activos: nodo={node}, f={frequency}, K={k_value}, MD={md_value}")

            volume = np.zeros_like(aal_data)
            for row in rows.itertuples(index=False):
                region = int(atlas["indices"][int(row.Nodo_AAL) - 1])
                volume[aal_data == region] = float(row.Valor_propagacion)

            propagated_img = nilearn.image.new_img_like(aal_img, volume, affine=aal_img.affine, copy_header=True)
            texture = surface.vol_to_surf(propagated_img, str(MESH))
            cmap = LinearSegmentedColormap.from_list(f"cmap_{frequency:g}", FREQUENCY_COLORS[frequency])

            for view in VIEWS:
                figure = plotting.plot_surf(
                    str(MESH), texture, hemi=NODE_HEMISPHERE[node], view=view,
                    threshold=0.1, cmap=cmap, bg_map=str(SULC_MAP), colorbar=False,
                    title=f"{NODE_NAMES[node]} | K={k_value:g}, MD={md_value:g} | f={frequency:g} Hz | {view.capitalize()}",
                )
                figure.savefig(
                    combination_dir / f"Node_{node}_K_{k_value:g}_MD_{md_value:g}_Fstim_{frequency:g}_{view}.png",
                    dpi=300, bbox_inches="tight", pad_inches=0.05, transparent=True,
                )
                plt.close(figure)



