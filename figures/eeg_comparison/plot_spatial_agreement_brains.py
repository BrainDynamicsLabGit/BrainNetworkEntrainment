from pathlib import Path
import difflib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from nilearn import datasets, image, plotting


folder = Path(__file__).resolve().parent
excel_file = folder / "visual_eeg_simulation_correspondence.xlsx"
output_folder = folder / "brain_figures"
output_folder.mkdir(exist_ok=True)

best = pd.read_excel(excel_file, sheet_name="mejor_amplitud")
regions = pd.read_excel(excel_file, sheet_name="regiones_84")


# Atlas AAL de Nilearn.
aal = datasets.fetch_atlas_aal(version="SPM12")
atlas_img = image.load_img(aal.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = list(aal.labels)
atlas_indices = list(aal.indices)
label_to_index = {name: i for i, name in enumerate(atlas_labels)}


def atlas_name(region):
    if region in label_to_index:
        return region

    equivalents = {
        "ParaHippocamp_L": "ParaHippocampal_L",
        "ParaHippocamp_R": "ParaHippocampal_R",
    }

    if region in equivalents:
        return equivalents[region]

    hemisphere = region[-2:] if region.endswith(("_L", "_R")) else ""
    candidates = [
        name for name in atlas_labels
        if not hemisphere or name.endswith(hemisphere)
    ]
    match = difflib.get_close_matches(region, candidates, n=1, cutoff=0.75)
    return match[0] if match else None


def create_mask(groups):
    mask = np.zeros_like(atlas_data, dtype=float)

    for region_names, value in groups:
        for region in region_names:
            name = atlas_name(region)
            if name is not None:
                code = int(atlas_indices[label_to_index[name]])
                mask[atlas_data == code] = value

    return image.new_img_like(atlas_img, mask)


cmap = ListedColormap([
    (0.92, 0.92, 0.92, 1.0),
    "#2F6BFF",  # EEG only
    "#00FFFF",  # Simulation only
    "#2CA02C",  # Agreement
])

legend = [
    Patch(facecolor="#2F6BFF", label="EEG only"),
    Patch(facecolor="#00FFFF", label="Simulation only"),
    Patch(facecolor="#2CA02C", label="Agreement"),
]


for row_index, row in best.iterrows():
    frequency = float(row["frecuencia_hz"])
    amplitude = float(row["amplitud"])

    data = regions[
        np.isclose(regions["frecuencia_hz"], frequency)
        & np.isclose(regions["amplitud"], amplitude)
    ]

    eeg_active = set(data.loc[data["eeg_activa"] == 1, "region"].astype(str))
    TP = set(data.loc[(data["eeg_activa"] == 1) & (data["simulada_activa"] == 1), "region"].astype(str))
    FP = set(data.loc[(data["eeg_activa"] == 0) & (data["simulada_activa"] == 1), "region"].astype(str))
    FN = set(data.loc[(data["eeg_activa"] == 1) & (data["simulada_activa"] == 0), "region"].astype(str))

    # Las metricas se calculan, pero no se crea una figura para ellas.
    best.loc[row_index, "TP_calculated"] = len(TP)
    best.loc[row_index, "FP_calculated"] = len(FP)
    best.loc[row_index, "FN_calculated"] = len(FN)
    best.loc[row_index, "matched_eeg_regions_percent"] = 100 * len(TP) / len(eeg_active)

    brain_mask = create_mask([
        (FN, 1),
        (FP, 2),
        (TP, 3),
    ])

    fig, ax = plt.subplots(figsize=(16, 5))

    plotting.plot_glass_brain(
        brain_mask,
        display_mode="lyrz",
        cmap=cmap,
        threshold=0.1,
        vmin=0,
        vmax=3,
        colorbar=False,
        axes=ax,
    )

    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=16,
    )

    plt.tight_layout(rect=[0, 0.12, 1, 0.90])
    plt.savefig(
        output_folder / f"glass_brain_{frequency:g}Hz_A{amplitude:g}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


print("\nMetrics for the best amplitude of each frequency:")
print(
    best[
        [
            "frecuencia_hz",
            "amplitud",
            "pearson_r",
            "spearman_rho",
            "dice",
            "TP_calculated",
            "FP_calculated",
            "FN_calculated",
            "matched_eeg_regions_percent",
        ]
    ]
)
