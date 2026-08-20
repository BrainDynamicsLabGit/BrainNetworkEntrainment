from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


folder = Path(__file__).resolve().parent
data = pd.read_excel(folder / "amplitude_sensitivity.xlsx")

frequencies = [13.0, 29.4, 43.0]
amplitudes = list(range(100, 1100, 100))
colors = {13.0: "tab:orange", 29.4: "tab:red", 43.0: "tab:purple"}


def create_figure(nodes, part):
    fig, axes = plt.subplots(5, 9, figsize=(20, 12), sharex=True, sharey=True)

    for ax, node in zip(axes.flat, nodes):
        node_data = data[data["indice_nodo"] == node]

        for frequency in frequencies:
            values = (
                node_data[node_data["frecuencia"] == frequency]
                .sort_values("amplitud")
            )

            ax.plot(
                values["amplitud"],
                values["numero_nodos_entrenados"],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=colors[frequency],
            )

            maximum = values["numero_nodos_entrenados"].max()
            maximum_values = values[values["numero_nodos_entrenados"] == maximum]

            ax.scatter(
                maximum_values["amplitud"],
                maximum_values["numero_nodos_entrenados"],
                marker="D",
                s=35,
                color=colors[frequency],
                zorder=5,
            )

        node_name = node_data["nodo_estimulado"].iloc[0]
        ax.text(
            0.97,
            0.93,
            node_name,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "lightgray"},
        )

        ax.set_xticks(amplitudes)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, 90)
        ax.grid(alpha=0.25)

    legend = [
        Line2D([0], [0], color=colors[f], marker="o", label=f"{f:g} Hz")
        for f in frequencies
    ]

    fig.legend(
        handles=legend,
        title="Stimulation frequency",
        loc="upper center",
        ncol=3,
        frameon=False,
    )
    fig.supxlabel("Stimulation amplitude")
    fig.supylabel("Number of entrained nodes")
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.94])
    plt.savefig(folder / f"amplitude_sensitivity_part_{part}.png", dpi=300)
    plt.show()


create_figure(range(0, 45), part=1)
create_figure(range(45, 90), part=2)
