from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


folder = Path(__file__).resolve().parent
excel_file = folder / "onset_phase_entrainment_data.xlsx"

sheets = {"13Hz": 13.0, "29.4Hz": 29.4, "43Hz": 43.0}
colors = {13.0: "tab:orange", 29.4: "tab:red", 43.0: "tab:purple"}
markers = {13.0: "o", 29.4: "s", 43.0: "^"}

fig, ax = plt.subplots(figsize=(8, 5))

for sheet, frequency in sheets.items():
    data = pd.read_excel(excel_file, sheet_name=sheet)

    summary = (
        data.groupby("onset_deg")["numero_nodos_entrenados"]
        .agg(["mean", "std"])
        .reset_index()
    )

    x = summary["onset_deg"].to_numpy()
    mean = summary["mean"].to_numpy()
    sd = summary["std"].to_numpy()

    ax.plot(
        x,
        mean,
        color=colors[frequency],
        marker=markers[frequency],
        linewidth=2,
        label=f"{frequency:g} Hz",
    )

    ax.fill_between(
        x,
        mean - sd,
        mean + sd,
        color=colors[frequency],
        alpha=0.18,
    )

ax.set_xticks(
    [0, 45, 90, 135, 180, 225, 270, 315],
    ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$", r"$5\pi/4$", r"$3\pi/2$", r"$7\pi/4$"],
)

ax.set_xlabel("Stimulation onset phase [rad]")
ax.set_ylabel("Number of entrained nodes")
ax.set_ylim(0, 90)
ax.set_title("Effect of stimulation onset phase (Right Precuneus)")
ax.legend(title="Stimulation frequency", frameon=False)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()
plt.savefig(folder / "effect_onset_phase_right_precuneus.png", dpi=300)
plt.show()
