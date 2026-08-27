from pathlib import Path

import numpy as np
import scipy.io as sio

from KuramotoClassFor import Kuramoto


# PARAMETROS
N_NODES = 90
K = 4
MEAN_DELAY = 0.021
NATURAL_FREQUENCY = 40

SIMULATION_PERIOD = 80
DT = 1e-3

STIM_TSTART = 30
STIM_TEND = 50
STIM_FREQUENCY = 43
STIM_AMPLITUDE = 400
STIMULATED_NODES = [67]

SEED = 789


# CARGAR LAS MATRICES C Y D
PROJECT_ROOT = Path(__file__).resolve().parent
data = sio.loadmat(PROJECT_ROOT / "input_data" / "AAL_matrices.mat")

C = data["C"].astype(float)
D = data["D"].astype(float)

C[~np.eye(N_NODES, dtype=bool)] /= C[~np.eye(N_NODES, dtype=bool)].mean()
D /= 1000


# CREAR Y EJECUTAR EL MODELO
np.random.seed(SEED)

model = Kuramoto(
    struct_connectivity=C,
    delays_matrix=D,
    K=K,
    dt=DT,
    simulation_period=SIMULATION_PERIOD,
    StimTstart=STIM_TSTART,
    StimTend=STIM_TEND,
    StimFreq=STIM_FREQUENCY,
    StimWeigth=STIM_AMPLITUDE,
    n_nodes=N_NODES,
    natfreqs=NATURAL_FREQUENCY,
    GenerateRandom=False,
    SEED=SEED,
    mean_delay=MEAN_DELAY,
)

model.initializeForcingNodes(STIMULATED_NODES)
R, Dynamics = model.simulate(Forced=True)


# GUARDAR EL RESULTADO
OUTPUT_DIR = PROJECT_ROOT / "output_example"
OUTPUT_DIR.mkdir(exist_ok=True)

sio.savemat(
    OUTPUT_DIR / "simulacion_ejemplo.mat",
    {
        "theta": Dynamics,
        "kop": np.asarray(R),
        "nodes": STIMULATED_NODES,
        "fstim": STIM_FREQUENCY,
        "weight": STIM_AMPLITUDE,
        "K": K,
        "mean_delay": MEAN_DELAY,
    },
)

print("Simulacion terminada.")
