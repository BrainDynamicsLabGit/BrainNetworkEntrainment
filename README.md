# BrainNetworkEntrainment

Code associated with the study **Selective brain network stimulation by frequency entrainment**.

## Overview

This repository implements a forced Kuramoto model with delays to
simulate frequency entrainment in structural brain networks. 

The model is described by:

$$
\dot{\theta}_n(t)=\omega_n
+A\delta_{n,E}\sin(2\pi f t-\theta_n(t))
+\frac{K}{N}\sum_{p=1}^{N}c_{np}
\sin\left[\theta_p(t-\tau_{np})-\theta_n(t)\right].
$$

where $\theta_n$ is the phase of node $n$, $\omega_n$ is its natural
frequency, $A$ and $f$ are the stimulation amplitude and frequency,
$c_{np}$ is the structural connectivity, and $\tau_{np}$ is the transmission
delay between nodes.

## Requirements

- Python 3
- NumPy
- SciPy
- pandas
- Matplotlib
- openpyxl
- tqdm
- JiTCDDE
- SymEngine
- Nilearn

Install the dependencies with:

```bash
pip install numpy scipy pandas matplotlib openpyxl tqdm jitcdde symengine nilearn
```

## Input data

### Simulation

The file `input_data/AAL_matrices.mat` must contain:

- `C`: structural connectivity matrix.
- `D`: inter-regional distance matrix.

The example is prepared for the AAL90 parcellation.

## Running the simulation example

From the main repository folder, run:

```bash
python examples/ejecutar_simulacion_ejemplo.py
```

The example simulates a network of 90 nodes, uses a natural frequency of 40 Hz,
and applies a 43 Hz stimulation to a selected node. The simulation parameters
can be changed directly at the beginning of the script.

## Output

The simulation produces a mat file containing the phase time series and the
Kuramoto order parameter. 

## Citation

If you use this code, please cite:

Otero, M., Poo, E., Torres, F., Mendoza, C., Lea-Carnall, C., Weinstein, A., ... & El-Deredy, W. (2025). Selective brain network stimulation by frequency entrainment. bioRxiv, 2025-04.


