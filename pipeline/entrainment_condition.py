
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: neurocomp
"""
"""Functions used to calculate entrainment"""

import numpy as np
from scipy import signal


def calculate_psd(phases, fs, nperseg):
    """Calculate the PSD"""

    frequencies, psd = signal.welch(
        np.sin(phases),
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        axis=1,
    )

    return frequencies, psd


def calculate_global_plv(
    stimulation_phases,
    stimulus_phase,
    fs,
    stimulation_frequency,
    filter_width,
    filter_order,
    edge_samples,
):
    """Filter the node signals and calculate PLV."""

    low_frequency = stimulation_frequency - filter_width
    high_frequency = stimulation_frequency + filter_width

    b, a = signal.butter(
        filter_order,
        [low_frequency, high_frequency],
        btype="bandpass",
        fs=fs,
    )

    filtered_signals = signal.filtfilt(
        b,
        a,
        np.sin(stimulation_phases),
        axis=1,
    )

    if edge_samples > 0:
        filtered_signals = filtered_signals[
            :, edge_samples:-edge_samples
        ]
        stimulus_phase = stimulus_phase[
            edge_samples:-edge_samples
        ]

    node_phases = np.unwrap(
        np.angle(
            signal.hilbert(
                filtered_signals,
                axis=1,
            )
        ),
        axis=1,
    )

    phase_difference = node_phases - stimulus_phase

    return np.abs(
        np.mean(
            np.exp(1j * phase_difference),
            axis=1,
        )
    )


def calculate_entrainment(
    baseline_phases,
    stimulation_phases,
    stimulus_phase,
    fs,
    stimulation_frequency,
    nperseg,
    filter_width,
    filter_order,
    edge_samples,
):
    """Identify nodes that satisfy the PSD and global PLV conditions."""

    frequencies, baseline_psd = calculate_psd(
        baseline_phases,
        fs,
        nperseg,
    )

    _, stimulation_psd = calculate_psd(
        stimulation_phases,
        fs,
        nperseg,
    )

    frequency_index = np.argmin(
        np.abs(frequencies - stimulation_frequency)
    )

    psd_ratio = (
        stimulation_psd[:, frequency_index]
        / baseline_psd[:, frequency_index]
    )

    plv = calculate_global_plv(
        stimulation_phases,
        stimulus_phase,
        fs,
        stimulation_frequency,
        filter_width,
        filter_order,
        edge_samples,
    )

    entrained = (psd_ratio >= 2) & (plv > 0.7)

    return entrained, psd_ratio, plv
