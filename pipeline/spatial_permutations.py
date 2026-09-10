# -*- coding: utf-8 -*-
"""


@author: elida
"""

"""Restricted spatial permutations with Excel output.

EEG stays fixed while the simulated continuous and binary maps are shuffled
using the same indices within each anatomical group. 
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def grupo_anatomico(region, subcorticales):
    """Assign the region to its tissue-type and hemisphere group."""

    if region.endswith("_L"):
        hemisferio = "L"
    elif region.endswith("_R"):
        hemisferio = "R"
    else:
        raise ValueError(
            f"Could not identify the hemisphere of {region}."
        )

    tipo = (
        "subcortical"
        if region in subcorticales
        else "cortical"
    )

    return f"{tipo}_{hemisferio}"


def permutar_dentro_de_grupos(grupos, rng):
    """Shuffle indices independently within each anatomical group."""

    grupos = np.asarray(grupos)
    indices = np.arange(len(grupos))

    for grupo in np.unique(grupos):
        posiciones = np.where(grupos == grupo)[0]
        indices[posiciones] = rng.permutation(posiciones)

    return indices


def dice_binario(real, simulado):
    """Calculate Dice overlap between two binary maps."""

    real = np.asarray(real, dtype=int)
    simulado = np.asarray(simulado, dtype=int)

    tp = np.sum((real == 1) & (simulado == 1))
    fp = np.sum((real == 0) & (simulado == 1))
    fn = np.sum((real == 1) & (simulado == 0))

    denominador = 2 * tp + fp + fn

    return (
        2 * tp / denominador
        if denominador > 0
        else np.nan
    )


def p_espacial(distribucion_nula, observado):
    """Calculate an upper-tail permutation p-value with the +1 correction."""

    return (
        1 + np.sum(distribucion_nula >= observado)
    ) / (
        len(distribucion_nula) + 1
    )



def spatial_permutations(
    archivo_resultados,
    carpeta_salida,
    subcorticales,
    hoja_mejores="mejor_amplitud",
    hoja_regiones="regiones",
    nombre_salida="permutaciones_espaciales_bilateral.xlsx",
    n_permutaciones=10000,
    seed=1102,
    n_regiones=84,
):
    """Run the original permutation analysis and save its three Excel sheets.

    archivo_resultados: input workbook containing the selected amplitudes and
        regional maps. The selected-amplitude sheet needs frecuencia_hz and
        amplitud. The regional sheet also needs region, valor_eeg,
        valor_simulado, eeg_activa and simulada_activa.
    carpeta_salida: destination folder (created if needed).
    hoja_mejores, hoja_regiones: input sheet names.
    nombre_salida: output workbook name.
    n_permutaciones, seed: permutation count and random seed.
    subcorticales: subcortical regions not represented in the EEG map.
    n_regiones: expected number of regions per condition; None skips this check.

    Returns the output path, summary, null distributions and anatomical groups.

    """
    ARCHIVO_RESULTADOS = Path(archivo_resultados)
    CARPETA_SALIDA = Path(carpeta_salida)
    HOJA_MEJORES = hoja_mejores
    HOJA_REGIONES = hoja_regiones
    NOMBRE_SALIDA = nombre_salida
    N_PERMUTACIONES = n_permutaciones
    SEED = seed
    N_REGIONES = n_regiones
    # Subcortical regions not represented in the EEG map.
    SUBCORTICALES = set(subcorticales)

    CARPETA_SALIDA.mkdir(
        parents=True,
        exist_ok=True
    )

    mejores = pd.read_excel(
        ARCHIVO_RESULTADOS,
        sheet_name=HOJA_MEJORES
    )

    regiones = pd.read_excel(
        ARCHIVO_RESULTADOS,
        sheet_name=HOJA_REGIONES
    )

    columnas_necesarias = {
        "frecuencia_hz",
        "amplitud",
        "region",
        "valor_eeg",
        "valor_simulado",
        "eeg_activa",
        "simulada_activa",
    }

    faltantes = columnas_necesarias - set(regiones.columns)

    if faltantes:
        raise ValueError(
            f"Missing columns in '{HOJA_REGIONES}': {sorted(faltantes)}"
        )


    # ============================================================
    # TEST EACH FREQUENCY
    # ============================================================

    rng = np.random.default_rng(SEED)

    resumen = []
    distribuciones = []
    grupos_guardados = []

    for _, fila_mejor in mejores.iterrows():

        frecuencia = float(fila_mejor["frecuencia_hz"])
        amplitud = float(fila_mejor["amplitud"])

        datos = regiones[
            np.isclose(
                regiones["frecuencia_hz"].astype(float),
                frecuencia
            )
            &
            np.isclose(
                regiones["amplitud"].astype(float),
                amplitud
            )
        ].copy()

        datos = (
            datos
            .drop_duplicates(subset="region")
            .reset_index(drop=True)
        )

        if N_REGIONES is not None and len(datos) != N_REGIONES:
            raise ValueError(
                f"{frecuencia:g} Hz, A={amplitud:g}: "
                f"expected {N_REGIONES} regions, found {len(datos)}."
            )

        datos["grupo_permutacion"] = datos[
            "region"
        ].apply(
            lambda region: grupo_anatomico(region, SUBCORTICALES)
        )

        conteos = datos[
            "grupo_permutacion"
        ].value_counts()

        eeg_continuo = datos[
            "valor_eeg"
        ].to_numpy(dtype=float)

        sim_continuo = datos[
            "valor_simulado"
        ].to_numpy(dtype=float)

        eeg_binario = datos[
            "eeg_activa"
        ].to_numpy(dtype=int)

        sim_binario = datos[
            "simulada_activa"
        ].to_numpy(dtype=int)

        grupos = datos[
            "grupo_permutacion"
        ].to_numpy()

        pearson_obs = pearsonr(
            eeg_continuo,
            sim_continuo
        ).statistic

        spearman_obs = spearmanr(
            eeg_continuo,
            sim_continuo
        ).statistic

        dice_obs = dice_binario(
            eeg_binario,
            sim_binario
        )

        pearson_perm = np.empty(N_PERMUTACIONES)
        spearman_perm = np.empty(N_PERMUTACIONES)
        dice_perm = np.empty(N_PERMUTACIONES)

        for i in range(N_PERMUTACIONES):

            indices_perm = permutar_dentro_de_grupos(
                grupos,
                rng
            )

            sim_cont_perm = sim_continuo[
                indices_perm
            ]

            sim_bin_perm = sim_binario[
                indices_perm
            ]

            pearson_perm[i] = pearsonr(
                eeg_continuo,
                sim_cont_perm
            ).statistic

            spearman_perm[i] = spearmanr(
                eeg_continuo,
                sim_cont_perm
            ).statistic

            dice_perm[i] = dice_binario(
                eeg_binario,
                sim_bin_perm
            )

        p_pearson = p_espacial(
            pearson_perm,
            pearson_obs
        )

        p_spearman = p_espacial(
            spearman_perm,
            spearman_obs
        )

        p_dice = p_espacial(
            dice_perm,
            dice_obs
        )

        resumen.append({
            "frecuencia_hz": frecuencia,
            "amplitud": amplitud,
            "n_permutaciones": N_PERMUTACIONES,
            "pearson_observado": pearson_obs,
            "pearson_p_espacial": p_pearson,
            "spearman_observado": spearman_obs,
            "spearman_p_espacial": p_spearman,
            "dice_observado": dice_obs,
            "dice_p_espacial": p_dice,
            "cortical_L": int(conteos.get("cortical_L", 0)),
            "cortical_R": int(conteos.get("cortical_R", 0)),
            "subcortical_L": int(conteos.get("subcortical_L", 0)),
            "subcortical_R": int(conteos.get("subcortical_R", 0)),
        })

        distribuciones.append(pd.DataFrame({
            "frecuencia_hz": frecuencia,
            "amplitud": amplitud,
            "permutacion": np.arange(1, N_PERMUTACIONES + 1),
            "pearson_permutado": pearson_perm,
            "spearman_permutado": spearman_perm,
            "dice_permutado": dice_perm,
        }))

        grupos_tmp = datos[
            ["region", "grupo_permutacion"]
        ].copy()

        grupos_tmp.insert(
            0,
            "amplitud",
            amplitud
        )

        grupos_tmp.insert(
            0,
            "frecuencia_hz",
            frecuencia
        )

        grupos_guardados.append(
            grupos_tmp
        )

        print(
            f"\n{frecuencia:g} Hz | A={amplitud:g}"
            f"\nPearson:  r={pearson_obs:.4f}, "
            f"spatial p={p_pearson:.4e}"
            f"\nSpearman: rho={spearman_obs:.4f}, "
            f"spatial p={p_spearman:.4e}"
            f"\nDice:     {dice_obs:.4f}, "
            f"spatial p={p_dice:.4e}"
            f"\n{conteos.to_string()}"
        )


    # ============================================================
    # SAVE RESULTS
    # ============================================================

    df_resumen = pd.DataFrame(resumen)

    df_distribuciones = pd.concat(
        distribuciones,
        ignore_index=True
    )

    df_grupos = pd.concat(
        grupos_guardados,
        ignore_index=True
    )

    ruta_salida = (
        CARPETA_SALIDA
        / NOMBRE_SALIDA
    )

    with pd.ExcelWriter(ruta_salida) as writer:

        df_resumen.to_excel(
            writer,
            sheet_name="resumen",
            index=False
        )

        df_distribuciones.to_excel(
            writer,
            sheet_name="distribuciones_nulas",
            index=False
        )

        df_grupos.to_excel(
            writer,
            sheet_name="grupos_anatomicos",
            index=False
        )

    print(
        f"\nResults saved to:\n"
        f"{ruta_salida}"
    )


    return ruta_salida, df_resumen, df_distribuciones, df_grupos
