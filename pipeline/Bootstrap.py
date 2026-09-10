

"""EEG-simulation spatial correspondence across seeds (Bootstrap 95% confidence interval of the mean, calculated using 10,000 bootstrap resamples.)"""

import difflib
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats

def normalizar_nombre(nombre):
    """Normalize region names and hemisphere suffixes."""
    while isinstance(nombre, np.ndarray):
        nombre = nombre.flat[0]
    texto = ' '.join(str(nombre).strip().split())
    if texto.endswith(('_L', '_R')):
        cuerpo = texto[:-2]
        hemi = texto[-1]
    else:
        partes = texto.replace('_', ' ').split()
        if partes[0].upper() in ('L', 'R'):
            hemi = partes[0].upper()
            cuerpo = '_'.join(partes[1:])
        elif partes[-1].upper() in ('L', 'R'):
            hemi = partes[-1].upper()
            cuerpo = '_'.join(partes[:-1])
        else:
            return texto.replace(' ', '_')
    cuerpo = cuerpo.replace('Tempr', 'Temporal').replace('Pol', 'Pole').replace('Front', 'Frontal')
    return f'{cuerpo}_{hemi}'

def mejorar_match(dic_simulado, dic_eeg, labels84, cutoff=0.8):
    """Align EEG and simulated regions using the original name matching."""
    eeg_norm = {normalizar_nombre(nombre): valor for nombre, valor in dic_eeg.items()}
    disponibles = set(eeg_norm)
    sim_ordenado = {}
    eeg_ordenado = {}
    for region in labels84:
        if region not in dic_simulado:
            raise ValueError(f'Region {region} is absent from the simulation.')
        if region in disponibles:
            region_eeg = region
        else:
            candidatos = difflib.get_close_matches(region, list(disponibles), n=1, cutoff=cutoff)
            if not candidatos:
                raise ValueError(f'No EEG match found for {region}.')
            region_eeg = candidatos[0]
        sim_ordenado[region] = dic_simulado[region]
        eeg_ordenado[region] = eeg_norm[region_eeg]
        disponibles.remove(region_eeg)
    return (sim_ordenado, eeg_ordenado)

def metricas_binarias(regiones_eeg, regiones_simuladas, universo):
    """Calculate binary overlap metrics within the comparable regions."""
    eeg = set(regiones_eeg) & set(universo)
    sim = set(regiones_simuladas) & set(universo)
    tp = len(eeg & sim)
    fp = len(sim - eeg)
    fn = len(eeg - sim)
    tn = len(set(universo) - (eeg | sim))

    def dividir(a, b):
        return a / b if b else np.nan
    return {'n_eeg_activas': len(eeg), 'n_simuladas_activas': len(sim), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'overlap': tp, 'dice': dividir(2 * tp, 2 * tp + fp + fn), 'jaccard': dividir(tp, tp + fp + fn), 'precision': dividir(tp, tp + fp), 'recall': dividir(tp, tp + fn)}

def bootstrap_ci_mean(valores, n_bootstrap=10000, seed=1102):
    """Estimate the percentile bootstrap 95% confidence interval of the mean."""
    valores = np.asarray(valores, dtype=float)
    valores = valores[np.isfinite(valores)]
    if len(valores) == 0:
        return (np.nan, np.nan)
    if len(valores) == 1:
        return (valores[0], valores[0])
    rng = np.random.default_rng(seed)
    medias_bootstrap = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        muestra = rng.choice(valores, size=len(valores), replace=True)
        medias_bootstrap[i] = np.mean(muestra)
    ci_low = np.percentile(medias_bootstrap, 2.5)
    ci_high = np.percentile(medias_bootstrap, 97.5)
    return (ci_low, ci_high)

def seed_correspondence(
    archivo_semillas,
    ruta_aal90,
    archivo_valores_eeg,
    archivo_nombres_eeg,
    archivo_eeg_activadas,
    carpeta_simulaciones,
    carpeta_salida,
    amplitud_por_frecuencia,
    nodos,
    experiment_name,
    n_nodes,
    k,
    md,
    indices_excluidos,
    multiplicar_k_por_n=True,
    umbral_simulado=0.0,
    cutoff_match=0.80,
    n_bootstrap=10000,
    semilla_bootstrap=1102,
    nombre_salida="metricas_correspondencia_seeds_bilateral.xlsx",
    clave_labels="label90",
):
    """Calculate correspondence across seeds and export results to Excel.

    Paths are supplied by the caller. The seed file contains one integer per
    line. amplitud_por_frecuencia maps each frequency (Hz) to its amplitude.
    nodos is the target string used in filenames, e.g. "42-43". md uses the
    same units and filename convention as the original simulation output.
    indices_excluidos contains zero-based indices of subcortical regions not
    represented in the EEG map. These regions are excluded from comparison.

    The label MAT file contains clave_labels. EEG values are read from the
    first text column and matched to names in archivo_nombres_eeg, one per
    line. The first column of archivo_eeg_activadas lists active EEG regions
    without a header. NPY files contain frequency-to-propagation dictionaries.


    The summary reports the mean and its bootstrap 95% confidence interval
    using 10,000 resamples by default. Returns the workbook path and the five
    result DataFrames.
    """
    ARCHIVO_SEMILLAS = Path(archivo_semillas)
    RUTA_AAL90 = Path(ruta_aal90)
    ARCHIVO_VALORES_EEG = Path(archivo_valores_eeg)
    ARCHIVO_NOMBRES_EEG = Path(archivo_nombres_eeg)
    ARCHIVO_EEG_ACTIVADAS = Path(archivo_eeg_activadas)
    CARPETA_SIMULACIONES = Path(carpeta_simulaciones)
    CARPETA_SALIDA = Path(carpeta_salida)
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    AMPLITUD_POR_FRECUENCIA = amplitud_por_frecuencia
    FRECUENCIAS = list(amplitud_por_frecuencia)
    NODOS, EXPERIMENT_NAME = nodos, experiment_name
    N_NODES, MD = n_nodes, md
    k_model = k * n_nodes if multiplicar_k_por_n else k

    INDICES_EXCLUIDOS = indices_excluidos
    n_regiones = n_nodes - len(set(indices_excluidos))
    UMBRAL_SIMULADO, CUTOFF_MATCH = umbral_simulado, cutoff_match
    N_BOOTSTRAP, SEMILLA_BOOTSTRAP = n_bootstrap, semilla_bootstrap
    NOMBRE_SALIDA = nombre_salida
    with open(ARCHIVO_SEMILLAS) as archivo:
        SEEDS = [int(linea.strip()) for linea in archivo if linea.strip()]

    labels_raw = sio.loadmat(str(RUTA_AAL90))[clave_labels].ravel()

    labels90 = [normalizar_nombre(nombre) for nombre in labels_raw]

    labels84 = [region for i, region in enumerate(labels90) if i not in INDICES_EXCLUIDOS]

    if len(labels84) != n_regiones:
        raise ValueError(f'Unexpected number of comparable regions: {len(labels84)}.')

    print('Number of comparable regions:', len(labels84))

    valores_eeg = np.loadtxt(ARCHIVO_VALORES_EEG, usecols=0)

    with open(ARCHIVO_NOMBRES_EEG, 'r', encoding='utf-8') as f:
        nombres_eeg = [line.strip() for line in f]

    dicc_eeg = {}

    for nombre, valor in zip(nombres_eeg, valores_eeg):
        nombre_norm = normalizar_nombre(nombre)
        dicc_eeg.setdefault(nombre_norm, []).append(float(valor))

    promedios_eeg = {region: np.mean(valores) for region, valores in dicc_eeg.items()}

    tabla_activadas = pd.read_excel(ARCHIVO_EEG_ACTIVADAS, header=None)

    regiones_eeg_activadas = set()

    for nombre in tabla_activadas.iloc[:, 0].dropna():
        nombre_norm = normalizar_nombre(nombre)
        if nombre_norm in labels84:
            regiones_eeg_activadas.add(nombre_norm)
        else:
            match = difflib.get_close_matches(nombre_norm, labels84, n=1, cutoff=CUTOFF_MATCH)
            if match:
                regiones_eeg_activadas.add(match[0])
            else:
                print('Warning: unmatched active EEG region:', nombre)

    print('Number of active EEG regions:', len(regiones_eeg_activadas))

    resultados = []

    detalle_regiones = []

    archivos_faltantes = []

    for fstim in FRECUENCIAS:
        amplitud = AMPLITUD_POR_FRECUENCIA[fstim]
        print('\n' + '=' * 70)
        print(f'Frequency: {fstim:g} Hz')
        print(f'Fixed amplitude: {amplitud:g}')
        print('=' * 70)
        for seed in SEEDS:
            nombre_npy = f'propagation_fstim_{fstim:.2f}_weight_{amplitud:.1f}_nodes_{NODOS}_{EXPERIMENT_NAME}_N{N_NODES}_K{k_model:.3f}_MD{MD:.3f}_seed{int(seed)}.npy'
            ruta_npy = CARPETA_SIMULACIONES / nombre_npy
            if not ruta_npy.exists():
                print('File not found:', nombre_npy)
                archivos_faltantes.append({'seed': int(seed), 'frecuencia_hz': float(fstim), 'amplitud': float(amplitud), 'archivo': str(ruta_npy), 'error': 'File not found'})
                continue
            try:
                propagata = np.load(ruta_npy, allow_pickle=True).item()
            except Exception as error:
                print('Error loading:', nombre_npy, error)
                archivos_faltantes.append({'seed': int(seed), 'frecuencia_hz': float(fstim), 'amplitud': float(amplitud), 'archivo': str(ruta_npy), 'error': str(error)})
                continue
            if fstim in propagata:
                clave = fstim
            else:
                clave = min(propagata.keys(), key=lambda x: abs(float(x) - fstim))
            valores_simulados = np.asarray(propagata[clave], dtype=float).ravel()
            if len(valores_simulados) != N_NODES:
                print('Incorrect region count:', nombre_npy)
                archivos_faltantes.append({'seed': int(seed), 'frecuencia_hz': float(fstim), 'amplitud': float(amplitud), 'archivo': str(ruta_npy), 'error': f'{len(valores_simulados)} regions; unexpected node count'})
                continue
            dicc_simulado = dict(zip(labels90, valores_simulados))
            sim84, eeg84 = mejorar_match(dicc_simulado, promedios_eeg, labels84, cutoff=CUTOFF_MATCH)
            vector_sim84 = np.array([sim84[region] for region in labels84], dtype=float)
            vector_eeg84 = np.array([eeg84[region] for region in labels84], dtype=float)
            if np.std(vector_sim84) == 0 or np.std(vector_eeg84) == 0:
                pearson_r = np.nan
                pearson_p = np.nan
            else:
                pearson_r, pearson_p = stats.pearsonr(vector_sim84, vector_eeg84)
            if np.std(vector_sim84) == 0 or np.std(vector_eeg84) == 0:
                spearman_rho = np.nan
                spearman_p = np.nan
            else:
                spearman_rho, spearman_p = stats.spearmanr(vector_sim84, vector_eeg84)
            regiones_simuladas_activadas = {region for region in labels84 if dicc_simulado[region] > UMBRAL_SIMULADO}
            binarias = metricas_binarias(regiones_eeg_activadas, regiones_simuladas_activadas, labels84)
            resultados.append({'seed': int(seed), 'frecuencia_hz': float(fstim), 'amplitud': float(amplitud), 'archivo_npy': ruta_npy.name, 'n_regiones': n_regiones, 'pearson_r': float(pearson_r), 'pearson_p_parametrico': float(pearson_p), 'spearman_rho': float(spearman_rho), 'spearman_p_parametrico': float(spearman_p), **binarias})
            for region in labels84:
                eeg_activa = int(region in regiones_eeg_activadas)
                simulada_activa = int(region in regiones_simuladas_activadas)
                tp_region = int(eeg_activa == 1 and simulada_activa == 1)
                fp_region = int(eeg_activa == 0 and simulada_activa == 1)
                fn_region = int(eeg_activa == 1 and simulada_activa == 0)
                tn_region = int(eeg_activa == 0 and simulada_activa == 0)
                detalle_regiones.append({'seed': int(seed), 'frecuencia_hz': float(fstim), 'amplitud': float(amplitud), 'region': region, 'valor_eeg': float(eeg84[region]), 'valor_simulado': float(sim84[region]), 'eeg_activa': eeg_activa, 'simulada_activa': simulada_activa, 'TP': tp_region, 'FP': fp_region, 'FN': fn_region, 'TN': tn_region})
            print(f"seed={seed} | Pearson={pearson_r:.4f} | Spearman={spearman_rho:.4f} | Dice={binarias['dice']:.4f}")

    df_resultados = pd.DataFrame(resultados)

    df_detalle = pd.DataFrame(detalle_regiones)

    df_faltantes = pd.DataFrame(archivos_faltantes)

    if df_resultados.empty:
        raise ValueError('No simulation could be processed. Check the simulation folder and filename format.')

    print('\n' + '=' * 70)

    print('SEEDS PROCESSED PER FREQUENCY')

    print('=' * 70)

    print(df_resultados.groupby('frecuencia_hz')['seed'].nunique())

    METRICAS_PRINCIPALES = {'Pearson': 'pearson_r', 'Spearman': 'spearman_rho', 'Dice': 'dice'}

    resumen = []

    for fstim in FRECUENCIAS:
        datos_freq = df_resultados[np.isclose(df_resultados['frecuencia_hz'], fstim)]
        for nombre_metrica, columna in METRICAS_PRINCIPALES.items():
            valores = datos_freq[columna].dropna().to_numpy(dtype=float)
            if len(valores) == 0:
                continue
            bootstrap_low, bootstrap_high = bootstrap_ci_mean(valores, n_bootstrap=N_BOOTSTRAP, seed=SEMILLA_BOOTSTRAP)
            resumen.append({
                'frecuencia_hz': float(fstim),
                'amplitud': float(AMPLITUD_POR_FRECUENCIA[fstim]),
                'metrica': nombre_metrica,
                'n_seeds': len(valores),
                'mean': np.mean(valores),
                'bootstrap_mean_CI95_low': bootstrap_low,
                'bootstrap_mean_CI95_high': bootstrap_high,
            })

    df_resumen = pd.DataFrame(resumen)

    resumen_wide = []

    for fstim in FRECUENCIAS:
        fila = {'frecuencia_hz': float(fstim), 'amplitud': float(AMPLITUD_POR_FRECUENCIA[fstim])}
        for nombre_metrica in METRICAS_PRINCIPALES.keys():
            datos_metrica = df_resumen[np.isclose(df_resumen['frecuencia_hz'], fstim) & (df_resumen['metrica'] == nombre_metrica)]
            if datos_metrica.empty:
                continue
            r = datos_metrica.iloc[0]
            prefijo = nombre_metrica.lower()
            fila[f'{prefijo}_mean'] = r['mean']
            fila[f'{prefijo}_bootstrap_CI95_low'] = r['bootstrap_mean_CI95_low']
            fila[f'{prefijo}_bootstrap_CI95_high'] = r['bootstrap_mean_CI95_high']
        resumen_wide.append(fila)

    df_resumen_wide = pd.DataFrame(resumen_wide)

    ruta_salida = CARPETA_SALIDA / NOMBRE_SALIDA

    with pd.ExcelWriter(ruta_salida) as writer:
        df_resultados.to_excel(writer, sheet_name='metricas_por_seed', index=False)
        df_resumen.to_excel(writer, sheet_name='resumen_estadistico', index=False)
        df_resumen_wide.to_excel(writer, sheet_name='resumen_por_frecuencia', index=False)
        df_detalle.to_excel(writer, sheet_name='regiones_84_por_seed', index=False)
        df_faltantes.to_excel(writer, sheet_name='archivos_faltantes', index=False)

    return ruta_salida, df_resultados, df_resumen, df_resumen_wide, df_detalle, df_faltantes
