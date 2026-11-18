"""Datos de la clase 8: electrocardiogramas (PTB-XL) y cantos de aves (BirdCLEF+ 2025).

Aquí están las tablas de metadatos. Las señales se piden con `taller.clase08_senales`.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

from . import clase08_senales

PTBXL = "https://physionet.org/files/ptb-xl/1.0.3/"


def _csv(url: str) -> pd.DataFrame:
    respuesta = requests.get(url, timeout=180)
    respuesta.raise_for_status()
    return pd.read_csv(io.BytesIO(respuesta.content))


def construir_ptbxl() -> pd.DataFrame:
    """La tabla `ptbxl_database.csv` tal como la publica PhysioNet: un registro por fila.

    Se dejan por fuera los nombres seudonimizados de quien validó y la ruta de 500 Hz, que
    el curso no usa. Los códigos diagnósticos (`scp_codes`) llegan como texto y se
    interpretan en el cuaderno, a la vista.
    """
    tabla = _csv(PTBXL + "ptbxl_database.csv")
    columnas = ["ecg_id", "patient_id", "age", "sex", "height", "weight", "site", "device",
                "recording_date", "report", "scp_codes", "heart_axis", "infarction_stadium1",
                "infarction_stadium2", "validated_by_human", "baseline_drift", "static_noise",
                "burst_noise", "electrodes_problems", "extra_beats", "pacemaker", "strat_fold",
                "filename_lr"]
    return tabla[columnas]


def construir_ptbxl_codigos() -> pd.DataFrame:
    """`scp_statements.csv`: qué significa cada código SCP-ECG y a qué superclase pertenece."""
    tabla = _csv(PTBXL + "scp_statements.csv")
    tabla = tabla.rename(columns={tabla.columns[0]: "codigo"})
    return tabla[["codigo", "description", "diagnostic", "form", "rhythm",
                  "diagnostic_class", "diagnostic_subclass"]]


def construir_aves() -> pd.DataFrame:
    """Metadatos de las 1.000 grabaciones: especie, colección, autor, licencia, coordenadas.

    `indice` es la posición de la grabación dentro del archivo, y con ella la busca
    `clase08_senales.grabaciones`. La duración se lee del encabezado de cada audio.
    """
    import pyarrow.parquet as pq
    import soundfile as sf

    archivo = clase08_senales.archivo_aves()
    columnas = ["primary_label", "scientific_name", "common_name", "collection", "rating", "url",
                "latitude", "longitude", "author", "license"]
    tabla = pq.read_table(archivo, columns=columnas).to_pandas()
    audios = pq.read_table(archivo, columns=["recording"]).column("recording").to_pylist()
    tabla.insert(0, "indice", range(len(tabla)))
    tabla.insert(1, "archivo", [a["path"] for a in audios])
    tabla["segundos"] = [round(sf.info(io.BytesIO(a["bytes"])).duration, 2) for a in audios]
    return tabla


FICHAS = {
    "ptbxl": {
        "titulo": "PTB-XL 1.0.3: metadatos de 21.799 electrocardiogramas de 12 derivaciones",
        "fuente": "PhysioNet, PTB-XL 1.0.3 (Wagner y otros, 2020; doi 10.13026/kfzx-aw45)",
        "url": "https://physionet.org/content/ptb-xl/1.0.3/",
        "licencia": "Creative Commons Attribution 4.0 (CC BY 4.0). Citar a los autores y a PhysioNet.",
        "constructor": construir_ptbxl,
        "md5": "aea32fccef6b50388de9f27dfc5a2376",
    },
    "ptbxl_codigos": {
        "titulo": "PTB-XL 1.0.3: diccionario de códigos SCP-ECG",
        "fuente": "PhysioNet, PTB-XL 1.0.3, archivo scp_statements.csv",
        "url": "https://physionet.org/content/ptb-xl/1.0.3/",
        "licencia": "Creative Commons Attribution 4.0 (CC BY 4.0).",
        "constructor": construir_ptbxl_codigos,
        "md5": "fda60589c88f5a42a3e3b9b0eb208a49",
    },
    "aves_magdalena": {
        "titulo": "BirdCLEF+ 2025: muestra de 1.000 grabaciones de 50 especies de aves (20 por especie)",
        "fuente": "Hugging Face, mteb/birdclef25-mini, tomado de train_audio de BirdCLEF+ 2025 (xeno-canto e iNaturalist)",
        "url": "https://huggingface.co/datasets/mteb/birdclef25-mini",
        "licencia": ("Cada grabación conserva la licencia Creative Commons de su autor (columna license; "
                     "la mayoría CC BY-NC-SA 4.0). Uso académico sin fines comerciales, con cita al autor. "
                     "El repositorio de Hugging Face no declara licencia propia: se descarga, no se republica."),
        "constructor": construir_aves,
        "md5": "426fc6c700bed541b463932830f5639f",
    },
}
