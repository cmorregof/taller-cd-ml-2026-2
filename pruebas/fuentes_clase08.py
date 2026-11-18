"""Revisa las fuentes de la clase 8 con consultas pequeñas.

    python pruebas/fuentes_clase08.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests

from taller import clase08_senales, fuentes_clase08


def probar():
    """Devuelve una lista de mensajes. Lanza AssertionError con un mensaje claro si una fuente cambió."""
    mensajes = []

    # PTB-XL: encabezado de la tabla (solo los primeros 2 kB), diccionario de códigos y un trazo
    cabeza = requests.get(fuentes_clase08.PTBXL + "ptbxl_database.csv", headers={"Range": "bytes=0-2047"}, timeout=60)
    assert cabeza.status_code in (200, 206), f"PhysioNet respondió {cabeza.status_code} al pedir ptbxl_database.csv"
    columnas = cabeza.text.splitlines()[0].split(",")
    for columna in ["ecg_id", "patient_id", "age", "sex", "scp_codes", "strat_fold", "filename_lr"]:
        assert columna in columnas, f"PTB-XL: falta la columna {columna} en ptbxl_database.csv"
    codigos = fuentes_clase08.construir_ptbxl_codigos()
    assert {"NORM", "MI", "STTC", "CD", "HYP"} <= set(codigos["diagnostic_class"].dropna()), "PTB-XL: cambiaron las superclases"
    trazo = clase08_senales._un_trazo(requests.Session(), "records100/00000/00001_lr")
    assert trazo.shape == (1000, 12), f"PTB-XL: el trazo 00001 tiene forma {trazo.shape}"
    mensajes.append(f"PTB-XL: tabla con {len(columnas)} columnas, {len(codigos)} códigos, trazo 00001 de {trazo.shape}")

    # BirdCLEF+ 2025 (muestra en Hugging Face): que el archivo exista y pese lo mismo
    arbol = requests.get("https://huggingface.co/api/datasets/mteb/birdclef25-mini/tree/main/data", timeout=60)
    assert arbol.status_code == 200, f"Hugging Face respondió {arbol.status_code} para mteb/birdclef25-mini"
    archivos = {a["path"]: a["size"] for a in arbol.json()}
    nombre = "data/train-00000-of-00001.parquet"
    assert nombre in archivos, "mteb/birdclef25-mini: ya no está el archivo parquet"
    assert archivos[nombre] == 257378505, f"mteb/birdclef25-mini: el parquet cambió de tamaño ({archivos[nombre]} bytes)"
    mensajes.append(f"BirdCLEF+ 2025 (muestra): parquet de {archivos[nombre] / 1e6:.0f} MB en Hugging Face")
    return mensajes


if __name__ == "__main__":
    for mensaje in probar():
        print("OK    ", mensaje)
