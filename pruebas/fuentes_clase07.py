"""Revisa las fuentes de la clase 7 con consultas pequeñas.

    python pruebas/fuentes_clase07.py

`probar()` devuelve una lista de mensajes y lanza AssertionError, con un mensaje
claro, si alguna fuente cambió de forma. No necesita pyarrow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from taller import fuentes_clase07 as f


def _restmex():
    for particion, archivo in f.RESTMEX_ARCHIVOS.items():
        respuesta = requests.get(f.RESTMEX + archivo, headers={**f.AGENTE, "Range": "bytes=0-3"}, timeout=120)
        assert respuesta.status_code in (200, 206) and respuesta.content[:4] == b"PAR1", f"Rest-Mex: el archivo de {particion} ya no es un parquet en su sitio"
    ficha = requests.get("https://huggingface.co/api/datasets/javilonso/rest23_sentiment_data", headers=f.AGENTE, timeout=120).json()
    descripcion = ficha["cardData"]["dataset_info"]
    faltan = {"Title", "Review", "Polarity", "Country", "Type"} - {c["name"] for c in descripcion["features"]}
    assert not faltan, f"Rest-Mex: ya no vienen las columnas {sorted(faltan)}"
    filas = {p["name"]: p["num_examples"] for p in descripcion["splits"]}
    assert filas == {"train": 201_032, "test": 50_259}, f"Rest-Mex: las particiones tenían 201.032 y 50.259 filas y ahora tienen {filas}"
    return "Rest-Mex 2023 (Hugging Face): los dos parquet en su sitio, columnas presentes, 201.032 y 50.259 filas"


def _secop():
    pagina = f.secop_pagina("Caldas", 2022, 0, limite=20)
    assert len(pagina) == 20, f"SECOP II: esperaba 20 contratos de Caldas en 2022 y llegaron {len(pagina)}"
    faltan = set(f.SECOP_COLUMNAS) - set(pagina.columns)
    assert not faltan, f"SECOP II: ya no vienen las columnas {sorted(faltan)}"
    assert pagina["objeto_del_contrato"].notna().all(), "SECOP II: hay contratos sin objeto en la muestra"
    return f"SECOP II (jbjy-vk9h): {len(f.SECOP_COLUMNAS)} columnas presentes, consulta por departamento y año correcta"


def probar():
    return [_restmex(), _secop()]


if __name__ == "__main__":
    try:
        for mensaje in probar():
            print("OK    ", mensaje)
    except Exception as error:
        print("FALLA ", error)
        sys.exit(1)
