"""Señales de la clase 8: trazos de PTB-XL y grabaciones de aves de BirdCLEF+ 2025.

Las tablas de metadatos se piden con `descargar(...)`. Las señales no caben en una
tabla, así que se piden aquí:

    from taller.clase08_senales import trazos, grabaciones
    X = trazos(tabla)                 # arreglo (registros, 1000 muestras, 12 derivaciones), en mV
    for fila, onda, fs in grabaciones(tabla): ...

Las dos funciones buscan primero en `datos/cache/`, después en la fuente y, si
la fuente no responde, en `respaldo/`.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "datos" / "cache"
RESPALDO = RAIZ / "respaldo"

# PhysioNet redirige cada archivo a este depósito público de Amazon. Se usa directo
# porque así cada trazo cuesta una consulta y no dos.
PTBXL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
DERIVACIONES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FS_ECG = 100          # muestras por segundo en la versión reducida (records100)

AVES_URL = "https://huggingface.co/datasets/mteb/birdclef25-mini/resolve/main/data/train-00000-of-00001.parquet"
AVES_ARCHIVO = "birdclef25_mini.parquet"


def _un_trazo(sesion: requests.Session, ruta: str) -> np.ndarray:
    """Lee un .dat de PTB-XL: enteros de 16 bits, 12 derivaciones intercaladas, 1000 unidades por mV."""
    ultimo = None
    for intento in range(4):
        try:
            respuesta = sesion.get(PTBXL + ruta + ".dat", timeout=60)
            respuesta.raise_for_status()
            crudo = np.frombuffer(respuesta.content, dtype="<i2")
            if crudo.size != 12 * 1000:
                raise ValueError(f"{ruta}: {crudo.size} valores, esperaba 12000")
            return crudo.reshape(1000, 12)
        except Exception as error:
            ultimo = error
            time.sleep(1 + 2 * intento)
    raise RuntimeError(f"PhysioNet no entregó {ruta}") from ultimo


def trazos(tabla: pd.DataFrame, hilos: int = 48) -> np.ndarray:
    """Devuelve los trazos de las filas de `tabla` (columna `filename_lr`), en milivoltios.

    El resultado tiene forma (registros, 1000, 12): diez segundos a 100 Hz y doce derivaciones,
    en el orden de `DERIVACIONES`. La primera vez descarga un archivo de 24 kB por registro.
    """
    rutas = list(tabla["filename_lr"])
    huella = hashlib.md5("\n".join(rutas).encode()).hexdigest()[:8]
    archivo = f"ptbxl_trazos_{len(rutas)}_{huella}.npz"
    en_cache, en_respaldo = CACHE / archivo, RESPALDO / archivo
    if en_cache.exists():
        print(f"[trazos] Ya estaban descargados. Leo la copia de esta sesión ({len(rutas)} registros).")
        return np.load(en_cache)["trazos"].astype("float32") / 1000.0
    try:
        print(f"[trazos] Descargo {len(rutas)} trazos de PhysioNet (PTB-XL 1.0.3, records100)…", flush=True)
        inicio = time.time()
        sesion = requests.Session()
        sesion.mount("https://", requests.adapters.HTTPAdapter(pool_connections=hilos, pool_maxsize=hilos))
        with ThreadPoolExecutor(max_workers=hilos) as grupo:
            crudos = np.stack(list(grupo.map(lambda r: _un_trazo(sesion, r), rutas)))
        print(f"[trazos] Descarga en {time.time() - inicio:.0f} s.")
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(en_cache, trazos=crudos)
    except Exception as error:
        if not en_respaldo.exists():
            raise RuntimeError(
                "No pude descargar los trazos de PhysioNet y no hay copia en respaldo/. Revisa la "
                f"conexión y vuelve a correr la celda. Detalle técnico: {error!r}") from error
        print(f"[trazos] La fuente no respondió ({type(error).__name__}). Uso la copia de respaldo.")
        crudos = np.load(en_respaldo)["trazos"]
    return crudos.astype("float32") / 1000.0


def archivo_aves() -> Path:
    """Ruta local del parquet de BirdCLEF+ 2025 (muestra de 1.000 grabaciones, 257 MB)."""
    en_cache, en_respaldo = CACHE / AVES_ARCHIVO, RESPALDO / AVES_ARCHIVO
    if en_cache.exists():
        return en_cache
    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        print("[aves] Descargo la muestra de BirdCLEF+ 2025 desde Hugging Face (257 MB)…", flush=True)
        inicio = time.time()
        parcial = en_cache.with_suffix(".parcial")
        with requests.get(AVES_URL, stream=True, timeout=120) as respuesta:
            respuesta.raise_for_status()
            with open(parcial, "wb") as salida:
                shutil.copyfileobj(respuesta.raw, salida, length=1 << 20)
        parcial.rename(en_cache)
        print(f"[aves] Descarga en {time.time() - inicio:.0f} s.")
        return en_cache
    except Exception as error:
        if not en_respaldo.exists():
            raise RuntimeError(
                "No pude descargar las grabaciones de Hugging Face y no hay copia en respaldo/. "
                f"Detalle técnico: {error!r}") from error
        print(f"[aves] La fuente no respondió ({type(error).__name__}). Uso la copia de respaldo.")
        return en_respaldo


def grabaciones(tabla: pd.DataFrame):
    """Recorre las grabaciones de `tabla` y entrega (fila, onda, fs) una por una.

    `onda` es un arreglo de numpy en una sola pista, con valores entre -1 y 1, y `fs` es
    la frecuencia de muestreo (32.000 Hz en BirdCLEF+ 2025). Se entregan de a una para no
    llenar la memoria: las mil grabaciones suman más de nueve horas de sonido.
    """
    import pyarrow.parquet as pq
    import soundfile as sf

    columna = pq.read_table(archivo_aves(), columns=["recording"]).column("recording").to_pylist()
    for _, fila in tabla.iterrows():
        onda, fs = sf.read(io.BytesIO(columna[int(fila["indice"])]["bytes"]), dtype="float32")
        if onda.ndim > 1:
            onda = onda.mean(axis=1)
        yield fila, onda, fs
