"""Datos de la clase 9: noticias de El País (corpus MLSUM) y focos de calor de NASA FIRMS."""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

from . import clase09_firms

MLSUM = "https://huggingface.co/datasets/mteb/mlsum/resolve/main/es/test-00000-of-00001.parquet"
FIRMS = "https://firms.modaps.eosdis.nasa.gov/data/country/modis/{anio}/modis_{anio}_Colombia.csv"


def _bajar(url: str) -> bytes:
    ultimo = None
    for intento in range(3):
        try:
            respuesta = requests.get(url, timeout=180)
            respuesta.raise_for_status()
            return respuesta.content
        except Exception as error:
            ultimo = error
            time.sleep(3 * (intento + 1))
    raise RuntimeError(f"No respondió {url}") from ultimo


def construir_mlsum() -> pd.DataFrame:
    """Las 13.920 noticias de El País de la partición de prueba de MLSUM en español.

    Cada fila trae el texto, el resumen que escribió el periódico (la entradilla), el tema,
    el título, la fecha y la dirección de la noticia. `indice` es la posición en el archivo.
    """
    tabla = pd.read_parquet(io.BytesIO(_bajar(MLSUM)))
    tabla.insert(0, "indice", range(len(tabla)))
    return tabla[["indice", "date", "topic", "title", "summary", "text", "url"]]


def construir_firms(desde: int = 2016, hasta: int = 2025) -> pd.DataFrame:
    """Focos de calor MODIS (Terra y Aqua) sobre Colombia, un archivo anual por año, con su departamento.

    Las columnas son las de FIRMS. Se agrega `departamento`, calculado con los polígonos de
    geoBoundaries; queda vacío cuando el centro del píxel cae fuera de todos (mar o frontera).
    """
    partes = []
    for anio in range(int(desde), int(hasta) + 1):
        print(f"   FIRMS, MODIS {anio}…", flush=True)
        partes.append(pd.read_csv(io.BytesIO(_bajar(FIRMS.format(anio=anio))), dtype={"acq_time": str}))
    tabla = pd.concat(partes, ignore_index=True)
    tabla["departamento"] = clase09_firms.asignar_departamento(tabla["latitude"], tabla["longitude"])
    return tabla[["acq_date", "acq_time", "latitude", "longitude", "departamento", "satellite", "daynight",
                  "confidence", "frp", "brightness", "bright_t31", "scan", "track", "type", "version"]]


FICHAS = {
    "mlsum_es": {
        "titulo": "MLSUM en español: 13.920 noticias de El País con su entradilla (partición de prueba)",
        "fuente": "Hugging Face, mteb/mlsum, configuración es (Scialom y otros, 2020, corpus MLSUM)",
        "url": "https://huggingface.co/datasets/mteb/mlsum",
        "licencia": ("La ficha del conjunto declara la licencia como desconocida y los autores de MLSUM lo "
                     "publicaron para investigación. Los textos son de El País: se usan en clase con cita "
                     "y enlace a la noticia, y no se republican."),
        "constructor": construir_mlsum,
        "md5": None,
    },
    "firms_colombia": {
        "titulo": "NASA FIRMS: focos de calor MODIS sobre Colombia, 2016 a 2025, con departamento",
        "fuente": "NASA FIRMS, MODIS Collection 6.1 (MCD14ML), archivos anuales por país; límites de geoBoundaries (OpenStreetMap)",
        "url": "https://firms.modaps.eosdis.nasa.gov/country/",
        "licencia": ("Datos abiertos de la NASA: citar «NASA FIRMS». Límites departamentales de geoBoundaries, "
                     "tomados de OpenStreetMap, con licencia ODbL 1.0."),
        "constructor": construir_firms,
        "md5": None,
    },
}
