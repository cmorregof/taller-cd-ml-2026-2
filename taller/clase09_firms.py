"""Asigna a cada foco de calor el departamento donde cayó.

Los polígonos de los 33 departamentos (32 y Bogotá) vienen de geoBoundaries (William & Mary
geoLab), que para Colombia toma los límites de OpenStreetMap. No son la cartografía oficial
del DANE: sirven para contar focos por departamento, no para dirimir un límite.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import requests
from matplotlib.path import Path as Contorno

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "datos" / "cache"
RESPALDO = RAIZ / "respaldo"
ARCHIVO = "departamentos_geoboundaries.geojson"
GEO_URL = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/releaseData/"
           "gbOpen/COL/ADM1/geoBoundaries-COL-ADM1_simplified.geojson")

# geoBoundaries trae dos nombres que no son los de uso en Colombia.
NOMBRES = {"Bogota Capital District": "Bogotá, D. C.",
           "Archipiélago de San Andrés, Providencia y Santa Catalina": "San Andrés y Providencia"}


def geojson() -> dict:
    en_cache, en_respaldo = CACHE / ARCHIVO, RESPALDO / ARCHIVO
    if en_cache.exists():
        return json.loads(en_cache.read_text(encoding="utf-8"))
    try:
        respuesta = requests.get(GEO_URL, timeout=120)
        respuesta.raise_for_status()
        datos = respuesta.json()
        assert len(datos["features"]) == 33, f"esperaba 33 departamentos y llegaron {len(datos['features'])}"
        CACHE.mkdir(parents=True, exist_ok=True)
        en_cache.write_text(respuesta.text, encoding="utf-8")
        return datos
    except Exception as error:
        if not en_respaldo.exists():
            raise RuntimeError(f"No pude descargar los límites departamentales de geoBoundaries. Detalle: {error!r}") from error
        print("[departamentos] La fuente no respondió. Uso la copia de respaldo.")
        return json.loads(en_respaldo.read_text(encoding="utf-8"))


def poligonos() -> dict[str, list[np.ndarray]]:
    """Devuelve {departamento: [anillos exteriores]}, cada anillo como arreglo (n, 2) de longitud y latitud."""
    salida = {}
    for rasgo in geojson()["features"]:
        nombre = rasgo["properties"]["shapeName"]
        geometria = rasgo["geometry"]
        partes = geometria["coordinates"] if geometria["type"] == "MultiPolygon" else [geometria["coordinates"]]
        salida[NOMBRES.get(nombre, nombre)] = [np.asarray(parte[0], dtype=float) for parte in partes]
    return salida


def asignar_departamento(latitud, longitud) -> np.ndarray:
    """Para cada punto, el nombre del departamento que lo contiene, o None si no cae en ninguno."""
    puntos = np.column_stack([np.asarray(longitud, dtype=float), np.asarray(latitud, dtype=float)])
    resultado = np.full(len(puntos), None, dtype=object)
    for nombre, anillos in poligonos().items():
        for anillo in anillos:
            (x0, y0), (x1, y1) = anillo.min(axis=0), anillo.max(axis=0)
            candidatos = np.flatnonzero((puntos[:, 0] >= x0) & (puntos[:, 0] <= x1) & (puntos[:, 1] >= y0) & (puntos[:, 1] <= y1))
            if len(candidatos):
                adentro = Contorno(anillo).contains_points(puntos[candidatos])
                resultado[candidatos[adentro]] = nombre
    return resultado
