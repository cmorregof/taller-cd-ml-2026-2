"""Los datos del parte: focos de calor MODIS de NASA FIRMS sobre Colombia, con su departamento.

Este archivo ya está escrito y no hay que cambiarlo. Uso:

    from datos import cargar_focos
    focos = cargar_focos()          # DataFrame de pandas, un foco por fila

La primera vez descarga los archivos anuales de FIRMS y los límites departamentales
(unos 20 segundos) y guarda una copia en `focos.csv.gz`. Las siguientes veces lee la copia.

Columnas: fecha (datetime), departamento, latitude, longitude, confidence (0 a 100),
frp (potencia radiativa del fuego, en megavatios), daynight (D o N), satellite, type.
Según la documentación de FIRMS, type = 0 es presunto incendio de vegetación, 1 volcán
activo, 2 otra fuente fija en tierra y 3 mar adentro.

Fuentes: NASA FIRMS, MODIS Collection 6.1 (citar «NASA FIRMS»). Límites: geoBoundaries,
tomados de OpenStreetMap (ODbL 1.0). No son la cartografía oficial del DANE.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from matplotlib.path import Path as Contorno

FIRMS = "https://firms.modaps.eosdis.nasa.gov/data/country/modis/{anio}/modis_{anio}_Colombia.csv"
LIMITES = ("https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/releaseData/"
           "gbOpen/COL/ADM1/geoBoundaries-COL-ADM1_simplified.geojson")
NOMBRES = {"Bogota Capital District": "Bogotá, D. C.",
           "Archipiélago de San Andrés, Providencia y Santa Catalina": "San Andrés y Providencia"}
COPIA = Path(__file__).resolve().parent / "focos.csv.gz"


def departamentos() -> list[str]:
    """Los 33 nombres (32 departamentos y Bogotá), en orden alfabético."""
    rasgos = requests.get(LIMITES, timeout=120).json()["features"]
    return sorted(NOMBRES.get(r["properties"]["shapeName"], r["properties"]["shapeName"]) for r in rasgos)


def _asignar(latitud, longitud) -> np.ndarray:
    puntos = np.column_stack([np.asarray(longitud, float), np.asarray(latitud, float)])
    salida = np.full(len(puntos), None, dtype=object)
    for rasgo in requests.get(LIMITES, timeout=120).json()["features"]:
        nombre = NOMBRES.get(rasgo["properties"]["shapeName"], rasgo["properties"]["shapeName"])
        geometria = rasgo["geometry"]
        partes = geometria["coordinates"] if geometria["type"] == "MultiPolygon" else [geometria["coordinates"]]
        for parte in partes:
            anillo = np.asarray(parte[0], float)
            (x0, y0), (x1, y1) = anillo.min(axis=0), anillo.max(axis=0)
            cerca = np.flatnonzero((puntos[:, 0] >= x0) & (puntos[:, 0] <= x1) & (puntos[:, 1] >= y0) & (puntos[:, 1] <= y1))
            if len(cerca):
                salida[cerca[Contorno(anillo).contains_points(puntos[cerca])]] = nombre
    return salida


def cargar_focos(desde: int = 2016, hasta: int = 2025) -> pd.DataFrame:
    if COPIA.exists():
        return pd.read_csv(COPIA, parse_dates=["fecha"])
    partes = []
    for anio in range(desde, hasta + 1):
        respuesta = requests.get(FIRMS.format(anio=anio), timeout=180)
        respuesta.raise_for_status()
        partes.append(pd.read_csv(io.BytesIO(respuesta.content)))
    tabla = pd.concat(partes, ignore_index=True)
    tabla["departamento"] = _asignar(tabla["latitude"], tabla["longitude"])
    tabla["fecha"] = pd.to_datetime(tabla["acq_date"])
    tabla = tabla[["fecha", "departamento", "latitude", "longitude", "confidence", "frp", "daynight", "satellite", "type"]]
    tabla.to_csv(COPIA, index=False)
    return pd.read_csv(COPIA, parse_dates=["fecha"])
