"""Datos del mercado eléctrico colombiano, tomados de la API pública SiMEM de XM.

La API entrega como máximo 31 días por consulta en conjuntos horarios y
diarios. Por eso se pide mes por mes, con pocas consultas simultáneas para
no saturar el servicio.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

API = "https://www.simem.co/backend-files/api/PublicData"


def _mes(conjunto: str, inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.DataFrame:
    parametros = {"datasetid": conjunto, "startDate": f"{inicio:%Y-%m-%d}", "endDate": f"{fin:%Y-%m-%d}"}
    ultimo = None
    for intento in range(4):
        try:
            respuesta = requests.get(API, params=parametros, timeout=120)
            respuesta.raise_for_status()
            return pd.DataFrame(respuesta.json()["result"]["records"])
        except Exception as error:  # se reintenta con espera creciente
            ultimo = error
            time.sleep(3 * (intento + 1))
    raise RuntimeError(f"SiMEM no respondió para {conjunto} entre {inicio:%Y-%m-%d} y {fin:%Y-%m-%d}") from ultimo


def _rango(conjunto: str, desde: str, hasta: str) -> pd.DataFrame:
    inicios = pd.date_range(desde, hasta, freq="MS")
    limite = pd.Timestamp(hasta)
    tramos = [(i, min(i + pd.offsets.MonthEnd(0), limite)) for i in inicios]
    print(f"   SiMEM, conjunto {conjunto}: {len(tramos)} meses…", flush=True)
    with ThreadPoolExecutor(max_workers=4) as grupo:
        partes = list(grupo.map(lambda t: _mes(conjunto, *t), tramos))
    return pd.concat([p for p in partes if len(p)], ignore_index=True)


def crudo_precio(desde: str, hasta: str) -> pd.DataFrame:
    """Precio de bolsa horario tal como lo entrega XM, sin limpiar."""
    return _rango("EC6945", desde, hasta)


def construir_precio(desde: str = "2023-01-01", hasta: str = "2026-10-31") -> pd.DataFrame:
    """Precio de bolsa horario, sin limpiar: tres variables y varias versiones por hora.

    La limpieza se hace en el cuaderno, a la vista, porque es una decisión de modelación.
    """
    tabla = crudo_precio(desde, hasta)
    tabla = tabla[["FechaHora", "CodigoVariable", "Version", "Valor"]]
    return tabla.sort_values(["FechaHora", "CodigoVariable", "Version"]).reset_index(drop=True)


def construir_embalses(desde: str = "2023-01-01", hasta: str = "2026-10-31") -> pd.DataFrame:
    """Energía útil almacenada y capacidad útil, por embalse y por día."""
    tabla = _rango("c51127", desde, hasta)
    tabla = tabla[["Fecha", "CodigoEmbalse", "RegionHidrologica",
                   "VolumenUtilDiarioEnergia", "CapacidadUtilEnergia"]]
    return tabla.sort_values(["Fecha", "CodigoEmbalse"]).reset_index(drop=True)
