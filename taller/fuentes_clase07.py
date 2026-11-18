"""Datos de la clase 7: reseñas de turismo (Rest-Mex 2023) y contratos públicos (SECOP II).

Los dos son datos de texto y caben en una tabla: `descargar` los entrega completos.
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

AGENTE = {"User-Agent": "taller-cd-ml-2026-2"}


def _pedir(url: str, intentos: int = 4, **opciones) -> requests.Response:
    ultimo = None
    opciones.setdefault("timeout", 300)
    for intento in range(intentos):
        try:
            respuesta = requests.get(url, headers=AGENTE, **opciones)
            respuesta.raise_for_status()
            return respuesta
        except Exception as error:
            ultimo = error
            time.sleep(3 * (intento + 1))
    raise RuntimeError(f"La fuente no respondió: {url[:120]}") from ultimo


# ------------------------------------------------------------------ Rest-Mex
RESTMEX = "https://huggingface.co/datasets/javilonso/rest23_sentiment_data/resolve/main/data/"
RESTMEX_ARCHIVOS = {"train": "train-00000-of-00001-cfa5b64c3684b67f.parquet",
                    "test": "test-00000-of-00001-aa4ca60ff714d920.parquet"}
RESTMEX_POR_PAIS = 20_000


def restmex_completo() -> pd.DataFrame:
    """Las 251.291 reseñas de las dos particiones publicadas, con sus columnas originales."""
    partes = []
    for particion, archivo in RESTMEX_ARCHIVOS.items():
        tabla = pd.read_parquet(io.BytesIO(_pedir(RESTMEX + archivo).content))
        partes.append(tabla.assign(particion_original=particion))
    return pd.concat(partes, ignore_index=True)


def construir_restmex() -> pd.DataFrame:
    """Muestra de 60.000 reseñas: 20.000 por cada valor de `Country`.

    Dentro de cada país las reseñas se ordenan por su índice original y se toman
    20.000 repartidas de manera uniforme. No hay sorteo: la muestra es siempre la
    misma. Las columnas conservan los nombres y los códigos numéricos originales,
    porque el conjunto no publica su diccionario y la decodificación se hace en el
    cuaderno, a la vista.
    """
    completo = restmex_completo().rename(columns={"__index_level_0__": "indice_original"})
    esperadas = {"Title", "Review", "Polarity", "Country", "Type", "indice_original"}
    if not esperadas <= set(completo.columns):
        raise ValueError(f"Rest-Mex cambió de columnas: {sorted(completo.columns)}")
    partes = []
    for _, grupo in completo.groupby("Country"):
        grupo = grupo.sort_values("indice_original")
        partes.append(grupo.iloc[np.linspace(0, len(grupo) - 1, RESTMEX_POR_PAIS).astype(int)])
    muestra = pd.concat(partes).sort_values("indice_original").reset_index(drop=True)
    return muestra[["indice_original", "particion_original", "Title", "Review", "Polarity", "Country", "Type"]]


# ------------------------------------------------------------------ SECOP II
SECOP = "https://www.datos.gov.co/resource/jbjy-vk9h.csv"
# Columnas que se piden. Quedan por fuera los nombres y documentos de contratistas,
# representantes legales, supervisores y ordenadores del gasto, y los datos bancarios.
SECOP_COLUMNAS = ["id_contrato", "nombre_entidad", "nit_entidad", "departamento", "ciudad", "orden", "sector",
                  "tipo_de_contrato", "modalidad_de_contratacion", "estado_contrato", "codigo_de_categoria_principal",
                  "fecha_de_firma", "fecha_de_inicio_del_contrato", "fecha_de_fin_del_contrato",
                  "valor_del_contrato", "dias_adicionados", "es_pyme", "objeto_del_contrato"]
SECOP_ANIOS = (2022, 2023)


def secop_pagina(departamento: str, anio: int, desplazamiento: int, limite: int = 10_000) -> pd.DataFrame:
    parametros = {
        "$select": ",".join(SECOP_COLUMNAS),
        "$where": f"departamento='{departamento}' AND fecha_de_firma between '{anio}-01-01T00:00:00' and '{anio}-12-31T23:59:59'",
        "$order": "id_contrato", "$limit": limite, "$offset": desplazamiento,
    }
    return pd.read_csv(io.StringIO(_pedir(SECOP, params=parametros).text), dtype={"nit_entidad": str, "codigo_de_categoria_principal": str})


def construir_secop(departamento: str = "Caldas") -> pd.DataFrame:
    """Contratos electrónicos de SECOP II firmados en 2022 y 2023 por entidades de un departamento.

    Se pagina de a 10.000 filas, por año, porque las consultas grandes al portal se
    caen por tiempo. Se escogieron 2022 y 2023 para que los contratos hayan tenido
    tiempo de terminar y de registrar sus prórrogas.
    """
    partes = []
    for anio in SECOP_ANIOS:
        desplazamiento = 0
        while True:
            pagina = secop_pagina(departamento, anio, desplazamiento)
            partes.append(pagina)
            print(f"   SECOP II, {departamento} {anio}: {desplazamiento + len(pagina)} contratos…", flush=True)
            if len(pagina) < 10_000:
                break
            desplazamiento += 10_000
    tabla = pd.concat(partes, ignore_index=True)
    if tabla.empty:
        raise ValueError(f"SECOP II no tiene contratos de «{departamento}» en {SECOP_ANIOS}. Revisa cómo está escrito el departamento.")
    return tabla.drop_duplicates("id_contrato").sort_values("id_contrato").reset_index(drop=True)


FICHAS = {
    "restmex": {
        "titulo": "Rest-Mex 2023: reseñas de turismo en español de México, Cuba y Colombia (muestra de 60.000)",
        "fuente": "Rest-Mex 2023 (IberLEF), copia en Hugging Face: javilonso/rest23_sentiment_data",
        "url": "https://huggingface.co/datasets/javilonso/rest23_sentiment_data",
        "licencia": "Uso académico. No se redistribuye: se descarga de la fuente.",
        "constructor": construir_restmex,
        "md5": "c5e601a99dc0c8e5e7eb0901223ba211",
    },
    "secop": {
        "titulo": "SECOP II: contratos electrónicos firmados en 2022 y 2023, por departamento",
        "fuente": "Colombia Compra Eficiente, datos.gov.co, conjunto jbjy-vk9h",
        "url": "https://www.datos.gov.co/d/jbjy-vk9h",
        "licencia": "CC BY-SA 4.0. Citar a Colombia Compra Eficiente.",
        "constructor": construir_secop,
        "md5": None,   # el portal actualiza los contratos: el MD5 se registra y no se compara
    },
}
