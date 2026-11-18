"""Datos de la clase 2: Saber 11 (ICFES) y dengue (SIVIGILA de Bucaramanga y Tuluá).

Las tres tablas salen de datos.gov.co, que usa la API de Socrata. Esa API corta las
consultas largas, así que todo se pide por páginas de 50.000 filas, con un orden
fijo para que las páginas no se pisen.

Las tablas se entregan casi como las publica la fuente. La limpieza se hace en el
cuaderno, a la vista, porque limpiar es tomar decisiones de modelación.
"""
from __future__ import annotations

import io
import time

import pandas as pd
import requests

SOCRATA = "https://www.datos.gov.co/resource/{}.csv"
PAGINA = 50_000

# ------------------------------------------------------------------ Saber 11
SABER11 = "rnvb-vnyh"
# De las 81 columnas se piden 37. Quedan por fuera los nombres de colegio y sede,
# los códigos de municipio repetidos y varias preguntas del hogar que no se usan.
COLUMNAS_SABER11 = [
    "estu_consecutivo", "estu_genero", "estu_fechanacimiento", "estu_nacionalidad", "estu_tieneetnia",
    "fami_estratovivienda", "fami_personashogar", "fami_cuartoshogar", "fami_educacionpadre",
    "fami_educacionmadre", "fami_tieneinternet", "fami_tienecomputador", "fami_tieneautomovil",
    "fami_numlibros", "estu_dedicacionlecturadiaria", "estu_horassemanatrabaja",
    "cole_cod_dane_establecimiento", "cole_naturaleza", "cole_bilingue", "cole_caracter",
    "cole_area_ubicacion", "cole_jornada", "cole_mcpio_ubicacion", "cole_depto_ubicacion",
    "punt_lectura_critica", "punt_matematicas", "punt_c_naturales", "punt_sociales_ciudadanas",
    "punt_ingles", "desemp_matematicas", "punt_global", "percentil_global",
    "estu_inse_individual", "estu_nse_individual", "estu_nse_establecimiento",
    "estu_estadoinvestigacion", "estu_generacion_e",
]

# ------------------------------------------------------------------ Dengue
DENGUE = {"bucaramanga": "qzc7-jbg3", "tulua": "dq2k-gub9"}
# Columnas que no se piden: nombres y códigos de aseguradora, ocupación, versión del
# programa de captura y las ocho columnas de muestras de tejido (solo aplican a muertes).
SOBRAN_DENGUE = {
    "orden", "grupo", "ocupacion_", "cod_ase_", "aseguradora", "version", "nom_eve",
    "nombre_nacionalidad", "muesttejid", "mueshigado", "muesbazo", "muespulmon",
    "muescerebr", "muesmiocar", "muesmedula", "muesri_on",
}


def _pagina(conjunto: str, parametros: dict) -> pd.DataFrame:
    ultimo = None
    for intento in range(4):
        try:
            respuesta = requests.get(SOCRATA.format(conjunto), params=parametros, timeout=180)
            respuesta.raise_for_status()
            # Todo se lee como texto: adivinar tipos aquí escondería las trampas del dato.
            return pd.read_csv(io.StringIO(respuesta.text), dtype=str)
        except Exception as error:  # se reintenta con espera creciente
            ultimo = error
            time.sleep(3 * (intento + 1))
    raise RuntimeError(f"datos.gov.co no respondió para el conjunto {conjunto}") from ultimo


def _todo(conjunto: str, **parametros) -> pd.DataFrame:
    partes, desde = [], 0
    while True:
        parte = _pagina(conjunto, {**parametros, "$limit": PAGINA, "$offset": desde})
        partes.append(parte)
        if len(parte) < PAGINA:
            break
        desde += PAGINA
        print(f"   {desde:,} filas…".replace(",", "."), flush=True)
    return pd.concat(partes, ignore_index=True)


def columnas(conjunto: str) -> list[str]:
    """Nombres de las columnas de un conjunto, leídos de una fila."""
    return list(_pagina(conjunto, {"$limit": 1}).columns)


def construir_saber11(departamento: str = "CALDAS") -> pd.DataFrame:
    """Resultados de Saber 11 2020-2 de los colegios de un departamento.

    El nombre va como lo escribe el ICFES: en mayúsculas y casi siempre sin tilde
    («CALDAS», «RISARALDA», «CHOCO», «BOGOTÁ»).
    """
    tabla = _todo(SABER11, **{
        "$select": ",".join(COLUMNAS_SABER11),
        "$where": f"cole_depto_ubicacion='{departamento.upper()}'",
        "$order": "estu_consecutivo",
    })
    if tabla.empty:
        raise ValueError(f"No hay colegios con cole_depto_ubicacion = «{departamento.upper()}». "
                         "Revisa cómo escribe el ICFES el nombre del departamento.")
    return tabla


def construir_dengue(ciudad: str = "bucaramanga") -> pd.DataFrame:
    """Casos de dengue notificados al SIVIGILA, como los publica cada alcaldía."""
    ciudad = ciudad.lower()
    if ciudad not in DENGUE:
        raise ValueError(f"Solo hay datos de {', '.join(DENGUE)}. Recibí «{ciudad}».")
    conjunto = DENGUE[ciudad]
    pedir = [c for c in columnas(conjunto) if c not in SOBRAN_DENGUE]
    return _todo(conjunto, **{"$select": ",".join(pedir), "$order": ":id"})


FICHAS = {
    "saber11": {
        "titulo": "Saber 11 2020-2: resultados anonimizados por estudiante, un departamento",
        "fuente": "ICFES, en datos.gov.co, conjunto rnvb-vnyh",
        "url": "https://www.datos.gov.co/d/rnvb-vnyh",
        "licencia": "CC BY-SA 4.0. Citar al ICFES como fuente.",
        "constructor": construir_saber11,
        "md5": None,  # depende del departamento que se pida: se registra, no se compara
    },
    "dengue": {
        "titulo": "Dengue: casos notificados al SIVIGILA en Bucaramanga o en Tuluá",
        "fuente": "Alcaldías de Bucaramanga y de Tuluá, en datos.gov.co, conjuntos qzc7-jbg3 y dq2k-gub9",
        "url": "https://www.datos.gov.co/d/qzc7-jbg3",
        "licencia": "CC BY-SA 4.0. Citar a la alcaldía respectiva como fuente.",
        "constructor": construir_dengue,
        "md5": None,  # las alcaldías actualizan la tabla: se registra, no se compara
    },
}
