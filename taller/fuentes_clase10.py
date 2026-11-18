"""Datos de la clase 10: réplicas de reclutamiento (JEP, CEV y HRDAG en el DANE) y tráfico aéreo (Aerocivil)."""
from __future__ import annotations

import io
import time
import zipfile

import pandas as pd
import requests

RECLUTAMIENTO = "https://microdatos.dane.gov.co/index.php/catalog/795/download/22641"     # Reclutamiento.parquet.zip
VUELOS = "https://www.datos.gov.co/resource/gb6w-ynu4.json"
AEROPUERTOS = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def _bajar(url: str, **opciones) -> requests.Response:
    ultimo = None
    for intento in range(3):
        try:
            respuesta = requests.get(url, timeout=300, headers={"User-Agent": "taller-cd-ml-2026-2"}, **opciones)
            respuesta.raise_for_status()
            return respuesta
        except Exception as error:
            ultimo = error
            time.sleep(4 * (intento + 1))
    raise RuntimeError(f"No respondió {url}") from ultimo


def construir_reclutamiento(replicas: int = 10) -> pd.DataFrame:
    """Las primeras `replicas` réplicas (de 100) de víctimas documentadas de reclutamiento, apiladas.

    Cada réplica trae las mismas 19.226 víctimas documentadas. Los campos que alguna fuente
    registró son iguales en todas las réplicas; los que faltaban están imputados y cambian de
    una réplica a otra. Las columnas `*_imputed` dicen cuáles son. Las 34 columnas `in_k`
    dicen qué fuentes (anonimizadas) documentaron a la víctima; aquí se agrega `n_fuentes`.
    """
    replicas = int(replicas)
    if not 2 <= replicas <= 100:
        raise ValueError("El número de réplicas debe estar entre 2 y 100.")
    archivo = zipfile.ZipFile(io.BytesIO(_bajar(RECLUTAMIENTO).content))
    partes = []
    for k in range(1, replicas + 1):
        parte = pd.read_parquet(io.BytesIO(archivo.read(f"Reclutamiento.parquet/verdata-reclutamiento-R{k}.parquet")))
        partes.append(parte)
    tabla = pd.concat(partes, ignore_index=True)
    fuentes = [c for c in tabla.columns if c.startswith("in_")]
    tabla[fuentes] = tabla[fuentes].astype("int8")
    tabla.insert(2, "n_fuentes", tabla[fuentes].sum(axis=1))
    for columna in ["etnia", "p_str", "sexo"]:
        tabla[columna] = tabla[columna].astype(str)
    tabla["yymm_hecho"] = tabla["yymm_hecho"].astype(int)
    tabla["is_conflict"] = tabla["is_conflict"].astype(int)
    return tabla


def construir_vuelos(aeropuerto: str = "MZL") -> pd.DataFrame:
    """Tráfico por origen y destino de todos los trayectos que salen de `aeropuerto` o llegan a él (sigla IATA)."""
    aeropuerto = str(aeropuerto).upper()
    partes, desde = [], 0
    while True:
        lote = _bajar(VUELOS, params={"$where": f"origen='{aeropuerto}' OR destino='{aeropuerto}'",
                                      "$order": "a_o, n_mero_de_mes, sigla_empresa, origen, destino, tipo_vuelo",
                                      "$limit": 20000, "$offset": desde}).json()
        partes.extend(lote)
        if len(lote) < 20000:
            break
        desde += 20000
    if not partes:
        raise ValueError(f"Aerocivil no tiene trayectos con la sigla «{aeropuerto}».")
    tabla = pd.DataFrame(partes).rename(columns={"a_o": "anio", "n_mero_de_mes": "mes", "tr_fico_n_i": "trafico"})
    for columna in ["anio", "mes", "pasajeros"]:
        tabla[columna] = pd.to_numeric(tabla[columna]).astype(int)
    tabla["carga_correo_kg"] = pd.to_numeric(tabla["carga_correo_kg"])
    return tabla[["anio", "mes", "sigla_empresa", "nombre", "origen", "nombre_origen", "ciudad_origen", "pais_origen",
                  "destino", "nombre_destino", "ciudad_destino", "pais_destino", "trafico", "tipo_vuelo",
                  "pasajeros", "carga_correo_kg"]]


def construir_aeropuertos() -> pd.DataFrame:
    """Coordenadas de los aeropuertos con sigla IATA, tomadas de OurAirports."""
    tabla = pd.read_csv(io.BytesIO(_bajar(AEROPUERTOS).content), keep_default_na=False)
    tabla = tabla[tabla["iata_code"] != ""]
    tabla = tabla[["iata_code", "name", "municipality", "iso_country", "type", "latitude_deg", "longitude_deg"]]
    return tabla.sort_values("iata_code").reset_index(drop=True)


FICHAS = {
    "reclutamiento": {
        "titulo": "Reclutamiento en el conflicto armado, 1990 a 2017: víctimas documentadas con campos imputados (réplicas)",
        "fuente": "JEP, Comisión de la Verdad y HRDAG; publicado por el DANE, Microdatos, catálogo 795",
        "url": "https://microdatos.dane.gov.co/index.php/catalog/795",
        "licencia": ("Creative Commons Atribución 4.0 Internacional, según la ficha del DANE. Citar a la Comisión de la "
                     "Verdad, la JEP y HRDAG. Los datos están anonimizados por el DANE."),
        "constructor": construir_reclutamiento,
        "md5": None,
    },
    "vuelos": {
        "titulo": "Aerocivil: tráfico aéreo comercial por origen y destino, 2020 a junio de 2026, de un aeropuerto",
        "fuente": "Aeronáutica Civil, datos.gov.co, conjunto gb6w-ynu4",
        "url": "https://www.datos.gov.co/d/gb6w-ynu4",
        "licencia": "Creative Commons Attribution Share Alike 4.0. Citar a la Aeronáutica Civil.",
        "constructor": construir_vuelos,
        "md5": None,
    },
    "aeropuertos": {
        "titulo": "OurAirports: coordenadas de los aeropuertos con sigla IATA",
        "fuente": "OurAirports (ourairports.com), archivo airports.csv",
        "url": "https://ourairports.com/data/",
        "licencia": "Dominio público, según OurAirports. No es una fuente oficial de la Aeronáutica Civil.",
        "constructor": construir_aeropuertos,
        "md5": None,
    },
}
