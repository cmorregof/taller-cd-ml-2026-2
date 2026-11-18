"""SIPSA mensual 2013 a 2024, construido desde los ocho ZIP oficiales del DANE.

Adaptado de `congelar_sipsa.py` del repositorio labs-calculo-2026-2. Cada CSV
anual usa nombres de columna y formatos de fecha distintos. Aquí se normalizan
a cinco columnas: fecha, ciudad, mercado, producto y precio_kg.
"""
from __future__ import annotations

import csv
import io
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime

import pandas as pd

FUENTES = {
    "2013-2017": "23898", "2018": "23897", "2019": "23896", "2020": "23895",
    "2021": "23894", "2022": "23893", "2023": "23892", "2024": "23891",
}
URL = "https://microdatos.dane.gov.co/index.php/catalog/776/download/{}"
MESES = {"ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
         "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12}


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def _fecha(valor: str) -> str:
    """Convierte 1/01/2013 o ene-24 al formato AAAA-MM."""
    valor = valor.strip()
    if "/" in valor:
        return datetime.strptime(valor, "%d/%m/%Y").strftime("%Y-%m")
    mes, anio = valor.lower().split("-")
    return f"20{anio}-{MESES[mes]:02d}"


def _precio(valor: str) -> float:
    """Convierte separadores colombianos de miles a un número."""
    valor = valor.replace("\xa0", "").strip().replace(" ", "")
    if "," in valor and "." in valor:
        valor = valor.replace(".", "").replace(",", ".")
    elif "," in valor:
        valor = valor.replace(",", ".")
    elif "." in valor:
        entero, decimal = valor.rsplit(".", 1)
        if len(decimal) == 3:
            valor = entero.replace(".", "") + decimal
    return float(valor)


def _leer_zip(contenido: bytes) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(contenido)) as archivo:
        nombre = next(n for n in archivo.namelist() if n.lower().endswith(".csv"))
        bruto = archivo.read(nombre)
    try:
        texto = bruto.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = bruto.decode("latin-1")
    lector = csv.reader(io.StringIO(texto), delimiter=";")
    encabezado = next(lector)
    pos = {_sin_tildes(n): i for i, n in enumerate(encabezado)}
    i_fecha, i_producto = pos["fecha"], pos["producto"]
    i_mercado = pos["mercado"] if "mercado" in pos else pos["fuente"]
    i_precio = next(i for n, i in pos.items() if n.startswith("precio"))
    filas = []
    for fila in lector:
        if not fila or len(fila) <= max(i_fecha, i_producto, i_mercado, i_precio):
            continue
        mercado, producto = fila[i_mercado].strip(), fila[i_producto].strip()
        if not mercado or not producto:
            continue
        filas.append({"fecha": _fecha(fila[i_fecha]),
                      "ciudad": mercado.split(",", 1)[0].strip(),
                      "mercado": mercado, "producto": producto,
                      "precio_kg": _precio(fila[i_precio])})
    return filas


def construir() -> pd.DataFrame:
    filas = []
    for etiqueta, identificador in FUENTES.items():
        print(f"   DANE, archivo {etiqueta}…", flush=True)
        solicitud = urllib.request.Request(URL.format(identificador),
                                           headers={"User-Agent": "taller-cd-ml-2026-2"})
        with urllib.request.urlopen(solicitud, timeout=120) as respuesta:
            filas.extend(_leer_zip(respuesta.read()))
    # El archivo de 2021 repite algunas observaciones. Se conserva una sola por
    # mes, mercado y producto, promediando el precio repetido.
    grupos = defaultdict(list)
    for fila in filas:
        grupos[(fila["fecha"], fila["mercado"], fila["producto"])].append(fila)
    limpias = []
    for grupo in grupos.values():
        fila = dict(grupo[0])
        fila["precio_kg"] = round(sum(o["precio_kg"] for o in grupo) / len(grupo), 2)
        limpias.append(fila)
    limpias.sort(key=lambda f: (f["fecha"], f["ciudad"], f["mercado"], f["producto"]))
    return pd.DataFrame(limpias, columns=["fecha", "ciudad", "mercado", "producto", "precio_kg"])
