"""Lector mínimo de hojas grandes de Excel (.xlsx).

Un .xlsx es un ZIP con un XML por hoja. `pandas.read_excel` arma toda la hoja
en memoria y con 278.000 filas tarda varios minutos. Aquí se recorre el XML de
la hoja fila por fila, que es más rápido y usa poca memoria.
"""
from __future__ import annotations

import re
import zipfile
from xml.etree.ElementTree import iterparse

import pandas as pd

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _letras_a_numero(referencia: str) -> int:
    numero = 0
    for letra in referencia:
        if not letra.isalpha():
            break
        numero = numero * 26 + ord(letra) - 64
    return numero - 1


def hojas(ruta) -> dict[str, str]:
    """Nombre de cada hoja y el archivo interno donde vive."""
    with zipfile.ZipFile(ruta) as paquete:
        libro = paquete.read("xl/workbook.xml").decode("utf-8")
        relaciones = paquete.read("xl/_rels/workbook.xml.rels").decode("utf-8")
    destino = dict(re.findall(r'<Relationship[^>]*Id="([^"]+)"[^>]*Target="([^"]+)"', relaciones))
    destino.update({i: t for t, i in re.findall(r'<Relationship[^>]*Target="([^"]+)"[^>]*Id="([^"]+)"', relaciones)})
    return {nombre: "xl/" + destino[rid].lstrip("/").removeprefix("xl/")
            for nombre, rid in re.findall(r'<sheet[^>]*name="([^"]+)"[^>]*r:id="([^"]+)"', libro)}


def _textos_compartidos(paquete: zipfile.ZipFile) -> list[str]:
    textos = []
    with paquete.open("xl/sharedStrings.xml") as flujo:
        for _, nodo in iterparse(flujo):
            if nodo.tag == NS + "si":
                textos.append("".join(t.text or "" for t in nodo.iter(NS + "t")))
                nodo.clear()
    return textos


def leer_hoja(ruta, hoja: str, columnas: list[str] | None = None, max_filas: int | None = None) -> pd.DataFrame:
    """Lee una hoja completa como texto. La primera fila es el encabezado.

    `columnas` deja solo esas columnas. Las celdas vacías quedan como faltantes.
    """
    interno = hojas(ruta)[hoja]
    with zipfile.ZipFile(ruta) as paquete:
        textos = _textos_compartidos(paquete)
        encabezado, posiciones, filas = None, None, []
        with paquete.open(interno) as flujo:
            for _, nodo in iterparse(flujo):
                if nodo.tag != NS + "row":
                    continue
                valores = {}
                for celda in nodo:
                    v = celda.find(NS + "v")
                    if v is None or v.text is None:
                        en_linea = celda.find(NS + "is")
                        if en_linea is None:
                            continue
                        texto = "".join(t.text or "" for t in en_linea.iter(NS + "t"))
                    else:
                        texto = textos[int(v.text)] if celda.get("t") == "s" else v.text
                    valores[_letras_a_numero(celda.get("r"))] = texto
                nodo.clear()
                if encabezado is None:
                    encabezado = {posicion: nombre.strip() for posicion, nombre in valores.items()}
                    posiciones = [p for p, n in encabezado.items() if columnas is None or n in columnas]
                    continue
                filas.append([valores.get(p) for p in posiciones])
                if max_filas and len(filas) >= max_filas:
                    break
    return pd.DataFrame(filas, columns=[encabezado[p] for p in posiciones])
