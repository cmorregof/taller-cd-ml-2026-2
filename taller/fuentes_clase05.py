"""Datos de la clase 5: lunares (HAM10000) y hojas de café (RoCoLe).

Los dos conjuntos son de imágenes. `descargar(nombre)` entrega la tabla de
metadatos y deja las fotos, reducidas a 256 píxeles de lado corto, en
`datos/cache/<nombre>/`. La carpeta se obtiene con `imagenes(nombre)`.
"""
from __future__ import annotations

import io
import re

import pandas as pd

from .clase05_imagenes import CACHE, ZipRemoto, en_paralelo, pedir, reducir

# ------------------------------------------------------------------ HAM10000
DATAVERSE = "https://dataverse.harvard.edu/api/access/datafile/{}"
HAM_METADATOS = "4338392"
HAM_PARTES = ["3172585", "3172584"]      # HAM10000_images_part_1.zip y part_2.zip


def ham_metadatos() -> pd.DataFrame:
    """Las 10.015 filas de metadatos tal como las publica Dataverse."""
    return pd.read_csv(io.BytesIO(pedir(DATAVERSE.format(HAM_METADATOS)).content), sep="\t")


def ham_en_muestra(tabla: pd.DataFrame) -> pd.Series:
    """El subconjunto del curso: melanomas y nevus confirmados por histopatología."""
    return tabla["dx"].isin(["mel", "nv"]) & (tabla["dx_type"] == "histo")


def construir_ham10000() -> pd.DataFrame:
    """Metadatos completos de HAM10000 y las fotos del subconjunto del curso.

    Los dos ZIP de imágenes pesan 2,8 GB. Aquí se baja solo el subconjunto
    (3.611 fotos, cerca de 1 GB de tráfico) leyendo cada foto del ZIP remoto por
    rango de bytes. Las filas fuera del subconjunto quedan en la tabla, sin foto,
    para poder explorar el conjunto entero.
    """
    tabla = ham_metadatos()
    tabla["en_muestra"] = ham_en_muestra(tabla)
    tabla["archivo"] = (tabla["image_id"] + ".jpg").where(tabla["en_muestra"], "")
    carpeta = CACHE / "ham10000"
    carpeta.mkdir(parents=True, exist_ok=True)
    pendientes = [a for a in tabla.loc[tabla["en_muestra"], "archivo"] if not (carpeta / a).exists()]
    if pendientes:
        print("   Dataverse: leo el índice de los dos ZIP de imágenes…", flush=True)
        partes = [ZipRemoto(DATAVERSE.format(p)) for p in HAM_PARTES]

        def bajar(nombre):
            parte = next(p for p in partes if nombre in p.indice)
            (carpeta / nombre).write_bytes(reducir(parte.leer(nombre)))

        en_paralelo(bajar, pendientes, hilos=16, aviso="fotos de HAM10000", cada=500)
    return tabla.sort_values(["lesion_id", "image_id"]).reset_index(drop=True)


# -------------------------------------------------------------------- RoCoLe
ROCOLE_API = "https://data.mendeley.com/public-api/datasets/c5yvn32dzg"
PATRON = re.compile(r"^C(\d+)P(\d+)([A-Z]+)(\d*)\.jpg$")


def rocole_listado() -> list[dict]:
    """Lista de archivos del conjunto, según la API pública de Mendeley Data."""
    return pedir(ROCOLE_API, headers={"Accept": "application/json"}).json()["files"]


def rocole_etiquetas(listado: list[dict]) -> pd.DataFrame:
    """Etiquetas publicadas por los autores en RoCoLe-classes.xlsx, leídas sin openpyxl."""
    import zipfile
    from xml.etree import ElementTree as ET

    url = next(f for f in listado if f["filename"] == "RoCoLe-classes.xlsx")["content_details"]["download_url"]
    libro = zipfile.ZipFile(io.BytesIO(pedir(url).content))
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    textos = ["".join(t.itertext()) for t in ET.fromstring(libro.read("xl/sharedStrings.xml")).findall("m:si", ns)]
    hoja = next(n for n in libro.namelist() if n.startswith("xl/worksheets/sheet"))
    filas = []
    for fila in ET.fromstring(libro.read(hoja)).iter(f"{{{ns['m']}}}row"):
        celdas = []
        for celda in fila.findall("m:c", ns):
            valor = celda.find("m:v", ns)
            if valor is None:
                celdas.append("")
            else:
                celdas.append(textos[int(valor.text)] if celda.get("t") == "s" else valor.text)
        filas.append(celdas)
    tabla = pd.DataFrame(filas[1:], columns=filas[0])
    esperadas = ["File", "Binary.Label", "Multiclass.Label"]
    if list(tabla.columns) != esperadas:
        raise ValueError(f"RoCoLe-classes.xlsx cambió de columnas: {list(tabla.columns)}")
    return tabla


def construir_rocole() -> pd.DataFrame:
    """Las 1.560 fotos de hojas de café robusta con su etiqueta y su planta.

    El nombre de cada foto tiene la forma C<n>P<m><letras><k>.jpg. El artículo de los
    autores no explica el código. La tabla conserva sus partes sin interpretarlas,
    salvo una: `planta` es el prefijo C<n>P<m>, que forma 390 grupos de 4 fotos, lo
    mismo que el artículo dice de las plantas (390 plantas, 4 imágenes por planta).
    """
    listado = rocole_listado()
    fotos = {f["filename"]: f for f in listado if f["filename"].lower().endswith(".jpg")}
    etiquetas = rocole_etiquetas(listado)
    faltan = set(etiquetas["File"]) ^ set(fotos)
    if faltan:
        raise ValueError(f"Las etiquetas y las fotos de RoCoLe no coinciden: {sorted(faltan)[:5]}")
    partes = etiquetas["File"].str.extract(PATRON)
    tabla = pd.DataFrame({
        "foto": etiquetas["File"],
        "planta": "C" + partes[0] + "P" + partes[1],
        "c": partes[0].astype(int),
        "p": partes[1].astype(int),
        "letras": partes[2],
        "estado": etiquetas["Binary.Label"],
        "clase": etiquetas["Multiclass.Label"],
        "bytes_original": [fotos[n]["size"] for n in etiquetas["File"]],
        "sha256_original": [fotos[n]["content_details"]["sha256_hash"] for n in etiquetas["File"]],
        "archivo": etiquetas["File"],
    })
    carpeta = CACHE / "rocole"
    carpeta.mkdir(parents=True, exist_ok=True)

    def bajar(nombre):
        (carpeta / nombre).write_bytes(reducir(pedir(fotos[nombre]["content_details"]["download_url"]).content))

    pendientes = [n for n in tabla["archivo"] if not (carpeta / n).exists()]
    en_paralelo(bajar, pendientes, hilos=16, aviso="fotos de RoCoLe", cada=250)
    return tabla.sort_values(["c", "p", "foto"]).reset_index(drop=True)


FICHAS = {
    "ham10000": {
        "titulo": "HAM10000: imágenes dermatoscópicas de lesiones pigmentadas (subconjunto: melanoma y nevus con histopatología)",
        "fuente": "Tschandl, Rosendahl y Kittler (2018), Harvard Dataverse, doi:10.7910/DVN/DBW86T",
        "url": "https://doi.org/10.7910/DVN/DBW86T",
        "documentacion": "https://doi.org/10.1038/sdata.2018.161",   # el artículo de los autores en Scientific Data
        "licencia": "CC BY-NC 4.0. Uso no comercial con cita a los autores.",
        "constructor": construir_ham10000,
        "md5": "16d84359162849ef638313c51e5f6345",   # MD5 de la tabla de metadatos; las fotos se verifican con el CRC32 del ZIP
    },
    "rocole": {
        "titulo": "RoCoLe: 1.560 fotos de hojas de café robusta, sanas y enfermas",
        "fuente": "Parraga-Alava, Cusme, Loor y Santander (2019), Mendeley Data, doi:10.17632/c5yvn32dzg.2",
        "url": "https://data.mendeley.com/datasets/c5yvn32dzg/2",
        "documentacion": "https://doi.org/10.1016/j.dib.2019.104414",   # el artículo de los autores en Data in Brief
        "licencia": "CC BY 4.0. Citar a los autores.",
        "constructor": construir_rocole,
        "md5": "d5e7fd580b8a3584fcc2e416bc7bea85",   # MD5 de la tabla; incluye el SHA-256 que Mendeley publica de cada foto
    },
}
