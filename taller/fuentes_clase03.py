"""Datos de la clase 3: los medidores de Londres, la lluvia del IDEAM y las emergencias de la UNGRD.

Los dos conjuntos grandes no caben cómodos en memoria si se leen de un golpe, así
que aquí nada se baja entero:

- Londres. La fuente oficial es un ZIP de 796 MB con 168 archivos CSV de un millón
  de filas cada uno (8,5 GB sin comprimir). El servidor acepta peticiones de rango,
  así que se lee el índice del ZIP y después solo los archivos que se pidan.
- IDEAM. La lluvia viene cada diez minutos. La API de datos.gov.co se cae con
  consultas grandes, así que se pide un pedazo por estación y por año,
  y cada año en cuatro trimestres.

La limpieza y la agregación se hacen en el cuaderno, a la vista.
"""
from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

# ------------------------------------------------------------------ Londres
LONDRES_ZIP = ("https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/"
               "Partitioned%20LCL%20Data.zip")
LONDRES_PAGINA = "https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d"
TARIFA_PLANA = 14.228          # peniques por kWh, según la ficha del conjunto en London Datastore
TOTAL_ARCHIVOS = 168


class ArchivoRemoto(io.RawIOBase):
    """Un archivo que vive en un servidor y se lee por pedazos con peticiones de rango.

    Le sirve a `zipfile` para abrir un ZIP sin bajarlo: primero pide el final, donde
    está el índice, y después solo los bytes del archivo interno que se quiera leer.
    """

    def __init__(self, url: str):
        self.url, self.posicion, self.sesion = url, 0, requests.Session()
        self.peticiones, self.bytes_bajados = 0, 0
        respuesta = self._pedir(0, 0)
        self.tamano = int(respuesta.headers["Content-Range"].split("/")[1])

    def _pedir(self, desde: int, hasta: int) -> requests.Response:
        ultimo = None
        for intento in range(4):
            try:
                respuesta = self.sesion.get(self.url, headers={"Range": f"bytes={desde}-{hasta}"}, timeout=180)
                if respuesta.status_code != 206:
                    raise RuntimeError(f"el servidor respondió {respuesta.status_code} y no 206")
                self.peticiones += 1
                self.bytes_bajados += len(respuesta.content)
                return respuesta
            except Exception as error:  # se reintenta con espera creciente
                ultimo = error
                time.sleep(2 * (intento + 1))
        raise RuntimeError("London Datastore no aceptó la petición de rango") from ultimo

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.posicion

    def seek(self, desplazamiento: int, desde: int = 0) -> int:
        base = {0: 0, 1: self.posicion, 2: self.tamano}[desde]
        self.posicion = base + desplazamiento
        return self.posicion

    def readinto(self, destino) -> int:
        if self.posicion >= self.tamano or len(destino) == 0:
            return 0
        hasta = min(self.posicion + len(destino), self.tamano) - 1
        contenido = self._pedir(self.posicion, hasta).content
        destino[:len(contenido)] = contenido
        self.posicion += len(contenido)
        return len(contenido)


def abrir_londres():
    """Abre el ZIP oficial sin bajarlo. Devuelve el ZIP, la lista ordenada de sus 168 archivos y el archivo remoto."""
    remoto = ArchivoRemoto(LONDRES_ZIP)
    paquete = zipfile.ZipFile(io.BufferedReader(remoto, buffer_size=1 << 20))
    internos = [i for i in paquete.infolist() if i.filename.lower().endswith(".csv")]
    internos.sort(key=lambda i: int(i.filename.rsplit("_", 1)[1].split(".")[0]))     # ..._0.csv, ..._1.csv, ..., ..._167.csv
    return paquete, internos, remoto


def cuales(archivos: int = 24) -> list[int]:
    """Los números de archivo del subconjunto del curso: `archivos` de los 168, repartidos por igual."""
    if not 1 <= archivos <= TOTAL_ARCHIVOS:
        raise ValueError(f"`archivos` debe estar entre 1 y {TOTAL_ARCHIVOS}.")
    paso = TOTAL_ARCHIVOS / archivos
    return sorted({int(k * paso) for k in range(archivos)})


def resumir_pedazo(crudo: pd.DataFrame) -> pd.DataFrame:
    """Lleva un pedazo de lecturas de media hora a una fila por hogar y por día.

    Es la misma regla que el cuaderno de la clase escribe a la vista: la lectura
    «Null» es faltante, las filas repetidas se cuentan una vez, y por cada hogar y
    día se guardan la suma de kWh y el número de lecturas.
    """
    crudo = crudo.rename(columns=lambda nombre: nombre.strip())
    crudo = crudo.rename(columns={"KWH/hh (per half hour)": "kwh"}).drop_duplicates()
    crudo["kwh"] = pd.to_numeric(crudo["kwh"], errors="coerce")
    crudo["fecha"] = crudo["DateTime"].str[:10]
    return (crudo.groupby(["LCLid", "stdorToU", "fecha"], observed=True)["kwh"]
            .agg(kwh="sum", lecturas="count").reset_index()
            .rename(columns={"LCLid": "hogar", "stdorToU": "tarifa"}))


def juntar(resumenes: list[pd.DataFrame]) -> pd.DataFrame:
    """Une los resúmenes. Un hogar puede quedar partido entre dos archivos: por eso se vuelve a sumar."""
    return (pd.concat(resumenes, ignore_index=True)
            .groupby(["hogar", "tarifa", "fecha"], observed=True)[["kwh", "lecturas"]].sum().reset_index()
            .sort_values(["hogar", "fecha"]).reset_index(drop=True))


def construir_londres(archivos: int = 24) -> pd.DataFrame:
    """Consumo diario por hogar, hecho con `archivos` de los 168 archivos oficiales.

    Es el mismo resultado que arma el cuaderno paso a paso. Sirve de copia de
    respaldo y para quien quiera la tabla sin repetir el proceso.
    """
    paquete, internos, _ = abrir_londres()
    resumenes = []
    for numero in cuales(archivos):
        print(f"   Londres, archivo {numero} de {TOTAL_ARCHIVOS}…", flush=True)
        with paquete.open(internos[numero]) as flujo:
            resumenes.append(resumir_pedazo(pd.read_csv(flujo, dtype=str)))
    tabla = juntar(resumenes)
    tabla["kwh"] = tabla["kwh"].round(4)
    return tabla


# ------------------------------------------------------------------ IDEAM
SOCRATA = "https://www.datos.gov.co/resource/{}.csv"
IDEAM_LLUVIA = "s54a-sgyg"
ESTACIONES = {
    "0026155502": "Cenicafé",
    "0023025502": "Almacafé Letras",
    "0026155230": "EMAS",
}
_caida = {"ideam": False}     # si un pedazo falla, los siguientes no insisten: van directo al respaldo


def _socrata(conjunto: str, parametros: dict, espera: int = 90, intentos: int = 2) -> pd.DataFrame:
    ultimo = None
    for intento in range(intentos):
        try:
            respuesta = requests.get(SOCRATA.format(conjunto), params=parametros, timeout=espera)
            respuesta.raise_for_status()
            return pd.read_csv(io.StringIO(respuesta.text), dtype=str)
        except Exception as error:
            ultimo = error
            time.sleep(3)
    raise RuntimeError(f"datos.gov.co no respondió a tiempo para el conjunto {conjunto}") from ultimo


def _trimestre(estacion: str, desde: str, hasta: str) -> pd.DataFrame:
    donde = (f"codigoestacion='{estacion}' AND fechaobservacion >= '{desde}T00:00:00' "
             f"AND fechaobservacion < '{hasta}T00:00:00'")
    return _socrata(IDEAM_LLUVIA, {"$select": "fechaobservacion,valorobservado", "$where": donde, "$limit": 60000})


def construir_lluvia(estacion: str = "0026155502", anio: int = 2022) -> pd.DataFrame:
    """Lluvia cada pocos minutos de una estación del IDEAM en un año, tal como la publica la fuente.

    El año se pide en cuatro trimestres a la vez: la API responde en segundos a un
    trimestre y se cae con el año entero.
    """
    if _caida["ideam"]:
        raise RuntimeError("La API del IDEAM ya falló en esta sesión: se usa la copia de respaldo.")
    anio = int(anio)
    cortes = [f"{anio}-01-01", f"{anio}-04-01", f"{anio}-07-01", f"{anio}-10-01", f"{anio + 1}-01-01"]
    try:
        with ThreadPoolExecutor(max_workers=4) as grupo:
            partes = list(grupo.map(lambda par: _trimestre(estacion, *par), zip(cortes[:-1], cortes[1:])))
    except Exception:
        _caida["ideam"] = True
        raise
    tabla = pd.concat(partes, ignore_index=True)
    return tabla.sort_values("fechaobservacion", kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------ UNGRD
# Tres conjuntos, uno por periodo, con nombres de columna distintos. Se piden las
# cuatro columnas que comparten.
UNGRD = {"2019 a 2022": "wwkg-r6te", "2023 y 2024": "rgre-6ak4", "2025": "2343-nuqp"}


def construir_emergencias(departamento: str = "CALDAS") -> pd.DataFrame:
    """Emergencias reportadas a la UNGRD en un departamento, de 2019 a 2025, sin limpiar."""
    partes = []
    for periodo, conjunto in UNGRD.items():
        print(f"   UNGRD, {periodo}…", flush=True)
        parte = _socrata(conjunto, {"$select": "fecha,departamento,municipio,evento",
                                    "$where": f"upper(departamento) like '%{departamento.upper()}%'", "$limit": 50000},
                         espera=120, intentos=3)
        parte["conjunto"] = conjunto
        partes.append(parte)
    return pd.concat(partes, ignore_index=True)


FICHAS = {
    "londres_diario": {
        "titulo": "Londres: consumo diario por hogar, de los medidores inteligentes de Low Carbon London",
        "fuente": "UK Power Networks, en London Datastore, «SmartMeter Energy Consumption Data in London Households»",
        "url": LONDRES_PAGINA,
        "licencia": "CC BY 4.0. Citar a UK Power Networks.",
        "constructor": construir_londres,
        "md5": None,  # depende de cuántos archivos se pidan: se registra, no se compara
    },
    "lluvia_ideam": {
        "titulo": "IDEAM: precipitación cada diez minutos, una estación y un año",
        "fuente": "IDEAM, en datos.gov.co, conjunto s54a-sgyg",
        "url": "https://www.datos.gov.co/d/s54a-sgyg",
        "licencia": "CC BY-SA 4.0. Citar al IDEAM como fuente.",
        "constructor": construir_lluvia,
        "md5": None,  # el IDEAM actualiza y corrige la serie: se registra, no se compara
    },
    "emergencias_ungrd": {
        "titulo": "UNGRD: emergencias reportadas en un departamento, 2019 a 2025",
        "fuente": "UNGRD, en datos.gov.co, conjuntos wwkg-r6te, rgre-6ak4 y 2343-nuqp",
        "url": "https://www.datos.gov.co/d/wwkg-r6te",
        "licencia": "CC BY-SA 4.0 (2019 a 2024) y CC BY 4.0 (2025). Citar a la UNGRD como fuente.",
        "constructor": construir_emergencias,
        "md5": None,
    },
}
