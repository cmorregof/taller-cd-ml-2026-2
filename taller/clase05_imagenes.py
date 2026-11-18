"""Piezas para los datos de imágenes (clases 5 y 6).

Una tabla cabe en un CSV y una carpeta de fotos no. Por eso los datos de imágenes
viajan en dos partes:

- la tabla de metadatos, que entrega `descargar(nombre)` como cualquier otro dato, y
- las fotos, que el constructor deja en `datos/cache/<nombre>/`, reducidas a 256
  píxeles en su lado corto y guardadas como JPEG.

`imagenes(nombre)` devuelve esa carpeta. Si la carpeta no existe (porque la fuente
se cayó y la tabla salió del respaldo), la arma desde `respaldo/<nombre>_imagenes.zip`.

Este módulo no importa `torch`: la red neuronal se carga en el cuaderno, a la vista.
"""
from __future__ import annotations

import io
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "datos" / "cache"
RESPALDO = RAIZ / "respaldo"
AGENTE = {"User-Agent": "taller-cd-ml-2026-2"}
LADO = 256  # lado corto, en píxeles, de las copias que se guardan


def pedir(url: str, intentos: int = 4, **opciones) -> requests.Response:
    """GET con reintentos y espera creciente."""
    ultimo = None
    opciones.setdefault("timeout", 120)
    encabezados = {**AGENTE, **opciones.pop("headers", {})}
    for intento in range(intentos):
        try:
            respuesta = requests.get(url, headers=encabezados, **opciones)
            respuesta.raise_for_status()
            return respuesta
        except Exception as error:
            ultimo = error
            time.sleep(2 * (intento + 1))
    raise RuntimeError(f"La fuente no respondió: {url[:120]}") from ultimo


def reducir(contenido: bytes, lado: int = LADO) -> bytes:
    """Devuelve la foto como JPEG RGB con el lado corto de `lado` píxeles.

    Respeta la orientación que el teléfono anotó en los metadatos EXIF y no agranda
    fotos que ya son pequeñas.
    """
    from PIL import Image, ImageOps

    foto = Image.open(io.BytesIO(contenido))
    foto.draft("RGB", (2 * lado, 2 * lado))   # decodifica el JPEG a menor escala: mucho más rápido
    foto = ImageOps.exif_transpose(foto).convert("RGB")
    ancho, alto = foto.size
    escala = lado / min(ancho, alto)
    if escala < 1:
        foto = foto.resize((max(lado, round(ancho * escala)), max(lado, round(alto * escala))),
                           Image.Resampling.LANCZOS)
    salida = io.BytesIO()
    foto.save(salida, "JPEG", quality=90)
    return salida.getvalue()


class _ArchivoHTTP(io.RawIOBase):
    """Archivo remoto de solo lectura que pide por HTTP los bytes que le leen."""

    def __init__(self, url: str, cola: int = 3_000_000):
        self.url = url
        encabezado = pedir(url, headers={"Range": "bytes=0-0"}).headers["Content-Range"]
        self.largo = int(encabezado.rsplit("/", 1)[1])
        self.inicio_cola = max(0, self.largo - cola)
        self.cola = self.rango(self.inicio_cola, self.largo - 1)
        self.posicion = 0

    def rango(self, desde: int, hasta: int) -> bytes:
        return pedir(self.url, headers={"Range": f"bytes={desde}-{hasta}"}).content

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.posicion

    def seek(self, desplazamiento, origen=0):
        base = {0: 0, 1: self.posicion, 2: self.largo}[origen]
        self.posicion = base + desplazamiento
        return self.posicion

    def read(self, n=-1):
        fin = self.largo if n is None or n < 0 else min(self.largo, self.posicion + n)
        if fin <= self.posicion:
            return b""
        if self.posicion >= self.inicio_cola:
            datos = self.cola[self.posicion - self.inicio_cola: fin - self.inicio_cola]
        else:
            datos = self.rango(self.posicion, fin - 1)
        self.posicion = fin
        return datos


class ZipRemoto:
    """Lee archivos sueltos de un ZIP publicado en la web sin bajarlo entero.

    Un ZIP guarda al final un índice con la posición de cada archivo. Con ese índice
    y pedidos HTTP por rango de bytes se trae solo lo que hace falta. Así se saca un
    subconjunto definido de un archivo de 1,4 GB sin bajar los 1,4 GB.
    """

    def __init__(self, url: str):
        self.url = url
        self.resolver()
        self.indice = {Path(i.filename).name: i
                       for i in zipfile.ZipFile(_ArchivoHTTP(self.directa)).infolist() if not i.is_dir()}

    def resolver(self):
        """Dataverse responde con una redirección a una dirección firmada que vence: se pide de nuevo."""
        respuesta = requests.get(self.url, headers=AGENTE, allow_redirects=False, timeout=120)
        self.directa = respuesta.headers.get("Location", self.url)

    def leer(self, nombre: str) -> bytes:
        info = self.indice[nombre]
        hasta = info.header_offset + 30 + len(info.filename.encode()) + 1024 + info.compress_size
        ultimo = None
        for intento in range(4):
            try:
                respuesta = requests.get(self.directa, timeout=120,
                                         headers={**AGENTE, "Range": f"bytes={info.header_offset}-{hasta}"})
                respuesta.raise_for_status()
                bloque = respuesta.content
                break
            except Exception as error:
                ultimo = error
                time.sleep(2 * (intento + 1))
                self.resolver()
        else:
            raise RuntimeError(f"No pude leer {nombre} del ZIP remoto") from ultimo
        firma, largo_nombre, largo_extra = bloque[:4], *struct.unpack("<HH", bloque[26:30])
        if firma != b"PK\x03\x04":
            raise ValueError(f"El ZIP remoto cambió: no encuentro {nombre} donde decía el índice")
        inicio = 30 + largo_nombre + largo_extra
        crudo = bloque[inicio: inicio + info.compress_size]
        datos = crudo if info.compress_type == 0 else zlib.decompressobj(-15).decompress(crudo)
        if zlib.crc32(datos) != info.CRC:
            raise ValueError(f"{nombre}: la suma de verificación CRC32 no coincide")
        return datos


def en_paralelo(funcion, elementos, hilos: int = 12, aviso: str = "", cada: int = 250) -> list:
    """Aplica `funcion` a cada elemento con varios hilos y va contando el avance."""
    elementos = list(elementos)
    resultados = []
    inicio = time.time()
    with ThreadPoolExecutor(max_workers=hilos) as grupo:
        for i, resultado in enumerate(grupo.map(funcion, elementos), 1):
            resultados.append(resultado)
            if aviso and (i % cada == 0 or i == len(elementos)):
                print(f"   {aviso}: {i} de {len(elementos)} ({time.time() - inicio:.0f} s)", flush=True)
    return resultados


def imagenes(nombre: str, archivos=None) -> Path:
    """Carpeta local con las fotos del dato `nombre`.

    Si se pasa la lista de `archivos` esperados (la columna `archivo` de la tabla),
    comprueba que estén todos. Si falta alguno, intenta armar la carpeta desde
    `respaldo/<nombre>_imagenes.zip`.
    """
    carpeta = CACHE / nombre
    esperados = None if archivos is None else [str(a) for a in archivos]

    def completa():
        if not carpeta.is_dir():
            return False
        if esperados is None:
            return any(carpeta.iterdir())
        return all((carpeta / a).exists() for a in esperados)

    if not completa():
        copia = RESPALDO / f"{nombre}_imagenes.zip"
        if copia.exists():
            print(f"[{nombre}] Faltan fotos en la caché. Las saco de la copia de respaldo.")
            carpeta.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(copia) as archivo:
                archivo.extractall(carpeta)
        if not completa():
            raise RuntimeError(
                f"No encuentro las fotos de «{nombre}» en {carpeta} y no hay copia completa en respaldo/. "
                f"Corre descargar(\"{nombre}\", forzar=True) con buena conexión o avísale al profesor."
            )
    return carpeta


def empacar_respaldo(nombre: str) -> Path:
    """Para el profesor: guarda en respaldo/ un ZIP con las fotos reducidas de la caché."""
    carpeta = CACHE / nombre
    destino = RESPALDO / f"{nombre}_imagenes.zip"
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_STORED) as archivo:
        for foto in sorted(f for f in carpeta.rglob("*") if f.suffix.lower() in (".jpg", ".png")):
            archivo.write(foto, foto.relative_to(carpeta).as_posix())
    return destino


def rasgos_guardados(nombre: str, filas: int):
    """Devuelve las características ya calculadas (matriz de numpy) o None si no existen.

    Busca primero en la caché de la sesión y después en respaldo/. Solo las acepta si
    tienen tantas filas como fotos hay en la tabla.
    """
    import numpy as np

    for carpeta, origen in [(CACHE, "esta sesión"), (RESPALDO, "la copia de respaldo")]:
        archivo = carpeta / f"{nombre}.npy"
        if archivo.exists():
            matriz = np.load(archivo)
            if len(matriz) == filas:
                print(f"[{nombre}] Características ya calculadas en {origen}: {matriz.shape}.")
                return matriz
    return None


def guardar_rasgos(nombre: str, matriz) -> None:
    import numpy as np

    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / f"{nombre}.npy", matriz)
