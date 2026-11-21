"""Una sola puerta de entrada a los datos del curso.

Uso en cualquier cuaderno:

    from taller import descargar
    tabla = descargar("sipsa")

La función busca el dato en tres lugares, en este orden:

1. La caché local (`datos/cache/`), si ya se descargó en esta sesión.
2. La fuente oficial, que es el camino normal.
3. La carpeta `respaldo/`, donde el profesor deja una copia para el salón
   cuando la fuente oficial se cae.
4. Una copia en línea del profesor (Google Drive), si la ficha del dato la
   tiene. Sirve en Colab cuando la fuente oficial no atiende a los
   servidores de Google, como pasa con el DANE.

Cada descarga deja un registro de procedencia (fecha, filas y MD5) en
`datos/cache/procedencia.json`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import sipsa, xm

RAIZ = Path(__file__).resolve().parent.parent
CACHE = RAIZ / "datos" / "cache"
RESPALDO = RAIZ / "respaldo"

CATALOGO = {
    "sipsa": {
        "titulo": "SIPSA: precios mayoristas mensuales de alimentos, 2013 a 2024",
        "fuente": "DANE, Microdatos, catálogo 776",
        "url": "https://microdatos.dane.gov.co/index.php/catalog/776",
        "licencia": "Uso con cita al DANE. El DANE no permite republicar el archivo: por eso se descarga y no se guarda en el repositorio.",
        "constructor": sipsa.construir,
        # MD5 del CSV normalizado (sin comprimir). La fuente es histórica y no cambia.
        "md5": "4db8e0a7d1befea2502a8ce129d76ca4",
        # El DANE no atiende a los servidores de Google (error 401 en Colab): ahí se usa esta copia.
        "respaldo_drive": "1GYEJEveDwYinH2PxgpqMPd_Cb3qzFpgh",
    },
    "xm_precio": {
        "titulo": "XM: precio de bolsa nacional, promedio diario",
        "fuente": "XM, SiMEM, conjunto EC6945",
        "url": "https://www.simem.co",
        "licencia": "Datos públicos de XM. Citar a XM como fuente.",
        "constructor": xm.construir_precio,
        "md5": None,  # la fuente se actualiza a diario: el MD5 se registra, no se compara
        "respaldo_drive": "1ZWRCo4rWfNkWhXXzLahxHaMSnDj3a90-",
    },
    "xm_embalses": {
        "titulo": "XM: nivel agregado de los embalses, diario",
        "fuente": "XM, SiMEM, conjunto c51127",
        "url": "https://www.simem.co",
        "licencia": "Datos públicos de XM. Citar a XM como fuente.",
        "constructor": xm.construir_embalses,
        "md5": None,
        "respaldo_drive": "1WcsxPvZAob12vKV0OmWHBgUbDAOhVmCT",
    },
}


# Cada clase registra sus datos en un módulo propio, taller/fuentes_claseNN.py,
# que define un diccionario FICHAS con el mismo formato de CATALOGO.
import importlib
import pkgutil

for _modulo in pkgutil.iter_modules([str(Path(__file__).resolve().parent)]):
    if _modulo.name.startswith("fuentes_clase"):
        CATALOGO.update(importlib.import_module(f"{__package__}.{_modulo.name}").FICHAS)


def _md5(tabla: pd.DataFrame) -> str:
    return hashlib.md5(tabla.to_csv(index=False).encode("utf-8")).hexdigest()


def _registrar(nombre: str, tabla: pd.DataFrame, origen: str, clave: str | None = None) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    archivo = CACHE / "procedencia.json"
    registro = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}
    registro[clave or nombre] = {
        "titulo": CATALOGO[nombre]["titulo"],
        "fuente": CATALOGO[nombre]["fuente"],
        "origen": origen,
        "descargado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "filas": int(len(tabla)),
        "md5": _md5(tabla),
    }
    if clave and clave != nombre:      # el nombre a secas apunta siempre a la última variante descargada
        registro[nombre] = dict(registro[clave], variante=clave)
    archivo.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")


def _bajar_respaldo(identificador: str, destino: Path) -> bool:
    """Baja la copia del profesor desde Google Drive. Devuelve True si quedó guardada."""
    import requests

    url = f"https://drive.google.com/uc?export=download&id={identificador}"
    try:
        respuesta = requests.get(url, timeout=120)
        respuesta.raise_for_status()
        if respuesta.content[:2] != b"\x1f\x8b":   # un .gz empieza así; si no, Drive devolvió otra cosa
            return False
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_bytes(respuesta.content)
        return True
    except Exception:
        return False


def procedencia() -> dict:
    """Devuelve el registro de lo descargado en esta sesión."""
    archivo = CACHE / "procedencia.json"
    return json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}


def descargar(nombre: str, forzar: bool = False, **opciones) -> pd.DataFrame:
    """Entrega la tabla `nombre` como DataFrame de pandas.

    `forzar=True` ignora la caché y vuelve a la fuente oficial.
    """
    if nombre not in CATALOGO:
        raise KeyError(
            f"No conozco el dato «{nombre}». Los disponibles son: {', '.join(CATALOGO)}."
        )
    ficha = CATALOGO[nombre]
    sufijo = "_".join(f"{k}-{v}" for k, v in sorted(opciones.items()))
    archivo = f"{nombre}{'_' + sufijo if sufijo else ''}.csv.gz"
    en_cache, en_respaldo = CACHE / archivo, RESPALDO / archivo

    if en_cache.exists() and not forzar:
        print(f"[{nombre}] Ya estaba descargado. Leo la copia de esta sesión.")
        return pd.read_csv(en_cache)

    try:
        print(f"[{nombre}] Descargo de la fuente oficial: {ficha['fuente']}.")
        tabla = ficha["constructor"](**opciones)
        origen = "fuente oficial"
    except Exception as error:  # la red falla de muchas maneras: se informa y se sigue
        # La copia en Drive es un identificador si el dato no tiene variantes, o un
        # diccionario {nombre del archivo sin extensión: identificador} si las tiene.
        copia = ficha.get("respaldo_drive")
        if isinstance(copia, dict):
            copia = copia.get(archivo.removesuffix(".csv.gz"))
        elif opciones:
            copia = None
        if not en_respaldo.exists() and copia:
            print(f"[{nombre}] Bajo la copia del profesor desde Google Drive.")
            _bajar_respaldo(copia, en_respaldo)
        if not en_respaldo.exists():
            raise RuntimeError(
                f"No pude descargar «{nombre}» de {ficha['fuente']} y no hay copia en la "
                f"carpeta respaldo/. Revisa la conexión y vuelve a correr la celda. "
                f"Si sigue fallando, avísale al profesor. Detalle técnico: {error!r}"
            ) from error
        print(f"[{nombre}] Uso la copia de respaldo. La fuente oficial no respondió ({type(error).__name__}).")
        tabla = pd.read_csv(en_respaldo)
        origen = "copia de respaldo"

    if ficha["md5"] and origen == "fuente oficial":
        if _md5(tabla) == ficha["md5"]:
            print(f"[{nombre}] MD5 verificado: el dato es idéntico al que se usó para preparar el curso.")
        else:
            print(
                f"[{nombre}] Aviso: el MD5 no coincide con el esperado. La fuente cambió "
                f"desde que se preparó el curso. Puedes seguir, pero cuéntale al profesor."
            )

    CACHE.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(en_cache, index=False)
    _registrar(nombre, tabla, origen, clave=archivo.removesuffix(".csv.gz"))
    print(f"[{nombre}] Listo: {len(tabla):,} filas.".replace(",", "."))
    return pd.read_csv(en_cache)
