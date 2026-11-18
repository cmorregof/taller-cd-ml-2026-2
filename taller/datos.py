"""Una sola puerta de entrada a los datos del curso.

Uso en cualquier cuaderno:

    from taller import descargar
    tabla = descargar("sipsa")

La función busca el dato en tres lugares, en este orden:

1. La caché local (`datos/cache/`), si ya se descargó en esta sesión.
2. La fuente oficial, que es el camino normal.
3. La carpeta `respaldo/`, donde el profesor deja una copia para el salón
   cuando la fuente oficial se cae.

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
    },
    "xm_precio": {
        "titulo": "XM: precio de bolsa nacional, promedio diario",
        "fuente": "XM, SiMEM, conjunto EC6945",
        "url": "https://www.simem.co",
        "licencia": "Datos públicos de XM. Citar a XM como fuente.",
        "constructor": xm.construir_precio,
        "md5": None,  # la fuente se actualiza a diario: el MD5 se registra, no se compara
    },
    "xm_embalses": {
        "titulo": "XM: nivel agregado de los embalses, diario",
        "fuente": "XM, SiMEM, conjunto c51127",
        "url": "https://www.simem.co",
        "licencia": "Datos públicos de XM. Citar a XM como fuente.",
        "constructor": xm.construir_embalses,
        "md5": None,
    },
}


def _md5(tabla: pd.DataFrame) -> str:
    return hashlib.md5(tabla.to_csv(index=False).encode("utf-8")).hexdigest()


def _registrar(nombre: str, tabla: pd.DataFrame, origen: str) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    archivo = CACHE / "procedencia.json"
    registro = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else {}
    registro[nombre] = {
        "titulo": CATALOGO[nombre]["titulo"],
        "fuente": CATALOGO[nombre]["fuente"],
        "origen": origen,
        "descargado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "filas": int(len(tabla)),
        "md5": _md5(tabla),
    }
    archivo.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")


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
        if not en_respaldo.exists():
            raise RuntimeError(
                f"No pude descargar «{nombre}» de {ficha['fuente']} y no hay copia en la "
                f"carpeta respaldo/. Revisa la conexión y vuelve a correr la celda. "
                f"Si sigue fallando, avísale al profesor. Detalle técnico: {error!r}"
            ) from error
        print(f"[{nombre}] La fuente oficial no respondió ({type(error).__name__}). Uso la copia de respaldo.")
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
    _registrar(nombre, tabla, origen)
    print(f"[{nombre}] Listo: {len(tabla):,} filas.".replace(",", "."))
    return pd.read_csv(en_cache)
