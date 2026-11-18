"""Revisa las fuentes de la clase 3 con consultas pequeñas.

    python pruebas/fuentes_clase03.py

`probar()` devuelve una lista de mensajes y lanza AssertionError, con un mensaje
claro, si alguna fuente cambió de forma.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from taller import fuentes_clase03 as f


def _londres():
    paquete, internos, remoto = f.abrir_londres()
    assert len(internos) == f.TOTAL_ARCHIVOS, f"Londres: el ZIP tenía 168 archivos y ahora tiene {len(internos)}"
    assert remoto.tamano == 795_722_689, f"Londres: el ZIP pesaba 795.722.689 bytes y ahora pesa {remoto.tamano}"
    with paquete.open(internos[0]) as flujo:
        cabeza = pd.read_csv(flujo, dtype=str, nrows=1000)
    esperadas = ["LCLid", "stdorToU", "DateTime", "KWH/hh (per half hour)"]
    assert [c.strip() for c in cabeza.columns] == esperadas, f"Londres: las columnas ahora son {list(cabeza.columns)}"
    return f"Londres: ZIP de {remoto.tamano / 1e6:.0f} MB con {len(internos)} archivos; se leyeron 1.000 filas con {remoto.bytes_bajados / 1e3:.0f} kB"


def _ideam():
    tabla = f._trimestre("0026155502", "2020-03-01", "2020-03-08")
    assert list(tabla.columns) == ["fechaobservacion", "valorobservado"], f"IDEAM: las columnas ahora son {list(tabla.columns)}"
    assert len(tabla) > 1500, f"IDEAM: Cenicafé tenía unas 2.000 lecturas en la primera semana de marzo de 2020 y ahora tiene {len(tabla)}"
    return f"IDEAM (s54a-sgyg): {len(tabla)} lecturas de Cenicafé en una semana de 2020"


def _ungrd():
    mensajes = []
    for periodo, conjunto in f.UNGRD.items():
        tabla = f._socrata(conjunto, {"$select": "fecha,departamento,municipio,evento",
                                      "$where": "upper(departamento) like '%CALDAS%'", "$limit": 5}, espera=120, intentos=3)
        assert len(tabla) == 5, f"UNGRD {conjunto}: no llegaron emergencias de Caldas"
        mensajes.append(f"UNGRD {periodo} ({conjunto}): responde con fecha, departamento, municipio y evento")
    return mensajes


def probar():
    return [_londres(), _ideam(), *_ungrd()]


if __name__ == "__main__":
    try:
        for mensaje in probar():
            print("OK    ", mensaje)
    except Exception as error:
        print("FALLA ", error)
        sys.exit(1)
