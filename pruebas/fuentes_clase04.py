"""Revisa las fuentes de la clase 4 con consultas pequeñas.

    python pruebas/fuentes_clase04.py

`probar()` devuelve una lista de mensajes y lanza AssertionError, con un mensaje
claro, si alguna fuente cambió de forma. No baja el Excel de Bogotá: solo
confirma que sigue publicado y que pesa lo mismo.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from taller import fuentes_clase04 as f


def _statsbomb():
    calendario = f.partidos()
    assert len(calendario) == 64, f"StatsBomb: el Mundial de 2022 tenía 64 partidos y ahora tiene {len(calendario)}"
    final = calendario.iloc[-1]
    remates = f.remates_de(int(final["partido"]))
    assert len(remates) > 20, f"StatsBomb: la final trae solo {len(remates)} remates"
    for campo in ["x", "y", "tipo", "parte_del_cuerpo", "gol", "statsbomb_xg", "rivales_en_el_cono"]:
        assert campo in remates[0], f"StatsBomb: ya no se puede calcular el campo {campo}"
    return f"StatsBomb: 64 partidos; la final ({final['local']} contra {final['visitante']}) trae {len(remates)} remates"


def _bogota():
    respuesta = requests.head(f.ANUARIO, timeout=60, allow_redirects=True)
    assert respuesta.status_code == 200, f"Bogotá: el anuario respondió {respuesta.status_code}"
    peso = int(respuesta.headers.get("Content-Length", 0))
    assert abs(peso - 126_822_000) < 3_000_000, f"Bogotá: el anuario pesaba unos 127 MB y ahora pesa {peso / 1e6:.0f} MB"
    return f"Bogotá: el anuario sigue publicado y pesa {peso / 1e6:.1f} MB"


def probar():
    return [_statsbomb(), _bogota()]


if __name__ == "__main__":
    try:
        for mensaje in probar():
            print("OK    ", mensaje)
    except Exception as error:
        print("FALLA ", error)
        sys.exit(1)
