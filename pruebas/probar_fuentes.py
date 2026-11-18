"""Revisa que las fuentes oficiales respondan y entreguen lo que el curso espera.

Hace consultas pequeñas, no la descarga completa. Corre solo cada mañana
(ver .github/workflows/fuentes.yml) y también a mano:

    python pruebas/probar_fuentes.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

from taller import sipsa, xm

fallas = []


def prueba(nombre, funcion):
    try:
        print(f"OK     {nombre}: {funcion()}")
    except Exception as error:
        fallas.append(nombre)
        print(f"FALLA  {nombre}: {error!r}")


def dane():
    solicitud = urllib.request.Request(sipsa.URL.format(sipsa.FUENTES["2024"]),
                                       headers={"User-Agent": "taller-cd-ml-2026-2"})
    with urllib.request.urlopen(solicitud, timeout=120) as respuesta:
        filas = sipsa._leer_zip(respuesta.read())
    assert len(filas) > 50_000, f"solo {len(filas)} filas en 2024"
    assert any(f["ciudad"] == "Manizales" for f in filas), "no aparece Manizales"
    return f"{len(filas)} filas en el archivo de 2024, con Manizales"


def precio():
    tabla = xm._mes("EC6945", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))
    for columna in ["FechaHora", "CodigoVariable", "Version", "Valor"]:
        assert columna in tabla.columns, f"falta la columna {columna}"
    assert "PB_Nal" in set(tabla["CodigoVariable"]), "no aparece PB_Nal"
    return f"{len(tabla)} registros en dos días, versiones {sorted(tabla['Version'].unique())}"


def embalses():
    tabla = xm._mes("c51127", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))
    for columna in ["Fecha", "CodigoEmbalse", "VolumenUtilDiarioEnergia", "CapacidadUtilEnergia"]:
        assert columna in tabla.columns, f"falta la columna {columna}"
    return f"{tabla['CodigoEmbalse'].nunique()} embalses"


prueba("DANE, SIPSA 2024", dane)
prueba("XM, precio de bolsa (EC6945)", precio)
prueba("XM, embalses (c51127)", embalses)

if fallas:
    print(f"\n{len(fallas)} fuente(s) con problemas: {', '.join(fallas)}")
    sys.exit(1)
print("\nTodas las fuentes responden.")
