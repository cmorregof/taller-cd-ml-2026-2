"""Revisa las fuentes de la clase 2 con consultas pequeñas.

    python pruebas/fuentes_clase02.py

`probar()` devuelve una lista de mensajes y lanza AssertionError, con un mensaje
claro, si alguna fuente cambió de forma.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from taller import fuentes_clase02 as f


def _saber11():
    tabla = f._pagina(f.SABER11, {"$select": ",".join(f.COLUMNAS_SABER11),
                                  "$where": "cole_depto_ubicacion='CALDAS'", "$limit": 50})
    assert len(tabla) == 50, f"Saber 11: esperaba 50 filas de Caldas y llegaron {len(tabla)}"
    faltan = set(f.COLUMNAS_SABER11) - set(tabla.columns)
    assert not faltan, f"Saber 11: ya no vienen las columnas {sorted(faltan)}"
    cuenta = f._pagina(f.SABER11, {"$select": "count(*) as n", "$where": "cole_depto_ubicacion='CALDAS'"})
    n = int(cuenta["n"].iloc[0])
    assert n == 9942, f"Saber 11: Caldas tenía 9.942 estudiantes y ahora tiene {n}"
    return f"Saber 11 (rnvb-vnyh): {len(f.COLUMNAS_SABER11)} columnas presentes, Caldas con {n} estudiantes"


def _dengue(ciudad, minimo, necesarias):
    conjunto = f.DENGUE[ciudad]
    presentes = set(f.columnas(conjunto))
    faltan = set(necesarias) - presentes
    assert not faltan, f"Dengue {ciudad}: ya no vienen las columnas {sorted(faltan)}"
    n = int(f._pagina(conjunto, {"$select": "count(*) as n"})["n"].iloc[0])
    assert n >= minimo, f"Dengue {ciudad}: había {minimo} fichas y ahora hay {n}"
    return f"Dengue {ciudad} ({conjunto}): {len(presentes)} columnas, {n} fichas"


COMUNES = ["cod_eve", "fec_not", "sexo_", "tip_ss_", "estrato_", "fec_con_", "ini_sin_", "pac_hos_", "ajuste_",
           "fiebre", "cefalea", "dolrretroo", "malgias", "artralgia", "erupcionr", "dolor_abdo", "vomito",
           "diarrea", "somnolenci", "hipotensio", "hepatomeg", "hem_mucosa", "hipotermia", "aum_hemato",
           "caida_plaq", "acum_liqui", "clasfinal", "conducta"]


def probar():
    return [
        _saber11(),
        _dengue("bucaramanga", 28_626, COMUNES + ["def_clas_edad"]),
        _dengue("tulua", 24_547, COMUNES + ["clas_edad"]),
    ]


if __name__ == "__main__":
    try:
        for mensaje in probar():
            print("OK    ", mensaje)
    except Exception as error:
        print("FALLA ", error)
        sys.exit(1)
