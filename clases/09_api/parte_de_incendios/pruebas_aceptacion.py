"""Pruebas de aceptación de «El parte de incendios».

    python pruebas_aceptacion.py

Cada prueba corresponde a un criterio de README.md. Las cifras esperadas se calculan aquí,
directamente sobre los datos, sin usar el código de ustedes. Si una prueba falla, lean el
mensaje: dice qué se pidió, qué se esperaba y qué llegó.
"""
import sys
import unicodedata

import pandas as pd
from fastapi.testclient import TestClient

from datos import cargar_focos, departamentos
from main import app

cliente = TestClient(app)
focos = cargar_focos()
vegetacion = focos[(focos["type"] == 0) & focos["departamento"].notna()]
resultados = []


def criterio(numero, descripcion):
    def decorar(funcion):
        try:
            funcion()
            resultados.append((numero, descripcion, True, ""))
        except AssertionError as error:
            resultados.append((numero, descripcion, False, str(error)))
        except Exception as error:
            resultados.append((numero, descripcion, False, f"{type(error).__name__}: {error}"))
        return funcion
    return decorar


def pedir(ruta, esperado=200, **parametros):
    respuesta = cliente.get(ruta, params=parametros)
    assert respuesta.status_code == esperado, f"GET {ruta} {parametros or ''}: esperaba {esperado} y llegó {respuesta.status_code}"
    return respuesta.json()


def cuenta(departamento, desde, hasta, confianza=0):
    tramo = vegetacion[(vegetacion["departamento"] == departamento) & vegetacion["fecha"].between(desde, hasta)
                       & (vegetacion["confidence"] >= confianza)]
    return tramo


@criterio(1, "GET /salud dice que el servicio está vivo y qué datos tiene")
def _():
    cuerpo = pedir("/salud")
    assert cuerpo.get("estado") == "ok", f"esperaba estado 'ok' y llegó {cuerpo.get('estado')!r}"
    assert cuerpo.get("focos") == len(vegetacion), f"esperaba focos = {len(vegetacion)} (tipo 0, con departamento) y llegó {cuerpo.get('focos')}"
    assert cuerpo.get("desde") == str(vegetacion["fecha"].min().date()), f"desde: llegó {cuerpo.get('desde')!r}"
    assert cuerpo.get("hasta") == str(vegetacion["fecha"].max().date()), f"hasta: llegó {cuerpo.get('hasta')!r}"


@criterio(2, "GET /departamentos entrega los 33 nombres en orden alfabético")
def _():
    cuerpo = pedir("/departamentos")
    assert cuerpo == departamentos(), f"la lista no coincide; llegaron {len(cuerpo)} nombres y el primero es {cuerpo[:1]}"


@criterio(3, "GET /parte/{departamento} cuenta bien los focos de un periodo")
def _():
    for departamento, desde, hasta in [("Vichada", "2024-02-01", "2024-02-29"), ("Caldas", "2016-01-01", "2025-12-31")]:
        cuerpo = pedir(f"/parte/{departamento}", desde=desde, hasta=hasta)
        tramo = cuenta(departamento, desde, hasta)
        assert cuerpo.get("departamento") == departamento, f"departamento: llegó {cuerpo.get('departamento')!r}"
        assert cuerpo.get("focos") == len(tramo), f"{departamento} {desde} a {hasta}: esperaba {len(tramo)} focos y llegaron {cuerpo.get('focos')}"
        assert abs(cuerpo.get("frp_total", -1) - tramo["frp"].sum()) < 1, f"frp_total: esperaba {tramo['frp'].sum():.1f} y llegó {cuerpo.get('frp_total')}"
        por_dia = {d["fecha"]: d["focos"] for d in cuerpo.get("por_dia", [])}
        esperado = {str(f.date()): int(n) for f, n in tramo.groupby("fecha").size().items()}
        assert por_dia == esperado, f"por_dia: esperaba {len(esperado)} días con focos y llegaron {len(por_dia)}, o las cuentas no coinciden"


@criterio(4, "El filtro confianza_minima deja solo los focos con esa confianza o más")
def _():
    cuerpo = pedir("/parte/Meta", desde="2023-01-01", hasta="2023-03-31", confianza_minima=80)
    esperado = len(cuenta("Meta", "2023-01-01", "2023-03-31", 80))
    assert cuerpo.get("focos") == esperado, f"esperaba {esperado} focos con confianza de 80 o más y llegaron {cuerpo.get('focos')}"


@criterio(5, "El nombre del departamento se entiende sin tildes y sin importar mayúsculas")
def _():
    referencia = pedir("/parte/Bolívar", desde="2024-01-01", hasta="2024-12-31")["focos"]
    for variante in ["bolivar", "BOLÍVAR", "Bolivar"]:
        cuerpo = pedir(f"/parte/{variante}", desde="2024-01-01", hasta="2024-12-31")
        assert cuerpo.get("focos") == referencia, f"«{variante}» dio {cuerpo.get('focos')} focos y «Bolívar» dio {referencia}"
        assert cuerpo.get("departamento") == "Bolívar", f"«{variante}»: el nombre devuelto debe ser el oficial, llegó {cuerpo.get('departamento')!r}"


@criterio(6, "Un departamento que no existe responde 404 con un mensaje que ayuda")
def _():
    cuerpo = pedir("/parte/Narnia", esperado=404, desde="2024-01-01", hasta="2024-12-31")
    assert "detail" in cuerpo and len(str(cuerpo["detail"])) > 15, "el 404 debe traer un mensaje en 'detail' que explique qué hacer"


@criterio(7, "Fechas mal escritas o al revés responden 422")
def _():
    pedir("/parte/Caldas", esperado=422, desde="ayer", hasta="2024-12-31")
    pedir("/parte/Caldas", esperado=422, desde="2024-12-31", hasta="2024-01-01")


@criterio(8, "GET /ranking ordena los departamentos por focos en un periodo")
def _():
    cuerpo = pedir("/ranking", desde="2024-01-01", hasta="2024-03-31", n=5)
    tramo = vegetacion[vegetacion["fecha"].between("2024-01-01", "2024-03-31")]
    esperado = tramo["departamento"].value_counts().head(5)
    assert [d["departamento"] for d in cuerpo] == list(esperado.index), f"esperaba {list(esperado.index)} y llegó {[d.get('departamento') for d in cuerpo]}"
    assert [d["focos"] for d in cuerpo] == [int(v) for v in esperado.values], "las cuentas del ranking no coinciden"
    pedir("/ranking", esperado=422, desde="2024-01-01", hasta="2024-03-31", n=0)
    pedir("/ranking", esperado=422, desde="2024-01-01", hasta="2024-03-31", n=99)


@criterio(9, "GET /brigadas propone a dónde mandar las brigadas la semana que empieza en una fecha")
def _():
    cuerpo = pedir("/brigadas", fecha="2025-02-03", n=5)
    assert isinstance(cuerpo.get("metodo"), str) and len(cuerpo["metodo"]) > 20, "falta 'metodo': una frase que diga cómo se calcula lo esperado"
    assert cuerpo.get("semana_desde") == "2025-02-03" and cuerpo.get("semana_hasta") == "2025-02-09", f"la semana debe ir de la fecha pedida a seis días después; llegó {cuerpo.get('semana_desde')} a {cuerpo.get('semana_hasta')}"
    lista = cuerpo.get("departamentos", [])
    assert len(lista) == 5, f"esperaba 5 departamentos y llegaron {len(lista)}"
    esperados = [d["esperado"] for d in lista]
    assert all(e >= 0 for e in esperados) and esperados == sorted(esperados, reverse=True), "'esperado' debe ser no negativo y venir de mayor a menor"
    assert set(d["departamento"] for d in lista) <= set(departamentos()), "hay nombres que no son departamentos"


@criterio(10, "/brigadas no usa datos de la semana que pronostica ni posteriores")
def _():
    # Para una fecha anterior a todos los datos no hay pasado: la respuesta honesta es 422, no un pronóstico.
    pedir("/brigadas", esperado=422, fecha="2015-06-01", n=5)
    # Y la propuesta para una semana pasada no puede coincidir siempre con lo que de verdad ocurrió.
    aciertos = 0
    for fecha in ["2024-02-05", "2024-08-05", "2025-01-06", "2025-03-03", "2025-09-01"]:
        cuerpo = pedir("/brigadas", fecha=fecha, n=5)
        fin = str((pd.Timestamp(fecha) + pd.Timedelta(days=6)).date())
        real = list(vegetacion[vegetacion["fecha"].between(fecha, fin)]["departamento"].value_counts().head(5).index)
        aciertos += [d["departamento"] for d in cuerpo["departamentos"]] == real
    assert aciertos < 5, "la propuesta coincide siempre, y en el mismo orden, con lo que ocurrió: revisen si están usando datos del futuro"


if __name__ == "__main__":
    for numero, descripcion, paso, detalle in resultados:
        print(f"{'PASA ' if paso else 'FALLA'} {numero:>2}. {descripcion}" + ("" if paso else f"\n          {detalle}"))
    pasaron = sum(r[2] for r in resultados)
    print(f"\n{pasaron} de {len(resultados)} criterios cumplidos.")
    sys.exit(0 if pasaron == len(resultados) else 1)
