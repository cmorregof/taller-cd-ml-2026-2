"""El índice del sancocho: una canasta fija con precios de plaza, medida como mide el DANE.

Aquí vive la plomería del cuaderno «El IPC del sancocho»: escoger cada mes la variedad
más barata, encadenar faltantes y armar el índice de cada ciudad. El cuaderno muestra las
decisiones; estas funciones las ejecutan.

    from taller.canasta import RECETA, precios_ciudad, indice, ipc_total, ipc_alimentos, indice_colombia
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
MESES = pd.date_range("2013-01-01", "2024-12-01", freq="MS")
BASE = pd.Timestamp("2018-12-01")          # la misma base del IPC del DANE

# Un almuerzo para seis: sancocho, arroz, banano y ensalada. Kilos por olla.
RECETA = {
    "espinazo": 1.5, "yuca": 1.0, "papa": 1.0, "mazorca": 0.9, "plátano verde": 1.4,
    "arroz": 0.75, "banano": 1.0, "zanahoria": 0.5, "tomate": 0.5, "lechuga": 0.5,
}

# Qué productos del SIPSA cuentan como cada ingrediente. El espinazo no tiene sustituto.
VARIEDADES = {
    "espinazo": r"^Carne de cerdo, espinazo$",
    "yuca": r"^Yuca ",
    "papa": r"^Papa ",
    "mazorca": r"^Chócolo mazorca$",
    "plátano verde": r"^Plátano.*verde",
    "arroz": r"^Arroz (?:de primera|excelso|de segunda|blanco)",
    "banano": r"^Banano",
    "zanahoria": r"^Zanahoria",
    "tomate": r"^Tomate (?:chonto|Riogrande|larga vida|riñón)",
    "lechuga": r"^Lechuga",
}


def _por_ciudad(sipsa: pd.DataFrame) -> pd.Series:
    tabla = sipsa[sipsa["precio_kg"] > 0].assign(fecha=lambda t: pd.to_datetime(t["fecha"]))
    return tabla.groupby(["ciudad", "producto", "fecha"])["precio_kg"].median()


def precios_ciudad(sipsa: pd.DataFrame, ciudad: str, criterio: str = "barata") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precio por kilo de cada ingrediente, mes a mes, en una ciudad.

    criterio="barata": cada mes se toma la variedad más barata que haya en la plaza.
    Devuelve los precios y, aparte, qué variedad se usó cada mes.
    """
    serie = _por_ciudad(sipsa).loc[ciudad]
    productos = serie.index.get_level_values(0).unique()
    precios, cual = {}, {}
    for ingrediente, patron in VARIEDADES.items():
        opciones = [p for p in productos if re.search(patron, p)]
        if not opciones:
            precios[ingrediente] = pd.Series(np.nan, index=MESES)
            cual[ingrediente] = pd.Series(None, index=MESES, dtype=object)
            continue
        tabla = pd.DataFrame({p: serie.loc[p].reindex(MESES) for p in opciones})
        precios[ingrediente] = tabla.min(axis=1)
        cual[ingrediente] = tabla.dropna(how="all").idxmin(axis=1).reindex(MESES)
    return pd.DataFrame(precios), pd.DataFrame(cual)


def indice(precios: pd.DataFrame, receta: dict = RECETA, base=BASE) -> pd.Series:
    """Índice de Laspeyres: costo de la misma canasta, con base = 100."""
    costo = (precios[list(receta)] * pd.Series(receta)).sum(axis=1, min_count=len(receta))
    return 100 * costo / costo.loc[base]


def ipc_total() -> pd.Series:
    tabla = pd.read_csv(RAIZ / "datos" / "ipc" / "ipc_total_indice.csv")
    return pd.Series(tabla["indice"].values, index=pd.to_datetime(tabla["fecha"]), name="IPC total").reindex(MESES)


def ipc_alimentos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """IPC de alimentos por ciudad, encadenado desde las variaciones mensuales del DANE.

    Los meses sin dato de una ciudad (junio a diciembre de 2016) se llenan con la
    variación nacional del mismo mes. Devuelve el índice y una tabla que marca qué se imputó.
    """
    tabla = pd.read_csv(RAIZ / "datos" / "ipc" / "ipc_alimentos_variacion_mensual.csv")
    tabla["fecha"] = pd.to_datetime(tabla["fecha"])
    variacion = tabla.pivot(index="fecha", columns="ambito", values="variacion_mensual").reindex(MESES)
    imputado = variacion.isna()
    variacion = variacion.apply(lambda c: c.fillna(variacion["Nacional"]))
    nivel = (1 + variacion / 100).cumprod()
    return 100 * nivel / nivel.loc[BASE], imputado


def _cambio_mediano(series: dict) -> pd.DataFrame:
    """Cambio mensual mediano de cada ingrediente en todo el país (en logaritmo)."""
    return pd.DataFrame({ing: pd.DataFrame({c: np.log(s[ing]) for c, s in series.items()}).diff().median(axis=1)
                         for ing in RECETA})


def _completar(serie: pd.Series, cambio_pais: pd.Series) -> pd.Series:
    """Llena huecos de una serie de precios: interpola por dentro y encadena con el país en los bordes."""
    if serie.notna().sum() < 2:
        return pd.Series(np.nan, index=MESES)
    log = np.log(serie).interpolate(limit_area="inside")
    primero, ultimo = MESES.get_loc(log.first_valid_index()), MESES.get_loc(log.last_valid_index())
    for i in range(primero - 1, -1, -1):
        log.iloc[i] = log.iloc[i + 1] - cambio_pais.iloc[i + 1]
    for i in range(ultimo + 1, len(MESES)):
        log.iloc[i] = log.iloc[i - 1] + cambio_pais.iloc[i]
    return np.exp(log)


def indice_colombia(sipsa: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Índice del sancocho para cada plaza del SIPSA, con la variedad más barata de cada mes.

    - Los huecos internos se interpolan; los bordes se encadenan con el cambio mediano del país.
    - Un ingrediente que la plaza nunca reporta toma el cambio del país.
    - Todas las plazas usan los mismos pesos: cantidades de la receta por el precio
      mediano nacional en la base. Así se comparan cambios y no niveles.

    Devuelve el índice de cada plaza, la fracción de datos imputados de cada una y la
    cobertura (fracción de meses con dato) por plaza e ingrediente.
    """
    ciudades = sorted(sipsa["ciudad"].unique())
    series = {c: precios_ciudad(sipsa, c)[0] for c in ciudades}
    cobertura = pd.DataFrame({c: s.notna().mean() for c, s in series.items()}).T
    cambio = _cambio_mediano(series)
    relativo_pais = np.exp(cambio.fillna(0).cumsum())
    relativo_pais = relativo_pais / relativo_pais.loc[BASE]
    precio_base = pd.Series({ing: np.nanmedian([s.loc[BASE, ing] for s in series.values()]) for ing in RECETA})
    pesos = pd.Series(RECETA) * precio_base
    pesos = pesos / pesos.sum()
    indices, imputado = {}, {}
    for c, s in series.items():
        relativos = {}
        for ing in RECETA:
            completa = _completar(s[ing], cambio[ing])
            relativos[ing] = completa / completa.loc[BASE] if completa.notna().all() else relativo_pais[ing]
        indices[c] = 100 * (pd.DataFrame(relativos) * pesos).sum(axis=1)
        imputado[c] = 1 - cobertura.loc[c].mean()
    return pd.DataFrame(indices), pd.Series(imputado, name="imputado"), cobertura
