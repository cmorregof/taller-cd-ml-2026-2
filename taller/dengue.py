"""Dengue en Bucaramanga como serie de tiempo: casos por semana de inicio de síntomas.

    from taller.dengue import serie_semanal
    serie = serie_semanal()

Sale de las mismas fichas de SIVIGILA del cuaderno de la clase 2 (datos.gov.co, qzc7-jbg3).
"""
from __future__ import annotations

import pandas as pd

from .datos import descargar

# Última semana de inicio de síntomas que se usa. Las siguientes están incompletas,
# porque una ficha tarda en notificarse (mediana de 4 días, y el 1 % tarda más de un mes).
CORTE = "2025-08-10"


def serie_semanal(ciudad: str = "bucaramanga", corte: str | None = CORTE, avisar: bool = False) -> pd.Series:
    """Casos de dengue por semana epidemiológica (de domingo a sábado) de inicio de síntomas.

    Salen las fichas de mortalidad (cod_eve 580), que registran otro evento, y las descartadas
    (ajuste_ 6). La fecha es la de inicio de síntomas, que coincide con la semana epidemiológica
    de la ficha; la de notificación trae día y mes invertidos en parte de las fichas de 2015 a 2022.
    """
    fichas = descargar("dengue", ciudad=ciudad)

    def fecha(columna):
        valores = fichas[columna].astype(str)
        iso = pd.to_datetime(valores.str[:10], format="%Y-%m-%d", errors="coerce")
        con_barras = pd.to_datetime(valores, format="%d/%m/%Y", errors="coerce")
        resultado = iso.fillna(con_barras)
        return resultado.where(resultado.dt.year > 1900)

    sintomas = fecha("ini_sin_")
    evento = pd.to_numeric(fichas["cod_eve"], errors="coerce")
    ajuste = pd.to_numeric(fichas["ajuste_"], errors="coerce")
    validas = (evento != 580) & (ajuste != 6) & sintomas.notna()
    semana = sintomas[validas].dt.to_period("W-SAT").dt.start_time
    conteo = semana.value_counts().sort_index()
    todas = pd.date_range(conteo.index.min(), conteo.index.max(), freq="7D")
    serie = conteo.reindex(todas, fill_value=0).astype(int).rename("casos")
    if avisar:
        print(f"Fichas: {len(fichas)} | de mortalidad: {int((evento == 580).sum())} | descartadas: {int((ajuste == 6).sum())} | quedan: {int(validas.sum())}")
    return serie[:corte] if corte else serie
