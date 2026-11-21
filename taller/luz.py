"""Plomería del cuaderno «El precio de la luz»: arma las variables, entrena los candidatos
con el protocolo de tres tramos y deja todo listo para comparar y decidir.

    from taller.luz import competir, pronostico
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mape(real, pronostico) -> float:
    return float((abs(real - pronostico) / real).mean() * 100)


def modelos(vecinos: int = 5) -> dict:
    return {
        "lineal": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=vecinos)),
        "bosque": RandomForestRegressor(n_estimators=300, random_state=0),
        "boosting": GradientBoostingRegressor(random_state=0),
    }


def variables(datos: pd.DataFrame) -> pd.DataFrame:
    """Lo que ve cada modelo el día de hoy. Todo en proporción al precio de hoy, salvo los embalses."""
    p = datos["precio"]
    return pd.DataFrame({
        "ayer": p.shift(1) / p,
        "promedio_7": p.rolling(7).mean() / p,
        "promedio_30": p.rolling(30).mean() / p,
        "embalse": datos["embalse"],
        "embalse_cambio_7": datos["embalse"] - datos["embalse"].shift(7),
        "dia_semana": datos.index.dayofweek,
    })


@dataclass
class Resultado:
    errores: pd.DataFrame          # MAPE de validación y de prueba por candidato
    pronosticos: pd.DataFrame      # pronósticos en el tramo de prueba
    real: pd.Series                # precio real en el tramo de prueba
    importancias: pd.Series        # importancia de cada variable en el bosque
    alfa: float                    # alfa escogido para el suavizado exponencial
    H: int
    tramos: dict
    _datos: pd.DataFrame
    _vecinos: int


def competir(datos: pd.DataFrame, H: int = 1, vecinos: int = 5, fin_validacion: str = "2025-06-30") -> Resultado:
    """Entrena ocho candidatos con tres tramos: entrenamiento hasta 2024, validación hasta `fin_validacion`
    y prueba después. Los modelos de aprendizaje pronostican el cambio relativo P(t+H) / P(t)."""
    X = variables(datos)
    base = X.join(datos["precio"].rename("hoy")).join(datos["precio"].shift(-H).rename("real")).dropna()
    nombres = list(X.columns)
    razon = base["real"] / base["hoy"]
    objetivo = base.index + pd.Timedelta(days=H)
    E = objetivo <= pd.Timestamp("2024-12-31")
    V = (base.index >= "2025-01-01") & (objetivo <= pd.Timestamp(fin_validacion))
    P = base.index > pd.Timestamp(fin_validacion)

    errores, pron = {}, {}
    for nombre, modelo in modelos(vecinos).items():
        modelo.fit(base[nombres][E], razon[E])
        en_validacion = mape(base["real"][V], modelo.predict(base[nombres][V]) * base["hoy"][V])
        modelo.fit(base[nombres][E | V], razon[E | V])
        pron[nombre] = pd.Series(modelo.predict(base[nombres][P]) * base["hoy"][P].values, index=base.index[P])
        errores[nombre] = {"validación": en_validacion, "prueba": mape(base["real"][P], pron[nombre])}

    alfas = np.round(np.arange(0.05, 1.0, 0.05), 2)
    suave = lambda a: datos["precio"].ewm(alpha=a, adjust=False).mean()
    alfa = float(min(alfas, key=lambda a: mape(base["real"][E], suave(a)[base.index[E]])))
    nivel = suave(alfa)
    simples = {"suavizado exponencial": nivel, "ingenuo": datos["precio"], "promedio_7": datos["precio"].rolling(7).mean()}
    for nombre, serie in simples.items():
        errores[nombre] = {"validación": mape(base["real"][V], serie[base.index[V]]),
                           "prueba": mape(base["real"][P], serie[base.index[P]])}
        pron[nombre] = serie[base.index[P]]

    bosque = modelos(vecinos)["bosque"].fit(base[nombres], razon)
    return Resultado(pd.DataFrame(errores).T, pd.DataFrame(pron), base["real"][P],
                     pd.Series(bosque.feature_importances_, index=nombres).sort_values(), alfa, H,
                     {"entrenamiento": int(E.sum()), "validación": int(V.sum()), "prueba": int(P.sum())},
                     datos, vecinos)


def pronostico(resultado: Resultado, metodo: str) -> float:
    """Pronóstico para H días después del último dato, con el método escogido y todos los datos."""
    datos, H = resultado._datos, resultado.H
    if metodo == "ingenuo":
        return float(datos["precio"].iloc[-1])
    if metodo == "promedio_7":
        return float(datos["precio"].rolling(7).mean().iloc[-1])
    if metodo == "suavizado exponencial":
        return float(datos["precio"].ewm(alpha=resultado.alfa, adjust=False).mean().iloc[-1])
    X = variables(datos)
    base = X.join(datos["precio"].shift(-H).rename("real")).dropna()
    modelo = modelos(resultado._vecinos)[metodo].fit(base[list(X.columns)], base["real"] / datos["precio"][base.index])
    return float(modelo.predict(X.iloc[[-1]])[0]) * float(datos["precio"].iloc[-1])
