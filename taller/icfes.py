"""Plomería del cuaderno «La brecha del ICFES»: variables, tramos por colegio, captación,
competencia de candidatos y auditoría por grupos.

    from taller.icfes import preparar, tasa_por, tramos, competir, asignar_cupos, auditar
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

EDUCACION = {
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9,
}
LIBROS = {"0 A 10 LIBROS": 0, "11 A 25 LIBROS": 1, "26 A 100 LIBROS": 2, "MÁS DE 100 LIBROS": 3}
PERSONAS = {"1 a 2": 1.5, "3 a 4": 3.5, "5 a 6": 5.5, "7 a 8": 7.5, "9 o más": 9}
CUARTOS = {"Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis o mas": 6}
LECTURA = {"No leo por entretenimiento": 0, "30 minutos o menos": 1, "Entre 30 y 60 minutos": 2,
           "Entre 1 y 2 horas": 3, "Más de 2 horas": 4}
TRABAJO = {"0": 0, "Menos de 10 horas": 1, "Entre 11 y 20 horas": 2, "Entre 21 y 30 horas": 3, "Más de 30 horas": 4}


def preparar(crudo: pd.DataFrame, edad_minima: int = 13, edad_maxima: int = 70):
    """Convierte las respuestas del formulario en números. Solo se usa lo que se sabe en la inscripción.

    Devuelve X (variables), y (1 si quedó en el cuarto más bajo del país), el código del colegio
    y la tabla original, sin los estudiantes con puntaje global igual a cero.
    """
    edad = 2020 - pd.to_numeric(crudo["estu_fechanacimiento"].astype(str).str[:4], errors="coerce")
    si = lambda col: (crudo[col] == "Si").astype(float).where(crudo[col].notna())
    X = pd.DataFrame(index=crudo.index)
    X["mujer"] = (crudo["estu_genero"] == "F").astype(float).where(crudo["estu_genero"].notna())
    X["edad"] = edad.where(edad.between(edad_minima, edad_maxima))
    X["etnia"] = si("estu_tieneetnia")
    X["estrato"] = pd.to_numeric(crudo["fami_estratovivienda"].str.extract(r"(\d)")[0], errors="coerce")
    X["sin_estrato"] = (crudo["fami_estratovivienda"] == "Sin Estrato").astype(float)
    X["educacion_madre"] = crudo["fami_educacionmadre"].map(EDUCACION)
    X["educacion_padre"] = crudo["fami_educacionpadre"].map(EDUCACION)
    for cosa in ["internet", "computador", "automovil"]:
        X[cosa] = si(f"fami_tiene{cosa}")
    X["libros"] = crudo["fami_numlibros"].map(LIBROS)
    X["personas_hogar"] = crudo["fami_personashogar"].map(PERSONAS)
    X["cuartos"] = crudo["fami_cuartoshogar"].map(CUARTOS)
    X["lectura_diaria"] = crudo["estu_dedicacionlecturadiaria"].map(LECTURA)
    X["horas_trabajo"] = crudo["estu_horassemanatrabaja"].astype(str).map(TRABAJO)
    X["colegio_oficial"] = (crudo["cole_naturaleza"] == "OFICIAL").astype(float)
    X["colegio_rural"] = (crudo["cole_area_ubicacion"] == "RURAL").astype(float)
    X["nocturna_o_sabatina"] = crudo["cole_jornada"].isin(["NOCHE", "SABATINA"]).astype(float)
    X["jornada_completa"] = crudo["cole_jornada"].isin(["COMPLETA", "UNICA"]).astype(float)

    y = (pd.to_numeric(crudo["percentil_global"], errors="coerce") <= 25).astype(int).rename("bajo")
    colegio = crudo["cole_cod_dane_establecimiento"]
    validas = pd.to_numeric(crudo["punt_global"], errors="coerce") > 0
    return X[validas], y[validas], colegio[validas], crudo[validas]


def tasa_por(y: pd.Series, grupo: pd.Series, orden: list | None = None) -> pd.DataFrame:
    """Porcentaje en el cuarto más bajo del país por grupo, con su error estándar."""
    tabla = pd.DataFrame({"n": y.groupby(grupo).size(), "tasa": y.groupby(grupo).mean() * 100})
    tabla["ee"] = np.sqrt(tabla["tasa"] * (100 - tabla["tasa"]) / tabla["n"])
    return tabla.reindex(orden).dropna() if orden else tabla.sort_values("tasa")


def tramos(X: pd.DataFrame, y: pd.Series, colegio: pd.Series, semilla: int = 0):
    """Entrenamiento, validación y prueba, con colegios enteros en cada tramo (60 / 20 / 20)."""
    resto, prueba = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=semilla).split(X, y, colegio))
    entrena, valida = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semilla)
                           .split(X.iloc[resto], y.iloc[resto], colegio.iloc[resto]))
    E, V, P = (np.zeros(len(X), dtype=bool) for _ in range(3))
    E[resto[entrena]] = True
    V[resto[valida]] = True
    P[prueba] = True
    return E, V, P


def captacion(real, puntaje, cupo: float = 0.25) -> float:
    """Fracción de los que quedaron abajo que cae entre los primeros `cupo` del orden. Empates por sorteo."""
    real, puntaje = np.asarray(real), np.asarray(puntaje, dtype=float)
    k = int(round(cupo * len(real)))
    orden = np.lexsort((np.random.default_rng(0).random(len(real)), -puntaje))
    return float(real[orden[:k]].sum() / real.sum())


def modelos() -> dict:
    llenar = lambda: SimpleImputer(strategy="median")
    return {
        "logística": make_pipeline(llenar(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "árbol": make_pipeline(llenar(), DecisionTreeClassifier(max_depth=5, random_state=0)),
        "bosque": make_pipeline(llenar(), RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                                  random_state=0, n_jobs=-1)),
    }


def competir(X, y, E, V, P, cupo: float = 0.25):
    """Dos reglas sin modelo y tres modelos. Se escoge en validación y se reporta en prueba.

    Devuelve la tabla de resultados y el puntaje de riesgo que cada candidato da en la prueba.
    """
    mediana_madre = X.loc[E, "educacion_madre"].median()
    reglas = {
        "sorteo": pd.Series(np.random.default_rng(1).random(len(X)), index=X.index),
        "regla: educación de la madre": -X["educacion_madre"].fillna(mediana_madre),
    }
    tabla, puntajes = {}, {}
    for nombre, regla in reglas.items():
        puntajes[nombre] = regla[P]
        tabla[nombre] = {"captación en validación": captacion(y[V], regla[V], cupo),
                         "captación en prueba": captacion(y[P], regla[P], cupo),
                         "AUC en prueba": roc_auc_score(y[P], regla[P])}
    for nombre, modelo in modelos().items():
        modelo.fit(X[E], y[E])
        en_validacion = captacion(y[V], modelo.predict_proba(X[V])[:, 1], cupo)
        modelo.fit(X[E | V], y[E | V])
        puntajes[nombre] = pd.Series(modelo.predict_proba(X[P])[:, 1], index=X.index[P])
        tabla[nombre] = {"captación en validación": en_validacion,
                         "captación en prueba": captacion(y[P], puntajes[nombre], cupo),
                         "AUC en prueba": roc_auc_score(y[P], puntajes[nombre])}
    return pd.DataFrame(tabla).T, puntajes


def asignar_cupos(puntaje: pd.Series, cupo: float = 0.25) -> pd.Series:
    """Marca con True a los primeros `cupo` del orden de riesgo."""
    k = int(round(cupo * len(puntaje)))
    orden = np.lexsort((np.random.default_rng(0).random(len(puntaje)), -puntaje.values))
    marcado = pd.Series(False, index=puntaje.index)
    marcado.iloc[orden[:k]] = True
    return marcado


def auditar(y: pd.Series, cupos: pd.Series, grupo: pd.Series) -> pd.DataFrame:
    """Por grupo: cuántos quedaron abajo, cuántos recibieron cupo y qué fracción de los de abajo lo recibió."""
    grupo = grupo.loc[cupos.index]
    filas = {}
    for valor, indices in grupo.groupby(grupo).groups.items():
        yy, cc = y.loc[indices], cupos.loc[indices]
        filas[valor] = {"n": len(yy), "% que quedó abajo": yy.mean() * 100, "% con cupo": cc.mean() * 100,
                        "captación": cc[yy == 1].mean()}
    return pd.DataFrame(filas).T
