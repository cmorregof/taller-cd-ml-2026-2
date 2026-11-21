"""Plomería del cuaderno «La factura de Londres»: leer un pedazo del ZIP oficial, la tabla
congelada del curso, las huellas de versión, la serie por hogar, la competencia de
pronósticos y la tabla de avisos.

    from taller.londres import abrir, leer_pedazo, congelada, huella, versionar, serie_por_hogar, competir, avisos
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from . import datos
from .fuentes_clase03 import RESPALDO_LONDRES, TARIFA_PLANA, TOTAL_ARCHIVOS, abrir_londres, cuales

ARCHIVO_CONGELADO = "londres_diario_archivos-24"
MD5_CONGELADO = "c87108a03bb3720eb56aaecc57bd1c8b"      # del archivo .csv.gz que está en Drive
VERSIONES = datos.CACHE / "versiones"


# ------------------------------------------------------------------ leer
def abrir():
    """Abre el ZIP oficial sin bajarlo. Devuelve el ZIP, la lista de sus 168 archivos y el contador de lo bajado."""
    return abrir_londres()


def leer_pedazo(paquete, internos, numero: int) -> pd.DataFrame:
    """Lee uno de los 168 archivos, con todo como texto, tal como viene."""
    with paquete.open(internos[numero]) as flujo:
        return pd.read_csv(flujo, dtype=str)


def congelada() -> pd.DataFrame:
    """La tabla diaria del curso: 24 de los 168 archivos, ya resumidos, una fila por hogar y día.

    Es una versión fija, guardada el 21 de noviembre de 2026. Si el archivo que llega no
    tiene el MD5 esperado, la función se detiene: no se trabaja con un dato distinto sin saberlo.
    """
    destino = datos.CACHE / f"{ARCHIVO_CONGELADO}.csv.gz"
    if not destino.exists():
        print("[londres] Bajo la tabla congelada del curso desde Google Drive.")
        if not datos._bajar_respaldo(RESPALDO_LONDRES[ARCHIVO_CONGELADO], destino):
            raise RuntimeError("No pude bajar la tabla congelada de Londres. Revisa la conexión y vuelve a correr la celda.")
    md5 = hashlib.md5(destino.read_bytes()).hexdigest()
    if md5 != MD5_CONGELADO:
        raise RuntimeError(f"La tabla congelada llegó con otro MD5 ({md5}). Borra {destino} y vuelve a correr la celda.")
    tabla = pd.read_csv(destino)
    print(f"[londres] Tabla congelada: {len(tabla):,} filas, {tabla['hogar'].nunique()} hogares. MD5 verificado.".replace(",", "."))
    return tabla


# ------------------------------------------------------------------ versiones
def huella(tabla: pd.DataFrame) -> str:
    """Diez caracteres que identifican el contenido de una tabla. Si cambia una celda, cambia la huella."""
    return hashlib.md5(tabla.to_csv(index=False).encode("utf-8")).hexdigest()[:10]


def versionar(tabla: pd.DataFrame, regla: str, madre: str | None = None, nombre: str = "londres_diario") -> str:
    """Guarda la tabla en parquet con su huella en el nombre y la anota en el registro de versiones."""
    VERSIONES.mkdir(parents=True, exist_ok=True)
    h = huella(tabla)
    tabla.to_parquet(VERSIONES / f"{nombre}_{h}.parquet", index=False)
    archivo = VERSIONES / "registro.json"
    registro = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else []
    registro = [v for v in registro if v["huella"] != h]
    registro.append({"nombre": nombre, "huella": h, "filas": int(len(tabla)), "regla": regla, "madre": madre,
                     "guardada": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
    archivo.write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    return h


def registro(nombre: str = "londres_diario") -> pd.DataFrame:
    """El registro de versiones como tabla."""
    archivo = VERSIONES / "registro.json"
    filas = json.loads(archivo.read_text(encoding="utf-8")) if archivo.exists() else []
    return pd.DataFrame([f for f in filas if f["nombre"] == nombre])


# ------------------------------------------------------------------ serie
def ventana(hogares_por_dia: pd.Series, fraccion: float = 0.5):
    """Primer y último día con al menos `fraccion` del máximo de hogares midiendo."""
    suficientes = hogares_por_dia[hogares_por_dia >= hogares_por_dia.max() * fraccion]
    return suficientes.index.min(), suficientes.index.max()


def serie_por_hogar(tabla: pd.DataFrame) -> pd.Series:
    """Consumo promedio por hogar y día. Si falta un día suelto, se interpola."""
    return tabla.groupby("fecha")["kwh"].mean().asfreq("D").interpolate().rename("kwh")


def mape(real, pronostico) -> float:
    """Error porcentual absoluto medio, en %."""
    real, pronostico = np.asarray(real, dtype=float), np.asarray(pronostico, dtype=float)
    return float(np.mean(np.abs(pronostico - real) / real) * 100)


# ------------------------------------------------------------------ pronóstico
def modelos() -> dict:
    return {
        "regresión lineal": LinearRegression(),
        "bosque aleatorio": RandomForestRegressor(n_estimators=300, random_state=0),
        "boosting": GradientBoostingRegressor(random_state=0),
    }


def competir(serie: pd.Series, fin_entrenamiento: str = "2013-06-30", fin_validacion: str = "2013-10-31"):
    """Pronóstico del consumo de mañana: dos reglas sin modelo y tres modelos.

    Los modelos pronostican la razón entre mañana y hoy con el consumo reciente, el día de
    la semana y la época del año. Se escoge en validación y se reporta en prueba. Devuelve
    la tabla de errores (MAPE, en %) y los pronósticos de la prueba junto al valor real.
    """
    manana = serie.index + pd.Timedelta(days=1)
    X = pd.DataFrame({
        "ayer": serie.shift(1) / serie,
        "hace_6": serie.shift(6) / serie,
        "promedio_7": serie.rolling(7).mean() / serie,
        "promedio_28": serie.rolling(28).mean() / serie,
        "manana_es_finde": (manana.dayofweek >= 5).astype(int),
        "manana_es_lunes": (manana.dayofweek == 0).astype(int),
        "epoca_sen": np.sin(2 * np.pi * manana.dayofyear / 365.25),
        "epoca_cos": np.cos(2 * np.pi * manana.dayofyear / 365.25),
    }, index=serie.index)
    base = X.join(serie.rename("hoy")).join(serie.shift(-1).rename("real")).dropna()
    variables, razon = list(X.columns), base["real"] / base["hoy"]
    objetivo = base.index + pd.Timedelta(days=1)
    E = objetivo <= pd.Timestamp(fin_entrenamiento)
    V = (base.index > pd.Timestamp(fin_entrenamiento)) & (objetivo <= pd.Timestamp(fin_validacion))
    P = base.index > pd.Timestamp(fin_validacion)

    reglas = {"igual que hoy": base["hoy"], "igual que hace una semana": serie.shift(6).reindex(base.index)}
    errores, pronosticos = {}, {"real": pd.Series(base["real"][P].values, index=objetivo[P])}
    for nombre, pronostico in reglas.items():
        errores[nombre] = {"validación": mape(base["real"][V], pronostico[V]), "prueba": mape(base["real"][P], pronostico[P])}
        pronosticos[nombre] = pd.Series(pronostico[P].values, index=objetivo[P])
    for nombre, modelo in modelos().items():
        modelo.fit(base[variables][E], razon[E])
        en_validacion = mape(base["real"][V], modelo.predict(base[variables][V]) * base["hoy"][V])
        modelo.fit(base[variables][E | V], razon[E | V])
        en_prueba = modelo.predict(base[variables][P]) * base["hoy"][P]
        errores[nombre] = {"validación": en_validacion, "prueba": mape(base["real"][P], en_prueba)}
        pronosticos[nombre] = pd.Series(en_prueba.values, index=objetivo[P])
    tabla = pd.DataFrame(errores).T
    tabla.attrs["dias"] = {"entrenamiento": int(E.sum()), "validación": int(V.sum()), "prueba": int(P.sum())}
    return tabla, pd.DataFrame(pronosticos)


# ------------------------------------------------------------------ decisión
def avisos(real: pd.Series, pronostico: pd.Series, nivel: float, factores=(0.90, 0.95, 1.00, 1.05)) -> pd.DataFrame:
    """Una política por renglón: se avisa la víspera si el pronóstico supera `nivel` por el factor."""
    es_alto, filas = real > nivel, []
    for factor in factores:
        aviso = pronostico > nivel * factor
        filas.append({"umbral (kWh)": nivel * factor, "avisos": int(aviso.sum()),
                      "días altos avisados": int((aviso & es_alto).sum()),
                      "días altos sin aviso": int((~aviso & es_alto).sum()),
                      "avisos en falso": int((aviso & ~es_alto).sum())})
    return pd.DataFrame(filas)
