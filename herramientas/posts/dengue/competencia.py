"""Pronóstico de casos semanales con origen rodante: SIR reajustado cada semana contra modelos de aprendizaje automático.

    python herramientas/posts/dengue/serie.py
    OMP_NUM_THREADS=1 python herramientas/posts/dengue/competencia.py validacion     (unos 6 minutos con dos núcleos)
    OMP_NUM_THREADS=1 python herramientas/posts/dengue/competencia.py prueba         (unos 8 minutos)
"""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from multiprocessing import Pool
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sir import ajustar
SALIDA = Path(__file__).resolve().parents[3] / "documentos" / "posts" / "dengue"

s = pd.read_csv(SALIDA / "serie.csv", index_col=0, parse_dates=True)["casos"]
y = s.values.astype(float); fechas = s.index; n = len(y)
H = [1, 2, 4, 8]
TRAMOS = {"validacion": ("2019-01-06", "2020-12-27"), "prueba": ("2023-01-01", "2025-08-10")}
def origenes(tramo):
    a, b = TRAMOS[tramo]
    # el origen t es la última semana conocida; el objetivo t+h debe caer dentro del tramo
    return [t for t in range(n) if fechas[t] >= pd.Timestamp(a) - pd.Timedelta(weeks=1) and fechas[t] < pd.Timestamp(b)]

# ---------- SIR ----------
def sir_un_origen(args):
    t, W, gamma = args
    ventana = y[max(0, t - W + 1): t + 1]
    a = ajustar(ventana, gamma=gamma, extra=max(H))
    return t, W, gamma, [float(a["ajuste"][len(ventana) + h - 1]) for h in H], a["R0"], a["N"]

# ---------- aprendizaje automático ----------
L = 8
def fila(t):
    lv = np.log1p(y[t - L + 1: t + 1])[::-1]
    ang = 2 * np.pi * fechas[t].dayofyear / 365.25
    return np.concatenate([lv, [lv[0] - lv[1], lv[0] - lv[3], np.sin(ang), np.cos(ang)]])
FILAS = {t: fila(t) for t in range(L - 1, n)}
def modelos():
    return {"ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "bosque": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=1),
            "boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0)}
def ml_un_origen(t):
    sal = {}
    for h in H:
        tr = [u for u in range(L - 1, t - h + 1)]            # pares (u, u+h) con u+h <= t: solo pasado
        X = np.array([FILAS[u] for u in tr]); base = np.log1p(y[tr]); obj = np.log1p(y[np.array(tr) + h])
        x = FILAS[t][None, :]
        for nombre, m in modelos().items():
            sal[(nombre, "nivel", h)] = float(np.expm1(m.fit(X, obj).predict(x)[0]))
            sal[(nombre, "cambio", h)] = float(np.expm1(np.log1p(y[t]) + m.fit(X, obj - base).predict(x)[0]))
    return t, sal

if __name__ == "__main__":
    tramo = sys.argv[1]
    O = origenes(tramo); a, b = map(pd.Timestamp, TRAMOS[tramo])
    reg = []
    def anotar(modelo, t, h, pred):
        if t + h < n and a <= fechas[t + h] <= b:
            reg.append({"modelo": modelo, "origen": fechas[t], "h": h, "fecha": fechas[t + h], "real": y[t + h], "pred": max(pred, 0.0)})
    for t in O:
        for h in H:
            anotar("ingenuo", t, h, y[t])
            anotar("ingenuo estacional", t, h, y[t + h - 52])
    t0 = time.time()
    with Pool(2) as p:
        for t, sal in p.imap_unordered(ml_un_origen, O, chunksize=4):
            for (nombre, variante, h), pred in sal.items(): anotar(f"{nombre} ({variante})", t, h, pred)
        print("ML listo", round(time.time() - t0), "s", flush=True)
        tareas = [(t, W, g) for t in O for W in (16, 26, 52) for g in (1.0, 0.5, None)]
        extra = []
        for t, W, g, preds, R0, N in p.imap_unordered(sir_un_origen, tareas, chunksize=4):
            nombre = f"SIR (ventana {W}, " + ("gamma libre" if g is None else f"1/gamma = {1 / g:g} sem") + ")"
            for h, pr in zip(H, preds): anotar(nombre, t, h, pr)
            extra.append({"modelo": nombre, "origen": fechas[t], "R0": R0, "N": N})
        print("SIR listo", round(time.time() - t0), "s", flush=True)
    pd.DataFrame(reg).round({"pred": 3}).to_csv(SALIDA / f"pred_{tramo}.csv.gz", index=False)
    pd.DataFrame(extra).to_csv(SALIDA / f"sir_parametros_{tramo}.csv", index=False)
