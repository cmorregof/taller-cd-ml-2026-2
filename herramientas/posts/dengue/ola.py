import numpy as np, pandas as pd, time
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sir import ajustar, incidencia
s = pd.read_csv(__import__("pathlib").Path(__file__).resolve().parents[3] / "documentos/posts/dengue/serie.csv", index_col=0, parse_dates=True)["casos"]
ola = s["2023-01-01":]
print("ola:", len(ola), "semanas,", int(ola.sum()), "casos, pico", int(ola.max()), "en", ola.idxmax().date())
t = time.time(); libre = ajustar(ola.values); print("libre", {k: round(float(v), 4) for k, v in libre.items() if k not in ("x", "ajuste")}, round(time.time() - t, 1), "s")
for dur in (0.5, 1, 2, 3):
    a = ajustar(ola.values, gamma=1 / dur)
    print(f"1/gamma={dur} sem", {k: round(float(v), 3) for k, v in a.items() if k not in ("x", "ajuste")}, "pico modelo", round(a["ajuste"].max()), "semana", ola.index[a["ajuste"].argmax()].date())
