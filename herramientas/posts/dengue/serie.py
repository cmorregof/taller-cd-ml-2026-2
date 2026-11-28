"""Serie semanal de dengue en Bucaramanga por semana de inicio de síntomas. Escribe documentos/posts/dengue/serie.csv."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
RAIZ = Path(__file__).resolve().parents[3]
SALIDA = RAIZ / "documentos" / "posts" / "dengue"
sys.path.insert(0, str(RAIZ))
import numpy as np, pandas as pd
from taller import descargar

CORTE = "2025-08-10"     # última semana de inicio de síntomas que se usa (las siguientes están incompletas por el retraso de notificación)

def serie_semanal(verbose=False):
    c = descargar("dengue", ciudad="bucaramanga")
    def fecha(col):
        v = c[col].astype(str)
        iso = pd.to_datetime(v.str[:10], format="%Y-%m-%d", errors="coerce")
        b = pd.to_datetime(v, format="%d/%m/%Y", errors="coerce")
        r = iso.fillna(b); return r.where(r.dt.year > 1900)
    sint = fecha("ini_sin_")
    cod = pd.to_numeric(c["cod_eve"], errors="coerce"); aj = pd.to_numeric(c["ajuste_"], errors="coerce")
    ok = (cod != 580) & (aj != 6) & sint.notna()
    sem = sint[ok].dt.to_period("W-SAT").dt.start_time      # semana epidemiológica: domingo a sábado
    s = sem.value_counts().sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="7D")
    s = s.reindex(idx, fill_value=0).astype(int)
    if verbose:
        print("fichas:", len(c), "| mortalidad (580):", int((cod == 580).sum()), "| descartadas:", int((aj == 6).sum()), "| quedan:", int(ok.sum()))
        print("semanas:", len(s), s.index[0].date(), "a", s.index[-1].date(), "| se usa hasta", CORTE)
    return s[:CORTE]

if __name__ == "__main__":
    s = serie_semanal(True)
    print(len(s), "semanas;", int(s.sum()), "casos")
    print(s.groupby(s.index.year).agg(["sum", "max", "idxmax"]))
    SALIDA.mkdir(parents=True, exist_ok=True)
    s.to_csv(SALIDA / "serie.csv", header=["casos"])
