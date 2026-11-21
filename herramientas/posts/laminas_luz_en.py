"""LinkedIn carousel: Colombia's electricity price (English). Run from the repository root."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from taller import descargar, carrusel as c
from taller.luz import competir

SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else "posts_luz_en")
SALIDA.mkdir(parents=True, exist_ok=True)
c.preparar_fuentes()

pc, em = descargar("xm_precio"), descargar("xm_embalses")
n = pc[pc.CodigoVariable == "PB_Nal"].copy()
pref = ["TX8", "TX7", "TX6", "TX5", "TX4", "TX3", "TX2", "TX1", "TXF", "TXR"]
n["o"] = n.Version.map({v: i for i, v in enumerate(pref)})
h = n.sort_values(["FechaHora", "o"]).drop_duplicates("FechaHora")
h["fecha"] = pd.to_datetime(h.FechaHora).dt.normalize()
precio = h.groupby("fecha").Valor.mean().rename("precio")
em["fecha"] = pd.to_datetime(em.Fecha)
s = em[~em.CodigoEmbalse.isin(["AGREGADO_SIN", "NEUSA", "SISGA", "TOMINE"])]
d = s.groupby("fecha")[["VolumenUtilDiarioEnergia", "CapacidadUtilEnergia"]].sum()
embalse = (d.iloc[:, 0] / d.iloc[:, 1] * 100).rename("embalse")
datos = pd.concat([precio, embalse], axis=1).dropna()
mensual = datos.precio.resample("MS").mean()
PIE = "Source: XM, SiMEM (Colombia's power market operator). Own calculations.\nCarlos Manuel Orrego Franco"
num = lambda v: f"{v:,.0f}"

# 1. The cover carries a photo and is built separately with herramientas/posts/portada.py.
TOTAL = 6

# 2. The roller coaster
alto, bajo = mensual.idxmax(), mensual.idxmin()
fig = c.lamina_lineas({"Spot\nprice": mensual}, {"Spot\nprice": c.AZUL}, "COLOMBIA, 2023 TO 2026",
                      f"From {num(mensual.max())} to {num(mensual.min())} pesos\nin eight months",
                      f"Monthly average of the national spot price, COP per kWh.\nPeak in {alto:%B %Y}; "
                      f"low in {bajo:%B %Y}.", PIE, f"2/{TOTAL}", base=None)
fig.savefig(SALIDA / "02.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 3. Driest month against most expensive month
emb_m = datos.embalse.resample("MS").mean()
fig, ax = c._lienzo("WATER AND PRICE", "The driest month was April 2024.\nThe most expensive came\nsix months later",
                    "Each dot is a month: average reservoir level and\naverage spot price (COP per kWh).", PIE, f"3/{TOTAL}")
ax.scatter(emb_m, mensual, s=90, color=c.GRIS, alpha=0.8, zorder=2)
marcas = {emb_m.idxmin(): c.ROJO, mensual.idxmax(): c.ROJO, pd.Timestamp("2024-05-01"): c.AZUL, pd.Timestamp("2023-09-01"): c.AZUL}
for mes, color in marcas.items():
    ax.scatter(emb_m[mes], mensual[mes], s=260, color=color, zorder=3)
    ax.annotate(f"{mes:%b %Y}\n{emb_m[mes]:.0f} %, {num(mensual[mes])} COP", (emb_m[mes], mensual[mes]),
                xytext=(14, 0), textcoords="offset points", va="center", fontsize=17, family=c.COND,
                weight="semibold", color=color)
ax.set_xlabel("Reservoir level (%)", fontsize=18, family=c.COND, color="#3A3A3A")
ax.set_xlim(emb_m.min() - 4, emb_m.max() + 22)
ax.grid(axis="x", color="#DADDE1", lw=1)
c._fuente_ejes(ax)
fig.savefig(SALIDA / "03.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 4. The daily rhythm
h["hora"] = pd.to_datetime(h.FechaHora).dt.hour
por_hora = h.groupby("hora").Valor.median()
fig, ax = c._lienzo("THE DAILY RHYTHM", f"At 7 pm power costs twice\nwhat it costs at 3 am",
                    f"Median hourly spot price by hour of the day, COP per kWh.\n"
                    f"7 pm: {num(por_hora[19])}. 3 am: {num(por_hora[3])}.", PIE, f"4/{TOTAL}")
colores = [c.ROJO if k == 19 else (c.AZUL if k == 3 else c.GRIS) for k in por_hora.index]
ax.bar(por_hora.index, por_hora.values, color=colores, width=0.75)
for k in (3, 19):
    ax.text(k, por_hora[k] + 8, num(por_hora[k]), ha="center", va="bottom", fontsize=24, family=c.SERIF,
            weight="bold", color=colores[k])
ax.set_xticks([0, 3, 6, 12, 19, 23])
ax.set_xticklabels(["midnight", "3 am", "6 am", "noon", "7 pm", "11 pm"], fontsize=17)
ax.set_yticks([])
ax.grid(False)
ax.spines["bottom"].set_color("#3A3A3A")
c._fuente_ejes(ax)
fig.savefig(SALIDA / "04.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 5. Forecasting a week ahead
r1, r7 = competir(datos, H=1), competir(datos, H=7)
nombres = {"ingenuo": "Same as\ntoday", "lineal": "Linear\nregression", "boosting": "Gradient\nboosting", "bosque": "Random\nforest"}
errores = {nombres[k]: r7.errores.loc[k, "prueba"] for k in nombres}
fig = c.lamina_barras(errores, {k: (c.AZUL if k.startswith("Same") else c.GRIS) for k in errores}, "FORECASTING",
                      "A week ahead, nothing beats\n“same as today”",
                      "Mean absolute percentage error of 7-day forecasts on the test set.\n"
                      f"One day ahead, “same as today” is off by {r1.errores.loc['ingenuo', 'prueba']:.1f} %.",
                      PIE, f"5/{TOTAL}", idioma="en")
fig.savefig(SALIDA / "05.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 6. How it was done
fig = c.lamina_texto("HOW IT WAS DONE", "Four years, 33,600 hours\nand one cleaning rule",
                     ["Hourly spot prices and daily reservoir levels\nfrom XM's public SiMEM API.",
                      "Each hour is settled several times: I kept\nthe most recent numbered settlement.",
                      "Eight forecasting methods, chosen on a validation\nperiod and scored on months they never saw.",
                      "My graduate students in Statistics at Universidad\nNacional de Colombia, Manizales, solve it too."],
                     "github.com/cmorregof/taller-cd-ml-2026-2\nCarlos Manuel Orrego Franco", f"6/{TOTAL}")
fig.savefig(SALIDA / "06.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

print("Done:", SALIDA, "| hours:", por_hora[[3, 19]].round(0).to_dict(), "| marked:", {f"{m:%Y-%m}": (round(emb_m[m]), round(mensual[m])) for m in marcas})
