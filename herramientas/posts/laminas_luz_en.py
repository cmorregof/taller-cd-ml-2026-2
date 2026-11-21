"""LinkedIn carousel: Colombia's electricity price (English). Run from the repository root."""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
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

# 1. Cover
fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=c.FONDO)
fig.patches.append(plt.Rectangle((0, 0.9926), 1, 0.0074, transform=fig.transFigure, color=c.ROJO))
fig.text(0.067, 0.90, " COLOMBIA'S POWER MARKET ", fontsize=21, family=c.COND, weight="semibold", color="white",
         va="center", bbox=dict(boxstyle="square,pad=0.35", fc=c.ROJO, ec="none"))
fig.text(0.067, 0.85, "Can you predict\ntomorrow's price\nof electricity?", fontsize=62, family=c.SERIF,
         weight="bold", color=c.TINTA, va="top", linespacing=1.05)
fig.text(0.067, 0.565, "Four years of Colombia's wholesale power market,\nday by day, against the level of its reservoirs.",
         fontsize=25, family=c.SERIF, color="#464646", va="top", linespacing=1.3)
ax = fig.add_axes([0.067, 0.13, 0.866, 0.30], facecolor=c.FONDO)
ax.fill_between(mensual.index, mensual.values, color=c.AZUL, alpha=0.12)
ax.plot(mensual.index, mensual.values, color=c.AZUL, lw=4)
ax.axis("off")
fig.text(0.067, 0.06, "Carlos Manuel Orrego Franco", fontsize=19, family=c.COND, weight="semibold", color="#3A3A3A")
fig.text(0.933, 0.06, "Swipe  →", fontsize=19, family=c.COND, weight="semibold", color="#3A3A3A", ha="right")
fig.savefig(SALIDA / "01.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 2. The roller coaster
alto, bajo = mensual.idxmax(), mensual.idxmin()
fig = c.lamina_lineas({"Spot\nprice": mensual}, {"Spot\nprice": c.AZUL}, "COLOMBIA, 2023 TO 2026",
                      f"From {num(mensual.max())} to {num(mensual.min())} pesos\nin eight months",
                      f"Monthly average of the national spot price, COP per kWh.\nPeak in {alto:%B %Y}; "
                      f"low in {bajo:%B %Y}.", PIE, "2/5", base=None)
fig.savefig(SALIDA / "02.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 3. Reservoirs
q = datos.groupby(pd.cut(datos.embalse, [0, 40, 50, 60, 70, 100]), observed=True).precio.median()
medianas = {f"{int(i.left)}–{int(i.right)} %": v for i, v in q.items()}
fig = c.lamina_barras(medianas, {k: (c.ROJO if k.startswith("0") else c.AZUL) for k in medianas}, "WATER AND PRICE",
                      "When reservoirs drop below 40 %,\npower gets more expensive",
                      "Median daily spot price (COP per kWh) by the aggregate\nlevel of the country's hydro reservoirs.",
                      PIE, "3/5", sufijo="", decimales=0, idioma="en")
fig.savefig(SALIDA / "03.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 4. Forecasting a week ahead
r1, r7 = competir(datos, H=1), competir(datos, H=7)
nombres = {"ingenuo": "Same as\ntoday", "lineal": "Linear\nregression", "boosting": "Gradient\nboosting", "bosque": "Random\nforest"}
errores = {nombres[k]: r7.errores.loc[k, "prueba"] for k in nombres}
fig = c.lamina_barras(errores, {k: (c.AZUL if k.startswith("Same") else c.GRIS) for k in errores}, "FORECASTING",
                      "A week ahead, nothing beats\n“same as today”",
                      "Mean absolute percentage error of 7-day forecasts on the test set.\n"
                      f"One day ahead, “same as today” is off by {r1.errores.loc['ingenuo', 'prueba']:.1f} %.",
                      PIE, "4/5", idioma="en")
fig.savefig(SALIDA / "04.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 5. How it was done
fig = c.lamina_texto("HOW IT WAS DONE", "Four years, 33,600 hours\nand one cleaning rule",
                     ["Hourly spot prices and daily reservoir levels\nfrom XM's public SiMEM API.",
                      "Each hour is settled several times: I kept\nthe most recent numbered settlement.",
                      "Eight forecasting methods, chosen on a validation\nperiod and scored on months they never saw.",
                      "My graduate students in Statistics at Universidad\nNacional de Colombia, Manizales, solve it too."],
                     "github.com/cmorregof/taller-cd-ml-2026-2\nCarlos Manuel Orrego Franco", "5/5")
fig.savefig(SALIDA / "05.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

imgs = [Image.open(SALIDA / f"0{i}.png").convert("RGB") for i in range(1, 6)]
imgs[0].save(SALIDA / "carousel_colombia_power_price.pdf", save_all=True, append_images=imgs[1:], resolution=100)
print("Done:", SALIDA)
