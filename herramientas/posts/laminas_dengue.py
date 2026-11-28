"""Carrusel de LinkedIn: ¿quién pronostica mejor el dengue? Se corre desde la raíz del repositorio.

Lee lo que dejan herramientas/posts/dengue/serie.py y competencia.py en documentos/posts/dengue/.
La portada (01) lleva foto y se arma con herramientas/posts/portada.py dengue.

    python herramientas/posts/laminas_dengue.py            -> documentos/posts/dengue/02.png … 08.png y el PDF
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from taller import carrusel as c

DATOS = Path("documentos/posts/dengue")
SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else DATOS)
SALIDA.mkdir(parents=True, exist_ok=True)
c.preparar_fuentes()
num = lambda v, d=0: c._numero(v, d)

serie = pd.read_csv(DATOS / "serie.csv", index_col=0, parse_dates=True)["casos"]
val = pd.read_csv(DATOS / "pred_validacion.csv.gz", parse_dates=["origen", "fecha"])
pru = pd.read_csv(DATOS / "pred_prueba.csv.gz", parse_dates=["origen", "fecha"])
for t in (val, pru):
    t["ea"] = (t.pred - t.real).abs(); t["ep"] = t.ea / t.real.clip(lower=1) * 100; t["sesgo"] = t.pred - t.real

# El representante de cada familia se escoge en validación (2019-2020), por error porcentual medio en los cuatro horizontes.
mape_val = val.pivot_table(index="modelo", columns="h", values="ep").mean(axis=1)
def mejor(prefijo):
    return mape_val[[m for m in mape_val.index if m.startswith(prefijo)]].idxmin()
SIR, RIDGE, BOSQUE, BOOSTING, INGENUO = mejor("SIR"), mejor("ridge"), mejor("bosque"), mejor("boosting"), "ingenuo"
print("Escogidos en validación:", SIR, "|", RIDGE, "|", BOSQUE, "|", BOOSTING)
mae = pru.pivot_table(index="modelo", columns="h", values="ea")
print(mae.loc[[SIR, INGENUO, RIDGE, BOSQUE, BOOSTING]].round(1).to_string())
semanas = pru[pru.h == 1].fecha.nunique()

def diferencia(a, b, h, B=2000, bloque=8, semilla=0):
    """Diferencia de error absoluto medio entre dos modelos, con intervalo de 90 % por bootstrap de bloques de 8 semanas."""
    x = pru[(pru.modelo == a) & (pru.h == h)].set_index("fecha").ea; y = pru[(pru.modelo == b) & (pru.h == h)].set_index("fecha").ea
    d = (x - y).dropna().values; n = len(d); k = int(np.ceil(n / bloque)); rng = np.random.default_rng(semilla)
    reps = [np.concatenate([d[i:i + bloque] for i in rng.integers(0, n - bloque + 1, k)])[:n].mean() for _ in range(B)]
    return d.mean(), np.percentile(reps, 5), np.percentile(reps, 95)
contra_ingenuo = {h: diferencia(SIR, INGENUO, h) for h in (1, 2, 4, 8)}
for h, (m, lo, hi) in contra_ingenuo.items(): print(f"SIR menos ingenuo, h={h}: {m:+.1f} ({lo:+.1f} a {hi:+.1f})")
cruzan = all(lo < 0 < hi for _, lo, hi in contra_ingenuo.values())

PIE = "Fuente: fichas de SIVIGILA publicadas por la Alcaldía de Bucaramanga en datos.gov.co. Cálculos propios.\nCarlos Manuel Orrego Franco"
TOTAL = 8
ROJO, AZUL, GRIS, MORADO, TINTA = c.ROJO, c.AZUL, c.GRIS, c.MORADO, c.TINTA
def guardar(fig, n):
    fig.savefig(SALIDA / f"{n:02d}.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 2. Diez años de casos
pico, cuando = int(serie.max()), serie.idxmax()
antes = serie[:"2022-12-31"]; pico_antes, cuando_antes = int(antes.max()), antes.idxmax()
por_anio = serie.groupby(serie.index.year).sum(); parte_2024 = por_anio[2024] / serie.sum() * 100
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
fig, ax = c._lienzo("DENGUE EN BUCARAMANGA, 2015 A 2025", f"En 2024 la ciudad tuvo el {parte_2024:.0f} %\nde los casos de once años",
                    f"Casos notificados por semana de inicio de síntomas. El pico fue de {pico}\ncasos en una semana de {MESES[cuando.month - 1]} de 2024. "
                    f"El mayor anterior era de {pico_antes}, en {cuando_antes.year}.", PIE, f"2/{TOTAL}")
ax.fill_between(serie.index, serie.values, color=AZUL, alpha=0.18, lw=0); ax.plot(serie.index, serie.values, color=AZUL, lw=2.2)
ax.scatter([cuando], [pico], s=160, color=ROJO, zorder=3)
ax.annotate(f"{pico} casos", (cuando, pico), xytext=(-14, 0), textcoords="offset points", ha="right", va="center", fontsize=22, family=c.SERIF, weight="bold", color=ROJO)
ax.set_ylim(0, pico * 1.08); ax.margins(x=0.01); c._fuente_ejes(ax); guardar(fig, 2)

# 3. Los competidores
fig = c.lamina_texto("LA COMPETENCIA", "Tres maneras de pronosticar\nlos casos que vienen", [
    "«Igual que la última semana». No aprende nada.\nEs la vara que todos deben superar.",
    "Aprendizaje automático. Regresión, bosque y boosting\nque miran las ocho semanas anteriores.",
    "Un modelo SIR. Tres ecuaciones diferenciales que\ndescriben cómo se contagia y se agota una epidemia.",
    f"Todos pronostican a 1, 2, 4 y 8 semanas, cada semana,\ncon lo conocido hasta ese día. Son {semanas} semanas\nde prueba, de enero de 2023 a agosto de 2025.",
], PIE, f"3/{TOTAL}"); guardar(fig, 3)

# 4. El error por horizonte
fig, ax = c._lienzo("EL RESULTADO", "Nadie le gana con claridad a\n«igual que la última semana»",
                    "Error absoluto medio, en casos por semana, según con cuántas\nsemanas de anticipación se hace el pronóstico.", PIE, f"4/{TOTAL}")
grupos = [(SIR, "SIR", ROJO), (INGENUO, "Igual que la\núltima semana", GRIS), (RIDGE, "Regresión\ncon rezagos", AZUL)]
ancho = 0.26
for j, (m, nombre, color) in enumerate(grupos):
    xs = np.arange(4) + (j - 1) * ancho
    ax.bar(xs, mae.loc[m, [1, 2, 4, 8]].values, width=ancho * 0.92, color=color, label=nombre.replace("\n", " "))
    for x, v in zip(xs, mae.loc[m, [1, 2, 4, 8]].values):
        ax.text(x, v + 0.6, num(v, 1), ha="center", va="bottom", fontsize=15, family=c.COND, weight="semibold", color=color if color != GRIS else "#5F6B7A")
ax.set_xticks(range(4)); ax.set_xticklabels(["1 semana", "2 semanas", "4 semanas", "8 semanas"], fontsize=18)
ax.set_yticks([]); ax.grid(False); ax.spines["bottom"].set_color("#3A3A3A")
ax.legend(loc="upper left", frameon=False, prop={"family": c.COND, "size": 18}); c._fuente_ejes(ax); guardar(fig, 4)

# 5. El pronóstico a ocho semanas
def pronostico(m, h=8): return pru[(pru.modelo == m) & (pru.h == h)].sort_values("fecha").set_index("fecha").pred
techo_sir = pronostico(SIR).max(); cuando_techo = pronostico(SIR).idxmax()
fig, ax = c._lienzo("OCHO SEMANAS ANTES", "El ingenuo llega tarde.\nEl SIR se pasa de largo",
                    f"Pronóstico hecho ocho semanas antes, frente a los casos que hubo.\nTras el pico, el SIR llegó a anunciar {num(techo_sir)} casos en una semana.", PIE, f"5/{TOTAL}")
real = serie["2023-01-01":]
ax.plot(real.index, real.values, color=TINTA, lw=3.6, solid_capstyle="round", label="Casos que hubo")
ax.plot(pronostico(INGENUO), color=GRIS, lw=2.6, label="Igual que la última semana")
ax.plot(pronostico(SIR), color=ROJO, lw=2.6, label="SIR")
ax.legend(loc="upper left", frameon=False, prop={"family": c.COND, "size": 18}); ax.set_ylim(0, techo_sir * 1.05); ax.margins(x=0.01)
import matplotlib.dates as mdates
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y")); c._fuente_ejes(ax); guardar(fig, 5)

# 6. El sesgo en la bajada
bajada = pru[(pru.fecha > cuando) & (pru.h == 8)]
sesgo = bajada.groupby("modelo").sesgo.mean()
valores = {"Igual que la\núltima semana": sesgo[INGENUO], "Regresión\ncon rezagos": sesgo[RIDGE], "SIR": sesgo[SIR]}
fig = c.lamina_barras(valores, {"Igual que la\núltima semana": GRIS, "Regresión\ncon rezagos": AZUL, "SIR": ROJO}, "CUANDO LA EPIDEMIA BAJA",
                      "El SIR sabe que una\nepidemia se agota",
                      f"Casos de más que anuncia cada método, en promedio, ocho semanas\nantes. Semanas posteriores al pico de {MESES[cuando.month - 1]} de 2024.",
                      PIE, f"6/{TOTAL}", sufijo="", decimales=0)
fig.axes[0].tick_params(labelsize=19); guardar(fig, 6)
print("sesgo en la bajada a 8 semanas:", {k.replace(chr(10), " "): round(v, 1) for k, v in valores.items()})

# 7. Lo que el dato no deja medir (las cifras salen de herramientas/posts/dengue/ola.py)
fig = c.lamina_texto("LO QUE EL DATO NO DICE", "La curva mide la velocidad.\nEl R0 queda sin medir", [
    "Ajusté el SIR suponiendo que la infección dura media\nsemana, una, dos o tres. Las cuatro versiones explican\nlos casos casi igual de bien.",
    "Según ese supuesto, el número reproductivo R0\nsale entre 1,03 y 1,17.",
    "Los pronósticos casi no cambian. La curva de casos\nfija la velocidad de crecimiento y deja libre el resto.",
    "Para separar una cosa de la otra hace falta un dato\nde afuera, como cuánto dura la infección.",
], PIE, f"7/{TOTAL}"); guardar(fig, 7)

# 8. Cierre
fig = c.lamina_texto("CÓMO SE HIZO", "Método y salvedades", [
    "28.520 fichas de dengue de Bucaramanga, 2015 a 2025,\ncontadas por semana de inicio de síntomas.",
    "El SIR se resuelve con Runge-Kutta de orden 4 y se ajusta\npor máxima verosimilitud a las últimas 16 semanas.\nCada familia se afinó con la epidemia de 2019 y 2020.",
    "Las diferencias de error entre el SIR y el ingenuo caben\ndentro del margen de azar en los cuatro horizontes.",
    "Usé la serie ya consolidada. En tiempo real, las últimas\nsemanas llegan incompletas, y eso no lo medí.",
], "github.com/cmorregof/taller-cd-ml-2026-2\nCarlos Manuel Orrego Franco", f"8/{TOTAL}"); guardar(fig, 8)
assert cruzan, "La frase de la lámina 8 supone que los cuatro intervalos cruzan el cero."

from PIL import Image
portada = Path("documentos/posts/dengue_01_portada.png")
paginas = [Image.open(p).convert("RGB") for p in ([portada] if portada.exists() else []) + sorted(SALIDA.glob("0[2-9].png"))]
paginas[0].save(Path("documentos/posts") / "carrusel_quien_pronostica_el_dengue.pdf", save_all=True, append_images=paginas[1:], resolution=100)
print("PDF con", len(paginas), "páginas")
