"""Carrusel de LinkedIn: La brecha del ICFES en Caldas (español). Se corre desde la raíz del repositorio."""
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
from taller.icfes import preparar, tasa_por, tramos, competir, asignar_cupos, auditar

SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else "posts_icfes")
SALIDA.mkdir(parents=True, exist_ok=True)
c.preparar_fuentes()

crudo = descargar("saber11", departamento="CALDAS")
X, y, colegio, datos = preparar(crudo)
PIE = "Fuente: ICFES, Saber 11 2020-2, en datos.gov.co. Cálculos propios.\nCarlos Manuel Orrego Franco"

# 1. Portada
fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=c.FONDO)
fig.patches.append(plt.Rectangle((0, 0.9926), 1, 0.0074, transform=fig.transFigure, color=c.ROJO))
fig.text(0.067, 0.90, " SABER 11 EN CALDAS ", fontsize=21, family=c.COND, weight="semibold", color="white",
         va="center", bbox=dict(boxstyle="square,pad=0.35", fc=c.ROJO, ec="none"))
fig.text(0.067, 0.85, "¿Quién se queda\natrás en el ICFES?", fontsize=66, family=c.SERIF,
         weight="bold", color=c.TINTA, va="top", linespacing=1.05)
n_est = f"{len(X):,}".replace(",", ".")
fig.text(0.067, 0.64, f"{n_est} estudiantes de grado once en Caldas, 2020.\nQué los separa y qué haría un modelo con ellos.",
         fontsize=26, family=c.SERIF, color="#464646", va="top", linespacing=1.3)
zona = tasa_por(y, datos["cole_area_ubicacion"].str.lower())
fig.text(0.067, 0.50, "En el cuarto más bajo del país, según la zona del colegio:", fontsize=19, family=c.COND, color="#3A3A3A")
ax = fig.add_axes([0.067, 0.14, 0.866, 0.33], facecolor=c.FONDO)
filas = [("Colegio urbano", zona.loc["urbano", "tasa"], c.AZUL), ("Colegio rural", zona.loc["rural", "tasa"], c.ROJO)]
for k, (nombre, valor, color) in enumerate(filas):
    base = 1 - k
    ax.barh(base, valor, color=color, height=0.42)
    ax.text(0, base + 0.36, nombre, fontsize=22, family=c.COND, weight="semibold", color="#3A3A3A", va="bottom")
    ax.text(valor + 1, base, f"{valor:.0f} %", va="center", fontsize=36, family=c.SERIF, weight="bold", color=color)
ax.set_xlim(0, 50); ax.set_ylim(-0.4, 1.75); ax.axis("off")
fig.text(0.067, 0.06, "Carlos Manuel Orrego Franco", fontsize=19, family=c.COND, weight="semibold", color="#3A3A3A")
fig.text(0.933, 0.06, "Desliza  →", fontsize=19, family=c.COND, weight="semibold", color="#3A3A3A", ha="right")
fig.savefig(SALIDA / "01.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 2. Colegio y zona
z = tasa_por(y, datos["cole_naturaleza"].str.capitalize() + ", " + datos["cole_area_ubicacion"].str.lower()).query("n >= 50")["tasa"]
fig = c.lamina_barras(z.round(1).to_dict(), {k: (c.ROJO if "rural" in k else c.AZUL) for k in z.index}, "COLEGIO Y ZONA",
                      f"En los colegios oficiales rurales,\n{z.max():.0f} % queda en el cuarto\nmás bajo del país",
                      "Porcentaje de estudiantes con percentil nacional 25 o menos.", PIE, "2/6")
fig.savefig(SALIDA / "02.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 3. La familia
madre = {"Ninguno": "ninguna", "Primaria incompleta": "primaria", "Primaria completa": "primaria",
         "Secundaria (Bachillerato) incompleta": "bachillerato", "Secundaria (Bachillerato) completa": "bachillerato",
         "Técnica o tecnológica incompleta": "técnica", "Técnica o tecnológica completa": "técnica",
         "Educación profesional incompleta": "profesional", "Educación profesional completa": "profesional", "Postgrado": "posgrado"}
m = tasa_por(y, datos["fami_educacionmadre"].map(madre), ["ninguna", "primaria", "bachillerato", "técnica", "profesional", "posgrado"])["tasa"]
fig = c.lamina_barras(m.round(1).to_dict(), {k: c.MORADO for k in m.index}, "LA FAMILIA PESA",
                      f"Si la madre no estudió, {m.iloc[0]:.0f} %\nqueda abajo; si tiene posgrado, {m.iloc[-1]:.0f} %",
                      "Porcentaje en el cuarto más bajo del país\nsegún la educación de la madre.", PIE, "3/6")
fig.savefig(SALIDA / "03.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 4. Municipios
mun = tasa_por(y, datos["cole_mcpio_ubicacion"].str.title())
mun = mun[mun["n"] >= 80]
ext = pd.concat([mun.nlargest(4, "tasa"), mun.nsmallest(4, "tasa")]).sort_values("tasa")["tasa"]
fig = c.lamina_ranking(ext, "MUNICIPIOS", f"En {ext.idxmax()}, {ext.max():.0f} %.\nEn {ext.idxmin()}, {ext.min():.0f} %",
                       "Porcentaje en el cuarto más bajo del país. Los cuatro más\naltos y los cuatro más bajos con 80 o más estudiantes.",
                       PIE, "4/6", resaltar=["Manizales"])
fig.savefig(SALIDA / "04.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 5. El sesgo del modelo
E, V, P = tramos(X, y, colegio)
todos, puntajes = competir(X, y, E, V, P, 0.25)
mejor = todos.drop(index="sorteo")["captación en validación"].idxmax()
cupos = asignar_cupos(puntajes[mejor], 0.25)
jornada = auditar(y, cupos, X["nocturna_o_sabatina"].map({1.0: "Nocturna\no sabatina", 0.0: "Diurna"}))["captación"] * 100
fig = c.lamina_barras(jornada.round(0).to_dict(), {"Diurna": c.ROJO, "Nocturna\no sabatina": c.AZUL}, "EL MODELO TAMBIÉN SESGA",
                      f"Con cupo para uno de cada cuatro,\nel modelo alcanza solo al {jornada['Diurna']:.0f} %\nde los de jornada diurna que lo necesitan",
                      "De los estudiantes que quedaron en el cuarto más bajo,\nqué porcentaje habría recibido el cupo, por jornada.",
                      PIE, "5/6", decimales=0)
fig.savefig(SALIDA / "05.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

# 6. Cómo se hizo
fig = c.lamina_texto("CÓMO SE HIZO", "Un examen, 9.941 estudiantes\ny una pregunta incómoda",
                     ["Resultados anonimizados de Saber 11 2020-2 que\npublica el ICFES en datos.gov.co.",
                      "Cuarto más bajo: percentil nacional 25 o menos.\nSolo se quitó un puntaje en cero.",
                      "Un bosque aleatorio reparte cupos de refuerzo con lo\nque se sabe en la inscripción, y se audita por grupos.",
                      "Lo trabajamos en clase en la Especialización en\nEstadística de la UNAL Manizales."],
                     "github.com/cmorregof/taller-cd-ml-2026-2\nCarlos Manuel Orrego Franco", "6/6")
fig.savefig(SALIDA / "06.png", dpi=100, facecolor=c.FONDO); plt.close(fig)

imgs = [Image.open(SALIDA / f"0{i}.png").convert("RGB") for i in range(1, 7)]
imgs[0].save(SALIDA / "carrusel_la_brecha_del_icfes.pdf", save_all=True, append_images=imgs[1:], resolution=100)
print("Listo:", SALIDA, "| ganador:", mejor, "| captación por jornada:", jornada.round(0).to_dict(),
      "| zona:", zona["tasa"].round(1).to_dict(), "| madre:", m.round(1).to_dict(), "| municipios:", ext.round(0).to_dict())
