"""Carrusel de LinkedIn: La brecha del ICFES en Caldas (español). Se corre desde la raíz del repositorio.

Hace las láminas 2 a 7. La portada lleva una foto y se arma aparte con herramientas/posts/portada.py.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from taller import descargar, carrusel as c
from taller.icfes import preparar, tasa_por, tramos, competir, asignar_cupos, auditar

SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else "posts_icfes")
SALIDA.mkdir(parents=True, exist_ok=True)
c.preparar_fuentes()

crudo = descargar("saber11", departamento="CALDAS")
X, y, colegio, datos = preparar(crudo)
PIE = "Fuente: ICFES, Saber 11 2020-2, en datos.gov.co. Cálculos propios.\nCarlos Manuel Orrego Franco"
ATRAS = "Se queda atrás: puntaje entre el 25 % más bajo del país."
TOTAL = 7


def guardar(fig, n):
    fig.savefig(SALIDA / f"0{n}.png", dpi=100, facecolor=c.FONDO)
    plt.close(fig)


def pares_verticales(grupos, series, colores, etiqueta, titular, bajada, pagina, notas=None):
    """Barras verticales agrupadas con el valor encima y una leyenda escrita arriba."""
    fig, ax = c._lienzo(etiqueta, titular, bajada, PIE, pagina)
    ancho = 0.36
    for j, (nombre, valores) in enumerate(series.items()):
        for i, g in enumerate(grupos):
            x = i + (j - 0.5) * (ancho + 0.04)
            ax.bar(x, valores[g], width=ancho, color=colores[nombre])
            ax.text(x, valores[g] + 1, c._numero(valores[g], 1) + " %", ha="center", va="bottom",
                    fontsize=25, family=c.SERIF, weight="bold", color=colores[nombre])
            if notas and (nombre, g) in notas:
                ax.text(x, valores[g] / 2, notas[(nombre, g)], ha="center", va="center", fontsize=15,
                        family=c.COND, weight="semibold", color="white", rotation=90)
    tope = max(max(v.values()) for v in series.values())
    for j, nombre in enumerate(series):
        alto = 0.97 - 0.065 * j
        ax.add_patch(plt.Rectangle((0.0, alto - 0.018), 0.03, 0.036, transform=ax.transAxes, color=colores[nombre],
                                   clip_on=False))
        ax.text(0.045, alto, nombre, transform=ax.transAxes, va="center",
                fontsize=19, family=c.COND, weight="semibold", color=colores[nombre])
    ax.set_xticks(range(len(grupos)))
    ax.set_xticklabels(grupos, fontsize=22)
    ax.set_ylim(0, tope * 1.5)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["bottom"].set_color("#3A3A3A")
    c._fuente_ejes(ax)
    return fig


zona = datos["cole_area_ubicacion"].str.lower()
madre = datos["fami_educacionmadre"]
baja = madre.isin(["Ninguno", "Primaria incompleta", "Primaria completa"])
profesional = madre.isin(["Educación profesional incompleta", "Educación profesional completa", "Postgrado"])

# 2. Madre y zona
tasa = {(nombre, z): 100 * y[(zona == z) & grupo].mean()
        for nombre, grupo in [("primaria", baja), ("profesional", profesional)] for z in ["urbano", "rural"]}
n_rural_prof = int(((zona == "rural") & profesional).sum())
series = {"Madre con primaria o menos": {"Colegio urbano": tasa[("primaria", "urbano")], "Colegio rural": tasa[("primaria", "rural")]},
          "Madre profesional": {"Colegio urbano": tasa[("profesional", "urbano")], "Colegio rural": tasa[("profesional", "rural")]}}
fig = pares_verticales(["Colegio urbano", "Colegio rural"], series,
                       {"Madre con primaria o menos": c.ROJO, "Madre profesional": c.AZUL}, "LA MADRE Y LA ZONA",
                       f"Con madre profesional, se queda\natrás el {c._numero(tasa[('profesional', 'urbano')], 1)} % en la ciudad\ny el {c._numero(tasa[('profesional', 'rural')], 1)} % en el campo",
                       "Porcentaje que se queda atrás según la zona del colegio\ny la educación de la madre.",
                       f"2/{TOTAL}", notas={("Madre profesional", "Colegio rural"): f"{n_rural_prof} estudiantes"})
fig.text(0.067, 0.078, ATRAS + " Madre profesional: llegó a la universidad,\nterminara o no, o hizo posgrado.",
         fontsize=15, family=c.COND, color="#5F6B7A", va="bottom", linespacing=1.35)
guardar(fig, 2)

# 3. Lo rural junta desventajas
internet = {z: 100 * (datos.loc[zona == z, "fami_tieneinternet"] == "Si").mean() for z in ["urbano", "rural"]}
madre_baja = {z: 100 * baja[zona == z].mean() for z in ["urbano", "rural"]}
series = {"Colegio urbano": {"Internet en casa": internet["urbano"], "Madre con\nprimaria o menos": madre_baja["urbano"]},
          "Colegio rural": {"Internet en casa": internet["rural"], "Madre con\nprimaria o menos": madre_baja["rural"]}}
fig = pares_verticales(["Internet en casa", "Madre con\nprimaria o menos"], series,
                       {"Colegio urbano": c.AZUL, "Colegio rural": c.ROJO}, "LO RURAL SE ACUMULA",
                       f"En el campo, solo el {c._numero(internet['rural'], 1)} %\ntiene internet en casa",
                       "Porcentaje de estudiantes, según la zona del colegio.\nQuienes no respondieron cuentan en el total.", f"3/{TOTAL}")
guardar(fig, 3)

# 4. Municipios
mun = tasa_por(y, datos["cole_mcpio_ubicacion"].str.title())
mun = mun[mun["n"] >= 80]
ext = pd.concat([mun.nlargest(4, "tasa"), mun.nsmallest(4, "tasa")]).sort_values("tasa")["tasa"]
fig = c.lamina_ranking(ext, "MUNICIPIOS", f"En {ext.idxmax()} se queda atrás\nel {ext.max():.0f} %. En {ext.idxmin()}, el {ext.min():.0f} %",
                       "Los cuatro municipios más altos y los cuatro más bajos,\ncon 80 o más estudiantes.",
                       PIE, f"4/{TOTAL}", resaltar=["Manizales"])
guardar(fig, 4)

# 5. El modelo frente a dos reglas
E, V, P = tramos(X, y, colegio)
todos, puntajes = competir(X, y, E, V, P, 0.25)
mejor = todos.drop(index="sorteo")["captación en validación"].idxmax()
captado = (todos["captación en prueba"] * 100)
nombres = {"sorteo": "Sorteo", "regla: educación de la madre": "Según la\neducación\nde la madre", mejor: "Modelo\n(" + mejor + ")"}
valores = {nombres[k]: captado[k] for k in ["sorteo", "regla: educación de la madre", mejor]}
fig = c.lamina_barras(valores, dict(zip(valores, [c.GRIS, c.MORADO, c.AZUL])), "UN MODELO REPARTE CUPOS",
                      f"Con cupo para uno de cada cuatro,\nel modelo encuentra al {captado[mejor]:.0f} % de\nquienes se iban a quedar atrás",
                      "Solo con lo que se sabe al inscribirse. Medido en\ncolegios que el modelo no vio al entrenar.",
                      PIE, f"5/{TOTAL}", decimales=0)
guardar(fig, 5)

# 6. La auditoría
cupos = asignar_cupos(puntajes[mejor], 0.25)
nocturna = X["nocturna_o_sabatina"] == 1
jornada = auditar(y, cupos, X["nocturna_o_sabatina"].map({1.0: "Nocturna\no sabatina", 0.0: "Diurna"}))["captación"] * 100
edad_n, edad_d = X.loc[nocturna, "edad"].median(), X.loc[~nocturna, "edad"].median()
noct, diur = jornada["Nocturna\no sabatina"], jornada["Diurna"]
fig = c.lamina_barras(jornada.round(0).to_dict(), {"Diurna": c.ROJO, "Nocturna\no sabatina": c.AZUL}, "LA AUDITORÍA",
                      f"Encontró al {noct:.0f} % en la jornada\nnocturna y al {diur:.0f} % en la diurna",
                      f"De quienes se iban a quedar atrás, qué porcentaje recibió\ncupo. Mediana de edad: {edad_n:.0f} años en la nocturna\no sabatina y {edad_d:.0f} en la diurna.",
                      PIE, f"6/{TOTAL}", decimales=0)
guardar(fig, 6)

# 7. Cómo se hizo
n_est = f"{len(X):,}".replace(",", ".")
fig = c.lamina_texto("CÓMO SE HIZO", f"Un examen, {n_est} estudiantes\ny una auditoría",
                     ["Resultados anonimizados de Saber 11 2020-2 que\npublica el ICFES en datos.gov.co.",
                      "Se queda atrás quien tiene un puntaje entre el 25 %\nmás bajo del país. Solo se quitó un puntaje en cero.",
                      "Un modelo reparte cupos de refuerzo con lo que se sabe\nal inscribirse, y se audita grupo por grupo.",
                      "Lo trabajamos en clase en la Especialización en\nEstadística de la UNAL Manizales."],
                     "github.com/cmorregof/taller-cd-ml-2026-2\nCarlos Manuel Orrego Franco", f"7/{TOTAL}")
guardar(fig, 7)

print("Listo:", SALIDA, "| ganador:", mejor, "| captación:", captado.round(1).to_dict(),
      "| jornada:", jornada.round(1).to_dict(), "| madre y zona:", {k: round(v, 1) for k, v in tasa.items()},
      "| rural profesional n:", n_rural_prof, "| internet:", {k: round(v, 1) for k, v in internet.items()},
      "| madre baja:", {k: round(v, 1) for k, v in madre_baja.items()}, "| edades:", edad_n, edad_d,
      "| municipios:", ext.round(0).to_dict())
