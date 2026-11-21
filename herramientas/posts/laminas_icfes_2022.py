"""Carrusel de LinkedIn: ¿Quién se queda atrás en el ICFES? Caldas, Saber 11 de 2022 (español).

Se corre desde la raíz del repositorio. Hace las láminas 2 a 8; la portada sale de
herramientas/posts/portada.py. No toca el cuaderno de clase, que sigue con 2020.

Dato: «Resultados únicos Saber 11», datos.gov.co kgxf-xxbe, periodo 20224 (calendario A de 2022).
Trampa: en ese periodo cada estudiante aparece dos veces; se quitan los repetidos por estu_consecutivo.
El conjunto no trae el percentil nacional. Se cuenta con la API cuántos estudiantes del país tienen
puntaje global menor o igual a cada valor y se toma el umbral que deja al 25 % por debajo.
"""
import io, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
from taller import carrusel as c
from taller.icfes import EDUCACION, PERSONAS, CUARTOS, tasa_por, tramos, competir, asignar_cupos, auditar

SALIDA = Path(sys.argv[1] if len(sys.argv) > 1 else "posts_icfes_2022")
SALIDA.mkdir(parents=True, exist_ok=True)
CACHE = Path("datos/cache/saber11_2022_caldas.csv")
API = "https://www.datos.gov.co/resource/kgxf-xxbe"
PERIODO = "20224"
c.preparar_fuentes()


def contar(donde: str) -> int:
    r = requests.get(API + ".json", params={"$select": "count(*)", "$where": f"periodo='{PERIODO}' and {donde}"}, timeout=180)
    r.raise_for_status()
    return int(r.json()[0]["count"])


# Umbral nacional del 25 %: el mayor puntaje t con a lo sumo el 25 % del país en t o menos.
# Los repetidos del periodo no cambian la proporción, porque cada estudiante aparece dos veces.
total = contar("punt_global::number > 0")
bajo, alto = 150, 300
while bajo < alto:
    medio = (bajo + alto + 1) // 2
    if contar(f"punt_global::number > 0 and punt_global::number <= {medio}") / total <= 0.25:
        bajo = medio
    else:
        alto = medio - 1
UMBRAL = bajo
debajo = contar(f"punt_global::number > 0 and punt_global::number <= {UMBRAL}") / total
despues = contar(f"punt_global::number > 0 and punt_global::number <= {UMBRAL + 1}") / total
print(f"Umbral nacional 2022: {UMBRAL} | país en {UMBRAL} o menos: {100 * debajo:.2f} % | en {UMBRAL + 1} o menos: {100 * despues:.2f} %")

if not CACHE.exists():
    r = requests.get(API + ".csv", params={"$where": f"cole_depto_ubicacion='CALDAS' and periodo='{PERIODO}'",
                                          "$limit": 50000}, timeout=300)
    r.raise_for_status()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(r.text)
crudo = pd.read_csv(CACHE, dtype=str)
print("Filas:", len(crudo), "| estudiantes distintos:", crudo["estu_consecutivo"].nunique())
datos = crudo.drop_duplicates("estu_consecutivo").reset_index(drop=True)
puntaje = pd.to_numeric(datos["punt_global"], errors="coerce")
datos = datos[puntaje > 0].reset_index(drop=True)
puntaje = pd.to_numeric(datos["punt_global"])

edad = 2022 - pd.to_numeric(datos["estu_fechanacimiento"].str[-4:], errors="coerce")
si = lambda col: (datos[col] == "Si").astype(float).where(datos[col].notna())
X = pd.DataFrame(index=datos.index)
X["mujer"] = (datos["estu_genero"] == "F").astype(float).where(datos["estu_genero"].notna())
X["edad"] = edad.where(edad.between(13, 70))
X["estrato"] = pd.to_numeric(datos["fami_estratovivienda"].str.extract(r"(\d)")[0], errors="coerce")
X["sin_estrato"] = (datos["fami_estratovivienda"] == "Sin Estrato").astype(float)
X["educacion_madre"] = datos["fami_educacionmadre"].map(EDUCACION)
X["educacion_padre"] = datos["fami_educacionpadre"].map(EDUCACION)
for cosa in ["internet", "computador", "automovil", "lavadora"]:
    X[cosa] = si(f"fami_tiene{cosa}")
X["personas_hogar"] = datos["fami_personashogar"].map(PERSONAS)
X["cuartos"] = datos["fami_cuartoshogar"].map(CUARTOS)
X["colegio_oficial"] = (datos["cole_naturaleza"] == "OFICIAL").astype(float)
X["colegio_rural"] = (datos["cole_area_ubicacion"] == "RURAL").astype(float)
X["nocturna_o_sabatina"] = datos["cole_jornada"].isin(["NOCHE", "SABATINA"]).astype(float)
X["jornada_completa"] = datos["cole_jornada"].isin(["COMPLETA", "UNICA"]).astype(float)
y = (puntaje <= UMBRAL).astype(int).rename("bajo")
colegio = datos["cole_cod_dane_establecimiento"]

PIE = "Fuente: ICFES, Saber 11 2022 (calendario A), en datos.gov.co. Cálculos propios.\nCarlos Manuel Orrego Franco"
TOTAL = 8


def guardar(fig, n):
    fig.savefig(SALIDA / f"0{n}.png", dpi=100, facecolor=c.FONDO)
    plt.close(fig)


def pares_verticales(grupos, series, colores, etiqueta, titular, bajada, pagina, notas=None, pie=PIE, subir=0.0):
    """Barras verticales agrupadas con el valor encima y una leyenda arriba."""
    fig, ax = c._lienzo(etiqueta, titular, bajada, pie, pagina)
    if subir:
        caja = ax.get_position()
        ax.set_position([caja.x0, caja.y0 + subir, caja.width, caja.height - subir])
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

# 2. Madre y zona en 2022
tasa = {(nombre, z): 100 * y[(zona == z) & grupo].mean()
        for nombre, grupo in [("primaria", baja), ("profesional", profesional)] for z in ["urbano", "rural"]}
n_rural_prof = int(((zona == "rural") & profesional).sum())
por_zona = {z: 100 * y[zona == z].mean() for z in ["urbano", "rural"]}
series = {"Madre con primaria o menos": {"Colegio urbano": tasa[("primaria", "urbano")], "Colegio rural": tasa[("primaria", "rural")]},
          "Madre profesional": {"Colegio urbano": tasa[("profesional", "urbano")], "Colegio rural": tasa[("profesional", "rural")]}}
fig = pares_verticales(["Colegio urbano", "Colegio rural"], series,
                       {"Madre con primaria o menos": c.ROJO, "Madre profesional": c.AZUL}, "LA MADRE Y LA ZONA",
                       f"Si la madre solo hizo primaria,\nse queda atrás el {c._numero(tasa[('primaria', 'urbano')], 1)} % en\nla ciudad y el {c._numero(tasa[('primaria', 'rural')], 1)} % en el campo",
                       "Porcentaje que se queda atrás según la zona del colegio\ny la educación de la madre.",
                       f"2/{TOTAL}", notas={("Madre profesional", "Colegio rural"): f"{n_rural_prof} estudiantes"}, subir=0.06)
fig.text(0.067, 0.082, f"Se queda atrás: puntaje global de {UMBRAL} o menos, el 25 % más bajo del país en 2022.\n"
         "Madre profesional: llegó a la universidad, terminara o no, o hizo posgrado.",
         fontsize=15, family=c.COND, color="#5F6B7A", va="bottom", linespacing=1.35)
guardar(fig, 2)

# 3. 2020 frente a 2022, colegios rurales. Las cifras de 2020 salen del cuaderno de clase
# (datos.gov.co rnvb-vnyh, percentil nacional del ICFES), con la misma definición de madre profesional.
RURAL_2020 = {"primaria": 39.6, "profesional": 33.3, "n_profesional": 72}
series = {"2020, en pandemia": {"Madre con\nprimaria o menos": RURAL_2020["primaria"], "Madre\nprofesional": RURAL_2020["profesional"]},
          "2022": {"Madre con\nprimaria o menos": tasa[("primaria", "rural")], "Madre\nprofesional": tasa[("profesional", "rural")]}}
fig = pares_verticales(["Madre con\nprimaria o menos", "Madre\nprofesional"], series,
                       {"2020, en pandemia": c.GRIS, "2022": c.AZUL}, "EL CAMPO, DOS AÑOS",
                       f"En 2020, en el campo, tener madre\nprofesional casi no se notaba.\nEn 2022 sí",
                       "Porcentaje que se queda atrás en los colegios rurales de Caldas.",
                       f"3/{TOTAL}",
                       notas={("2020, en pandemia", "Madre\nprofesional"): f"{RURAL_2020['n_profesional']} estudiantes",
                              ("2022", "Madre\nprofesional"): f"{n_rural_prof} estudiantes"},
                       pie="Fuente: ICFES, Saber 11 2020-2 y 2022, en datos.gov.co. Cálculos propios.\nCarlos Manuel Orrego Franco", subir=0.06)
fig.text(0.067, 0.082, "Grupos pequeños de madres profesionales: las cifras cambian mucho con pocos estudiantes.\n"
         "2020 usa el percentil del ICFES; 2022, el umbral del 25 % calculado con todo el país.",
         fontsize=15, family=c.COND, color="#5F6B7A", va="bottom", linespacing=1.35)
guardar(fig, 3)

# 4. Lo rural junta desventajas
internet = {z: 100 * (datos.loc[zona == z, "fami_tieneinternet"] == "Si").mean() for z in ["urbano", "rural"]}
madre_baja = {z: 100 * baja[zona == z].mean() for z in ["urbano", "rural"]}
series = {"Colegio urbano": {"Internet en casa": internet["urbano"], "Madre con\nprimaria o menos": madre_baja["urbano"]},
          "Colegio rural": {"Internet en casa": internet["rural"], "Madre con\nprimaria o menos": madre_baja["rural"]}}
fig = pares_verticales(["Internet en casa", "Madre con\nprimaria o menos"], series,
                       {"Colegio urbano": c.AZUL, "Colegio rural": c.ROJO}, "LO RURAL SE ACUMULA",
                       f"En el campo, el {c._numero(internet['rural'], 1)} % tiene\ninternet en casa. En la ciudad,\nel {c._numero(internet['urbano'], 1)} %",
                       "Porcentaje de estudiantes, según la zona del colegio.\nQuienes no respondieron cuentan en el total.", f"4/{TOTAL}")
guardar(fig, 4)

# 5. Municipios
mun = tasa_por(y, datos["cole_mcpio_ubicacion"].str.title())
mun = mun[mun["n"] >= 80]
ext = pd.concat([mun.nlargest(4, "tasa"), mun.nsmallest(4, "tasa")]).sort_values("tasa")["tasa"]
fig = c.lamina_ranking(ext, "MUNICIPIOS", f"En {ext.idxmax()} se queda atrás\nel {ext.max():.0f} %. En {ext.idxmin()}, el {ext.min():.0f} %",
                       "Los cuatro municipios más altos y los cuatro más bajos,\ncon 80 o más estudiantes.",
                       PIE, f"5/{TOTAL}", resaltar=["Manizales"])
guardar(fig, 5)

# 6. El modelo frente a dos reglas
E, V, P = tramos(X, y, colegio)
todos, puntajes = competir(X, y, E, V, P, 0.25)
mejor = todos.drop(index="sorteo")["captación en validación"].idxmax()
captado = todos["captación en prueba"] * 100
nombres = {"sorteo": "Sorteo", "regla: educación de la madre": "Según la\neducación\nde la madre", mejor: "Modelo\n(" + mejor + ")"}
valores = {nombres[k]: captado[k] for k in ["sorteo", "regla: educación de la madre", mejor]}
fig = c.lamina_barras(valores, dict(zip(valores, [c.GRIS, c.MORADO, c.AZUL])), "UN MODELO REPARTE CUPOS",
                      f"Con cupo para uno de cada cuatro,\nel modelo encuentra al {captado[mejor]:.0f} % de\nquienes se iban a quedar atrás",
                      "Solo con lo que se sabe al inscribirse. Medido en\ncolegios que el modelo no vio al entrenar.",
                      PIE, f"6/{TOTAL}", decimales=0)
guardar(fig, 6)

# 7. La auditoría
cupos = asignar_cupos(puntajes[mejor], 0.25)
nocturna = X["nocturna_o_sabatina"] == 1
etiqueta_jornada = X["nocturna_o_sabatina"].map({1.0: "Nocturna\no sabatina", 0.0: "Diurna"})
auditoria = auditar(y, cupos, etiqueta_jornada)
jornada = auditoria["captación"] * 100
noct, diur = jornada["Nocturna\no sabatina"], jornada["Diurna"]
edad_n, edad_d = X.loc[nocturna, "edad"].median(), X.loc[~nocturna, "edad"].median()
fig = c.lamina_barras(jornada.round(0).to_dict(), {"Diurna": c.ROJO, "Nocturna\no sabatina": c.AZUL}, "LA AUDITORÍA",
                      f"Encontró al {noct:.0f} % en la jornada\nnocturna y al {diur:.0f} % en la diurna",
                      f"De quienes se iban a quedar atrás, qué porcentaje recibió\ncupo. Mediana de edad: {edad_n:.0f} años en la nocturna\no sabatina y {edad_d:.0f} en la diurna.",
                      PIE, f"7/{TOTAL}", decimales=0)
guardar(fig, 7)

# 8. Cómo se hizo
n_est = f"{len(X):,}".replace(",", ".")
fig = c.lamina_texto("CÓMO SE HIZO", f"Un examen, {n_est} estudiantes\ny una auditoría",
                     ["Resultados anonimizados de Saber 11 de 2022 que\npublica el ICFES en datos.gov.co.",
                      "Cada estudiante venía dos veces en la tabla. Se quitaron\nlos repetidos y los puntajes en cero.",
                      f"Se queda atrás quien saca {UMBRAL} o menos, el 25 % más\nbajo del país, contado con los datos de todo Colombia.",
                      "Un modelo reparte cupos con lo que se sabe al\ninscribirse, y se audita grupo por grupo."],
                     "github.com/cmorregof/taller-cd-ml-2026-2\nCarlos Manuel Orrego Franco", f"8/{TOTAL}")
guardar(fig, 8)

print("Listo:", SALIDA, "| estudiantes:", len(X), "| se quedan atrás:", round(100 * y.mean(), 1),
      "| por zona:", {k: round(v, 1) for k, v in por_zona.items()},
      "| madre y zona:", {k: round(v, 1) for k, v in tasa.items()}, "| rural profesional n:", n_rural_prof,
      "| internet:", {k: round(v, 1) for k, v in internet.items()}, "| madre baja:", {k: round(v, 1) for k, v in madre_baja.items()},
      "| ganador:", mejor, "| captación:", captado.round(1).to_dict(),
      "| auditoría:", auditoria.round(1).to_dict(), "| edades:", edad_n, edad_d,
      "| nocturna n:", int(nocturna.sum()), "| municipios:", ext.round(0).to_dict())
