"""Genera el gemelo de fútbol de la clase 4, La Premier del Leicester, y su hoja de bitácora.

    python herramientas/construir_clase04_premier.py            -> los dos cuadernos y la hoja
    python herramientas/construir_clase04_premier.py sin-hoja   -> solo los cuadernos (no necesita pdflatex)

El cuaderno de los estudiantes sale del mismo texto que la solución: lo que va
entre « » en el código se reemplaza por ____. Por eso en las celdas de código
las comillas españolas solo se usan para marcar huecos.

El cuaderno de la mañana (04a, El Mundial de Qatar) vive en construir_clase04.py.
Este archivo se lee solo: no importa nada de allá.
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno

NOMBRE = "04c_la_premier_del_leicester"

# ------------------------------------------------------- LA PREMIER DEL LEICESTER
PREMIER = [
("md", r'''
# Clase 4 · La Premier del Leicester (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *El Mundial de Qatar*: el mismo camino con una liga completa, la Premier League 2015/2016, que ganó el Leicester City.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la mañana con casi seis veces más goles: construir un gol esperado para una liga, detectar una fuga de información, comparar un catálogo de modelos sin dejarse engañar por el ganador y comprobar con el dato lo que un Mundial no deja comprobar, si las ocasiones de un jugador se repiten más que su puntería. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que les ayude a llenar un hueco o que interprete un error. Deben poder explicar sin ayuda con qué pregunta se detecta una fuga de información, por qué se parte por partido y por fecha, en qué celda se agrega un modelo, qué dice la gráfica de puestos, qué mide la correlación entre las dos mitades de la temporada y qué no permite concluir. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con al menos tres modelos agregados por ustedes al catálogo, y la foto de la hoja de bitácora llena en la tarea de Classroom. |

**Cómo se trabaja.** El código está escrito casi todo. Donde vean `____` falta una pieza, y el comentario de al lado dice qué va. Lean la celda entera antes de llenar. Si una celda falla, lean el error antes de pedir ayuda. Las apuestas, las respuestas y la decisión no se escriben en el cuaderno: van en la **hoja de bitácora** impresa. Al final le toman una foto a las dos caras y la adjuntan a la tarea de Classroom.
'''),
("md", r'''
> **Antes de empezar: conozcan la fuente.** Los datos son otra vez de [StatsBomb Open Data](https://github.com/statsbomb/open-data). En la mañana lo abrieron pensando en un Mundial. Ábranlo ahora pensando en una liga y dedíquenle cinco minutos.
>
> - En `data/competitions.json`, ¿qué temporadas de la Premier League están publicadas? Abran la carpeta `data/matches/2/` y miren cuántos partidos trae cada una. ¿Cuál es la única que parece una liga completa?
> - En la [especificación de los eventos](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Events%20v4.0.0.pdf), busquen dentro del evento *Shot* el campo `outcome`. ¿Qué valores puede tomar? ¿En qué momento se conoce cada uno, antes o después de que el balón sale del pie?
> - Lean de nuevo el [anuncio de la publicación del Mundial de 2022](https://blogarchive.statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/). Si mañana publican una gráfica hecha con los remates de la Premier, ¿qué tienen que poner en ella para cumplir lo que la empresa pide?
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

En la mañana construimos un gol esperado con los 64 partidos de un Mundial. El método funcionó, pero casi todas las conclusiones quedaron con una advertencia al lado: hay pocos goles. La validación tuvo 33. El ganador entre trece modelos cayó al quinto puesto en la prueba. Y la afirmación más importante para un club, que las ocasiones se repiten más que la puntería, quedó escrita como supuesto porque un solo torneo no la deja comprobar.

Una liga cambia esa cuenta. La Premier League 2015/2016 son 380 partidos, 38 jornadas por equipo. Es la temporada que ganó el Leicester City, y por eso la escogí: ante un campeón cualquiera, la primera pregunta de un analista es cuánto de ese título estaba en las ocasiones y cuánto en la puntería. Ian Graham parte de la misma sospecha en su libro. Para él, una temporada entera sigue mezclando suerte con rendimiento, porque el fútbol tiene muy pocos goles (capítulos 1 y 6).

Graham cuenta además que entre los pioneros de este trabajo en Inglaterra hubo otros dos clubes, el Brentford y el Brighton. A los dos los compraron apostadores profesionales, que llevaron al club lo que habían aprendido en las apuestas (capítulo 6). Graham nunca trabajó en el Brighton, y lo que cuenta de ese club lo sabe desde afuera. Los traigo porque ustedes van a hacer hoy el trabajo de esa gente: ponerle una probabilidad a cada remate y revisar después si esa probabilidad se sostiene.

Imagínense otra vez la dirección deportiva de un club, ahora con una temporada completa de la liga sobre la mesa. Las tres preguntas son las de la mañana, y la tercera es la que hoy sí se puede responder con datos.

1. **Predecir.** ¿Qué probabilidad de gol tiene cada remate?
2. **Explicar.** ¿Qué hace peligroso un remate, y qué parte de eso se ve solo cuando se sabe dónde estaban los defensas?
3. **Decidir.** ¿A qué delanteros manda a observar el club: a los que más goles hicieron o a los que más ocasiones tuvieron?

**La respuesta y la métrica** son las de la mañana y no las repito. $y_i=1$ si el remate $i$ fue gol, el modelo entrega $\hat p_i$, y se mide con la pérdida logarítmica y el puntaje de Brier porque las probabilidades se van a sumar.

### Predicción escrita

**Antes de correr la celda siguiente, escriban en la hoja sus apuestas A1 a A4.** Se escriben con esfero y no se corrigen.

- **A1.** En el Mundial, cerca de 11 de cada cien remates fueron gol. ¿Y en la Premier?
- **A2.** Si por descuido el modelo recibe la columna `resultado`, que dice en qué terminó cada remate, ¿qué le pasa a su pérdida logarítmica?
- **A3.** En la mañana ganó la validación el bosque, y la correlación de Spearman entre los puestos de validación y los de la prueba fue 0,74. ¿Cuál de los trece modelos gana con la Premier? ¿Esa correlación sube o baja?
- **A4.** Entre la primera y la segunda mitad de la temporada, ¿qué se repite más en un jugador: la calidad de sus ocasiones o su puntería?
'''),
("md", r'''
## 2. Datos

Una sola línea trae los remates de toda la temporada, con una fila por remate. Tarda cerca de medio minuto, porque lee 380 partidos. La tabla se sigue llamando `remates_mundial` por la clase de la mañana, y con los dos números de abajo se le dice qué torneo traer.
'''),
("code", r'''
COMPETICION, TEMPORADA = 2, 27        # Premier League 2015/2016. Al final lo cambiarán por 223, 282: la Copa América 2024

AZUL, ROJO, MORADO, GRIS, FONDO = "#1D4E89", "#D7263D", "#6C4AB6", "#9AA5B1", "#FBFAF7"
plt.rcParams.update({"figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
                     "axes.prop_cycle": plt.cycler(color=[AZUL, ROJO, MORADO, GRIS])})
'''),
("code", r'''
remates = descargar("remates_mundial", competicion=COMPETICION, temporada=TEMPORADA)
print("Remates:", len(remates), "| Partidos:", remates["partido"].nunique(), "| Goles:", remates["gol"].sum(), "| Columnas:", remates.shape[1])
print("Del", remates["fecha"].min(), "al", remates["fecha"].max())
remates.head(3).T
'''),
("md", r'''
### 2.1 Las trampas del dato

Las trampas de la mañana no se copian. Se comprueban de nuevo, porque el dato es otro.

**Primera trampa: las tandas de penales.** En el Mundial, el periodo 5 era la tanda y había que sacarlo. Una liga no tiene partidos de eliminación. La celda dice si aquí existe ese periodo.
'''),
("code", r'''
por_periodo = remates.groupby("periodo")["gol"].agg(remates="size", goles="sum", tasa="mean")
por_periodo["tasa"] = (por_periodo["tasa"] * 100).round(1)
print(por_periodo.to_string())
print("Periodos que aparecen:", sorted(int(p) for p in remates["periodo"].unique()), "| ¿hay tanda de penales?", "sí" if 5 in por_periodo.index else "no")

juego = remates[remates["periodo"] < «5»].reset_index(drop=True)      # HUECO 1: el número del periodo que es la tanda de penales. Se quedan los periodos menores que ese
print(f"Remates que se quedan: {len(juego)} de {len(remates)}")
'''),
("md", r'''
Anoten en la hoja lo que encontraron y lo que deciden. El filtro se deja escrito aunque aquí no quite nada, porque al final van a correr el cuaderno con una copa, y ahí sí hay tandas.

**Segunda trampa: una columna que trae la respuesta de otro.** `statsbomb_xg` es el gol esperado que calcula la propia StatsBomb. Sigue sin poder entrar como variable, por la razón de la mañana: mi modelo aprendería a copiar el de ellos. Se guarda como referencia profesional para el final.

**Tercera trampa: remates sin «foto».** Tres variables salen de la foto que StatsBomb toma en el instante del remate, con la posición de todos los jugadores. En la mañana les dije que los penales no traen foto. Compruébenlo con este dato antes de creerme.
'''),
("code", r'''
sin_foto = juego["rivales_en_el_cono"].isna()
print(pd.crosstab(juego["tipo"], sin_foto.map({False: "con foto", True: "sin foto"})).to_string())
penales = juego["tipo"] == "Penalty"
print(f"\nRemates sin foto: {int(sin_foto.sum())} | de ellos son penales: {int((sin_foto & penales).sum())}")
print(f"Penales: {int(penales.sum())} | penales que sí traen foto: {int((penales & ~sin_foto).sum())}")
print(f"Remates con foto en la que no aparece el arquero rival: {int((juego['distancia_del_arquero_al_arco'].isna() & ~sin_foto).sum())}")
'''),
("md", r'''
Lean los tres renglones y escriban en la hoja qué remates son los que no traen foto. La frase de la mañana necesita una corrección, y la tabla dice cuál. También hay unos pocos remates con foto donde el arquero rival no aparece, y en ellos falta la distancia del arquero a su arco. El faltante sigue siendo por diseño y no al azar, así que se llena con la mediana de entrenamiento dentro de cada modelo, y la variable `penal` le avisa al modelo que esos remates son otra cosa.

**La geometría.** La distancia y el ángulo se calculan como en la mañana. Con $d_x=120-x$ y $d_y=|40-y|$, la distancia al centro del arco es $d=\sqrt{d_x^2+d_y^2}$, en unidades de la cancha de StatsBomb, que son cercanas a una yarda.
'''),
("code", r'''
dx, dy = 120 - juego["x"], (40 - juego["y"]).abs()

X = pd.DataFrame({
    "distancia_m": «np.sqrt(dx**2 + dy**2)» * 0.9144,      # HUECO 2: la distancia d de la fórmula de arriba, escrita con np.sqrt, dx y dy
    "angulo": np.degrees(np.arctan2(8 * dx, dx**2 + dy**2 - 16)),
    "cabeza": (juego["parte_del_cuerpo"] == "Head").astype(int),
    "penal": (juego["tipo"] == "Penalty").astype(int),
    "tiro_libre": (juego["tipo"] == "Free Kick").astype(int),
    "bajo_presion": juego["bajo_presion"],
    "de_primera": juego["de_primera"],
    "mano_a_mano": juego["mano_a_mano"],
    "rivales_en_el_cono": juego["rivales_en_el_cono"],
    "arquero_en_el_cono": juego["arquero_en_el_cono"],
    "arquero_lejos_del_arco": juego["distancia_del_arquero_al_arco"],
})
y = juego["gol"]
variables = list(X.columns)
print(f"Remates en juego: {len(juego)} | goles: {y.sum()} | tasa de gol: {y.mean() * 100:.1f} %")
print("Faltantes por variable:"); print(X.isna().sum()[lambda s: s > 0].to_string())
'''),
("md", r'''
Comparen la tasa de gol con su apuesta A1 y anoten en la hoja lo que salió.

### 2.2 Conocer el dato antes de modelarlo

Las cuatro miradas de la mañana: dónde se remata, cómo cae la tasa de gol con la distancia, cómo sube con el ángulo y cuánto cambia según la clase de remate. Las barras de error son de dos errores estándar. Fíjense en cuánto se angostaron frente a las del Mundial.
'''),
("code", r'''
def tasa_por(grupo):
    tabla = pd.DataFrame({"n": y.groupby(grupo, observed=True).size(), "tasa": y.groupby(grupo, observed=True).mean() * 100})
    tabla["ee"] = np.sqrt(tabla["tasa"] * (100 - tabla["tasa"]) / tabla["n"])
    return tabla

fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
eje = ejes[0, 0]
eje.scatter(juego.loc[y == 0, "x"], juego.loc[y == 0, "y"], s=5, color=GRIS, alpha=0.25, label="no fue gol")
eje.scatter(juego.loc[y == 1, "x"], juego.loc[y == 1, "y"], s=10, color=ROJO, alpha=0.7, label="gol")
eje.plot([120, 120], [36, 44], color="black", lw=4); eje.plot([102, 102, 120], [18, 62, 62], color="gray", lw=1); eje.plot([102, 120], [18, 18], color="gray", lw=1)
eje.set(xlim=(75, 121), ylim=(80, 0), title="Dónde se remató (el arco está a la derecha)", xlabel="x", ylabel="y"); eje.legend(fontsize=8, loc="lower left")

por_distancia = tasa_por(pd.cut(X["distancia_m"], [0, 6, 9, 12, 16, 20, 25, 30, 60]))
ejes[0, 1].bar([f"{int(i.left)} a {int(i.right)}" for i in por_distancia.index], por_distancia["tasa"], yerr=2 * por_distancia["ee"], color=AZUL, ecolor=MORADO, capsize=3)
ejes[0, 1].set(title="Tasa de gol según la distancia (m)", ylabel="%")

por_angulo = tasa_por(pd.cut(X["angulo"], [0, 10, 15, 20, 30, 45, 180]))
ejes[1, 0].bar([f"{int(i.left)} a {int(i.right)}" for i in por_angulo.index], por_angulo["tasa"], yerr=2 * por_angulo["ee"], color=AZUL, ecolor=MORADO, capsize=3)
ejes[1, 0].set(title="Tasa de gol según el ángulo con que se ve el arco (grados)", ylabel="%")

clase = np.select([X["penal"] == 1, X["tiro_libre"] == 1, X["cabeza"] == 1, X["mano_a_mano"] == 1], ["penal", "tiro libre", "cabeza", "mano a mano"], "pie, en juego")
por_clase = tasa_por(pd.Series(clase, index=y.index)).sort_values("tasa")
ejes[1, 1].barh([f"{nombre} (n={int(n)})" for nombre, n in zip(por_clase.index, por_clase["n"])], por_clase["tasa"], xerr=2 * por_clase["ee"], color=MORADO, ecolor=AZUL, capsize=3)
ejes[1, 1].set(title="Tasa de gol según la clase de remate", xlabel="%")
plt.tight_layout(); plt.show()
print(f"Mediana de la distancia: {X['distancia_m'].median():.1f} m | remates desde más de 20 m: {(X['distancia_m'] > 20).mean() * 100:.0f} %, con tasa de gol de {y[X['distancia_m'] > 20].mean() * 100:.1f} %")
'''),
("md", r'''
La forma de las cuatro gráficas es la de la mañana. Cambió la precisión, porque detrás de cada barra hay muchos más remates. Los penales siguen siendo pocos, y su barra de error lo muestra.
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición y las líneas base

La partición es la de la mañana, por partido y por fecha. El primer 60 % de los partidos entrena, el 20 % siguiente sirve para escoger y el último 20 % se reserva para reportar. En una liga eso deja la prueba con las últimas jornadas de la temporada.
'''),
("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

partidos = juego["partido"].drop_duplicates().tolist()          # ya vienen en orden de fecha
corte_1, corte_2 = int(«0.6» * len(partidos)), int(0.8 * len(partidos))      # HUECO 3: la fracción de los partidos que entrena, como decimal
E = juego["partido"].isin(partidos[:corte_1]).values
V = juego["partido"].isin(partidos[corte_1:corte_2]).values
P = juego["partido"].isin(partidos[corte_2:]).values
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {juego['partido'][tramo].nunique():3d} partidos | {tramo.sum():4d} remates | {y[tramo].sum():3d} goles | tasa {y[tramo].mean() * 100:.1f} % | del {juego['fecha'][tramo].min()} al {juego['fecha'][tramo].max()}")
'''),
("md", r'''
Miren cuántos goles tiene ahora la validación y compárenlos con los 33 de la mañana. De ese número depende cuánto se le puede creer al orden de los modelos.

Las dos líneas base son las mismas. La primera le da a todo remate la tasa de gol. La segunda es una regresión logística con la distancia sola. No repito la demostración de la exactitud: con una clase así de rara, decir que ningún remate es gol acierta casi siempre y no sirve.
'''),
("code", r'''
def puntuar(real, prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return {"pérdida logarítmica": log_loss(real, prob), "Brier": brier_score_loss(real, prob),
            "AUC": roc_auc_score(real, prob) if len(np.unique(prob)) > 1 else 0.5, "goles que predice": prob.sum(), "goles que hubo": int(np.sum(real))}

solo_distancia = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
referencias_validacion = {
    "tasa base": np.full(V.sum(), y[«E»].mean()),       # HUECO 4: el tramo del que sale la tasa cuando se mide en validación (E, V o P). La validación no puede ayudarse a sí misma
    "solo distancia": solo_distancia.fit(X[["distancia_m"]][E], y[E]).predict_proba(X[["distancia_m"]][V])[:, 1],
}
probabilidades = {
    "tasa base": np.full(P.sum(), y[E | V].mean()),
    "solo distancia": solo_distancia.fit(X[["distancia_m"]][E | V], y[E | V]).predict_proba(X[["distancia_m"]][P])[:, 1],
}
errores = {nombre: {"validación": puntuar(y[V], referencias_validacion[nombre])["pérdida logarítmica"], **puntuar(y[P], probabilidades[nombre])} for nombre in probabilidades}
display(pd.DataFrame(errores).T.round(4))
'''),
("md", r'''
### 3.2 Una trampa nueva: la fuga por la columna `resultado`

Esta trampa no la vimos en la mañana y sale del libro. Graham explica que hay dos momentos para apretar pausa (capítulo 7). El gol esperado de siempre congela la imagen justo antes del golpeo: sabe dónde está el balón y dónde están los jugadores, y nada más. Hay otro modelo que la congela cuando el balón ya salió del pie y conoce la trayectoria. Con ese segundo modelo Graham valoraba arqueros (capítulo 2). Según él, les da probabilidad cero a unos dos tercios de los remates, porque van afuera, pegan en el palo o los bloquea un defensa. En las cuentas que él hace para la Premier, un remate cualquiera termina en gol cerca del 10 % de las veces y un remate al arco cerca del 30 %.

La tabla trae una columna que pertenece a ese segundo momento. `resultado` dice en qué terminó cada remate. La celda la muestra y repite las tres cuentas de Graham con esta temporada.
'''),
("code", r'''
print(juego["resultado"].value_counts().to_string())
al_arco = juego["resultado"].isin(["Goal", "Saved", "Saved to Post"])       # gol, atajado, o atajado y al palo: los remates que iban entre los tres palos
conversion_al_arco = y[al_arco].sum() / «al_arco.sum()»                   # HUECO 5: el denominador de esta tasa, es decir, cuántos remates fueron al arco
print(f"\nRemates que no fueron al arco: {(~al_arco).mean() * 100:.0f} % | goles por cada cien remates: {y.mean() * 100:.1f} | goles por cada cien remates al arco: {conversion_al_arco * 100:.1f}")
'''),
("md", r'''
Comparen los tres números con los del libro. Ahora la trampa. Voy a entrenar a propósito una regresión logística que recibe `resultado`, codificada con una columna de ceros y unos por cada valor, y a medirla en validación al lado de la misma regresión sin esa columna.
'''),
("code", r'''
con_fuga = pd.concat([X, pd.get_dummies(juego[«"resultado"»], prefix="resultado").astype(int)], axis=1)      # HUECO 6: la columna que se anota después del remate, entre comillas
print("Columnas que se le agregan al modelo:", [c for c in con_fuga.columns if c.startswith("resultado")])

comparacion = {}
for nombre, tabla in [("sin la columna resultado", X), ("con la columna resultado", con_fuga)]:
    modelo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(tabla[E], y[E])
    comparacion[nombre] = puntuar(y[V], modelo.predict_proba(tabla[V])[:, 1])
comparacion = pd.DataFrame(comparacion).T
display(comparacion.round(4))
print(f"La pérdida logarítmica en validación quedó dividida por {comparacion['pérdida logarítmica'].iloc[0] / comparacion['pérdida logarítmica'].iloc[1]:.0f}.")
'''),
("md", r'''
Comparen con su apuesta A2. Un resultado así de bueno es una alarma. El modelo no aprendió nada de fútbol. Leyó la respuesta, porque uno de los valores de `resultado` es `Goal`. Graham llama fuga de información a lo que pasa cuando un modelo aprende algo que uno no quería que aprendiera (capítulo 8).

La pregunta que detecta la fuga es la del curso desde la primera clase. **El día en que se toma la decisión, ¿esta columna ya existe?** El club quiere saber cuánto valía la ocasión en el instante del remate. En ese instante nadie sabe todavía si el balón va al arco. `resultado` se anota después, y por eso queda por fuera. Anótenlo en la tabla de trampas de la hoja, con sus palabras.

La celda siguiente deja la decisión escrita en código, para que un descuido posterior no la deshaga.
'''),
("code", r'''
PROHIBIDAS = ["resultado", «"statsbomb_xg"»]      # HUECO 7: la otra columna que no puede entrar, la de la segunda trampa, entre comillas
coladas = [columna for columna in X.columns if any(columna.startswith(prohibida) for prohibida in PROHIBIDAS)]
assert not coladas, f"Estas columnas no pueden entrar al modelo: {coladas}"
print(f"Las {len(variables)} variables del modelo se conocen en el instante del remate. Ninguna es de la lista prohibida: {PROHIBIDAS}")
'''),
("md", r'''
### 3.3 El catálogo de modelos

El catálogo es el de la mañana, con los mismos trece modelos y las mismas opciones. `MODELOS` es un diccionario, la función `tuberia` le pone delante a cada modelo el llenado de faltantes y la estandarización, y todo lo que sigue recorre el diccionario. Agregar un modelo es agregar una línea.

Dejo los 150 vecinos y las hojas grandes por la razón que vimos con el modelo de 75 vecinos. Un vecindario pequeño puede no tener ningún gol, y entonces el modelo dice probabilidad cero para un remate que después entra.

**Su turno, y es obligatorio.** Agreguen al menos tres modelos al diccionario, uno por línea, donde dice el comentario. Pueden repetir uno con otras opciones o traer uno nuevo de la [lista de scikit-learn](https://scikit-learn.org/stable/supervised_learning.html). La condición es que tenga `predict_proba`. Anoten en la hoja cuáles agregaron y cómo les fue.

En la mañana el catálogo corrió en un instante. Aquí hay varias veces más filas de entrenamiento, y la máquina de soporte es la que más lo siente. La celda del protocolo dice cuánto tardó cada modelo, para que sepan qué esperar de los que agreguen.
'''),
("code", r'''
import time, warnings
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
warnings.filterwarnings("ignore", category=FutureWarning)      # avisos de versiones futuras de scikit-learn que no cambian el resultado

def tuberia(modelo, estandarizar=True):
    """Llena los faltantes con la mediana, estandariza si el modelo lo necesita y pone el modelo al final."""
    pasos = [SimpleImputer(strategy="median")] + ([StandardScaler()] if estandarizar else []) + [modelo]
    return make_pipeline(*pasos)

MODELOS = {
    "logística":            tuberia(LogisticRegression(max_iter=2000)),
    "logística apretada":   tuberia(LogisticRegression(C=0.05, max_iter=2000)),              # C pequeño: coeficientes más encogidos
    "discriminante lineal": tuberia(LinearDiscriminantAnalysis()),
    "Bayes ingenuo":        tuberia(GaussianNB()),
    "k vecinos":            tuberia(KNeighborsClassifier(n_neighbors=«150»)),                  # HUECO 8: cuántos vecinos, para que casi ningún vecindario se quede sin goles. El número de la mañana
    "máquina de soporte":   tuberia(SVC(probability=True, random_state=0)),
    "red neuronal":         tuberia(MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, random_state=0)),
    "árbol":                tuberia(DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=0), estandarizar=False),
    "bosque":               tuberia(RandomForestClassifier(n_estimators=400, min_samples_leaf=10, random_state=0, n_jobs=-1), estandarizar=False),
    "árboles extra":        tuberia(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=10, random_state=0, n_jobs=-1), estandarizar=False),
    "boosting":             tuberia(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, random_state=0), estandarizar=False),
    "boosting clásico":     tuberia(GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=0), estandarizar=False),
    "AdaBoost":             tuberia(AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=0), estandarizar=False),
    # Agreguen aquí sus modelos, uno por línea:   "mi modelo": tuberia(...),
}
print(f"Modelos en el catálogo: {len(MODELOS)}")
'''),
("md", r'''
La celda siguiente es el protocolo, y no hay que cambiarle nada. Entrena cada modelo con el tramo de entrenamiento, lo mide en validación, lo vuelve a entrenar con entrenamiento y validación juntos, y lo mide en la prueba.
'''),
("code", r'''
segundos = {}
for nombre, receta in MODELOS.items():
    inicio = time.time()
    modelo = clone(receta)                               # una copia sin entrenar, para no ensuciar el catálogo
    modelo.fit(X[E], y[E])
    en_validacion = puntuar(y[V], modelo.predict_proba(X[V])[:, 1])["pérdida logarítmica"]
    modelo.fit(X[E | V], y[E | V])
    probabilidades[nombre] = modelo.predict_proba(X[P])[:, 1]
    errores[nombre] = {"validación": en_validacion, **puntuar(y[P], probabilidades[nombre])}
    segundos[nombre] = time.time() - inicio
todos = pd.DataFrame(errores).T
print(f"El catálogo tardó {sum(segundos.values()):.0f} segundos. Los tres más lentos:", ", ".join(f"{n} ({s:.0f} s)" for n, s in sorted(segundos.items(), key=lambda par: -par[1])[:3]))

todos[["validación", "pérdida logarítmica"]].rename(columns={"pérdida logarítmica": "prueba"}).sort_values("validación").plot.barh(figsize=(9, 0.42 * len(todos) + 1.5), color=[AZUL, MORADO])
plt.xlabel("pérdida logarítmica (menos es mejor)"); plt.title(f"{len(todos)} candidatos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(4))
'''),
("md", r'''
Antes de mirar quién ganó, miren la columna de los goles que predice cada modelo y compárenla con los goles que hubo. Algún modelo puede ordenar bien los remates, con un AUC decente, y aun así prometer muchos más goles de los que hubo. Esa es la diferencia entre ordenar y estar calibrado, y la tabla la muestra sin necesidad de balancear nada.

Ahora sí, el ganador. La regla es la de la clase 1. Se escoge con una sola columna, y si el ganador no le gana en la prueba a la línea base de la distancia, se sigue con la línea base.
'''),
("code", r'''
mejor = todos[«"validación"»].idxmin()                    # HUECO 9: la columna con la que se escoge al ganador, entre comillas
elegido = mejor if todos.loc[mejor, "pérdida logarítmica"] < todos.loc["solo distancia", "pérdida logarítmica"] else "solo distancia"
referencia = puntuar(y[P], juego["statsbomb_xg"][P])
print(f"Ganador en validación: {mejor}")
print(f"En prueba: pérdida logarítmica {todos.loc[mejor, 'pérdida logarítmica']:.4f} | Brier {todos.loc[mejor, 'Brier']:.4f} | predice {todos.loc[mejor, 'goles que predice']:.1f} goles y hubo {int(todos.loc[mejor, 'goles que hubo'])}")
print(f"Solo distancia: {todos.loc['solo distancia', 'pérdida logarítmica']:.4f} | tasa base: {todos.loc['tasa base', 'pérdida logarítmica']:.4f}")
print(f"El modelo de StatsBomb, en los mismos remates: pérdida logarítmica {referencia['pérdida logarítmica']:.4f} | Brier {referencia['Brier']:.4f} | predice {referencia['goles que predice']:.1f} goles")
print(f"Método con el que seguimos: {elegido}")
'''),
("md", r'''
### 3.4 Con más goles, ¿cuánto del ganador es azar?

Esta es la gráfica de la mañana. Ordeno los modelos del catálogo por su puesto en validación y por su puesto en la prueba. Si la validación escogiera sin error, los puntos caerían sobre la diagonal. El resumen es la correlación de Spearman entre los dos órdenes.
'''),
("code", r'''
del_catalogo = todos.loc[list(MODELOS)]
puestos = del_catalogo[["validación", "pérdida logarítmica"]].rank().rename(columns={"pérdida logarítmica": "prueba"})
spearman = puestos["validación"].corr(puestos["prueba"], method=«"spearman"»)       # HUECO 10: la correlación que compara dos órdenes y no dos valores, entre comillas y en minúscula
plt.figure(figsize=(6.8, 6.2))
plt.plot([1, len(puestos)], [1, len(puestos)], color=GRIS, ls=":")
plt.scatter(puestos["validación"], puestos["prueba"], s=60, color=AZUL, zorder=3)
for nombre, (pv, pp) in puestos.iterrows():
    plt.annotate(nombre, (pv, pp), xytext=(6, 4), textcoords="offset points", fontsize=8)
plt.scatter(*puestos.loc[mejor], s=160, facecolors="none", edgecolors=ROJO, lw=2, zorder=4)
plt.xlabel("puesto en validación (1 es el mejor)"); plt.ylabel("puesto en prueba"); plt.title("¿El orden de validación se sostiene en la prueba?")
plt.xticks(range(1, len(puestos) + 1)); plt.yticks(range(1, len(puestos) + 1)); plt.margins(0.12); plt.gca().invert_xaxis(); plt.gca().invert_yaxis(); plt.show()

separacion = del_catalogo["validación"].sort_values()
manana = {"goles en validación": "33", "Spearman entre los dos órdenes": "0,74", "ganador de validación": "bosque", "puesto del ganador en la prueba": "5 de 13",
          "pérdida logarítmica del ganador, en prueba": "0,3055", "pérdida logarítmica de StatsBomb, en prueba": "0,2744"}
tarde = {"goles en validación": str(int(y[V].sum())), "Spearman entre los dos órdenes": f"{spearman:.2f}", "ganador de validación": mejor,
         "puesto del ganador en la prueba": f"{int(puestos.loc[mejor, 'prueba'])} de {len(puestos)}",
         "pérdida logarítmica del ganador, en prueba": f"{todos.loc[mejor, 'pérdida logarítmica']:.4f}", "pérdida logarítmica de StatsBomb, en prueba": f"{referencia['pérdida logarítmica']:.4f}"}
display(pd.DataFrame({"Mundial, en la mañana": manana, "este torneo": {clave: valor.replace(".", ",") for clave, valor in tarde.items()}}))
print(f"Distancia en validación entre el primero y el quinto: {separacion.iloc[4] - separacion.iloc[0]:.4f}")
'''),
("md", r'''
El círculo rojo es el ganador de validación. La columna de la izquierda es lo que vimos en la mañana. La de la derecha la acaba de calcular su cuaderno. Comparen con su apuesta A3 y respondan en la hoja.

- ¿El ganador de validación fue también el mejor en la prueba? ¿Qué tan lejos quedó?
- ¿La gráfica se acercó a la diagonal frente a la de la mañana? ¿A qué se lo atribuyen?
- ¿La distancia frente al modelo de StatsBomb se achicó o se agrandó frente a la de la mañana?

Un aviso sobre los modelos que ustedes agregaron. Cada modelo nuevo es un boleto más en la rifa de la validación, y con suficientes boletos alguno gana por suerte. Con más goles en la validación ese efecto es menor, y sigue ahí.

### 3.5 Balanceo y calibración, en corto

En la mañana vimos que `class_weight="balanced"` deja casi igual el orden de los remates e infla todas las probabilidades, y que una recalibración de Platt deshace el daño. Aquí lo repiten en una celda para comprobar que no era un accidente del Mundial. Si el modelo elegido no sabe balancear, la demostración se hace con la logística.
'''),
("code", r'''
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def armar(nombre, peso=None):
    """Una copia sin entrenar del modelo con ese nombre, y las columnas que usa. `peso="balanced"` pide balancear las clases."""
    if nombre == "solo distancia":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=peso)), ["distancia_m"]
    modelo = clone(MODELOS[nombre])
    if peso is not None:
        modelo.steps[-1][1].set_params(class_weight=peso)
    return modelo, variables

sabe_balancear = elegido == "solo distancia" or "class_weight" in MODELOS[elegido].steps[-1][1].get_params()
demostracion = elegido if sabe_balancear else "logística"
(normal, columnas_demo), (balanceado, _) = armar(demostracion), armar(demostracion, peso=«"balanced"»)      # HUECO 11: el valor de class_weight que iguala el peso de las dos clases, entre comillas
normal.fit(X[columnas_demo][E], y[E]); balanceado.fit(X[columnas_demo][E], y[E])
p_normal, p_balanceado = normal.predict_proba(X[columnas_demo][P])[:, 1], balanceado.predict_proba(X[columnas_demo][P])[:, 1]
columnas = armar(elegido)[1]

platt = LogisticRegression().fit(logit(balanceado.predict_proba(X[columnas_demo][V])[:, 1]).reshape(-1, 1), y[V])      # la recalibración se ajusta con la validación
p_recalibrado = platt.predict_proba(logit(p_balanceado).reshape(-1, 1))[:, 1]

trampa = pd.DataFrame({"sin balancear": puntuar(y[P], p_normal), "balanceado": puntuar(y[P], p_balanceado),
                       "balanceado y recalibrado": puntuar(y[P], p_recalibrado)}).T
print("Modelo:", demostracion, "(entrenado solo con el tramo de entrenamiento, para dejar la validación libre)")
display(trampa.round(4))

def calibracion(real, prob, tramos=8):
    cortes = pd.qcut(pd.Series(prob), tramos, duplicates="drop")
    tabla = pd.DataFrame({"prob": np.asarray(prob), "real": np.asarray(real)}).groupby(cortes.values, observed=True).agg(n=("real", "size"), predicha=("prob", "mean"), observada=("real", "mean"))
    return tabla, float((tabla["n"] / tabla["n"].sum() * (tabla["observada"] - tabla["predicha"]).abs()).sum())

plt.figure(figsize=(6.5, 5.5))
for (nombre, prob), color in zip({"sin balancear": p_normal, "balanceado": p_balanceado, "StatsBomb": juego["statsbomb_xg"][P].values}.items(), [AZUL, ROJO, MORADO]):
    tabla, ece = calibracion(y[P], prob)
    plt.plot(tabla["predicha"], tabla["observada"], marker="o", color=color, label=f"{nombre} (ECE {ece:.3f})")
plt.plot([0, 1], [0, 1], color=GRIS, ls=":"); plt.xlim(0, 0.9); plt.ylim(0, 0.9)
plt.xlabel("probabilidad promedio que dio el modelo"); plt.ylabel("fracción de goles que hubo"); plt.title("Curva de calibración en la prueba"); plt.legend(loc="upper left"); plt.show()
'''),
("md", r'''
Lean la tabla en el orden de la mañana. Primero el AUC, que casi no se mueve. Después los goles que predice el modelo balanceado frente a los que hubo. La curva roja queda por debajo de la diagonal porque el modelo balanceado promete mucho más de lo que pasa. Miren qué tan cerca de la diagonal quedan las otras dos.

### 3.6 Explicar: lo que la foto le quitó a las etiquetas

Hay una historia del libro que se puede poner a prueba con este dato. Antes de la foto, los modelos de gol esperado usaban etiquetas de la jugada. Un remate de contraataque valía más y uno que venía de balón parado valía menos. Graham cuenta que, cuando su modelo recibió las posiciones de los defensas, esas etiquetas perdieron casi todo su peso (capítulo 7). Su explicación es que las etiquetas eran sustitutos de lo que de verdad importa. En un contraataque hay pocos defensas entre el balón y el arco, y después de un tiro de esquina hay muchos.

La celda ajusta dos regresiones logísticas con entrenamiento y validación juntos. La A tiene la geometría y dos etiquetas de la jugada. La B tiene lo mismo más las variables de la foto. Lo que se compara es la razón de probabilidades $e^{\beta}$ de cada etiqueta en los dos modelos. Una razón de 1 quiere decir que la etiqueta ya no aporta.
'''),
("code", r'''
etiquetas = pd.DataFrame({"de contraataque": (juego["jugada"] == "From Counter").astype(int),
                          "tras balón parado": juego["jugada"].isin(["From Free Kick", "From Corner"]).astype(int)})
geometria = ["distancia_m", "angulo", "cabeza", "penal", "tiro_libre"]
de_la_foto = ["mano_a_mano", "rivales_en_el_cono", "arquero_en_el_cono", "arquero_lejos_del_arco"]
filas = {}
for nombre, columnas_modelo in [("A: geometría y etiquetas", geometria), ("B: lo mismo, más la foto", geometria + de_la_foto)]:
    tabla = pd.concat([X[columnas_modelo], etiquetas], axis=1)
    modelo = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=5000)).fit(tabla[E | V], y[E | V])
    coeficientes = pd.Series(modelo[-1].coef_[0], index=tabla.columns)
    filas[nombre] = {"razón, de contraataque": np.exp(coeficientes["de contraataque"]), "razón, tras balón parado": np.exp(coeficientes["tras balón parado"]),
                     "pérdida logarítmica en prueba": log_loss(y[P], modelo.predict_proba(tabla[P])[:, 1])}
print(f"Remates de contraataque: {int(etiquetas['de contraataque'].sum())} | tras balón parado: {int(etiquetas['tras balón parado'].sum())} | total: {len(juego)}")
display(pd.DataFrame(filas).T.round(3))
'''),
("md", r'''
Lean la tabla. ¿Las dos razones se acercaron a 1 cuando entró la foto, como cuenta Graham? ¿Cuánto mejoró la pérdida logarítmica? Recuerden que esto es asociación. La razón de una etiqueta describe con qué viene acompañado un remate de esa clase, y no alcanza para decir que contraatacar cause goles.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Lo que una liga permite comprobar: ¿se repiten más las ocasiones o la puntería?

En la mañana, el argumento para la dirección deportiva descansaba en una frase que no pudimos comprobar: las ocasiones se repiten más que la puntería, y un club paga por lo que se repite. Con 38 jornadas ya no hace falta suponerlo.

El plan es este. Primero calculo el gol esperado de cada remate **fuera de muestra**, como en la mañana: parto los partidos en ocho grupos, y a cada remate lo califica un modelo entrenado con los otros siete. Dejo por fuera los penales. Después parto la temporada en dos mitades por fecha. Para cada jugador $j$ y cada mitad calculo tres medidas, con $n_j$ sus remates, $g_j$ sus goles y $\hat p_i$ el gol esperado de cada remate:

$$\text{volumen}=n_j,\qquad \text{calidad de las ocasiones}=\frac{1}{n_j}\sum_{i\in j}\hat p_i,\qquad \text{puntería}=\frac{1}{n_j}\Bigl(g_j-\sum_{i\in j}\hat p_i\Bigr).$$

La tercera es la diferencia entre goles y gol esperado, por remate. Si una medida describe algo propio del jugador, quien estuvo arriba en la primera mitad debería seguir arriba en la segunda. Eso se resume con la correlación entre mitades. Llamo $a_j$ al valor del jugador $j$ en la primera mitad y $b_j$ al de la segunda, sobre los $J$ jugadores que pasan un mínimo de remates:

$$r=\frac{\sum_{j=1}^{J}(a_j-\bar a)(b_j-\bar b)}{\sqrt{\sum_{j}(a_j-\bar a)^2}\,\sqrt{\sum_{j}(b_j-\bar b)^2}}.$$

Una correlación sola no dice cuánto confiar en ella. Para el intervalo uso **bootstrap sobre jugadores**. Saco $J$ jugadores con reemplazo, recalculo la correlación y repito $B=2000$ veces. El intervalo de 95 % va del percentil 2,5 al percentil 97,5 de esas correlaciones:

$$\bigl[\,r^{*}_{(0{,}025)},\ r^{*}_{(0{,}975)}\,\bigr],\qquad r^{*}_b=r\bigl(\text{muestra } b \text{ de } J \text{ jugadores con reemplazo}\bigr).$$

Se remuestrean jugadores y no remates porque la unidad que se quiere generalizar es el jugador.
'''),
("code", r'''
from sklearn.model_selection import GroupKFold

fuera_de_muestra = np.zeros(len(juego))
for entrena, prueba in GroupKFold(n_splits=8).split(X, y, juego[«"partido"»]):       # HUECO 12: la columna que define los grupos, para que ningún partido quede a los dos lados, entre comillas
    fuera_de_muestra[prueba] = armar(elegido)[0].fit(X[columnas].iloc[entrena], y.iloc[entrena]).predict_proba(X[columnas].iloc[prueba])[:, 1]
juego["xg"] = fuera_de_muestra
total = puntuar(y, fuera_de_muestra)
print(f"Todo el torneo, fuera de muestra: predice {total['goles que predice']:.1f} goles y hubo {total['goles que hubo']} | pérdida logarítmica {total['pérdida logarítmica']:.4f} | StatsBomb: {puntuar(y, juego['statsbomb_xg'])['pérdida logarítmica']:.4f}")

sin_penales = juego[juego["tipo"] != «"Penalty"»].copy()       # HUECO 13: el tipo de remate que se deja por fuera al comparar jugadores, como aparece en la tabla de la sección 2.1
sin_penales["mitad"] = np.where(sin_penales["partido"].isin(partidos[:len(partidos) // 2]), "primera", "segunda")
print(sin_penales.groupby("mitad").agg(partidos=("partido", "nunique"), remates=("gol", "size"), goles=("gol", "sum"), desde=("fecha", "min"), hasta=("fecha", "max")).to_string())

por_mitad = sin_penales.groupby(["jugador", "mitad"]).agg(remates=("gol", "size"), goles=("gol", "sum"), xg=("xg", "sum")).unstack("mitad").fillna(0)
print("\nJugadores que llegan a cada mínimo de remates en las DOS mitades:")
print(pd.Series({f"{minimo} o más": int((por_mitad["remates"] >= minimo).all(axis=1).sum()) for minimo in [5, 10, 15, 20, 25, 30]}).to_string())
'''),
("md", r'''
**El umbral.** Hay que escoger un mínimo de remates por mitad, y la tabla muestra lo que cuesta cada opción. Con un mínimo bajo entran muchos jugadores, pero la puntería de alguien con cinco remates es casi toda ruido. Con un mínimo alto quedan solo los delanteros titulares y el intervalo se ensancha por falta de jugadores. Escojo **15 remates en cada mitad**. Con la Premier deja varias decenas de jugadores, y quince remates sin penales son, con la tasa de gol de la sección 2, entre uno y dos goles esperados por mitad. Es poco, y esa pobreza es parte de lo que la celda va a medir. Para que el umbral no decida la conclusión, la celda repite la cuenta con otros tres.
'''),
("code", r'''
MINIMO = «15»          # HUECO 14: el mínimo de remates en cada mitad que escogió el texto de arriba
NOMBRES_MEDIDAS = ["remates (volumen)", "gol esperado por remate (calidad de las ocasiones)", "goles menos gol esperado, por remate (puntería)"]

def medidas(tabla, mitad):
    n = tabla["remates"][mitad]
    return dict(zip(NOMBRES_MEDIDAS, [n, tabla["xg"][mitad] / n, (tabla["goles"][mitad] - tabla["xg"][mitad]) / n]))

def correlacion_con_intervalo(a, b, repeticiones=2000, semilla=0):
    """La correlación entre a y b, y su intervalo de 95 % por bootstrap sobre jugadores."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    sorteo = np.random.default_rng(semilla).integers(0, len(a), (repeticiones, len(a)))
    with np.errstate(invalid="ignore", divide="ignore"):
        replicas = [np.corrcoef(a[indices], b[indices])[0, 1] for indices in sorteo]
    return np.corrcoef(a, b)[0, 1], np.nanpercentile(replicas, 2.5), np.nanpercentile(replicas, 97.5)

def entre_mitades(minimo):
    """Los jugadores que pasan el mínimo en las dos mitades y, por medida, la correlación entre mitades con su intervalo."""
    pasan = por_mitad[(por_mitad["remates"] >= minimo).all(axis=1)]
    if len(pasan) < 10:
        return pasan, None
    primera, segunda = medidas(pasan, "primera"), medidas(pasan, "segunda")
    return pasan, pd.DataFrame({nombre: correlacion_con_intervalo(primera[nombre], segunda[nombre]) for nombre in NOMBRES_MEDIDAS}, index=["correlación", "desde", "hasta"]).T

estables, repeticion = entre_mitades(MINIMO)
print(f"Jugadores con {MINIMO} remates o más en cada mitad: {len(estables)}")
if repeticion is None:
    print(f"Son menos de 10. Con tan pocos jugadores una correlación entre mitades no dice nada, y no la calculo.")
    print("El cuaderno funciona bien. Lo que pasa es que este torneo no alcanza para comprobar qué se repite más.")
else:
    display(repeticion.round(2))
    for nombre, fila in repeticion.iterrows():
        print(f"- {nombre}: {'el intervalo contiene el cero' if fila['desde'] <= 0 <= fila['hasta'] else 'el intervalo NO contiene el cero'}")

    primera, segunda = medidas(estables, "primera"), medidas(estables, "segunda")
    fig, ejes = plt.subplots(1, 3, figsize=(14, 4.6))
    for eje, nombre, color, escala in zip(ejes, NOMBRES_MEDIDAS, [GRIS, AZUL, ROJO], [1, 100, 100]):
        eje.scatter(primera[nombre] * escala, segunda[nombre] * escala, s=22, color=color, alpha=0.75)
        eje.set_title(f"{nombre.split(' (')[0]}\nr = {repeticion.loc[nombre, 'correlación']:.2f}  [{repeticion.loc[nombre, 'desde']:.2f} a {repeticion.loc[nombre, 'hasta']:.2f}]".replace(".", ","), fontsize=10)
        eje.set_xlabel("primera mitad" + (", %" if escala == 100 else "")); eje.set_ylabel("segunda mitad" + (", %" if escala == 100 else ""))
    fig.suptitle(f"Cada punto es un jugador con {MINIMO} remates o más en cada mitad, sin penales ({len(estables)} jugadores)", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.show()

    print("\nLa misma cuenta con otros mínimos (correlación e intervalo de 95 %):")
    otros = {}
    for minimo in [10, 15, 20, 25]:
        pasan, tabla = entre_mitades(minimo)
        if tabla is not None:
            otros[f"{minimo} o más ({len(pasan)} jugadores)"] = {nombre.split(" (")[0]: f"{fila['correlación']:.2f} [{fila['desde']:.2f} a {fila['hasta']:.2f}]" for nombre, fila in tabla.iterrows()}
    display(pd.DataFrame(otros).T)
'''),
("md", r'''
Comparen con su apuesta A4 y copien en la hoja las tres correlaciones con sus intervalos. Después respondan, también en la hoja, mirando la tabla principal y la de los otros mínimos.

- ¿Cuál de las tres medidas se repite más de una mitad a la otra? ¿En cuál el intervalo contiene el cero?
- ¿La respuesta cambia con el umbral? Miren también cómo se ensanchan los intervalos cuando quedan pocos jugadores, y desconfíen de la fila que tenga menos.
- ¿Qué **no** permite concluir este resultado?

Para la tercera pregunta les dejo tres cosas que hay que tener presentes.

La primera es de aritmética. Con quince o veinte remates por mitad, la puntería de un jugador se calcula con uno, dos o tres goles. Un gol más o un gol menos la cambia de signo. Una medida tan ruidosa se correlaciona poco consigo misma aunque debajo haya una habilidad real. La correlación baja dice que con media temporada no se alcanza a ver esa habilidad. No dice que no exista.

La segunda es del libro, y va en la misma dirección. Graham afirma que la habilidad para definir existe, y que en el Liverpool cada jugador tenía su propia estimación de conversión, que se ajustaba con el tiempo (capítulo 7). Cuenta también que Ben Torvaney encontró que el mejor pronóstico de los goles futuros mezclaba 70 % de gol esperado y 30 % de goles reales (capítulo 7). Los goles pesan menos que las ocasiones en esa mezcla, y aun así pesan. En otro capítulo sostiene que las tasas por noventa minutos, entre ellas la de gol esperado, se repiten de una temporada a la siguiente mucho más que las medidas de «impacto» (capítulo 15). Su comparación es entre temporadas y por noventa minutos. La de ustedes es entre mitades y por remate, así que se parecen sin ser la misma.

La tercera es una limitación del dato. La tabla no trae minutos jugados. El volumen de remates de un jugador depende de cuánto jugó en cada mitad, y una lesión o un cambio de técnico lo mueven sin que el jugador haya cambiado.

Graham trae un ejemplo para pensar en el tamaño de la muestra, en un recuadro sobre el teorema de Bayes (capítulo 5). Parte del supuesto de que un delantero hace un gol cada cuatro partidos y corrige la tasa observada de cada jugador según cuánto se ha visto de él. Con datos reales, la tasa de Le Fondre baja de 0,54 a 0,40 goles por noventa minutos. La de Diego Costa casi no cambia, de 0,53 a 0,49. Las dos tasas crudas eran casi iguales, y la corrección las separa. El umbral de la celda anterior es una versión tosca de esa misma idea.

### 4.2 Las dos listas

Con esa evidencia a la vista, las dos listas de la mañana. La lista A ordena por goles y la B por gol esperado, las dos sin penales. La franja de 90 % sale de simular cada remate diez mil veces con su probabilidad. No voy a decirles quién fue el goleador de esa Premier. Lo dice la tabla.
'''),
("code", r'''
sorteo = np.random.default_rng(0)
def franja(probabilidades_jugador):
    simulados = (sorteo.random((10000, len(probabilidades_jugador))) < probabilidades_jugador.values).sum(axis=1)
    return np.percentile(simulados, [5, 95])

jugadores = sin_penales.groupby(["jugador", "equipo"]).agg(remates=("gol", "size"), goles=("gol", "sum"), xg=("xg", "sum")).reset_index()
jugadores["xg por remate"] = jugadores["xg"] / jugadores["remates"]
jugadores["goles menos xg"] = jugadores["goles"] - jugadores["xg"]
candidatos = pd.concat([jugadores.sort_values(["goles", "xg"], ascending=False).head(8), jugadores.sort_values("xg", ascending=False).head(8)]).drop_duplicates(["jugador", "equipo"]).copy()
franjas = [franja(sin_penales.loc[(sin_penales["jugador"] == fila["jugador"]) & (sin_penales["equipo"] == fila["equipo"]), "xg"]) for _, fila in candidatos.iterrows()]
candidatos["franja de 90 %"] = [f"{int(baja)} a {int(alta)}" for baja, alta in franjas]
candidatos["dentro de la franja"] = [baja <= goles <= alta for (baja, alta), goles in zip(franjas, candidatos["goles"])]

por_goles = candidatos.sort_values(["goles", "xg"], ascending=False).head(8)
por_ocasiones = candidatos.sort_values("xg", ascending=False).head(8)
print("Lista A. Los que más goles hicieron (sin penales):"); display(por_goles.round(2).reset_index(drop=True))
print("Lista B. Los que más ocasiones tuvieron (gol esperado, sin penales):"); display(por_ocasiones.round(2).reset_index(drop=True))
en_las_dos = sorted(set(por_goles["jugador"]) & set(por_ocasiones["jugador"]))
print(f"En las dos listas: {len(en_las_dos)} de 8 | goleadores de la lista A dentro de su franja: {int(por_goles['dentro de la franja'].sum())} de 8")
print(f"Ancho medio de la franja en la lista A: {np.mean([alta - baja for baja, alta in franjas[:8]]):.1f} goles, con {por_goles['remates'].mean():.0f} remates en promedio")
'''),
("md", r'''
Miren tres cosas. Cuántos nombres están en las dos listas. Si alguno de la lista A tiene los goles por fuera de su franja. Y qué tan ancha es la franja de un jugador que remató cerca de cien veces, que era una de las preguntas que les dejé en la mañana. En la mañana vimos que, con quince o veinte remates, dos goles o seis podían ser el mismo jugador con distinta suerte.

La segunda mirada es la de volumen contra calidad. Graham la usa para comparar a Messi con Ronaldo (capítulo 12). Los dos tienen tasas de gol parecidas, pero el gol esperado por remate de Messi ronda el 14 % y el de Ronaldo el 11 %, porque Ronaldo prueba más desde lejos. Dos jugadores pueden llegar al mismo total por caminos distintos, y al club le importa cuál es el camino.
'''),
("code", r'''
frecuentes = jugadores.nlargest(60, "remates")
destacados = frecuentes.sort_values("xg", ascending=False).head(10)
plt.figure(figsize=(11, 6.2))
plt.scatter(frecuentes["remates"], frecuentes["xg por remate"] * 100, s=60 + 14 * frecuentes["goles"], color=AZUL, alpha=0.5, edgecolors="none")
plt.scatter(destacados["remates"], destacados["xg por remate"] * 100, s=60 + 14 * destacados["goles"], color=ROJO, edgecolors="none")
# Cada punto destacado lleva un número, y la lista de la derecha trae el nombre completo en letra pequeña. Así los nombres no se montan.
borde = frecuentes["remates"].max() * 1.05
alturas = np.linspace(frecuentes["xg por remate"].max() * 100, frecuentes["xg por remate"].min() * 100, len(destacados))
for numero, (altura, (_, fila)) in enumerate(zip(alturas, destacados.iterrows()), start=1):
    plt.text(fila["remates"], fila["xg por remate"] * 100, str(numero), fontsize=7, fontweight="bold", color="white", ha="center", va="center")
    plt.text(borde, altura, f"{numero}. {fila['jugador']} ({fila['equipo']})", fontsize=6.5, va="center")
promedio = sin_penales["xg"].mean() * 100
plt.axhline(promedio, color=GRIS, ls=":"); plt.text(borde * 1.41, promedio, "promedio del torneo", fontsize=8, color=GRIS, va="bottom", ha="right")
plt.xlim(frecuentes["remates"].min() * 0.92, borde * 1.42); plt.margins(y=0.12)
plt.xlabel("remates en el torneo, sin penales (volumen)"); plt.ylabel("gol esperado por remate, % (calidad)")
plt.title(f"Volumen contra calidad: los {len(frecuentes)} jugadores que más remataron\nEl tamaño del punto son los goles. En rojo y con número, los diez de más gol esperado"); plt.show()
'''),
("md", r'''
Busquen en la gráfica un jugador que remata mucho desde posiciones pobres y otro que remata poco desde posiciones muy buenas. Si su club necesita gol, ¿a cuál mandan a observar primero?

**¿Y el campeón?** El cuaderno lleva el nombre del Leicester y todavía no lo hemos mirado. La misma suma sirve por equipo: goles y gol esperado a favor, y goles y gol esperado en contra, sin penales. Los autogoles no son remates y no están en la tabla, así que estos goles no coinciden con los de la tabla oficial de posiciones.
'''),
("code", r'''
rival = np.where(sin_penales["equipo"] == sin_penales["local"], sin_penales["visitante"], sin_penales["local"])
a_favor = sin_penales.groupby("equipo").agg(goles=("gol", "sum"), xg=("xg", "sum"))
en_contra = sin_penales.groupby(rival).agg(goles=("gol", "sum"), xg=("xg", "sum"))
calendario = juego.drop_duplicates("partido")
equipos = pd.DataFrame({"partidos": pd.concat([calendario["local"], calendario["visitante"]]).value_counts(), "goles a favor": a_favor["goles"], "xg a favor": a_favor["xg"], "goles en contra": en_contra["goles"], "xg en contra": en_contra["xg"]})
equipos["diferencia de gol"] = equipos["goles a favor"] - equipos["goles en contra"]
equipos["diferencia de gol esperada"] = equipos["xg a favor"] - equipos["xg en contra"]
equipos["lo que no explican las ocasiones"] = equipos["diferencia de gol"] - equipos["diferencia de gol esperada"]
equipos = equipos.sort_values("diferencia de gol esperada", ascending=False)
equipos.insert(0, "puesto por ocasiones", range(1, len(equipos) + 1))
display(equipos.round(1))
'''),
("md", r'''
La tabla está ordenada por la diferencia de gol esperada, que es lo que el equipo generó menos lo que concedió. En una liga todos los equipos jugaron los mismos partidos y los totales se pueden comparar. En una copa no, y por eso la primera columna dice cuántos jugó cada uno. Busquen al Leicester City. ¿En qué puesto queda por ocasiones? ¿Cuánto de su diferencia de gol no lo explican las ocasiones, y eso es mucho o poco frente a los demás equipos?

Graham tiene un caso parecido, visto desde adentro. El Liverpool de 2013/2014 marcó 101 goles cuando su modelo esperaba 80. A la temporada siguiente marcó 10 menos de lo esperado (capítulo 7). El mismo libro dice que los mejores equipos suelen marcar algo más de lo que espera el modelo, así que una diferencia a favor mezcla suerte con calidad. La tabla de ustedes no puede decir cuánto hay de cada una. Sirve para saber qué equipos mirar con más cuidado.

### 4.3 La decisión en tres frases

Graham cuenta cómo fue su primera reunión con Klopp (capítulo 4). Empezó con una sola hoja sobre el gol esperado. Después pasó a partidos del Dortmund que Klopp recordaba, y Klopp reconoció enseguida que había merecido ganarlos. En otro capítulo recomienda trabajar con pocas métricas, porque con diez métricas la mitad de los mediocampistas resultan «óptimos» en alguna combinación (capítulo 15).

Escriban la decisión **en la hoja**, para un director deportivo que sabe de fútbol y no sabe qué es una pérdida logarítmica. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal. La evidencia debe usar a lo sumo dos números de este cuaderno, y uno de ellos debe salir de la sección 4.1.

**Premortem.** Tomen el primer nombre de su lista. Ha pasado un año y el jugador no rindió. Escriban en la hoja la razón más probable, y digan si una tabla de remates la habría podido ver.

### 4.4 Riesgos y mantenimiento

Discutan estas preguntas con quien tengan al lado. La última se responde en la hoja.

- **El rival.** Graham descartó la tasa goleadora de Leandro Damião porque muchos de sus goles eran contra clubes pequeños de los campeonatos estatales (capítulo 2), y ajustó los goles de Ronaldo con su selección porque 47 fueron contra rivales como Liechtenstein o Andorra (capítulo 12). En el Liverpool, toda valoración se corregía por la fuerza de los rivales (capítulo 8). Su gol esperado trata igual un remate contra el campeón que contra un equipo que descendió. ¿A quién favorece eso en sus listas?
- **Lo que la tabla no ve.** El modelo no sabe quién patea, ni cuántos minutos jugó, ni qué hizo el jugador cuando no remató. ¿Qué jugadores valiosos no aparecen nunca en una tabla de remates?
- **El modelo envejece.** Entrenaron con una temporada de hace varios años. Graham cuenta que en la Premier los remates se hacen cada vez más cerca del arco (capítulo 7). ¿Qué le pasa a la calibración de un modelo si el juego cambia y nadie lo vuelve a entrenar? ¿Cada cuánto lo revisarían y con qué gráfica?
- **La fuente.** ¿Qué pasaría con sus conclusiones si StatsBomb cambiara la manera de tomar la foto, o dejara de publicar?
- **La pregunta ética.** Supongan que el club usa la diferencia entre goles y gol esperado de una media temporada para fijar el salario de un delantero, o para despedirlo. Miren otra vez el intervalo de la puntería en la sección 4.1 y la franja de la sección 4.2. ¿Qué le exigirían al número antes de aceptar ese uso? ¿Quién debería poder verlo y discutirlo? ¿Qué le deben al jugador si el número se equivoca?

### 4.5 Procedencia
'''),
("code", r'''
print("Fuente: StatsBomb Open Data (github.com/statsbomb/open-data). Se cita a StatsBomb y no se redistribuye el dato.")
procedencia()["remates_mundial"]
'''),
("md", r'''
### 4.6 Cambien una línea

La celda siguiente junta en una columna los números que pide la tabla de la hoja. **Córranla y copien la columna en la hoja antes de cambiar nada.**
'''),
("code", r'''
resumen = pd.Series({
    "Torneo (competición, temporada)": f"{COMPETICION}, {TEMPORADA}",
    "Partidos, remates en juego y goles": f"{len(partidos)}, {len(juego)} y {int(y.sum())}",
    "Ganador de validación y su puesto en la prueba": f"{mejor}, {int(puestos.loc[mejor, 'prueba'])} de {len(puestos)}",
    "Spearman de los puestos": f"{spearman:.2f}",
    "Pérdida logarítmica en prueba: ganador": f"{todos.loc[mejor, 'pérdida logarítmica']:.4f}",
    "Pérdida logarítmica en prueba: solo distancia": f"{todos.loc['solo distancia', 'pérdida logarítmica']:.4f}",
    "Pérdida logarítmica en prueba: StatsBomb": f"{referencia['pérdida logarítmica']:.4f}",
    f"Jugadores con {MINIMO} remates o más en cada mitad": str(len(estables)),
}).str.replace(".", ",", regex=False)
display(resumen.to_frame("para la hoja"))
'''),
("md", r'''
Ahora vuelvan a la celda donde dice `COMPETICION, TEMPORADA = 2, 27`, pongan `223, 282` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Es la **Copa América 2024**, un torneo de 32 partidos. No cambien nada más. El cuaderno está escrito para correr igual, y donde el torneo no alcance para una cuenta, lo dice.

Llenen la segunda columna de la tabla de la hoja con la nueva corrida y respondan la pregunta de cierre. **¿Qué conclusiones de la liga sobreviven en un torneo de 32 partidos y cuáles no?** Piensen al menos en estas cuatro: el periodo 5, la fuga por `resultado`, el ganador del catálogo y la comparación entre mitades.

Dejen el cuaderno entregado con la Copa América corrida.

**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase04/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*. La foto de las dos caras de la hoja de bitácora va en la tarea de Classroom.
'''),
]

# La hoja de bitácora que se imprime. La arma herramientas/hoja.py con la función `armar`.
HOJA_PREMIER = {
    "archivo": NOMBRE,
    "titulo": "La Premier del Leicester",
    "clase": "Clase 4 · sábado 28 de noviembre de 2026",
    "apuestas": [
        ("A1", "En el Mundial, cerca de 11 de cada cien remates fueron gol. ¿Y en la Premier?"),
        ("A2", "Si el modelo recibe la columna resultado, ¿qué le pasa a su pérdida logarítmica?"),
        ("A3", "En la mañana ganó la validación el bosque, y el Spearman entre puestos fue 0,74. ¿Qué modelo gana ahora? ¿El Spearman sube o baja?"),
        ("A4", "Entre las dos mitades de la temporada, ¿qué se repite más en un jugador: la calidad de sus ocasiones o su puntería?"),
    ],
    "tablas": [
        ("Secciones 2.1 y 3.2. Las trampas, comprobadas con la Premier", ["Trampa", "Lo que encontré en el dato", "Decisión"],
         ["Periodo 5", "Remates sin foto", "resultado"]),
        ("Sección 3.3. Los tres modelos que agregué al catálogo (pérdida logarítmica)", ["Modelo y opciones", "Validación", "Prueba"], ["1.", "2.", "3."]),
        ("Sección 4.1. Primera mitad contra segunda mitad", ["Medida", "Correlación", "Intervalo de 95 %"],
         ["Remates", "Gol esperado por remate", "Goles menos gol esperado, por remate"]),
    ],
    "preguntas": [
        "Sección 3.4. ¿El ganador de validación fue el mejor en la prueba? ¿Qué cambió frente a la mañana y por qué?",
        "Sección 4.1. ¿Qué se repite más, las ocasiones o la puntería? ¿Qué no permite concluir ese resultado?",
        "Sección 4.3. Premortem: pasó un año y mi primer nombre no rindió. ¿Por qué? ¿La tabla de remates lo podía ver?",
        "Sección 4.4. Si este número fija un salario o un despido, ¿qué le exijo al número y qué le debo al jugador?",
    ],
    "decision": "Sección 4.3. La decisión en tres frases, para un director deportivo que no sabe qué es una pérdida logarítmica",
    "cambio": ("Sección 4.6. Cambien una línea: Premier League 2015/2016 (2, 27) y Copa América 2024 (223, 282)", ["", "Premier", "Copa América"],
               ["Partidos, remates en juego y goles", "Ganador de validación y puesto en prueba", "Spearman de los puestos",
                "Pérdida log. en prueba: ganador", "Pérdida log. en prueba: solo distancia", "Pérdida log. en prueba: StatsBomb",
                "Jugadores que pasan el umbral (4.1)"]),
    "cierre": "¿Qué conclusiones de la liga sobreviven en un torneo de 32 partidos y cuáles no?",
}


def escribir_hoja():
    """Arma el .tex y el PDF de la hoja en documentos/bitacoras/, con el mismo molde de herramientas/hoja.py."""
    import hoja
    hoja.DESTINO.mkdir(parents=True, exist_ok=True)
    fuente = hoja.DESTINO / f"{NOMBRE}_bitacora.tex"
    fuente.write_text(hoja.armar(HOJA_PREMIER), encoding="utf-8")
    for _ in range(2):
        salida = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", fuente.name], cwd=hoja.DESTINO, capture_output=True, text=True, errors="replace")
    if salida.returncode != 0:
        print(salida.stdout[-1500:]); raise SystemExit(f"No compiló {fuente.name}")
    for sobra in (".aux", ".log"):
        (hoja.DESTINO / f"{NOMBRE}_bitacora{sobra}").unlink(missing_ok=True)
    print("Hoja lista:", hoja.DESTINO / f"{NOMBRE}_bitacora.pdf")


if __name__ == "__main__":
    destino = RAIZ / "clases" / "04_desbalance"
    destino.mkdir(parents=True, exist_ok=True)
    nbf.write(cuaderno(PREMIER, True), destino / f"{NOMBRE}.ipynb")
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(PREMIER, False), RAIZ / "_profesor" / f"{NOMBRE}_SOLUCION.ipynb")
    print("Cuadernos de La Premier del Leicester escritos.")
    if "sin-hoja" not in sys.argv[1:]:
        escribir_hoja()
