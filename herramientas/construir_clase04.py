"""Genera los cuadernos de la clase 4: El Mundial de Qatar (demostración) y
La Premier del Leicester (gemelo de fútbol), además de Las calles de Bogotá.

El cuaderno de los estudiantes sale del mismo texto que la solución: lo que va
entre « » en el código se reemplaza por ____.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno

# ---------------------------------------------------------- EL MUNDIAL DE QATAR
MUNDIAL = [
("md", r'''
# Clase 4 · El Mundial de Qatar

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. Ustedes lo siguen, lo corren y me interrumpen. En la segunda mitad cada uno resuelve el gemelo, *La Premier del Leicester*, con la misma técnica sobre una liga completa.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Modelar un evento raro y entregar probabilidades en las que se pueda confiar: medir con reglas de puntuación en vez de exactitud, revisar la calibración, ver qué le pasa a la probabilidad cuando se «balancean» las clases y comparar muchos modelos sin dejarse engañar por el ganador. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo) y 5 (comunicar a quien no es técnico). |
| **Uso de IA** | Pueden pedirle a Claude que explique una línea de código, que derive una fórmula de geometría o que escriba una gráfica. Deben poder explicar sin ayuda por qué la exactitud no sirve con un evento raro, qué mide el puntaje de Brier, cómo se lee una curva de calibración, qué le hace el balanceo a las probabilidades, por qué una suma de probabilidades es un número esperado de goles y en qué celda se agrega un modelo nuevo. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con las predicciones escritas, al menos tres modelos agregados por ustedes al catálogo y la decisión con su premortem. |
'''),
("md", r'''
> **Antes de empezar: conozcan la fuente.** Los datos son de [StatsBomb Open Data](https://github.com/statsbomb/open-data), el repositorio donde la empresa StatsBomb publica gratis, partido por partido, lo mismo que les vende a los clubes. Ábranlo y dedíquenle cinco minutos antes de correr nada.
>
> - En la carpeta `data/` hay tres tipos de archivo: `competitions.json`, `matches/` y `events/`. ¿Qué es una fila en cada uno?
> - Abran la [especificación de los eventos](https://github.com/statsbomb/open-data/blob/master/doc/Open%20Data%20Events%20v4.0.0.pdf) y busquen el evento *Shot*. ¿Qué anotan de cada remate, además de si fue gol?
> - Lean el [anuncio de la publicación del Mundial de 2022](https://blogarchive.statsbomb.com/news/statsbomb-release-free-2022-world-cup-data/). ¿Qué piden a cambio de usar el dato?
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

La clase pasada les hablé de *How to Win the Premier League*, el libro de Ian Graham. Graham es físico y dirigió durante once años el departamento de investigación del Liverpool. Su punto de partida es que en el fútbol hay tan pocos goles que el resultado de un partido mezcla mucha suerte con lo que el equipo de verdad jugó (capítulos 1 y 6).

En el fútbol casi ningún remate es gol. Graham cuenta que ya en 1968 dos analistas, Reep y Benjamin, habían reportado que cerca de uno de cada diez remates termina en gol (capítulo 7). Por eso el marcador es un mal resumen de un partido: un equipo puede rematar veinte veces desde lejos y otro tres veces desde el punto penal. La pregunta que se hacen los analistas es cuánto **valía** cada remate, y la respuesta es una probabilidad. A eso se le llama gol esperado, o xG. El propio Graham admite que el nombre no ayuda: «‘Expected Goals’ is a terrible name» (capítulo 1).

Hoy vamos a construir ese número con nuestras manos. Imagínense la dirección deportiva de un club que usa el Mundial de Qatar como vitrina. Necesita tres respuestas:

1. **Predecir.** ¿Qué probabilidad de gol tiene cada remate, según dónde y cómo se pateó?
2. **Explicar.** ¿Qué hace peligroso un remate?
3. **Decidir.** ¿A qué delanteros manda a observar el club: a los que más goles hicieron o a los que más ocasiones de gol tuvieron?

**El dato.** La empresa StatsBomb publica el evento a evento de los 64 partidos del Mundial de 2022. De ahí sacamos todos los remates.

**La respuesta.** $y_i=1$ si el remate $i$ fue gol. El modelo entrega $\hat p_i$, la probabilidad estimada de gol.

**La métrica.** Aquí importa el número mismo, porque lo vamos a sumar. Si un jugador tuvo remates con probabilidades $\hat p_1,\dots,\hat p_m$, su número esperado de goles es

$$\widehat{\text{xG}}=\sum_{i=1}^{m}\hat p_i.$$

Esa suma solo significa algo si las probabilidades están **calibradas**: de los remates a los que el modelo les da 10 %, cerca del 10 % debe ser gol. Por eso medimos con dos reglas de puntuación, que castigan una probabilidad según qué tan lejos queda de lo que pasó:

$$\text{Brier}=\frac{1}{n}\sum_{i}(\hat p_i-y_i)^2,\qquad \text{pérdida logarítmica}=-\frac{1}{n}\sum_{i}\bigl[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\bigr].$$

En las dos, menos es mejor.

### Predicción escrita

Antes de ver un solo dato, escriban aquí su apuesta.

- De cada cien remates en un Mundial, ¿cuántos son gol? **Mi apuesta:**
- Un remate desde el borde del área grande, de frente al arco, ¿qué probabilidad de gol tiene? **Mi apuesta:**
- ¿Qué es más fácil de convertir desde la misma distancia, un remate con el pie o un cabezazo? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

Una sola línea trae los remates de los 64 partidos, ya organizados en una tabla con una fila por remate. Tarda unos diez segundos.
'''),
("code", r'''
COMPETICION, TEMPORADA = 43, 106      # Mundial de Qatar 2022. En el gemelo usarán 2, 27: la Premier League 2015/2016
'''),
("code", r'''
remates = descargar("remates_mundial", competicion=COMPETICION, temporada=TEMPORADA)
print("Remates:", len(remates), "| Partidos:", remates["partido"].nunique(), "| Columnas:", remates.shape[1])
remates.head(3).T
'''),
("md", r'''
### 2.1 Las trampas del dato

**Primera trampa: los penales de las tandas.** StatsBomb divide cada partido en periodos. Los periodos 1 y 2 son los dos tiempos, el 3 y el 4 son la prórroga, y el 5 es la tanda de penales.
'''),
("code", r'''
por_periodo = remates.groupby("periodo")["gol"].agg(remates="size", goles="sum", tasa="mean")
por_periodo["tasa"] = (por_periodo["tasa"] * 100).round(1)
print(por_periodo.to_string())
'''),
("md", r'''
En la tanda, casi todos los remates son gol, porque todos son penales. Si se dejan, inflan la tasa de gol y le enseñan al modelo un remate que no existe durante el juego. **Decisión:** el periodo 5 sale.

**Segunda trampa: una columna que ya trae la respuesta de otro.** `statsbomb_xg` es el gol esperado que calcula la propia StatsBomb con su modelo. Si se la doy al mío, mi modelo aprende a copiar. **Decisión:** no entra como variable. La guardo para compararme con ella al final, que es para lo que sí sirve: es la referencia profesional.

**Tercera trampa: coordenadas que todavía no son variables.** Cada remate trae la posición $(x, y)$ en una cancha de 120 por 80 unidades, con el arco rival en $x=120$ y los palos en $y=36$ y $y=44$. Esas unidades son cercanas a una yarda. La posición cruda le dice poco a un modelo. Lo que importa es la geometría del remate: qué tan lejos está el arco y qué tan abierto se ve.

Sea $d_x=120-x$ y $d_y=|40-y|$. La distancia al centro del arco y el ángulo $\theta$ con que se ven los dos palos son

$$d=\sqrt{d_x^2+d_y^2},\qquad \theta=\arctan\!\left(\frac{8\,d_x}{d_x^2+d_y^2-16}\right),$$

donde el 8 es el ancho del arco y el 16 es el cuadrado de su mitad. Si el denominador es negativo, a $\theta$ se le suma $\pi$. Multiplico la distancia por 0,9144 para hablar en metros, sabiendo que es una aproximación.
'''),
("code", r'''
juego = remates[remates["periodo"] < 5].reset_index(drop=True)
dx, dy = 120 - juego["x"], (40 - juego["y"]).abs()

X = pd.DataFrame({
    "distancia_m": np.hypot(dx, dy) * 0.9144,
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
print("Tipo de remate de las filas con faltantes:", juego.loc[X["rivales_en_el_cono"].isna(), "tipo"].value_counts().to_dict())
'''),
("md", r'''
Comparen con su primera apuesta, y con el uno de cada diez de 1968. Tres de las variables salen de la «foto» que StatsBomb toma en el instante del remate, con la posición de todos los jugadores: cuántos rivales hay dentro del triángulo que forman el balón y los dos palos, si el arquero está dentro de ese triángulo y qué tan lejos está de su arco. La descarga las calcula por ustedes.

Esa foto es reciente. Graham cuenta que los primeros datos solo decían quién remató y desde dónde, de modo que un remate lejano valía lo mismo con el arquero en su sitio que con el arco vacío. StatsBomb empezó en 2017 a anotar dónde estaba cada jugador en cada remate (capítulo 7).

Los faltantes tienen explicación: los penales no traen foto. No es un faltante al azar, es un faltante por diseño, y se llena con la mediana de entrenamiento dentro de cada modelo. La variable `penal` le avisa al modelo que esos remates son otra cosa.

### 2.2 Conocer el dato antes de modelarlo

Cuatro miradas: dónde se remata, cómo cae la tasa de gol con la distancia, cómo sube con el ángulo y qué tanto cambia según el tipo de remate. Con un evento raro hay que mirar siempre cuántos casos hay detrás de cada tasa. El error estándar de una tasa $\hat p$ estimada con $n$ remates es $\sqrt{\hat p(1-\hat p)/n}$.
'''),
("code", r'''
def tasa_por(grupo):
    tabla = pd.DataFrame({"n": y.groupby(grupo, observed=True).size(), "tasa": y.groupby(grupo, observed=True).mean() * 100})
    tabla["ee"] = np.sqrt(tabla["tasa"] * (100 - tabla["tasa"]) / tabla["n"])
    return tabla

fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
eje = ejes[0, 0]
eje.scatter(juego.loc[y == 0, "x"], juego.loc[y == 0, "y"], s=8, color="#5BC0EB", alpha=0.35, label="no fue gol")
eje.scatter(juego.loc[y == 1, "x"], juego.loc[y == 1, "y"], s=22, color="#D7263D", label="gol")
eje.plot([120, 120], [36, 44], color="black", lw=4); eje.plot([102, 102, 120], [18, 62, 62], color="gray", lw=1); eje.plot([102, 120], [18, 18], color="gray", lw=1)
eje.set(xlim=(75, 121), ylim=(80, 0), title="Dónde se remató (el arco está a la derecha)", xlabel="x", ylabel="y"); eje.legend(fontsize=8)

por_distancia = tasa_por(pd.cut(X["distancia_m"], [0, 6, 9, 12, 16, 20, 25, 30, 60]))
ejes[0, 1].bar([str(i) for i in por_distancia.index], por_distancia["tasa"], yerr=2 * por_distancia["ee"], color="#1D4E89", ecolor="#1B998B")
ejes[0, 1].set(title="Tasa de gol según la distancia (m)", ylabel="%"); ejes[0, 1].tick_params(axis="x", rotation=30, labelsize=8)

por_angulo = tasa_por(pd.cut(X["angulo"], [0, 10, 15, 20, 30, 45, 180]))
ejes[1, 0].bar([str(i) for i in por_angulo.index], por_angulo["tasa"], yerr=2 * por_angulo["ee"], color="#1D4E89", ecolor="#1B998B")
ejes[1, 0].set(title="Tasa de gol según el ángulo con que se ve el arco (grados)", ylabel="%"); ejes[1, 0].tick_params(axis="x", rotation=30, labelsize=8)

clase = np.select([X["penal"] == 1, X["tiro_libre"] == 1, X["cabeza"] == 1, X["mano_a_mano"] == 1], ["penal", "tiro libre", "cabeza", "mano a mano"], "pie, en juego")
por_clase = tasa_por(pd.Series(clase, index=y.index)).sort_values("tasa")
ejes[1, 1].barh([f"{nombre} (n={int(n)})" for nombre, n in zip(por_clase.index, por_clase["n"])], por_clase["tasa"], xerr=2 * por_clase["ee"], color="#1B998B", ecolor="#1D4E89")
ejes[1, 1].set(title="Tasa de gol según la clase de remate", xlabel="%")
plt.tight_layout(); plt.show()
print(f"Mediana de la distancia: {X['distancia_m'].median():.1f} m | remates desde más de 20 m: {(X['distancia_m'] > 20).mean() * 100:.0f} %, con tasa de gol de {y[X['distancia_m'] > 20].mean() * 100:.1f} %")
'''),
("md", r'''
La tasa de gol se desploma con la distancia y sube con el ángulo. Las barras de error son anchas donde hay pocos remates, que es cerca del arco y en los penales. Ese es el problema de fondo de un evento raro: hay muchos remates, pero pocos goles, y lo que se aprende depende de los pocos.

### 2.3 Agrupar sin etiquetas: ¿cuántas clases de remate hay?

Antes de mirar el gol, le pido a un algoritmo que agrupe los remates solo por su geometría. $K$-medias busca $K$ centros $\mu_1,\dots,\mu_K$ que minimicen la distancia de cada remate a su centro más cercano:

$$\min_{\mu_1,\dots,\mu_K}\ \sum_{i=1}^{n}\ \min_{k}\ \lVert z_i-\mu_k\rVert^2,$$

con $z_i$ la distancia y el ángulo estandarizados. El algoritmo no ve si el remate fue gol. Después miramos la tasa de gol de cada grupo.
'''),
("code", r'''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

geometria = StandardScaler().fit_transform(X[["distancia_m", "angulo"]])
grupo = pd.Series(KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(geometria), index=X.index)
resumen_grupos = pd.DataFrame({"remates": grupo.value_counts(), "distancia media (m)": X.groupby(grupo)["distancia_m"].mean(),
                               "ángulo medio": X.groupby(grupo)["angulo"].mean(), "tasa de gol (%)": y.groupby(grupo).mean() * 100}).sort_values("distancia media (m)")

colores = dict(zip(resumen_grupos.index, ["#D7263D", "#1B998B", "#5BC0EB", "#1D4E89"]))
plt.figure(figsize=(7, 5))
plt.scatter(juego["x"], juego["y"], s=10, c=grupo.map(colores))
plt.plot([120, 120], [36, 44], color="black", lw=4); plt.xlim(75, 121); plt.ylim(80, 0); plt.title("Cuatro grupos de remates, armados sin mirar el gol"); plt.show()
display(resumen_grupos.round(1))
'''),
("md", r'''
Los grupos salen ordenados por cercanía, y sus tasas de gol son muy distintas aunque el algoritmo nunca vio el gol. Eso dice dos cosas. La geometría sola ya separa remates buenos de malos, así que un modelo sencillo va a llegar lejos. Y la mayoría de los remates cae en los grupos donde casi nunca hay gol.
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición y la trampa de la exactitud

Los partidos se ordenan por fecha y se parten así:

$$\underbrace{\text{primer } 60\,\%\ \text{de los partidos}}_{\text{entrenamiento}}\ \Big|\ \underbrace{20\,\%\ \text{siguiente}}_{\text{validación: se escoge}}\ \Big|\ \underbrace{\text{último } 20\,\%}_{\text{prueba: se reporta}}$$

Se parte por partido y no por remate porque los remates de un mismo partido comparten rival, arquero y cancha. La prueba queda con las rondas finales.
'''),
("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

partidos = juego["partido"].drop_duplicates().tolist()          # ya vienen en orden de fecha
corte_1, corte_2 = int(0.6 * len(partidos)), int(0.8 * len(partidos))
E = juego["partido"].isin(partidos[:corte_1]).values
V = juego["partido"].isin(partidos[corte_1:corte_2]).values
P = juego["partido"].isin(partidos[corte_2:]).values
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {juego['partido'][tramo].nunique():2d} partidos | {tramo.sum():4d} remates | {y[tramo].sum():3d} goles | tasa {y[tramo].mean() * 100:.1f} %")
'''),
("code", r'''
logistica = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[E], y[E])
prob = logistica.predict_proba(X[V])[:, 1]
real = y[V].values

def tres_medidas(real, dice):
    vp, fp, fn = ((dice == 1) & (real == 1)).sum(), ((dice == 1) & (real == 0)).sum(), ((dice == 0) & (real == 1)).sum()
    return {"exactitud": (dice == real).mean(), "sensibilidad": vp / (vp + fn), "precisión": vp / (vp + fp) if vp + fp else np.nan}

display(pd.DataFrame({"decir que ningún remate es gol": tres_medidas(real, np.zeros(len(real))),
                      "regresión logística, umbral 0,5": tres_medidas(real, prob >= 0.5)}).T.round(3))
print(f"Remates de validación: {len(real)} | goles: {real.sum()} | remates con probabilidad mayor que 0,5: {(prob >= 0.5).sum()}")
'''),
("md", r'''
Decir que ningún remate es gol acierta casi nueve de cada diez veces y no sirve para nada. El modelo, puesto a decir sí o no con umbral de 0,5, casi nunca dice sí, porque casi ningún remate tiene más de 50 % de probabilidad. Esa es la marca de un problema **desbalanceado**: la clase que interesa es rara, y la exactitud la ignora.

La salida que uno encuentra primero en internet es «balancear las clases». La vamos a probar en la sección 3.4. Antes hay que cambiar la pregunta. Al club no le sirve que le digan gol o no gol. Le sirve la probabilidad, y por eso el modelo se mide con el puntaje de Brier y con la pérdida logarítmica de la sección 1.

### 3.2 La línea base

Dos referencias que no necesitan aprendizaje automático. La primera le da a todo remate la misma probabilidad, la tasa de gol del entrenamiento. La segunda es una regresión logística con una sola variable, la distancia.
'''),
("code", r'''
def puntuar(real, prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return {"pérdida logarítmica": log_loss(real, prob), "Brier": brier_score_loss(real, prob),
            "AUC": roc_auc_score(real, prob) if len(np.unique(prob)) > 1 else 0.5, "goles que predice": prob.sum(), "goles que hubo": int(np.sum(real))}

solo_distancia = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
referencias_validacion = {
    "tasa base": np.full(V.sum(), y[E].mean()),
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
La tasa base tiene un AUC de 0,5, porque no ordena nada, pero predice bien el **total** de goles. Esa propiedad vale mucho y es fácil perderla. Guárdenla para la sección 3.4.

Estas dos referencias son los primeros peldaños de una escalera que Graham describe en el capítulo 7. El modelo más pobre le da a todo remate el mismo 10 %. Su primer modelo en el Tottenham solo distinguía tres zonas de la cancha. Cada versión posterior aleja las probabilidades de ese promedio. Lo que sigue es subir la escalera y medir cuánto se gana en cada peldaño.

### 3.3 El catálogo de modelos

Esta es la celda del cuaderno que más les conviene aprender a tocar. `MODELOS` es un diccionario: a la izquierda va el nombre que ustedes le quieran poner al modelo y a la derecha el modelo de scikit-learn, con sus opciones. La función `tuberia` le pone delante a cada modelo los dos pasos que todos necesitan: llenar los faltantes con la mediana y, si el modelo lo pide, estandarizar las variables. Todo lo que viene después recorre el diccionario, así que agregar un modelo es agregar una línea.

| Familia | Modelos del catálogo | Idea en una línea |
|---|---|---|
| Lineales | logística, logística apretada, discriminante lineal | $\log\dfrac{p_i}{1-p_i}=\beta_0+x_i^\top\beta$, con distintas maneras de estimar $\beta$ |
| Probabilísticos | Bayes ingenuo | Supone que las variables son independientes dentro de cada clase |
| De vecindad | $k$ vecinos | La fracción de goles entre los $k$ remates de entrenamiento más parecidos |
| De margen y redes | máquina de soporte, red neuronal | Fronteras curvas entre goles y no goles |
| Un árbol | árbol | Preguntas de sí o no que separan grupos con tasas de gol distintas |
| Muchos árboles en paralelo | bosque, árboles extra | El promedio de muchos árboles distintos |
| Muchos árboles en fila | boosting, boosting clásico, AdaBoost | Una suma de árboles pequeños, cada uno corrigiendo a los anteriores |

Gana quien tenga menor pérdida logarítmica en **validación**, y se reporta lo que haga en **prueba**. Uso 150 vecinos y hojas grandes a propósito. Con un evento raro, un vecindario pequeño o una hoja pequeña pueden no tener ningún gol, y el modelo diría probabilidad cero. Los remates lejanos entran cerca de 2 de cada 100 veces, así que un vecindario de 150 espera unos tres goles y uno de 40 espera menos de uno. Una probabilidad de cero que resulta ser gol recibe un castigo enorme en la pérdida logarítmica, y con razón, porque es afirmar que algo es imposible.

**Su turno.** Agreguen al menos tres modelos al diccionario y vuelvan a correr desde esta celda. Pueden repetir uno con otras opciones (por ejemplo, un bosque con hojas más pequeñas) o traer uno nuevo de la [lista de scikit-learn](https://scikit-learn.org/stable/supervised_learning.html). La única condición es que el modelo sepa dar probabilidades, es decir, que tenga `predict_proba`.

**Predicción escrita:** de los trece, ¿cuál creen que gana en validación? ¿Será el mismo que gane en la prueba? **Mi apuesta:**
'''),
("code", r'''
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)

def tuberia(modelo, estandarizar=True):
    """Llena los faltantes con la mediana, estandariza si el modelo lo necesita y pone el modelo al final."""
    pasos = [SimpleImputer(strategy="median")] + ([StandardScaler()] if estandarizar else []) + [modelo]
    return make_pipeline(*pasos)

MODELOS = {
    "logística":            tuberia(LogisticRegression(max_iter=2000)),
    "logística apretada":   tuberia(LogisticRegression(C=0.05, max_iter=2000)),              # C pequeño: coeficientes más encogidos
    "discriminante lineal": tuberia(LinearDiscriminantAnalysis()),
    "Bayes ingenuo":        tuberia(GaussianNB()),
    "k vecinos":            tuberia(KNeighborsClassifier(n_neighbors=150)),
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
La celda siguiente es el protocolo, y no hay que cambiarle nada. Toma cada modelo del catálogo, lo entrena con el tramo de entrenamiento, lo mide en validación, lo vuelve a entrenar con entrenamiento y validación juntos, y lo mide en la prueba.
'''),
("code", r'''
for nombre, receta in MODELOS.items():
    modelo = clone(receta)                               # una copia sin entrenar, para no ensuciar el catálogo
    modelo.fit(X[E], y[E])
    en_validacion = puntuar(y[V], modelo.predict_proba(X[V])[:, 1])["pérdida logarítmica"]
    modelo.fit(X[E | V], y[E | V])
    probabilidades[nombre] = modelo.predict_proba(X[P])[:, 1]
    errores[nombre] = {"validación": en_validacion, **puntuar(y[P], probabilidades[nombre])}
todos = pd.DataFrame(errores).T

todos[["validación", "pérdida logarítmica"]].rename(columns={"pérdida logarítmica": "prueba"}).sort_values("validación").plot.barh(figsize=(9, 0.42 * len(todos) + 1.5), color=["#1D4E89", "#1B998B"])
plt.xlabel("pérdida logarítmica (menos es mejor)"); plt.title(f"{len(todos)} candidatos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(4))
'''),
("md", r'''
### Una confesión sobre los 150 vecinos

El número de vecinos de la celda anterior no salió de la validación. En la preparación de esta clase el primer intento fue con 75, y pasó lo que muestra la celda siguiente. Lo dejo a la vista porque es la lección más fina del día.
'''),
("code", r'''
vecinos_75 = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=75))
vecinos_75.fit(X[E], y[E])
validacion_75 = puntuar(y[V], vecinos_75.predict_proba(X[V])[:, 1])["pérdida logarítmica"]
vecinos_75.fit(X[E | V], y[E | V])
p_75 = vecinos_75.predict_proba(X[P])[:, 1]
prueba_75 = puntuar(y[P], p_75)["pérdida logarítmica"]

print(f"k vecinos con 75  | validación: {validacion_75:.4f} | prueba: {prueba_75:.4f}")
print(f"k vecinos con 150 | validación: {todos.loc['k vecinos', 'validación']:.4f} | prueba: {todos.loc['k vecinos', 'pérdida logarítmica']:.4f}")
print(f"Mejor de todos en validación: {todos['validación'].min():.4f} ({todos['validación'].idxmin()})")
goles = np.asarray(y[P]) == 1
print(f"Goles de la prueba a los que el modelo de 75 les dio probabilidad cero: {int((p_75[goles] == 0).sum())} de {int(goles.sum())}")
'''),
("md", r'''
Lean los cuatro renglones juntos. Con 75 vecinos el modelo puede quedar muy bien en validación y caer en la prueba. La causa está en el último renglón: basta un remate que terminó en gol y recibió probabilidad cero, porque ninguno de sus 75 vecinos había sido gol. La pérdida logarítmica castiga ese error más que cualquier otro, ya que $-\log\hat p$ crece sin límite cuando $\hat p$ se acerca a cero y el evento ocurre.

Subir a 150 vecinos corrige el síntoma, pero la decisión se tomó después de mirar la prueba, y eso es justo lo que el protocolo prohíbe. Por eso va a la bitácora como lo que fue. La enseñanza es doble. Con un evento raro, ganar la validación no garantiza nada. Y un modelo que puede decir «imposible» es peligroso cuando su salida se va a usar como probabilidad.
'''),
("code", r'''
mejor = todos["validación"].idxmin()
elegido = mejor if todos.loc[mejor, "pérdida logarítmica"] < todos.loc["solo distancia", "pérdida logarítmica"] else "solo distancia"
referencia = puntuar(y[P], juego["statsbomb_xg"][P])
print(f"Ganador en validación: {mejor}")
print(f"En prueba: pérdida logarítmica {todos.loc[mejor, 'pérdida logarítmica']:.4f} | Brier {todos.loc[mejor, 'Brier']:.4f} | predice {todos.loc[mejor, 'goles que predice']:.1f} goles y hubo {int(todos.loc[mejor, 'goles que hubo'])}")
print(f"Solo distancia: {todos.loc['solo distancia', 'pérdida logarítmica']:.4f} | tasa base: {todos.loc['tasa base', 'pérdida logarítmica']:.4f}")
print(f"El modelo de StatsBomb, en los mismos remates: pérdida logarítmica {referencia['pérdida logarítmica']:.4f} | Brier {referencia['Brier']:.4f} | predice {referencia['goles que predice']:.1f} goles")
print(f"Método con el que seguimos: {elegido}")
'''),
("md", r'''
La regla es la de la clase 1: si el ganador de validación no le gana en la prueba a la línea base de la distancia, seguimos con la línea base, que es más simple.

Lean los tres renglones. La distancia sola ya recorre buena parte del camino desde la tasa base. En el Mundial de 2022 el ganador mejora un poco más. Y el modelo de StatsBomb, que es el producto de una empresa que vive de esto y que se entrenó con muchos más remates que los de un Mundial, queda por delante. La distancia entre el nuestro y el de ellos es lo que cuestan más datos y más variables.

### Con tantos candidatos, ¿cuánto del ganador es azar?

Ordeno los modelos del catálogo dos veces, por su puesto en validación y por su puesto en la prueba. Si la validación sirviera para escoger sin error, los puntos caerían sobre la diagonal.
'''),
("code", r'''
del_catalogo = todos.loc[list(MODELOS)]
puestos = del_catalogo[["validación", "pérdida logarítmica"]].rank().rename(columns={"pérdida logarítmica": "prueba"})
plt.figure(figsize=(6.5, 6))
plt.plot([1, len(puestos)], [1, len(puestos)], color="#9AA5B1", ls=":")
plt.scatter(puestos["validación"], puestos["prueba"], s=60, color="#1D4E89", zorder=3)
for nombre, (pv, pp) in puestos.iterrows():
    plt.annotate(nombre, (pv, pp), xytext=(6, 4), textcoords="offset points", fontsize=8)
plt.scatter(*puestos.loc[mejor], s=140, facecolors="none", edgecolors="#D7263D", lw=2, zorder=4)
plt.xlabel("puesto en validación (1 es el mejor)"); plt.ylabel("puesto en prueba"); plt.title("¿El orden de validación se sostiene en la prueba?")
plt.gca().invert_xaxis(); plt.gca().invert_yaxis(); plt.show()
separacion = del_catalogo["validación"].sort_values()
print(f"Correlación de Spearman entre los dos órdenes: {puestos['validación'].corr(puestos['prueba'], method='spearman'):.2f}")
print(f"Goles en validación: {int(y[V].sum())} | distancia en validación entre el primero y el quinto: {separacion.iloc[4] - separacion.iloc[0]:.4f}")
print(f"Puesto en la prueba del ganador de validación ({mejor}): {int(puestos.loc[mejor, 'prueba'])} de {len(puestos)}")
'''),
("md", r'''
El círculo rojo es el ganador de validación. Respondan con la gráfica y los tres renglones a la vista.

- ¿El ganador de validación fue también el mejor en la prueba? **Mi lectura:**
- La validación tiene unas pocas decenas de goles. Si uno solo de esos goles hubiera pegado en el palo, ¿cambiaría el orden de los cinco primeros? **Mi respuesta:**
- Cuando ustedes agreguen modelos, el mejor puntaje de validación solo puede mejorar o quedarse igual. ¿Eso quiere decir que el modelo final es mejor? **Mi respuesta:**

Con muchos candidatos y pocos goles, el primer puesto de validación tiene una parte de mérito y una parte de suerte. Es la misma idea con la que Graham mira a los futbolistas: de un delantero con cuatro goles en cuatro partidos se sabe menos que de uno con treinta en cuarenta (capítulo 5). En el gemelo van a repetir esta gráfica con una liga entera, que tiene diez veces más goles, y van a ver qué le pasa.

### 3.4 La trampa de la clase: balancear rompe la probabilidad

Volvamos a la receta de internet. `class_weight="balanced"` le dice al modelo que cada gol pese tanto como todos los no goles juntos. Es lo mismo que repetir los goles hasta que haya tantos como no goles.

**Predicción escrita:** si entreno con las clases balanceadas, ¿cuántos goles predecirá el modelo en la prueba, donde hubo unas pocas decenas? **Mi apuesta:**
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

# No todos los modelos del catálogo saben balancear. Si el elegido no sabe, la demostración se hace con la logística.
sabe_balancear = elegido == "solo distancia" or "class_weight" in MODELOS[elegido].steps[-1][1].get_params()
demostracion = elegido if sabe_balancear else "logística"
(normal, columnas_demo), (balanceado, _) = armar(demostracion), armar(demostracion, peso="balanced")
normal.fit(X[columnas_demo][E], y[E]); balanceado.fit(X[columnas_demo][E], y[E])
p_normal, p_balanceado = normal.predict_proba(X[columnas_demo][P])[:, 1], balanceado.predict_proba(X[columnas_demo][P])[:, 1]
columnas = armar(elegido)[1]

# Recalibración de Platt: una regresión logística de una variable, ajustada en validación, que corrige la escala.
platt = LogisticRegression().fit(logit(balanceado.predict_proba(X[columnas_demo][V])[:, 1]).reshape(-1, 1), y[V])
p_recalibrado = platt.predict_proba(logit(p_balanceado).reshape(-1, 1))[:, 1]

trampa = pd.DataFrame({"sin balancear": puntuar(y[P], p_normal), "balanceado": puntuar(y[P], p_balanceado),
                       "balanceado y recalibrado": puntuar(y[P], p_recalibrado)}).T
trampa["sensibilidad con umbral 0,5"] = [(p[y[P] == 1] >= 0.5).mean() for p in (p_normal, p_balanceado, p_recalibrado)]
print("Modelo:", demostracion, "(entrenado solo con el tramo de entrenamiento, para dejar la validación libre)")
display(trampa.round(4))
'''),
("md", r'''
Miren primero el AUC. Casi no cambia: el modelo balanceado **ordena** los remates igual de bien. Miren después los goles que predice. El balanceo le enseñó que la mitad de los remates son gol, y todas sus probabilidades quedaron infladas. La sensibilidad con umbral de 0,5 sube, y eso es lo que celebra quien recomienda la receta. Pero la suma ya no significa nada, y la pérdida logarítmica y el Brier lo castigan.

Hay una cuenta que lo explica. Si en el dato original la razón entre goles y no goles es $\pi/(1-\pi)$ y el balanceo la vuelve 1, las probabilidades quedan multiplicadas en la escala de la razón de probabilidades:

$$\frac{p_{\text{bal}}}{1-p_{\text{bal}}}\approx\frac{p}{1-p}\cdot\frac{1-\pi}{\pi}.$$

Con $\pi$ cercano a 0,11, ese factor es cercano a 8. La tercera fila deshace el daño con una **recalibración**: una regresión logística de una sola variable, ajustada con la validación, que devuelve las probabilidades a su escala. Funciona, pero es un rodeo para llegar adonde ya estaba el modelo sin balancear.

La regla que queda: balancear cambia el umbral, no la calidad del orden. Si lo que se necesita es una probabilidad, no se balancea. Si se necesita una decisión de sí o no, es más limpio dejar la probabilidad quieta y mover el umbral.

### 3.5 Calibración: ¿se le puede creer al número?

Agrupo los remates de la prueba en tramos según la probabilidad que les dio el modelo. En cada tramo $b$ comparo la probabilidad promedio $\bar p_b$ con la fracción real de goles $\bar y_b$. Un modelo calibrado cae sobre la diagonal. El resumen es el error de calibración esperado:

$$\text{ECE}=\sum_{b}\frac{n_b}{n}\,\bigl|\bar y_b-\bar p_b\bigr|.$$
'''),
("code", r'''
def calibracion(real, prob, tramos=5):
    cortes = pd.qcut(pd.Series(prob), tramos, duplicates="drop")
    tabla = pd.DataFrame({"prob": np.asarray(prob), "real": np.asarray(real)}).groupby(cortes.values, observed=True).agg(n=("real", "size"), predicha=("prob", "mean"), observada=("real", "mean"))
    return tabla, float((tabla["n"] / tabla["n"].sum() * (tabla["observada"] - tabla["predicha"]).abs()).sum())

curvas = {"sin balancear": p_normal, "balanceado": p_balanceado, "StatsBomb": juego["statsbomb_xg"][P].values}
plt.figure(figsize=(6.5, 5.5))
for (nombre, prob), color in zip(curvas.items(), ["#1D4E89", "#D7263D", "#5BC0EB"]):
    tabla, ece = calibracion(y[P], prob)
    plt.plot(tabla["predicha"], tabla["observada"], marker="o", color=color, label=f"{nombre} (ECE {ece:.3f})")
plt.plot([0, 1], [0, 1], color="gray", ls=":"); plt.xlim(0, 0.8); plt.ylim(0, 0.8)
plt.xlabel("probabilidad promedio que dio el modelo"); plt.ylabel("fracción de goles que hubo"); plt.title("Curva de calibración en la prueba"); plt.legend(); plt.show()
display(calibracion(y[P], p_normal)[0].round(3))
'''),
("md", r'''
La curva del modelo balanceado queda muy por debajo de la diagonal: promete mucho más de lo que pasa. Las otras dos andan cerca. La tabla recuerda cuántos remates hay en cada tramo, y por eso los puntos de la derecha, que son los de pocos remates, bailan más.

### 3.6 Explicar: ¿qué hace peligroso un remate?

Con las variables estandarizadas, $e^{\beta_j}$ es la razón de probabilidades por cada desviación estándar de la variable $j$, con las demás quietas. La segunda gráfica muestra la probabilidad que el modelo elegido le da a un remate de pie, de frente al arco y en juego, según la distancia.
'''),
("code", r'''
explicativo = clone(MODELOS["logística"]).fit(X[E | V], y[E | V])
razones = pd.Series(np.exp(explicativo[-1].coef_[0]), index=variables).sort_values()

modelo_final = armar(elegido)[0].fit(X[columnas][E | V], y[E | V])
distancias = np.arange(4, 36, 1.0)
de_frente = pd.DataFrame({"distancia_m": distancias, "angulo": np.degrees(np.arctan2(8 * distancias / 0.9144, (distancias / 0.9144) ** 2 - 16))})
for variable in variables[2:]:
    de_frente[variable] = X[variable][E | V].median()
de_frente["cabeza"] = 0; de_frente["penal"] = 0; de_frente["tiro_libre"] = 0

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.2))
razones.plot.barh(ax=izq, color=["#5BC0EB" if r < 1 else "#D7263D" for r in razones]); izq.axvline(1, color="gray")
izq.set(title="Razón de probabilidades por desviación estándar", xlabel="menor que 1 baja la probabilidad de gol; mayor que 1 la sube")
der.plot(distancias, modelo_final.predict_proba(de_frente[columnas])[:, 1] * 100, color="#D7263D", label="de pie")
der.plot(distancias, modelo_final.predict_proba(de_frente.assign(cabeza=1)[columnas])[:, 1] * 100, color="#5BC0EB", label="de cabeza")
der.axvline(16.5, color="gray", ls=":"); der.text(16.8, der.get_ylim()[1] * 0.9, "borde del área", fontsize=8, color="gray")
der.set(title=f"Probabilidad de gol de frente al arco ({elegido})", xlabel="distancia (m)", ylabel="%"); der.legend()
plt.tight_layout(); plt.show()
print(razones.round(2).to_string())
'''),
("md", r'''
Comparen con sus apuestas segunda y tercera. Si el modelo elegido es de árboles, la curva de la derecha sale con escalones, porque un árbol predice por tramos. Dos advertencias para leer la gráfica de la izquierda.

La distancia y el ángulo van juntos: casi todo remate cercano tiene el arco abierto. La regresión reparte entre las dos un efecto que comparten, y ninguno de los dos coeficientes se puede leer solo.

Y lo que se ve es asociación. Que los remates bajo presión entren menos no dice que el jugador falle por la presión. Puede decir que solo se remata bajo presión cuando no había nada mejor.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿a quién mandar a observar?

Graham cuenta sus errores con la misma calma que sus aciertos. Uno de los primeros fue Giovani dos Santos. Lo recomendó al Tottenham porque el modelo lo ponía muy arriba, pero lo había visto jugar unos 1.400 minutos, cerca de quince partidos, y no advirtió lo poco que era eso (capítulo 2). Un delantero que juega los siete partidos de un Mundial no llega ni a la mitad de esos minutos. Esa es la cuenta que hay que tener en la cabeza durante toda esta sección.

Para juzgar a un delantero no sirve el modelo que ya vio sus remates. Uso predicciones **fuera de muestra**: parto los partidos en ocho grupos, y la probabilidad de cada remate la calcula un modelo entrenado con los otros siete. Así cada remate recibe un número de un modelo que no lo conocía.

Con esas probabilidades, el gol esperado de un jugador es la suma de la sección 1. Dejo por fuera los penales, como hace Graham al comparar a Messi con Ronaldo (capítulo 12), porque un penal dice más de quién lo provocó que de quién lo patea. Para saber si los goles de un jugador se salen de lo esperable, simulo: cada remate entra con su probabilidad, diez mil veces, y miro entre qué valores cae el total el 90 % de las veces. Si los goles reales caen dentro de esa franja, la diferencia con el gol esperado se puede explicar por la suerte de un torneo corto.
'''),
("code", r'''
from sklearn.model_selection import GroupKFold

fuera_de_muestra = np.zeros(len(juego))
for entrena, prueba in GroupKFold(n_splits=8).split(X, y, juego["partido"]):
    fuera_de_muestra[prueba] = armar(elegido)[0].fit(X[columnas].iloc[entrena], y.iloc[entrena]).predict_proba(X[columnas].iloc[prueba])[:, 1]
juego["xg"] = fuera_de_muestra
total = puntuar(y, fuera_de_muestra)
print(f"Todo el torneo, fuera de muestra: predice {total['goles que predice']:.1f} goles y hubo {total['goles que hubo']} | pérdida logarítmica {total['pérdida logarítmica']:.4f} | StatsBomb: {puntuar(y, juego['statsbomb_xg'])['pérdida logarítmica']:.4f}")

sin_penales = juego[juego["tipo"] != "Penalty"]
sorteo = np.random.default_rng(0)
def franja(probabilidades_jugador):
    simulados = (sorteo.random((10000, len(probabilidades_jugador))) < probabilidades_jugador.values).sum(axis=1)
    return np.percentile(simulados, [5, 95])

jugadores = sin_penales.groupby(["jugador", "equipo"]).agg(remates=("gol", "size"), goles=("gol", "sum"), xg=("xg", "sum")).reset_index()
jugadores["xg por remate"] = jugadores["xg"] / jugadores["remates"]
jugadores["goles menos xg"] = jugadores["goles"] - jugadores["xg"]
franjas = {nombre: franja(sin_penales.loc[sin_penales["jugador"] == nombre, "xg"]) for nombre in jugadores["jugador"]}
jugadores["franja de 90 %"] = [f"{int(franjas[n][0])} a {int(franjas[n][1])}" for n in jugadores["jugador"]]
jugadores["dentro de la franja"] = [franjas[n][0] <= g <= franjas[n][1] for n, g in zip(jugadores["jugador"], jugadores["goles"])]

por_goles = jugadores.sort_values(["goles", "xg"], ascending=False).head(8)
por_ocasiones = jugadores.sort_values("xg", ascending=False).head(8)
print("\nLista A. Los que más goles hicieron (sin penales):"); display(por_goles.round(2).reset_index(drop=True))
print("Lista B. Los que más ocasiones tuvieron (gol esperado, sin penales):"); display(por_ocasiones.round(2).reset_index(drop=True))
en_las_dos = sorted(set(por_goles["jugador"]) & set(por_ocasiones["jugador"]))
print(f"En las dos listas: {len(en_las_dos)} de 8 | goleadores de la lista A dentro de su franja: {int(por_goles['dentro de la franja'].sum())} de 8")
'''),
("md", r'''
La lista A ordena por goles y la lista B por ocasiones. No coinciden del todo. Miren también lo anchas que son las franjas: con quince o veinte remates, hacer dos goles o hacer seis puede ser el mismo jugador con distinta suerte.

Hay una segunda manera de mirar a un delantero, que tampoco es contar goles. Dos jugadores pueden sumar el mismo gol esperado por caminos opuestos, uno rematando mucho desde lejos y otro rematando poco desde cerca. Graham lo ilustra con Coutinho, que tenía talento para el remate lejano y aun así pateaba, casi siempre, remates de baja probabilidad (capítulo 7). Y recuerda que un remate suele terminar la jugada, así que cada intento desde lejos le cuesta una opción a un compañero (capítulo 8).
'''),
("code", r'''
frecuentes = jugadores[jugadores["remates"] >= 8]
plt.figure(figsize=(9, 5.5))
plt.scatter(frecuentes["remates"], frecuentes["xg por remate"] * 100, s=30 + 60 * frecuentes["goles"], color="#1D4E89", alpha=0.55)
for _, fila in frecuentes.sort_values("xg", ascending=False).head(10).iterrows():
    plt.annotate(fila["jugador"], (fila["remates"], fila["xg por remate"] * 100), xytext=(6, 4), textcoords="offset points", fontsize=7)
plt.axhline(sin_penales["xg"].mean() * 100, color="#9AA5B1", ls=":"); plt.text(frecuentes["remates"].max() * 0.62, sin_penales["xg"].mean() * 100, "promedio del torneo", fontsize=8, color="#9AA5B1", va="bottom")
plt.xlabel("remates en el torneo (volumen)"); plt.ylabel("gol esperado por remate, % (calidad)")
plt.title("Volumen contra calidad: jugadores con ocho remates o más\nEl tamaño del punto son los goles"); plt.show()
'''),
("md", r'''
- ¿Qué jugador remata mucho y desde posiciones pobres? ¿Cuál remata poco y desde posiciones muy buenas? **Mi lectura:**
- Si su club necesita gol, ¿a cuál de los dos mandan a observar primero? **Mi respuesta:**

**Lo que dice el libro y lo que no.** El argumento para la dirección deportiva es que las ocasiones se repiten de un torneo a otro más que la puntería, y que un club paga por lo que se repite. Graham lo respalda con equipos: revisó nueve temporadas de la Bundesliga, y de 69 equipos que sacaron más puntos de los que merecían sus ocasiones, 34 sacaron menos al año siguiente (capítulo 4). Pero no dice que la puntería sea pura suerte. Afirma que la habilidad para definir existe, y que en el Liverpool cada jugador tenía una estimación propia de esa habilidad (capítulo 7). Tampoco cuenta que el Liverpool fichara mirando la diferencia entre goles y gol esperado. Su herramienta principal medía todo lo que un jugador hace con el balón, y a Firmino y a Mané los ficharon por lo que hacían además de marcar (capítulos 1, 4 y 8). El ejercicio de hoy, pasar del gol esperado de un torneo a una lista de fichajes, es nuestro. Este cuaderno no puede comprobar que las ocasiones se repiten, porque tiene un solo torneo. Va como supuesto en la decisión y como pregunta en los riesgos.

Un detalle de cómo trabajaban, para que lo copien. Los datos reducían miles de jugadores a una docena, y a cada uno de esa docena le veían entre veinte y treinta partidos en video (capítulos 2 y 8). La lista que sale de este cuaderno es eso: a quién vale la pena ir a ver.

### La decisión en tres frases

Escríbanla para un director deportivo que sabe de fútbol y no sabe qué es una pérdida logarítmica. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

**Premortem.** Antes de fichar a un delantero, en el Liverpool hacían un ejercicio: imaginar que el fichaje ya había fracasado y preguntarse por qué (capítulo 4). Háganlo con el primer nombre de su lista. Ha pasado un año y el jugador no rindió. Escriban la razón más probable, y digan si una tabla de remates la habría podido ver.

4. **Premortem:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Fuera el periodo 5 | Dejar las tandas de penales | No son remates del juego e inflan la tasa de gol |
| `statsbomb_xg` no es variable de entrada | Usarla | Es la salida de otro modelo: sirve de referencia, no de insumo |
| Distancia y ángulo en vez de $x$ e $y$ | Coordenadas crudas | La geometría del remate es lo que importa |
| Partición por partido y por fecha | Por remate, al azar | Los remates de un partido comparten rival y arquero |
| Pérdida logarítmica para escoger | Exactitud | Con 11 % de goles la exactitud premia decir siempre que no |
| 150 vecinos y hojas grandes | Valores por defecto; 75 vecinos | Un vecindario sin goles da probabilidad cero. El 150 se fijó después de ver caer al de 75 en la prueba, y esa es una falta contra el protocolo que queda registrada |
| No balancear las clases | `class_weight="balanced"` | Infla las probabilidades y la suma deja de ser un número de goles |
| Trece modelos en el catálogo | Escoger entre dos o tres | Comparar es barato; el costo es que el ganador de validación tiene una parte de suerte, y se mide con la gráfica de puestos |
| Predicciones fuera de muestra para juzgar jugadores | Las del modelo final | Un modelo no debe calificar los remates con que se entrenó |
| Penales fuera de la comparación entre jugadores | Contarlos | Un penal dice más de quién lo provocó que de quién lo patea |
| Franja por simulación | Comparar goles con xG a ojo | Con pocos remates la suerte pesa mucho |

### 4.3 Riesgos y mantenimiento

Respondan en dos o tres líneas cada una.

- El modelo se entrenó con unos mil remates y unos cien goles. ¿En qué clase de remate confían menos en su probabilidad? Miren la curva de calibración.
- El modelo no sabe quién patea. ¿A qué jugadores subestima de forma sistemática un gol esperado que trata igual a todos?
- Un Mundial son siete partidos por equipo, a lo sumo. ¿Qué habría que tener para comprobar que las ocasiones se repiten más que los goles?
- ¿Qué pasaría con sus conclusiones si StatsBomb cambiara la manera de anotar la posición del arquero?
- Si el club usara este número para fijar el salario de un jugador, ¿qué le exigirían al modelo que hoy no le exigen?

### 4.4 Procedencia
'''),
("code", r'''
print("Fuente: StatsBomb Open Data (github.com/statsbomb/open-data). Se cita a StatsBomb y no se redistribuye el dato.")
procedencia()["remates_mundial"]
'''),
("md", r'''
### 4.5 Lo que sigue: el gemelo

En la segunda mitad de la clase, cada uno repite este camino con la **Premier League 2015/2016**, la temporada en que el Leicester City salió campeón contra todos los pronósticos. Son 380 partidos en vez de 64, y cerca de diez veces más goles. Al final cambiarán una línea para correr lo mismo con la Copa América de 2024.

Lleven estas tres preguntas al gemelo. ¿Los modelos que hoy quedaron arriba siguen arriba con una liga entera? ¿La gráfica de puestos se acerca a la diagonal cuando hay más goles? ¿Qué tan anchas quedan las franjas de un delantero que remató más de cien veces?
'''),
]

# ------------------------------------------------------------ LAS CALLES DE BOGOTÁ
CALLES = [
("md", r'''
# Clase 4 · Las calles de Bogotá (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *El Mundial de Qatar*: la misma técnica en otro mundo.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase con un evento más raro y más serio que un gol: modelar la probabilidad de que un siniestro vial sea mortal, medirla con reglas de puntuación, revisar su calibración y usar la suma de probabilidades para decidir dónde intervenir. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que les ayude a llenar un hueco o que interprete un error. Deben poder explicar sin ayuda por qué se quitan los siniestros de solo daños, por qué la exactitud no sirve aquí, qué le hace el balanceo a la probabilidad, cómo se lee la curva de calibración y con qué criterio escogieron los cruces. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, la decisión en tres frases, la bitácora y la tabla de la sección 4.5 llenas. |

**Cómo se trabaja.** El código está escrito casi todo. Donde vean `____` falta una pieza, y el comentario de al lado dice qué va. Lean la celda entera antes de llenar. Si una celda falla, lean el error antes de pedir ayuda.
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

La Secretaría Distrital de Movilidad de Bogotá publica cada siniestro vial que registra la policía de tránsito: dónde fue, a qué hora, qué clase de siniestro fue, quiénes estaban involucrados y qué tan grave resultó. Detrás de cada fila hay personas, y detrás de algunas, una muerte. Conviene trabajar este dato con ese cuidado.

Imagínense el equipo de seguridad vial de la Secretaría. Tiene recursos para poner controles de velocidad y cruces seguros en cien puntos de la ciudad. Necesita tres respuestas:

1. **Predecir.** Cuando ocurre un siniestro con víctimas, ¿qué probabilidad hay de que sea mortal?
2. **Explicar.** ¿Qué condiciones acompañan a los siniestros mortales?
3. **Decidir.** ¿En qué cien cruces se ponen primero los controles?

**La respuesta.** $y_i=1$ si el siniestro $i$ dejó al menos una persona muerta, contando las muertes de los treinta días siguientes. El modelo entrega $\hat p_i$.

**La métrica.** Como en clase, lo que importa es la probabilidad, porque se va a sumar. El número esperado de siniestros mortales en un cruce $c$ es

$$\hat m_c=\sum_{i\in c}\hat p_i,$$

y esa suma solo vale si las probabilidades están calibradas. Medimos con el puntaje de Brier y con la pérdida logarítmica:

$$\text{Brier}=\frac{1}{n}\sum_i(\hat p_i-y_i)^2,\qquad \text{pérdida logarítmica}=-\frac{1}{n}\sum_i\bigl[y_i\log\hat p_i+(1-y_i)\log(1-\hat p_i)\bigr].$$

### Predicción escrita

Escriban su apuesta antes de correr nada.

- De cada cien siniestros con víctimas en Bogotá, ¿cuántos son mortales? **Mi apuesta:**
- ¿Qué es más letal: un siniestro con peatón, con motocicleta o con vehículo de carga? **Mi apuesta:**
- Cien cruces son una fracción mínima de la ciudad. ¿Qué porcentaje de los siniestros mortales futuros ocurrirá en los cien cruces que ustedes escojan? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

La fuente es un archivo de Excel de 127 MB con diez años de siniestros. **La descarga tarda cerca de un minuto y medio.** Córranla y lean la sección mientras termina.
'''),
("code", r'''
ACTOR = "todos"              # la población del análisis. Al final lo cambiarán por "Con_Peaton"
CONTROLES = 100              # cuántos cruces alcanza a intervenir la Secretaría
'''),
("code", r'''
crudo = descargar("siniestros_bogota")
print("Filas:", len(crudo), "| Columnas:", crudo.shape[1], "| ¿Algún siniestro repetido?", crudo["Codigo_Accidente"].duplicated().any())
crudo.head(3).T
'''),
("md", r'''
### 2.1 Las trampas del dato

**Primera trampa: una fecha que es un número.** Miren `Fecha_Acc`. Excel guarda las fechas como el número de días transcurridos desde el 30 de diciembre de 1899, y al leer el archivo llega ese número.
'''),
("code", r'''
crudo["fecha"] = pd.to_datetime(«"1899-12-30"») + pd.to_timedelta(crudo["Fecha_Acc"], unit="D")      # HUECO 1: el día cero de Excel, entre comillas, como año-mes-día
print("Primer siniestro:", crudo["fecha"].min().date(), "| último:", crudo["fecha"].max().date())
print("Filas en que el año de la fecha no coincide con la columna AA_Acc:", int((crudo["fecha"].dt.year != crudo["AA_Acc"]).sum()))
'''),
("md", r'''
**Compruébenlo ustedes mismos.** Las fechas deben ir de 2015 a 2024 y el año debe coincidir con `AA_Acc` en todas las filas. Si no, revisen el hueco 1.

**Segunda trampa: una categoría que desaparece.** La gravedad tiene tres valores. Miren cuántos siniestros hay de cada uno por año.
'''),
("code", r'''
por_anio = pd.crosstab(crudo["AA_Acc"], crudo["Gravedad_indicador_30d"])
display(por_anio)
por_anio.plot(figsize=(9, 3.4), color=["#1B998B", "#D7263D", "#1D4E89"], marker="o"); plt.title("Siniestros registrados por año, según la gravedad"); plt.xlabel(""); plt.show()
'''),
("md", r'''
Los siniestros de solo daños se desploman en 2023. La ciudad no se volvió más prudente de un año para otro. Lo que cambió fue la norma sobre el registro de los choques sin víctimas, y desde entonces casi no entran a esta tabla. Una serie que mezcle los tres valores mostraría una caída de la siniestralidad que nunca ocurrió.

**Decisión:** el análisis usa solo los siniestros **con víctimas**, que se registran igual durante los diez años.

**Tercera trampa: dos maneras de contar a los muertos.** Hay dos columnas de gravedad.
'''),
("code", r'''
print(pd.crosstab(crudo["Gravedad_Indicador_Tradicional"], crudo["Gravedad_indicador_30d"]).to_string())
'''),
("md", r'''
El diccionario del anuario dice que el segundo indicador mide la gravedad «tomando en cuenta los fallecimientos que ocurren dentro de los 30 días posteriores al siniestro». Las dos columnas discrepan en unos pocos cientos de filas. Miren en qué dirección: hay siniestros que el indicador tradicional marca con muertos y el de treinta días marca solo con heridos. El diccionario no explica esas filas, y yo no voy a inventar la explicación. **Decisión:** se usa el indicador de treinta días, que es el que el anuario define con precisión, y la duda queda anotada en los riesgos.

**Cuarta trampa: un vacío que significa «no».** Las columnas que empiezan por `Con_` dicen si en el siniestro había una moto, un peatón, un vehículo de carga.
'''),
("code", r'''
banderas = [c for c in crudo.columns if c.startswith("Con_")]
print(pd.DataFrame({"valores distintos": [crudo[c].dropna().unique().tolist() for c in banderas],
                    "% de filas con SI": [(crudo[c] == "SI").mean() * 100 for c in banderas]}, index=banderas).round(1).to_string())
'''),
("md", r'''
Solo aparece el valor `SI`. Donde no hubo moto la celda está vacía. En la clase 2 un vacío era un faltante. Aquí el vacío es la respuesta «no», y tratarlo como faltante borraría casi todo el dato. La limpieza depende de saber cómo se llenó la tabla.

Tres de esas columnas son distintas de las demás. `Con_Embriaguez`, `Con_Velocidad` y `Con_Huecos` no describen quién estaba en el siniestro. Registran, según el diccionario del anuario, que hubo una **hipótesis** sobre la causa. Una hipótesis se formula cuando ya se conoce el resultado del siniestro. **Decisión:** no entran al modelo. La sección 3.5 les pide medir cuánto cambiarían el resultado.

**Quinta trampa: un punto del mapa que no es un cruce.** Cada siniestro trae latitud y longitud. Miren cuáles pares de coordenadas se repiten más, idénticos hasta el último decimal, y de cuántas localidades distintas vienen sus siniestros.
'''),
("code", r'''
pares = (crudo.groupby(["Latitud", "Longitud"]).agg(siniestros=("Localidad", "size"), localidades=("Localidad", "nunique"), direcciones=("Direccion", "nunique"))
         .sort_values("siniestros", ascending=False))
display(pares.head(5))
sospechosos = pares[pares["localidades"] >= 5].index
en_sospechoso = pd.MultiIndex.from_frame(crudo[["Latitud", "Longitud"]]).isin(sospechosos)
print(f"Pares de coordenadas con siniestros de cinco o más localidades: {len(sospechosos)} | siniestros afectados: {en_sospechoso.sum()}")
print("Años de esos siniestros:", crudo.loc[en_sospechoso, "AA_Acc"].value_counts().sort_index().to_dict())
crudo.loc[en_sospechoso, ["Latitud", "Longitud"]] = np.nan
'''),
("md", r'''
Un mismo punto no puede quedar en muchas localidades a la vez, ni tener cientos de direcciones distintas. Ese par de coordenadas es un valor por defecto: lo que el sistema escribió cuando no pudo ubicar la dirección. Es otro faltante disfrazado, como el `-89` de la clase 2. Si se deja, aparece como el cruce más peligroso de Bogotá, y la Secretaría mandaría el primer control a un lugar que no existe.

**Decisión:** las coordenadas que reúnen siniestros de cinco o más localidades se declaran faltantes. El siniestro se conserva para el modelo y queda por fuera del mapa.

### 2.2 La tabla limpia
'''),
("code", r'''
victimas = crudo[crudo["Gravedad_indicador_30d"] != «"Solo Daños"»].copy()                 # HUECO 2: la categoría que se deja por fuera, como aparece en la tabla
if ACTOR != "todos":
    victimas = victimas[victimas[ACTOR] == "SI"]
victimas = victimas.sort_values("fecha").reset_index(drop=True)
y = (victimas[«"Gravedad_indicador_30d"»] == "Con Muertos").astype(int)                   # HUECO 3: la columna de gravedad que cuenta las muertes de los 30 días siguientes

hora = victimas["Hora_Acc"]
X = pd.DataFrame({
    "noche": ((hora >= 21) | (hora < 5)).astype(int),
    "hora_sen": np.sin(2 * np.pi * hora / 24), "hora_cos": np.cos(2 * np.pi * hora / 24),
    "fin_de_semana": victimas["Dia_Semana_Acc"].isin(["sábado", "domingo"]).astype(int),
    "atropello": (victimas["Clase_Acc"] == "Atropello").astype(int),
    "choque": (victimas["Clase_Acc"] == "Choque").astype(int),
    "volcamiento": (victimas["Clase_Acc"] == "Volcamiento").astype(int),
    "contra_objeto_fijo": (victimas["Elemento_Choque"] == "Objeto Fijo").astype(int),
})
actores = {"peaton": "Con_Peaton", "moto": "Con_Moto", "bicicleta": "Con_Bicicleta", "carga": "Con_Carga", "bus": "Con_Tpp",
           "taxi": "Con_Tpi", "persona_mayor": "Con_Persona_Mayor", "menor": "Con_Menores"}
for nombre, columna in actores.items():
    X[nombre] = (victimas[columna] == «"SI"»).astype(int)                                 # HUECO 4: el único valor que aparece en las columnas Con_
hipotesis = pd.DataFrame({nombre: (victimas[columna] == "SI").astype(int) for nombre, columna in
                          {"embriaguez": "Con_Embriaguez", "velocidad": "Con_Velocidad", "huecos": "Con_Huecos"}.items()})
variables = list(X.columns)

print(f"Población: {ACTOR} | siniestros con víctimas: {len(victimas)} | mortales: {y.sum()} | tasa: {y.mean() * 100:.2f} %")
print(f"Sin coordenadas: {victimas['Latitud'].isna().mean() * 100:.1f} %")
'''),
("md", r'''
Comparen con su primera apuesta. En el Mundial, uno de cada nueve remates era gol. Aquí el evento es bastante más raro, y todo lo que vimos en clase sobre el desbalance pega más fuerte.

### 2.3 Conocer el dato antes de modelarlo

Cuatro miradas: cuántos siniestros hay y qué tan letales son según el año, la hora y quién estaba involucrado, y dónde ocurren. El error estándar de una tasa $\hat p$ con $n$ casos es $\sqrt{\hat p(1-\hat p)/n}$.
'''),
("code", r'''
def tasa_por(grupo):
    tabla = pd.DataFrame({"n": y.groupby(grupo).size(), "tasa": y.groupby(grupo).«mean»() * 100})      # HUECO 5: la operación que convierte ceros y unos en una fracción
    tabla["ee"] = np.sqrt(tabla["tasa"] * (100 - tabla["tasa"]) / tabla["n"])
    return tabla

fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
anual = tasa_por(victimas["AA_Acc"])
ejes[0, 0].bar(anual.index, anual["n"], color="#1D4E89"); ejes[0, 0].set(title="Siniestros con víctimas por año", ylabel="siniestros")
otro = ejes[0, 0].twinx(); otro.errorbar(anual.index, anual["tasa"], yerr=2 * anual["ee"], color="#D7263D", marker="o"); otro.set_ylabel("% mortal", color="#D7263D"); otro.spines["right"].set_visible(True)

horaria = tasa_por(hora)
ejes[0, 1].bar(horaria.index, horaria["n"], color="#1D4E89"); ejes[0, 1].set(title="Por hora del día", xlabel="hora", ylabel="siniestros")
otro = ejes[0, 1].twinx(); otro.errorbar(horaria.index, horaria["tasa"], yerr=2 * horaria["ee"], color="#D7263D", marker="o"); otro.set_ylabel("% mortal", color="#D7263D"); otro.spines["right"].set_visible(True)

todas = pd.concat([X.drop(columns=["hora_sen", "hora_cos"]), hipotesis], axis=1)
letalidad = pd.DataFrame({"n": todas.sum(), "tasa": [y[todas[c] == 1].mean() * 100 if todas[c].sum() else np.nan for c in todas]}).dropna().sort_values("tasa")
letalidad["ee"] = np.sqrt(letalidad["tasa"] * (100 - letalidad["tasa"]) / letalidad["n"])
ejes[1, 0].barh([f"{nombre} (n={int(n)})" for nombre, n in zip(letalidad.index, letalidad["n"])], letalidad["tasa"], xerr=2 * letalidad["ee"],
                color=["#D7263D" if nombre in hipotesis else "#1B998B" for nombre in letalidad.index], ecolor="#1D4E89")
ejes[1, 0].axvline(y.mean() * 100, color="gray", ls=":"); ejes[1, 0].set(title="% mortal cuando la condición está presente (rojo: hipótesis)", xlabel="%"); ejes[1, 0].tick_params(axis="y", labelsize=8)

con_lugar = victimas["Latitud"].notna()
ejes[1, 1].hexbin(victimas.loc[con_lugar, "Longitud"], victimas.loc[con_lugar, "Latitud"], gridsize=70, cmap="Blues", mincnt=1, bins="log")
ejes[1, 1].scatter(victimas.loc[con_lugar & (y == 1), "Longitud"], victimas.loc[con_lugar & (y == 1), "Latitud"], s=2, color="#1D4E89", alpha=0.4)
ejes[1, 1].set(title="Dónde ocurren (puntos oscuros: mortales)", xlim=(-74.23, -74.0), ylim=(4.45, 4.84)); ejes[1, 1].set_xticks([]); ejes[1, 1].set_yticks([])
plt.tight_layout(); plt.show()
'''),
("md", r'''
Comparen con su segunda apuesta y respondan en una línea cada una.

- ¿A qué hora ocurren más siniestros y a qué hora son más letales? ¿Coinciden? **Mi lectura:**
- ¿Qué condición tiene la tasa de mortalidad más alta? ¿Es una condición del siniestro o una hipótesis del agente? **Mi lectura:**
- ¿La tasa de mortalidad cambia entre años más de lo que explican sus barras de error? **Mi lectura:**

**¿Quiénes están en la tabla y quiénes no?** Un siniestro entra al registro si llega la policía de tránsito. Respondan: ¿qué clase de siniestro con heridos puede quedar sin registrar? ¿Eso infla o desinfla la tasa de mortalidad que acaban de calcular? **Mi respuesta:**

### 2.4 Agrupar sin etiquetas: ¿cuántas clases de siniestro hay?

Como en clase, $K$-medias agrupa sin mirar la respuesta. Busca $K$ centros que minimicen

$$\sum_{i=1}^{n}\min_k\lVert z_i-\mu_k\rVert^2,$$

con $z_i$ las variables estandarizadas. Después miramos qué tan letal es cada grupo.
'''),
("code", r'''
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

describen = ["noche", "atropello", "choque", "peaton", "moto", "bicicleta", "carga", "bus", "persona_mayor"]
grupo = pd.Series(KMeans(n_clusters=«5», n_init=5, random_state=0).fit_predict(StandardScaler().fit_transform(X[describen])), index=X.index)   # HUECO 6: cuántos grupos pedir. Usen cinco
perfil = X[describen].groupby(grupo).mean().round(2)
perfil.insert(0, "% mortal", (y.groupby(grupo).mean() * 100).round(2))
perfil.insert(0, "siniestros", grupo.value_counts())
display(perfil.sort_values("% mortal", ascending=False))
'''),
("md", r'''
Cada fila es un grupo, y cada columna dice qué fracción de sus siniestros tiene esa condición. Pónganle un nombre a cada grupo en dos o tres palabras, y digan cuál es el más letal y cuál el más numeroso. **Mis nombres:**
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición y la trampa de la exactitud

Aquí manda el tiempo. La Secretaría va a usar el modelo con siniestros posteriores a los del entrenamiento:

$$\underbrace{2015\ \dots\ 2020}_{\text{entrenamiento}}\ \Big|\ \underbrace{2021\ \dots\ 2022}_{\text{validación: se escoge}}\ \Big|\ \underbrace{2023\ \dots\ 2024}_{\text{prueba: se reporta}}$$
'''),
("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

anio = victimas["AA_Acc"].values
E, V, P = anio <= 2020, (anio >= 2021) & (anio <= «2022»), anio >= 2023                 # HUECO 7: el último año de la validación
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {tramo.sum():6d} siniestros | {y[tramo].sum():5d} mortales | tasa {y[tramo].mean() * 100:.2f} %")

logistica = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X[E], y[E])
prob = logistica.predict_proba(X[V])[:, 1]
real = y[V].values
print(f"\nExactitud de decir que ningún siniestro es mortal: {(real == 0).mean():.3f}")
print(f"Exactitud de la regresión logística con umbral 0,5: {((prob >= 0.5) == real).mean():.3f}")
print(f"Siniestros a los que el modelo les da más de 50 %: {(prob >= 0.5).sum()} | probabilidad más alta que entrega: {prob.max():.2f}")
'''),
("md", r'''
Respondan: ¿por qué la exactitud de «ninguno es mortal» es tan alta? ¿Qué le diría esa cifra a un periodista que no conoce el dato? **Mi respuesta:**

### 3.2 La línea base

Dos referencias sin aprendizaje automático. La primera le da a todo siniestro la tasa de mortalidad del entrenamiento. La segunda le da la tasa de su clase de siniestro (atropello, choque, volcamiento y demás), calculada en el entrenamiento.
'''),
("code", r'''
def puntuar(real, prob):
    prob = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return {"pérdida logarítmica": log_loss(real, prob), "Brier": brier_score_loss(real, prob),
            "AUC": roc_auc_score(real, prob) if len(np.unique(prob)) > 1 else 0.5, "mortales que predice": prob.«sum»(), "mortales que hubo": int(np.sum(real))}    # HUECO 8: la operación que convierte probabilidades en un número esperado de casos

clase = victimas["Clase_Acc"]
def tasa_de_su_clase(tramo_entrena, tramo_destino):
    tasas = y[tramo_entrena].groupby(clase[tramo_entrena]).mean()
    return clase[tramo_destino].map(tasas).fillna(y[tramo_entrena].mean()).values

referencias_validacion = {"tasa base": np.full(V.sum(), y[E].mean()), "tasa de su clase": tasa_de_su_clase(E, V)}
probabilidades = {"tasa base": np.full(P.sum(), y[E | V].mean()), "tasa de su clase": tasa_de_su_clase(E | V, P)}
errores = {nombre: {"validación": puntuar(y[V], referencias_validacion[nombre])["pérdida logarítmica"], **puntuar(y[P], probabilidades[nombre])} for nombre in probabilidades}
display(pd.DataFrame(errores).T.round(4))
'''),
("md", r'''
### 3.3 Cinco modelos y el mismo protocolo de la clase

Los cinco candidatos son los de la clase. Gana quien tenga menor pérdida logarítmica en **validación** y se reporta lo que haga en **prueba**. Como el evento es muy raro, los vecindarios y las hojas son grandes, por la razón que vimos con los remates lejanos. Esta celda tarda cerca de un minuto.

**Predicción escrita:** ¿cuánto bajará la pérdida logarítmica frente a la tasa base: mucho, poco o casi nada? **Mi apuesta:**
'''),
("code", r'''
def cinco_modelos(peso=None):
    return {
        "logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=peso)),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=300)),
        "árbol": DecisionTreeClassifier(max_depth=6, min_samples_leaf=200, class_weight=peso, random_state=0),
        "bosque": RandomForestClassifier(n_estimators=200, min_samples_leaf=50, class_weight=peso, random_state=0, n_jobs=-1),
        "boosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, class_weight=peso, random_state=0),
    }

for nombre, modelo in cinco_modelos().items():
    modelo.fit(X[E], y[E])
    en_validacion = puntuar(y[V], modelo.predict_proba(X[V])[:, 1])["pérdida logarítmica"]
    modelo.fit(X[E | V], y[E | V])
    probabilidades[nombre] = modelo.predict_proba(X[P])[:, 1]
    errores[nombre] = {"validación": en_validacion, **puntuar(y[P], probabilidades[nombre])}
todos = pd.DataFrame(errores).T

todos[["validación", "pérdida logarítmica"]].rename(columns={"pérdida logarítmica": "prueba"}).sort_values("validación").plot.barh(figsize=(9, 4), color=["#1D4E89", "#1B998B"])
plt.xlabel("pérdida logarítmica (menos es mejor)"); plt.title("Siete candidatos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(4))

mejor = todos[«"validación"»].idxmin()                                                    # HUECO 9: la columna con la que se escoge al ganador
elegido = mejor if todos.loc[mejor, "pérdida logarítmica"] < todos.loc["tasa de su clase", "pérdida logarítmica"] else "tasa de su clase"
print(f"Ganador en validación: {mejor} | en prueba: pérdida logarítmica {todos.loc[mejor, 'pérdida logarítmica']:.4f}, predice {todos.loc[mejor, 'mortales que predice']:.0f} mortales y hubo {int(todos.loc[mejor, 'mortales que hubo'])}")
print(f"Tasa de su clase: {todos.loc['tasa de su clase', 'pérdida logarítmica']:.4f} | tasa base: {todos.loc['tasa base', 'pérdida logarítmica']:.4f} | método con el que seguimos: {elegido}")
'''),
("md", r'''
- El AUC del ganador puede ser alto y la mejora en pérdida logarítmica pequeña. ¿Cómo pueden ser ciertas las dos cosas con un evento tan raro? **Mi respuesta:**
- ¿El ganador predice bien el **total** de siniestros mortales de la prueba? **Mi lectura:**

### 3.4 La trampa de la clase: balancear rompe la probabilidad

En el Mundial, balancear multiplicó por tres los goles que el modelo predecía. La cuenta de la clase decía que las probabilidades se inflan por un factor cercano a $(1-\pi)/\pi$ en la escala de la razón de probabilidades, donde $\pi$ es la tasa del evento.

**Predicción escrita:** con la tasa de mortalidad que calcularon, ¿por cuánto se multiplicará el total de siniestros mortales que predice el modelo balanceado? **Mi apuesta:**
'''),
("code", r'''
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

nombre_modelo = elegido if elegido in cinco_modelos() else "logística"       # si ganó la línea base, se sigue con el modelo más simple de los cinco
normal = cinco_modelos()[nombre_modelo].fit(X[E], y[E])
balanceado = cinco_modelos(peso=«"balanced"»)[nombre_modelo].fit(X[E], y[E])               # HUECO 10: el valor de class_weight que iguala el peso de las dos clases, entre comillas
p_normal, p_balanceado = normal.predict_proba(X[P])[:, 1], balanceado.predict_proba(X[P])[:, 1]

platt = LogisticRegression().fit(logit(balanceado.predict_proba(X[V])[:, 1]).reshape(-1, 1), y[V])     # recalibración ajustada en validación
p_recalibrado = platt.predict_proba(logit(p_balanceado).reshape(-1, 1))[:, 1]

trampa = pd.DataFrame({"sin balancear": puntuar(y[P], p_normal), "balanceado": puntuar(y[P], p_balanceado),
                       "balanceado y recalibrado": puntuar(y[P], p_recalibrado)}).T
print("Modelo:", nombre_modelo, "| factor que anticipa la cuenta de la clase:", round((1 - y[E].mean()) / y[E].mean(), 1))
display(trampa.round(4))
'''),
("md", r'''
- ¿Qué columna casi no cambia entre las tres filas y cuál cambia muchísimo? **Mi lectura:**
- Si la Secretaría le reportara al Concejo el total de la segunda fila como «siniestros mortales esperados», ¿qué pasaría? **Mi respuesta:**

### 3.5 Calibración y la pregunta por las hipótesis

La curva de calibración agrupa los siniestros de la prueba en tramos según su probabilidad y compara la probabilidad promedio $\bar p_b$ con la fracción real de mortales $\bar y_b$. Su resumen es

$$\text{ECE}=\sum_b\frac{n_b}{n}\,\bigl|\bar y_b-\bar p_b\bigr|.$$
'''),
("code", r'''
def calibracion(real, prob, tramos=10):
    cortes = pd.qcut(pd.Series(prob), tramos, duplicates="drop")
    tabla = pd.DataFrame({"prob": np.asarray(prob), "real": np.asarray(real)}).groupby(cortes.values, observed=True).agg(n=("real", "size"), predicha=("prob", "mean"), observada=("real", "mean"))
    return tabla, float((tabla["n"] / tabla["n"].sum() * (tabla["observada"] - tabla["«predicha»"]).abs()).sum())      # HUECO 11: la columna que se compara con la fracción observada

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.6))
for eje, limite in [(izq, 1.0), (der, 0.25)]:
    for (nombre, prob), color in zip({"sin balancear": p_normal, "balanceado": p_balanceado, "balanceado y recalibrado": p_recalibrado}.items(), ["#1D4E89", "#D7263D", "#5BC0EB"]):
        tabla, ece = calibracion(y[P], prob)
        eje.plot(tabla["predicha"], tabla["observada"], marker="o", color=color, label=f"{nombre} (ECE {ece:.3f})")
    eje.plot([0, 1], [0, 1], color="gray", ls=":"); eje.set(xlim=(0, limite), ylim=(0, 0.25), xlabel="probabilidad promedio que dio el modelo", ylabel="fracción de mortales que hubo")
izq.set_title("Curva de calibración en la prueba"); der.set_title("La misma, ampliada"); izq.legend(fontsize=8); plt.tight_layout(); plt.show()

con_hipotesis = pd.concat([X, hipotesis], axis=1)
modelo_h = cinco_modelos()[nombre_modelo].fit(con_hipotesis[E | V], y[E | V])
comparacion = pd.DataFrame({"sin hipótesis": puntuar(y[P], probabilidades[nombre_modelo]), "con hipótesis": puntuar(y[P], modelo_h.predict_proba(con_hipotesis[P])[:, 1])}).T
display(comparacion.round(4))
'''),
("md", r'''
- ¿Cuál de las tres curvas promete más de lo que cumple? **Mi lectura:**
- La tabla compara el modelo con y sin las hipótesis del agente. ¿Cuánto mejoran? La hipótesis de exceso de velocidad se formula cuando ya se sabe cómo terminó el siniestro, y es razonable pensar que un siniestro con un muerto se investiga más a fondo. ¿Qué nombre tiene ese problema desde la clase 2, y por qué dejamos esas columnas por fuera aunque ayuden? **Mi respuesta:**

### 3.6 Explicar: ¿qué acompaña a un siniestro mortal?

Con las variables estandarizadas, $\text{OR}_j=e^{\beta_j}$ es la razón de probabilidades por cada desviación estándar de la variable $j$, con las demás quietas.
'''),
("code", r'''
explicativo = cinco_modelos()["logística"].fit(X[E | V], y[E | V])
razones = pd.Series(np.«exp»(explicativo[-1].coef_[0]), index=variables).sort_values()        # HUECO 12: la función que convierte un coeficiente en una razón de probabilidades
razones.plot.barh(figsize=(8, 5.5), color=["#5BC0EB" if r < 1 else "#D7263D" for r in razones]); plt.axvline(1, color="gray")
plt.xlabel("razón de probabilidades por desviación estándar"); plt.title("Qué acompaña a un siniestro mortal"); plt.show()
print(razones.round(2).to_string())
'''),
("md", r'''
- ¿Qué tres condiciones suben más la probabilidad de que el siniestro sea mortal? **Mi lectura:**
- «Peatón» y «atropello» casi siempre van juntos. ¿Qué le hace eso a la lectura de cada coeficiente por separado? **Mi respuesta:**
- Que los siniestros con menores sean menos letales, ¿quiere decir que los niños corren menos peligro en la calle? Propongan otra explicación. **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿en qué cien cruces?

Llamo **cruce** a una celda de unos 110 metros de lado, que sale de redondear la latitud y la longitud a tres decimales. Con los siniestros de 2015 a 2022 ordeno los cruces de tres maneras y me quedo con los primeros:

1. Por número de siniestros con víctimas.
2. Por número de siniestros mortales.
3. Por número **esperado** de siniestros mortales, $\hat m_c=\sum_{i\in c}\hat p_i$, con las probabilidades del modelo.

Después miro cuántos siniestros mortales de 2023 y 2024 ocurrieron en los cruces escogidos por cada criterio. Ningún criterio conoce esos dos años. La tercera manera es la que necesita probabilidades calibradas: una suma de probabilidades infladas no es un número de muertes.
'''),
("code", r'''
modelo_final = cinco_modelos()[nombre_modelo].fit(X[E | V], y[E | V])
victimas["p"] = modelo_final.predict_proba(X)[:, 1]
victimas["mortal"] = y
victimas["cruce"] = victimas["Latitud"].round(«3»).astype(str) + ", " + victimas["Longitud"].round(3).astype(str)    # HUECO 13: cuántos decimales definen una celda de unos 110 metros

con_lugar = victimas["Latitud"].notna().values
pasado, futuro = victimas[(E | V) & con_lugar], victimas[P & con_lugar]
por_cruce = pasado.groupby("cruce").agg(siniestros=("mortal", "size"), mortales=("mortal", "sum"), esperados=("p", "sum"))
despues = futuro.groupby("cruce")["mortal"].sum()

desempate = np.random.default_rng(0).random(len(por_cruce))
def primeros(criterio):
    return por_cruce.assign(desempate=desempate).sort_values([criterio, "desempate"], ascending=False).index[:CONTROLES]

sorteo = np.random.default_rng(1)
replicas = {criterio: [] for criterio in ["siniestros", "mortales", "esperados"]}
escogidos = {criterio: set(primeros(criterio)) for criterio in replicas}
cruces_futuros = futuro["cruce"].values; mortal_futuro = futuro["mortal"].values
for _ in range(300):                                           # bootstrap sobre los siniestros de la prueba
    filas = sorteo.integers(0, len(futuro), len(futuro))
    for criterio in replicas:
        replicas[criterio].append(mortal_futuro[filas][np.isin(cruces_futuros[filas], list(escogidos[criterio]))].sum())

decision = pd.DataFrame({criterio: {"mortales de 2023 y 2024 en esos cruces": int(despues.reindex(list(escogidos[criterio])).fillna(0).sum()),
                                    "intervalo de 90 %": f"{np.percentile(replicas[criterio], 5):.0f} a {np.percentile(replicas[criterio], 95):.0f}",
                                    "% de todos los mortales": despues.reindex(list(escogidos[criterio])).fillna(0).sum() / mortal_futuro.sum() * 100} for criterio in replicas}).T
al_azar = np.mean([despues.reindex(por_cruce.sample(CONTROLES, random_state=s).index).fillna(0).sum() for s in range(200)])
print(f"Cruces con algún siniestro entre 2015 y 2022: {len(por_cruce)} | se escogen {CONTROLES} ({CONTROLES / len(por_cruce) * 100:.1f} %)")
print(f"Siniestros mortales de 2023 y 2024 con coordenadas: {int(mortal_futuro.sum())} | en {CONTROLES} cruces al azar caerían {al_azar:.1f}")
display(decision.round(1))
print("Cruces que comparten el criterio 1 y el 3:", len(escogidos["siniestros"] & escogidos["esperados"]))
'''),
("md", r'''
Comparen con su tercera apuesta y respondan.

- ¿Cuál de los tres criterios atrapa más siniestros mortales futuros? ¿La diferencia entre ellos es mayor que sus intervalos? **Mi lectura:**
- Ordenar por los muertos del pasado parece lo más directo. ¿Por qué puede ser peor que ordenar por siniestros? Piensen en cuántos mortales hay por cruce. **Mi respuesta:**
- ¿Qué aporta el modelo frente a contar siniestros? Sean francos con lo que salió. **Mi respuesta:**
'''),
("code", r'''
recomendados = por_cruce.loc[list(primeros("esperados"))].head(10).round(2)
recomendados["dirección más frecuente"] = [pasado.loc[pasado["cruce"] == c, "Direccion"].mode().iloc[0] for c in recomendados.index]
display(recomendados)
'''),
("md", r'''
### La decisión en tres frases

Para la secretaria de movilidad, que conoce la ciudad y no sabe qué es una curva de calibración.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Solo siniestros con víctimas | Los tres niveles de gravedad | Los de solo daños dejaron de registrarse en 2023 |
| Gravedad a treinta días | | |
| El vacío en las columnas `Con_` es «no» | | |
| Hipótesis del agente fuera del modelo | | |
| Partición por años | | |
| Pérdida logarítmica para escoger | | |
| No balancear las clases | | |
| Coordenadas por defecto declaradas faltantes | | |
| Cruce: celda de tres decimales | | |
| Criterio para escoger los cruces | | |

### 4.3 Riesgos, ética y mantenimiento

- Un control en un cruce puede bajar los siniestros ahí y subirlos a dos cuadras. ¿Cómo lo sabrían?
- Hay siniestros que un indicador de gravedad marca con muertos y el otro no. ¿A quién le preguntarían cuál es el correcto, y qué resultado suyo cambiaría?
- Cerca del 4 % de los siniestros con víctimas no tiene coordenadas. ¿Qué pasa si faltan más en unas localidades que en otras?
- Los cruces con más siniestros registrados son también los de más tráfico y más presencia de policía. ¿Qué barrios pueden quedar por fuera de la lista por estar menos vigilados?
- Este modelo estima la probabilidad de muerte **dado** que hubo un siniestro con víctimas. ¿Por qué no sirve para decir en qué cruce es más probable que haya un siniestro?
- Detrás de cada fila mortal hay una familia. ¿Qué cuidados tendrían al publicar un mapa con estos puntos?

### 4.4 Procedencia
'''),
("code", r'''
procedencia()["siniestros_bogota"]
'''),
("md", r'''
### 4.5 Cambien una línea

Vuelvan a la celda donde dice `ACTOR = "todos"`, pongan `ACTOR = "Con_Peaton"` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Ahora la población son los siniestros en que hubo un peatón. Llenen la tabla con sus dos corridas.

| | Todos | Con peatón |
|---|---|---|
| Siniestros con víctimas y tasa de mortalidad | | |
| Ganador en validación | | |
| Pérdida logarítmica en prueba: ganador y tasa base | | |
| Mortales que predice el modelo balanceado, y mortales que hubo | | |
| Mortales futuros en los cruces escogidos, con cada criterio | | |
| % de todos los mortales que atrapan los cruces | | |

Con la tabla llena, respondan en tres líneas. ¿Los cien cruces para proteger peatones son los mismos que los cien cruces generales? ¿Qué le preguntarían a la Secretaría antes de decidir con cuál de las dos listas se queda? **Mi respuesta:**

Dejen el cuaderno entregado con `ACTOR = "Con_Peaton"` corrido y la tabla completa.

**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase04/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]

if __name__ == "__main__":
    destino = RAIZ / "clases" / "04_desbalance"
    destino.mkdir(parents=True, exist_ok=True)
    nbf.write(cuaderno(MUNDIAL, False), destino / "04a_el_mundial_de_qatar.ipynb")
    if CALLES:
        nbf.write(cuaderno(CALLES, True), destino / "04b_las_calles_de_bogota.ipynb")
        (RAIZ / "_profesor").mkdir(exist_ok=True)
        nbf.write(cuaderno(CALLES, False), RAIZ / "_profesor" / "04b_las_calles_de_bogota_SOLUCION.ipynb")
    print("Cuadernos de la clase 4 escritos.")
