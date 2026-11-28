"""Celdas del gemelo de la clase 7: La plata pública (SECOP II).

Cada ("tarea", comentario, solución) sale en el cuaderno del estudiante solo con el
comentario, y en la solución del profesor con el comentario y el código.
"""
from molde import ARRANQUE

PLATA = [
("md", r'''
# Clase 7 · La plata pública (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *Lo que dicen los turistas*: la misma técnica con otro texto.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase con el objeto de los contratos públicos: convertir texto en TF-IDF, comparar un catálogo de clasificadores contra una línea base, medir cuánto cambia el resultado según cómo se parta y convertir un puntaje en una lista de contratos para revisar. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (seleccionar y preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | El código lo escribe Claude y ustedes lo auditan. Deben poder explicar sin ayuda qué columnas no pueden entrar al modelo y por qué, por qué la partición es por fecha, qué significa la precisión entre los primeros cincuenta y qué no dice el modelo sobre un contrato. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, las respuestas, la auditoría, la decisión en tres frases y la tabla de la variante llenas. |

**Cómo se trabaja.** Como en las dos clases anteriores: cada celda de código trae solo un comentario que dice qué debe lograr. Péguenle a Claude el bloque «Lo que Claude necesita saber», pídanle el código por secciones, léanlo, córranlo y respondan las preguntas con sus palabras.
'''),
("md", r'''
> **Antes de empezar: conozcan la fuente.** Los contratos de hoy salen del conjunto [SECOP II, Contratos Electrónicos, en datos.gov.co](https://www.datos.gov.co/d/jbjy-vk9h). La misma página trae el diccionario de las columnas. Ábranla y explórenla cinco minutos antes de pedirle código a Claude. Respondan mirando la fuente:
>
> - ¿Qué entidad publica el conjunto y cuándo fue su última actualización?
> - ¿Qué es una fila: un proceso de contratación, un contrato o un pago?
> - ¿Cómo describe el diccionario la columna «Dias adicionados», y qué licencia tiene el conjunto?
'''),
("code", ARRANQUE),
("code", r'''
DEPARTAMENTO = "Caldas"     # departamento de las entidades que contratan. Lo cambiarán al final.
AUDITORES = 50              # cuántos contratos alcanza a revisar el equipo
SEMILLA = 0

import time
'''),
("md", r'''
### Lo que Claude necesita saber

> Trabajo en un cuaderno de Colab sin GPU. Ya están importados `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt` y `time`, y definidas las constantes `DEPARTAMENTO`, `AUDITORES` y `SEMILLA`.
>
> `tabla = descargar("secop", departamento=DEPARTAMENTO)` devuelve un DataFrame con una fila por contrato de SECOP II firmado en 2022 o 2023 por una entidad de ese departamento. Columnas: `id_contrato`, `nombre_entidad`, `nit_entidad`, `departamento`, `ciudad`, `orden`, `sector`, `tipo_de_contrato`, `modalidad_de_contratacion`, `estado_contrato`, `codigo_de_categoria_principal`, `fecha_de_firma`, `fecha_de_inicio_del_contrato`, `fecha_de_fin_del_contrato` (las tres fechas como texto), `valor_del_contrato`, `dias_adicionados`, `es_pyme` y `objeto_del_contrato` (texto libre de hasta 500 caracteres).
>
> Quiero predecir si un contrato tendrá prórroga (`dias_adicionados > 0`) usando **solo lo que se sabe el día de la firma**, sobre todo el texto de `objeto_del_contrato`, con TF-IDF y modelos de scikit-learn. Los candidatos están en un diccionario `MODELOS` que el cuaderno define en una celda propia, con el nombre de cada modelo como clave y un estimador de scikit-learn sin entrenar como valor. El código debe recorrer ese diccionario con `clone` y no nombrar modelos sueltos, salvo `MODELOS["logística"]` y `MODELOS["Bayes ingenuo"]` cuando una tarea los pida. No puedes usar como entrada `dias_adicionados`, `estado_contrato` ni `fecha_de_fin_del_contrato`. El vectorizador se ajusta solo con el entrenamiento. Escribe código corto, con comentarios en español, que corra en CPU en pocos minutos.
'''),
("md", r'''
## 1. Formulación

Imagínense el equipo de control interno de una gobernación. Tiene miles de contratos firmados cada año y gente para revisar a fondo cincuenta. Hoy los escoge al azar o por olfato. Quiere saber si el texto del contrato, que está disponible el mismo día de la firma, ayuda a escoger mejor.

Un contrato que se prorroga no es un contrato irregular. Las prórrogas son legales y muchas tienen buenas razones. Pero un contrato que se alarga es un contrato donde algo no salió como se planeó, y por eso es un lugar razonable para mirar primero.

1. **Predecir.** Con lo que se sabe el día de la firma, ¿qué tan probable es que el contrato termine con días adicionados?
2. **Explicar.** ¿Qué palabras del objeto se asocian con las prórrogas? ¿El modelo lee el tipo de trabajo o reconoce a la entidad que contrata?
3. **Decidir.** ¿Qué contratos revisan los cincuenta auditores?

**El dato.** SECOP II es la plataforma donde las entidades del Estado colombiano hacen su contratación en línea. Colombia Compra Eficiente publica los contratos en datos.gov.co (conjunto `jbjy-vk9h`, licencia CC BY-SA 4.0). La descarga trae los contratos firmados en 2022 y 2023 por entidades del departamento, con 18 de las 95 columnas. Dejé por fuera los nombres y documentos de contratistas, representantes legales, supervisores y ordenadores del gasto, y los datos bancarios. Son públicos, pero el análisis no los necesita.

**La respuesta.** `dias_adicionados` es, según el diccionario del portal, «días adicionados al contrato electrónico». Llamo prórroga a que sea mayor que cero.

**La métrica.** Las prórrogas son pocas, así que la exactitud no sirve, como vimos en la demostración. Uso la precisión promedio (AP) para comparar modelos. Para la decisión uso la precisión entre los primeros $k$: se ordenan los contratos de mayor a menor puntaje y se cuenta cuántos de los primeros $k$ tuvieron prórroga,

$$P@k=\frac{1}{k}\sum_{i\in\text{primeros }k} y_i .$$

Si $\pi$ es la proporción de prórrogas, escoger al azar da $P@k\approx\pi$. La razón $P@k/\pi$ dice cuántas veces mejor que el azar escoge el modelo.

### Predicciones escritas

- ¿Qué fracción de los contratos tiene prórroga? **Mi apuesta:**
- ¿Qué fracción de los objetos de contrato repite, palabra por palabra, el de otro contrato? **Mi apuesta:**
- De los 50 contratos que escoja el modelo, ¿cuántos tendrán prórroga? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

### 2.1 Descargar y mirar

La descarga tarda cerca de medio minuto.
'''),
("tarea", r'''
# TAREA 1. Descarguen la tabla con descargar("secop", departamento=DEPARTAMENTO). Conviertan fecha_de_firma
# a fecha. Reporten contratos, entidades, rango de fechas de firma y contratos por año.
''', r'''
tabla = descargar("secop", departamento=DEPARTAMENTO)
tabla["firma"] = pd.to_datetime(tabla["fecha_de_firma"])
tabla = tabla.sort_values("firma").reset_index(drop=True)
print(len(tabla), "contratos |", tabla["nombre_entidad"].nunique(), "entidades |", tabla["firma"].min().date(), "a", tabla["firma"].max().date())
print(tabla["firma"].dt.year.value_counts().sort_index().to_string())
tabla.head(3)
'''),
("tarea", r'''
# TAREA 2. Construyan la respuesta `prorroga` (dias_adicionados > 0). Reporten la proporción total, por tipo de
# contrato (tipos con 100 contratos o más), por modalidad (100 o más) y por trimestre de firma. Muestren
# además la tabla cruzada de estado_contrato contra prorroga.
''', r'''
tabla["prorroga"] = (tabla["dias_adicionados"] > 0).astype(int)
y = tabla["prorroga"].to_numpy()
print(f"Contratos con prórroga: {y.sum()} de {len(y)} ({y.mean() * 100:.1f} %)")

def tasa_por(columna, minimo=100):
    resumen = tabla.groupby(columna)["prorroga"].agg(["mean", "size"]).query("size >= @minimo").sort_values("mean", ascending=False)
    return (resumen.assign(mean=resumen["mean"] * 100).rename(columns={"mean": "% con prórroga", "size": "contratos"})).round(1)

display(tasa_por("tipo_de_contrato")); display(tasa_por("modalidad_de_contratacion"))
(tabla.groupby(tabla["firma"].dt.to_period("Q"))["prorroga"].mean() * 100).plot.bar(color="#1D4E89", rot=0, figsize=(8, 3))
plt.ylabel("% con prórroga"); plt.title("Prórrogas según el trimestre de firma"); plt.show()
display(pd.crosstab(tabla["estado_contrato"], tabla["prorroga"]))
'''),
("md", r'''
**Preguntas.**

- ¿Qué tipos de contrato se prorrogan más? ¿Coincide con lo que esperaban? **Mi respuesta:**
- Miren la tabla cruzada de `estado_contrato`. ¿Qué estado delata la respuesta? ¿Por qué esa columna no puede entrar al modelo, aunque mejore mucho el resultado? **Mi respuesta:**
- No sabemos si el portal cambia `fecha_de_fin_del_contrato` cuando hay una prórroga. ¿Qué hacen con una columna así? **Mi respuesta:**
- La proporción de prórrogas cambia con el trimestre de firma. Den dos explicaciones posibles, una del mundo y una del dato. **Mi respuesta:**
'''),
("md", r'''
### 2.2 El texto
'''),
("tarea", r'''
# TAREA 3. Exploren objeto_del_contrato: largo en caracteres, fracción escrita toda en mayúsculas, y textos
# repetidos después de normalizar (mayúsculas, sin espacios dobles). Muestren los 5 objetos más repetidos con
# su conteo, y la fracción de contratos cuyo objeto repite el de otro contrato.
''', r'''
objeto = tabla["objeto_del_contrato"].fillna("")
normal = objeto.str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
tabla["objeto_normal"] = normal
print(objeto.str.len().describe().round(0).to_string())
print(f"\nObjetos escritos todos en mayúsculas: {(objeto == objeto.str.upper()).mean() * 100:.0f} %")
repetido = normal.duplicated(keep=False)
print(f"Contratos cuyo objeto repite el de otro contrato: {repetido.mean() * 100:.1f} %")
print(f"Objetos distintos: {normal.nunique()} en {len(normal)} contratos")
for texto, veces in normal.value_counts().head(5).items():
    print(f"  {veces:4d} veces: {texto[:150]}")
'''),
("md", r'''
**Preguntas.**

- ¿Por qué hay tantos objetos repetidos? Piensen en cómo se redacta un contrato de prestación de servicios. **Mi respuesta:**
- Lean el objeto más repetido de su lista. ¿Describe algún trabajo? ¿Qué hacen con esos contratos: los quitan, los dejan, los marcan? Escriban la decisión en la bitácora. **Mi respuesta:**
- En la demostración, las reseñas repetidas eran pocas y movieron poco el resultado. Con la proporción que acaban de medir, ¿qué esperan aquí si parten al azar? **Mi respuesta:**
- A diferencia de las reseñas, aquí no pueden simplemente borrar las repeticiones: son contratos distintos. ¿Qué hacen entonces? **Mi respuesta:**

### 2.3 Partición y TF-IDF

Los auditores van a usar el modelo sobre contratos que todavía no se han firmado. Por eso la partición es **por fecha de firma**, como en la clase 1: se entrena con 2022, se valida con el primer semestre de 2023 y se prueba con el segundo.

Recuerden el peso de un término $t$ en un documento $d$:

$$w_{t,d}=\bigl(1+\ln \text{tf}(t,d)\bigr)\cdot\Bigl(\ln\frac{1+n}{1+\text{df}(t)}+1\Bigr).$$
'''),
("tarea", r'''
# TAREA 4. Partan por fecha de firma: E = 2022, V = enero a junio de 2023, P = julio a diciembre de 2023.
# Impriman contratos y proporción de prórrogas por tramo. Ajusten un TfidfVectorizer (palabras y pares,
# sin tildes, mínimo 2 documentos, tf sublineal) SOLO con E y transformen los tres tramos. Reporten el
# tamaño de la matriz y qué fracción de los objetos de V y de P ya aparece idéntica en E.
''', r'''
from sklearn.feature_extraction.text import TfidfVectorizer

E = np.flatnonzero((tabla["firma"] < "2023-01-01").to_numpy())
V = np.flatnonzero(((tabla["firma"] >= "2023-01-01") & (tabla["firma"] < "2023-07-01")).to_numpy())
P = np.flatnonzero((tabla["firma"] >= "2023-07-01").to_numpy())
for nombre, tramo in [("Entrenamiento (2022)", E), ("Validación (2023-I)", V), ("Prueba (2023-II)", P)]:
    print(f"{nombre:22s} {len(tramo):6d} contratos | {y[tramo].mean() * 100:.1f} % con prórroga")

def vectorizador():
    return TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")

textos = tabla["objeto_del_contrato"].fillna("").to_numpy()
tfidf = vectorizador().fit(textos[E])
X_E, X_V, X_P = (tfidf.transform(textos[tramo]) for tramo in (E, V, P))
terminos = np.array(tfidf.get_feature_names_out())
print(f"\nMatriz de entrenamiento: {X_E.shape[0]} contratos por {X_E.shape[1]} términos")
vistos = set(tabla["objeto_normal"].iloc[E])
print(f"Objetos de validación que ya estaban idénticos en el entrenamiento: {tabla['objeto_normal'].iloc[V].isin(vistos).mean() * 100:.1f} %")
print(f"Objetos de prueba que ya estaban idénticos en el entrenamiento:     {tabla['objeto_normal'].iloc[P].isin(vistos).mean() * 100:.1f} %")
'''),
("md", r'''
### 2.4 Sin supervisión: ¿qué contrata el departamento?

Factoricen la matriz TF-IDF del entrenamiento en temas con NMF, $A\approx WH$ con $W,H\ge 0$, y miren si las prórrogas se concentran en algún tema.
'''),
("tarea", r'''
# TAREA 5. NMF con 8 temas sobre los objetos del entrenamiento (vectorizador aparte, sin palabras vacías del
# español y solo palabras sueltas). Para cada tema: sus 8 términos de mayor peso, qué fracción de los
# contratos de E le corresponde y su proporción de prórrogas.
''', r'''
from sklearn.decomposition import NMF

VACIAS = ("de la el en y a los las que un una por con para del al lo se es su no como e o u sus asi segun "
          "prestacion servicios servicio prestar contratar contrato").split()
tfidf_temas = TfidfVectorizer(min_df=5, max_df=0.6, strip_accents="unicode", stop_words=VACIAS).fit(textos[E])
A = tfidf_temas.transform(textos[E])
nmf = NMF(n_components=8, init="nndsvd", random_state=SEMILLA, max_iter=400).fit(A)
tema_de = nmf.transform(A).argmax(axis=1)
nombres_temas = np.array(tfidf_temas.get_feature_names_out())
temas = pd.DataFrame([{"términos de mayor peso": ", ".join(nombres_temas[np.argsort(-nmf.components_[k])[:8]]),
                       "% de los contratos": 100 * (tema_de == k).mean(),
                       "% con prórroga": 100 * y[E][tema_de == k].mean()} for k in range(8)])
pd.set_option("display.max_colwidth", 110)
display(temas.round(1).sort_values("% con prórroga", ascending=False))
'''),
("md", r'''
**Pregunta.** Pónganle nombre a cada tema. ¿En cuáles se concentran las prórrogas? ¿Algún tema es en realidad una entidad y no un tipo de trabajo? **Mi respuesta:**
'''),
("md", r'''
## 3. Modelado

### 3.1 Líneas base

Dos líneas base. Escoger al azar, cuya precisión esperada es la proporción de prórrogas. Y una **planilla** sin texto: regresión logística con el tipo de contrato, la modalidad, el orden, el sector y el logaritmo del valor.
'''),
("tarea", r'''
# TAREA 6. Escriban una función que reciba etiquetas y puntajes y devuelva AP y precisión entre los primeros
# AUDITORES. Entrenen en E la planilla (one-hot de tipo_de_contrato, modalidad_de_contratacion, orden y sector,
# con categorías de menos de 20 contratos agrupadas; más log10 del valor estandarizado) y repórtenla en V y P
# junto con la línea base del azar.
''', r'''
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def medir(real, puntajes, k=None):
    k = AUDITORES if k is None else k
    primeros = np.argsort(-puntajes, kind="stable")[:k]
    return {"AP": average_precision_score(real, puntajes), f"P@{k}": float(real[primeros].mean())}

tabla["log_valor"] = np.log10(tabla["valor_del_contrato"].clip(lower=1))
CATEGORICAS = ["tipo_de_contrato", "modalidad_de_contratacion", "orden", "sector"]
planilla = make_pipeline(
    make_column_transformer((OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAS),
                            (StandardScaler(), ["log_valor"])),
    LogisticRegression(max_iter=2000, class_weight="balanced")).fit(tabla.iloc[E], y[E])

resultados = {}
def anotar(nombre, puntaje_V, puntaje_P):
    resultados[nombre] = {**{f"{c} (val)": v for c, v in medir(y[V], puntaje_V).items()},
                          **{f"{c} (prueba)": v for c, v in medir(y[P], puntaje_P).items()}}

anotar("al azar (valor esperado)", np.full(len(V), 0.0), np.full(len(P), 0.0))
resultados["al azar (valor esperado)"] = {"AP (val)": y[V].mean(), f"P@{AUDITORES} (val)": y[V].mean(),
                                          "AP (prueba)": y[P].mean(), f"P@{AUDITORES} (prueba)": y[P].mean()}
anotar("planilla (sin texto)", planilla.predict_proba(tabla.iloc[V])[:, 1], planilla.predict_proba(tabla.iloc[P])[:, 1])
display(pd.DataFrame(resultados).T.round(3))
'''),
("md", r'''
### 3.2 El catálogo de modelos

La celda siguiente es el catálogo de candidatos, el mismo de la demostración. Es un diccionario, `MODELOS`, con una línea por modelo, y ya viene escrita. La logística modela $P(y=1\mid d)=\sigma(\beta_0+\sum_t\beta_t\,w_{t,d})$.

- El texto entre comillas es el **nombre** con el que el modelo aparece en las tablas y en las gráficas.
- Lo que sigue es el **clasificador**. Cuando necesita una preparación propia, `make_pipeline(...)` la encadena delante. Aquí la usan los dos modelos que primero reducen la matriz a pocas dimensiones.
- Para **agregar un modelo** escriban una línea nueva con otro nombre, importen su clase de scikit-learn al comienzo de la celda y corran el cuaderno desde ahí. Las tareas que siguen deben recorrer lo que haya en el diccionario, sin nombrar modelos sueltos.

**Ejercicio.** Agreguen al menos tres modelos más de scikit-learn y vuelvan a correr. La matriz es dispersa y muy ancha, así que un modelo que no acepte matrices dispersas necesita la reducción delante. No les cambien el nombre a `"logística"` ni a `"Bayes ingenuo"`, porque una tarea los pide por ese nombre.
'''),
("code", r'''
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

MODELOS = {
    "Bayes ingenuo": ComplementNB(alpha=0.3),
    "Bayes multinomial": MultinomialNB(alpha=0.1),
    "Bayes de Bernoulli": BernoulliNB(alpha=0.1),
    "logística": LogisticRegression(C=3, max_iter=2000, class_weight="balanced"),
    "logística, castigo fuerte": LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"),
    "SVM lineal": LinearSVC(C=0.1, class_weight="balanced"),
    "SVM lineal, castigo fuerte": LinearSVC(C=0.01, class_weight="balanced"),
    "ridge": RidgeClassifier(alpha=3.0, class_weight="balanced"),
    "descenso estocástico": SGDClassifier(loss="log_loss", alpha=1e-6, class_weight="balanced", random_state=SEMILLA),
    "descenso estocástico, Huber": SGDClassifier(loss="modified_huber", alpha=1e-5, class_weight="balanced", random_state=SEMILLA),
    "LSA + boosting": make_pipeline(TruncatedSVD(n_components=100, random_state=SEMILLA), HistGradientBoostingClassifier(max_iter=100, class_weight="balanced", random_state=SEMILLA)),
    "LSA + discriminante lineal": make_pipeline(TruncatedSVD(n_components=100, random_state=SEMILLA), LinearDiscriminantAnalysis()),
}
print(len(MODELOS), "modelos en el catálogo:", ", ".join(MODELOS))
'''),
("tarea", r'''
# TAREA 7. Recorran el diccionario MODELOS: entrenen una copia de cada modelo (clone) sobre la matriz TF-IDF de E.
# Reporten AP y P@AUDITORES en V y en P en una tabla (con las líneas base), ordenada por AP de validación, y
# dibujen las curvas de precisión contra sensibilidad en validación. Escojan como ganador el modelo del catálogo
# con mayor AP en validación. Dibujen además una gráfica pequeña con el puesto de cada modelo en validación
# (eje horizontal) contra su puesto en prueba (eje vertical), según la AP, con la diagonal y el ganador en rojo.
# El código no debe nombrar ningún modelo.
''', r'''
from sklearn.base import clone

def puntaje(modelo, matriz):
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(matriz)
    return modelo.predict_proba(matriz)[:, 1]

inicio = time.time()
entrenados = {}
plt.figure(figsize=(8.5, 5))
for i, (nombre, receta) in enumerate(MODELOS.items()):
    modelo = clone(receta).fit(X_E, y[E])
    entrenados[nombre] = modelo
    anotar(nombre, puntaje(modelo, X_V), puntaje(modelo, X_P))
    precision, sensibilidad, _ = precision_recall_curve(y[V], puntaje(modelo, X_V))
    plt.plot(sensibilidad, precision, ls="-" if i < len(PALETA) else "--", label=f"{nombre} ({resultados[nombre]['AP (val)']:.3f})")
plt.axhline(y[V].mean(), color="gray", ls=":"); plt.ylim(0, 1)
plt.xlabel("sensibilidad"); plt.ylabel("precisión"); plt.title("Validación (punteada: escoger al azar)"); plt.legend(fontsize=7); plt.show()
print(f"Tiempo: {time.time() - inicio:.0f} s")
comparacion = pd.DataFrame(resultados).T.sort_values("AP (val)", ascending=False)
display(comparacion.round(3))
del_catalogo = comparacion.loc[list(MODELOS), ["AP (val)", "AP (prueba)"]]
mejor = del_catalogo["AP (val)"].idxmax()
ganador = entrenados[mejor]
print(f"Ganador en validación: {mejor}")

puestos = del_catalogo.rank(ascending=False, method="first").astype(int)        # el puesto 1 es la mejor AP
fig, eje = plt.subplots(figsize=(5, 5), facecolor="#FBFAF7")
eje.set_facecolor("#FBFAF7")
eje.plot([1, len(puestos)], [1, len(puestos)], color="#9AA5B1", ls=":")
eje.scatter(puestos.iloc[:, 0], puestos.iloc[:, 1], s=45, zorder=3, color=["#D7263D" if nombre == mejor else "#1D4E89" for nombre in puestos.index])
for nombre, fila in puestos.iterrows():
    eje.annotate(nombre, (fila.iloc[0], fila.iloc[1]), xytext=(6, 4), textcoords="offset points", fontsize=7)
eje.set(title="¿El orden de validación se repite en la prueba?", xlabel="puesto en validación (1 es el mejor)", ylabel="puesto en prueba",
        xticks=range(1, len(puestos) + 1), yticks=range(1, len(puestos) + 1))
plt.show()
print(f"AP en validación del primero y del segundo: {del_catalogo['AP (val)'].nlargest(2).round(3).tolist()}")
'''),
("md", r'''
**Preguntas.**

- El ganador salió de una lista larga. Miren cuánto se alejan los puntos de la diagonal y cuánta ventaja le sacó el primero al segundo en validación. Con tantos candidatos, ¿cuánto del ganador es azar? **Mi respuesta:**
- ¿El texto le gana a la planilla? ¿Por cuánto, y es consistente entre validación y prueba? **Mi respuesta:**
- La precisión entre los primeros 50 se calcula con 50 contratos. ¿Cuánto puede moverse por azar? **Mi respuesta:**

### 3.3 La trampa: tres maneras de partir

Midan el mismo modelo con tres exámenes: al azar, por fecha y por entidad (todas las entidades de la prueba son nuevas para el modelo). El vectorizador va dentro de un `Pipeline`, para que se ajuste solo con el entrenamiento de cada examen.
'''),
("tarea", r'''
# TAREA 8. Para la regresión logística y para Bayes ingenuo, con el TF-IDF dentro de un Pipeline, midan AP y
# P@AUDITORES en tres exámenes: (a) al azar 80/20, 5 semillas; (b) por fecha (el de la sección 2.3, entrenando
# con E y examinando en P); (c) por entidad con GroupShuffleSplit 80/20, 5 semillas. En cada examen reporten
# también qué fracción de los objetos de la prueba aparece idéntica en el entrenamiento. Muestren los promedios.
''', r'''
from sklearn.model_selection import GroupShuffleSplit, train_test_split

entidad = tabla["nombre_entidad"].to_numpy()
todas = np.arange(len(tabla))
objeto_normal = tabla["objeto_normal"].to_numpy()

def examen(entren, prueba):
    fila = {"% de la prueba ya visto": 100 * np.isin(objeto_normal[prueba], objeto_normal[entren]).mean()}
    for nombre in ["logística", "Bayes ingenuo"]:
        tuberia = make_pipeline(vectorizador(), clone(MODELOS[nombre])).fit(textos[entren], y[entren])
        cifras = medir(y[prueba], puntaje(tuberia, textos[prueba]))
        fila[f"{nombre}: AP"], fila[f"{nombre}: P@{AUDITORES}"] = cifras["AP"], cifras[f"P@{AUDITORES}"]
    fila["proporción de prórrogas"] = y[prueba].mean()
    return fila

inicio = time.time()
filas = []
for semilla in range(5):
    entren, prueba = train_test_split(todas, test_size=0.20, random_state=semilla, stratify=y)
    filas.append({"examen": "al azar", **examen(entren, prueba)})
    entren, prueba = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=semilla).split(todas, groups=entidad))
    filas.append({"examen": "por entidad", **examen(entren, prueba)})
filas.append({"examen": "por fecha", **examen(E, P)})
tres = pd.DataFrame(filas).groupby("examen").mean().loc[["al azar", "por fecha", "por entidad"]]
print(f"Tiempo: {time.time() - inicio:.0f} s")
display(tres.round(3))
tres[[f"logística: P@{AUDITORES}", f"Bayes ingenuo: P@{AUDITORES}", "proporción de prórrogas"]].plot.bar(color=["#D7263D", "#1B998B", "gray"], rot=0, figsize=(9, 3.8))
plt.ylabel(f"precisión entre los primeros {AUDITORES}"); plt.title("El mismo modelo, tres exámenes"); plt.show()
'''),
("md", r'''
**Preguntas.** Con los números de su tabla.

- ¿Cuánto vale la precisión entre los primeros 50 en cada examen? ¿Qué fracción de los objetos de la prueba ya estaba en el entrenamiento en cada uno? **Mi respuesta:**
- ¿Cuál de los tres exámenes se parece a lo que va a vivir el equipo de control interno el año entrante? ¿Cuál se parece a llevar el modelo a otro departamento? **Mi respuesta:**
- Si el modelo cae tanto en entidades nuevas, ¿qué estaba aprendiendo del texto? Respondan la pregunta de explicar. **Mi respuesta:**
- ¿Qué número habrían reportado si hubieran usado `train_test_split` sin pensar? **Mi respuesta:**

### 3.4 Explicar
'''),
("tarea", r'''
# TAREA 9. Con la regresión logística entrenada en E: los 20 términos con coeficiente más alto (empujan hacia
# prórroga) y los 20 más bajos, en dos gráficas de barras horizontales.
''', r'''
coeficientes = pd.Series(entrenados["logística"].coef_[0], index=terminos).sort_values()
fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 5.5))
coeficientes.tail(20).plot.barh(ax=izq, color="#D7263D"); izq.set(title="Empujan hacia «prórroga»", xlabel="coeficiente")
coeficientes.head(20).iloc[::-1].plot.barh(ax=der, color="#1D4E89"); der.set(title="Empujan en contra", xlabel="coeficiente")
plt.tight_layout(); plt.show()
'''),
("md", r'''
**Preguntas.**

- Separen los términos de la izquierda en dos montones: los que describen un tipo de trabajo (obra, interventoría, suministro) y los que nombran una entidad, un programa o un lugar. **Mi respuesta:**
- ¿Qué les dice el segundo montón sobre lo que pasará con el modelo cuando cambie la administración o el nombre de un programa? **Mi respuesta:**
- Un coeficiente alto no dice que esa palabra cause la prórroga. Escriban una frase que el equipo de control interno sí pueda usar. **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: la lista de los cincuenta

Ordenen los contratos de la prueba por el puntaje del modelo ganador y tomen los primeros `AUDITORES`. La incertidumbre se mide con *bootstrap* sobre los contratos de la prueba: en cada réplica se remuestrea la prueba, se vuelve a ordenar y se vuelve a contar, $\text{IC}_{95\%}=[\hat\theta^*_{(0{,}025)},\hat\theta^*_{(0{,}975)}]$.
'''),
("tarea", r'''
# TAREA 10. Con el ganador, en la prueba: P@AUDITORES con intervalo de 95 % por bootstrap (1.000 réplicas), la
# proporción de prórrogas de la prueba y cuántas veces mejor que el azar escoge el modelo. Dibujen P@k para k
# de 10 a 500. Muestren cuántos de los AUDITORES contratos escogidos son de cada entidad (las 6 primeras) y de
# cada tipo de contrato, y una tabla con los 10 primeros (entidad, tipo, valor, inicio del objeto, prórroga).
''', r'''
puntaje_P = puntaje(ganador, X_P)
orden = np.argsort(-puntaje_P, kind="stable")
elegidos = P[orden[:AUDITORES]]
aciertos = int(y[elegidos].sum())

rng = np.random.default_rng(SEMILLA)
replicas = []
for _ in range(1000):
    b = rng.integers(0, len(P), len(P))
    replicas.append(medir(y[P][b], puntaje_P[b])[f"P@{AUDITORES}"])
print(f"Modelo: {mejor} | Departamento: {DEPARTAMENTO} | Prueba: {len(P)} contratos del segundo semestre de 2023")
print(f"De los {AUDITORES} contratos escogidos, {aciertos} tuvieron prórroga: P@{AUDITORES} = {aciertos / AUDITORES:.2f} "
      f"(IC 95 %: {np.quantile(replicas, 0.025):.2f} a {np.quantile(replicas, 0.975):.2f})")
print(f"Proporción de prórrogas en la prueba: {y[P].mean():.3f} | el modelo escoge {aciertos / AUDITORES / y[P].mean():.1f} veces mejor que el azar")
print(f"Con la planilla sin texto: P@{AUDITORES} = {resultados['planilla (sin texto)'][f'P@{AUDITORES} (prueba)']:.2f}")

ks = np.arange(10, 501, 10)
plt.figure(figsize=(8, 3.8))
plt.plot(ks, [y[P][orden[:k]].mean() for k in ks], color="#D7263D", label=mejor)
plt.axhline(y[P].mean(), color="gray", ls=":", label="al azar"); plt.axvline(AUDITORES, color="#1B998B", ls="--")
plt.xlabel("contratos revisados (k)"); plt.ylabel("P@k"); plt.title("Precisión entre los primeros k, en la prueba"); plt.legend(); plt.show()

lista = tabla.iloc[elegidos]
print("Entidades de los contratos escogidos:"); print(lista["nombre_entidad"].value_counts().head(6).to_string())
print("\nTipos de contrato:"); print(lista["tipo_de_contrato"].value_counts().to_string())
display(lista.assign(objeto=lista["objeto_del_contrato"].str[:70])[["nombre_entidad", "tipo_de_contrato", "valor_del_contrato", "objeto", "prorroga"]].head(10))
'''),
("md", r'''
**Preguntas.**

- ¿Cuántos aciertos tuvo la lista y cuántos habría tenido una lista al azar? ¿El intervalo excluye al azar? **Mi respuesta:**
- ¿La lista se concentra en pocas entidades o en un tipo de contrato? ¿Eso es un hallazgo o un problema? **Mi respuesta:**
- Si los auditores solo revisan lo que el modelo señala, y el modelo se reentrena con lo que los auditores encuentran, ¿qué pasa con las entidades que el modelo nunca señala? **Mi respuesta:**

### La decisión en tres frases

Para la jefatura de control interno, que no sabe qué es TF-IDF.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Auditoría del código que les dio la IA

Comprueben estas tres cosas y escriban qué encontraron y en qué celda.

1. **Ninguna columna del futuro entró al modelo.** Revisen cada lista de columnas que use el código. No pueden estar `dias_adicionados`, `estado_contrato` ni `fecha_de_fin_del_contrato`. La celda siguiente mide cuánto «mejora» la planilla si se le cuela `estado_contrato`: ese es el tamaño del error que estarían firmando. **Lo que encontré:**
2. **El vocabulario se aprendió solo con el entrenamiento.** Busquen dónde se llama `fit` o `fit_transform` del vectorizador y con qué filas. En los exámenes repetidos debe ir dentro de un `Pipeline`. **Lo que encontré:**
3. **El número que se promete es el del examen por fecha.** Revisen que la lista de los cincuenta y su precisión salgan de la prueba del segundo semestre de 2023, que el ganador se haya escogido en validación y que la partición por fecha use la fecha de firma. **Lo que encontré:**
'''),
("tarea", r'''
# TAREA 11. Auditoría 1: entrenen de nuevo la planilla agregando estado_contrato a las columnas categóricas y
# comparen su AP y su P@AUDITORES en la prueba con los de la planilla correcta.
''', r'''
con_fuga = make_pipeline(
    make_column_transformer((OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORICAS + ["estado_contrato"]),
                            (StandardScaler(), ["log_valor"])),
    LogisticRegression(max_iter=2000, class_weight="balanced")).fit(tabla.iloc[E], y[E])
print("Planilla correcta:          ", {k: round(v, 3) for k, v in medir(y[P], planilla.predict_proba(tabla.iloc[P])[:, 1]).items()})
print("Planilla con estado_contrato:", {k: round(v, 3) for k, v in medir(y[P], con_fuga.predict_proba(tabla.iloc[P])[:, 1]).items()})
'''),
("md", r'''
### 4.3 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Prórroga es `dias_adicionados > 0` | Adiciones en valor | Es la columna que el portal publica; no hay columna de adición en valor en este conjunto |
| Contratos de 2022 y 2023 | | |
| Columnas que no entran al modelo | | |
| Partición por fecha de firma | | |
| No borrar los objetos repetidos | | |
| TF-IDF con palabras y pares | | |
| Precisión entre los primeros 50 | | |
| *Bootstrap* sobre los contratos de la prueba | | |

### 4.4 Riesgos y mantenimiento

- Una prórroga no es una irregularidad. ¿Cómo le presentarían la lista a la jefatura para que nadie la lea como una lista de sospechosos?
- El objeto del contrato lo redacta la entidad. Si se sabe que un modelo lee esas palabras, ¿qué puede pasar con la redacción?
- Los contratos firmados al final de 2023 han tenido menos tiempo para registrar prórrogas que los de 2022. ¿Hacia qué lado sesga eso la respuesta, y cómo lo comprobarían?
- El objeto a veces nombra personas. ¿Qué cuidado tendrían al publicar ejemplos?
- ¿Cada cuánto habría que reentrenar, y con qué examen comprobarían que el modelo sigue sirviendo?

### 4.5 Cambien una línea

Vuelvan a la celda del comienzo, cambien `DEPARTAMENTO = "Caldas"` por `DEPARTAMENTO = "Risaralda"` y corran todo de nuevo. La descarga nueva tarda medio minuto. Llenen la tabla.

| | Caldas | Risaralda |
|---|---|---|
| Contratos y entidades | | |
| Proporción de prórrogas | | |
| Objetos que repiten el de otro contrato | | |
| P@50 al azar, por fecha y por entidad (logística) | | |
| Ganador y su P@50 en la prueba, con intervalo | | |
| Veces mejor que el azar | | |

Con la tabla llena, respondan en tres líneas: ¿la receta sirve igual en los dos departamentos? ¿Entrenarían un solo modelo para los dos? **Mi respuesta:**

Dejen el cuaderno entregado con `DEPARTAMENTO = "Risaralda"` corrido y la tabla completa.

### 4.6 Entrega

**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase07/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]
