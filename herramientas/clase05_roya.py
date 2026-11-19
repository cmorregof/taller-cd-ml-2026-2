"""Celdas del gemelo de la clase 5: La roya (RoCoLe).

Cada ("tarea", comentario, solución) sale en el cuaderno del estudiante solo con el
comentario, y en la solución del profesor con el comentario y el código.
"""
from molde import ARRANQUE

ROYA = [
("md", r'''
# Clase 5 · La roya (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *Lunares*: la misma técnica con otras fotos.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase con fotos de hojas de café: extraer características con una red preentrenada, comparar cinco clasificadores contra una línea base, medir si partir al azar infla el resultado y convertir una probabilidad en una decisión con costos. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular), 2 (seleccionar y preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Desde esta clase el código lo escribe Claude y ustedes lo auditan. Pídanle el código de cada celda, léanlo y córranlo. Deben poder explicar sin ayuda por qué se parte por planta, qué midieron al comparar las particiones, de dónde sale el umbral y qué significa cada cifra que reportan. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, las respuestas a las preguntas, la sección de auditoría, la decisión en tres frases y la tabla de la variante llenas. |

**Cómo se trabaja.** Cada celda de código trae solo un comentario que dice qué debe lograr. Ustedes le piden el código a Claude, celda por celda o por secciones. Antes de pedirlo, péguenle el bloque «Lo que Claude necesita saber», porque Claude no conoce las funciones de este curso. Después de cada celda respondan la pregunta que la acompaña con sus propias palabras. Un cuaderno con código perfecto y preguntas sin responder no cumple el entregable.
'''),
("code", ARRANQUE),
("code", r'''
COSTO_FALSO_NEGATIVO = 5    # cuántas veces más cuesta no revisar una hoja enferma que revisar una sana. La cambiarán al final.
SEMILLA = 0                 # fija los sorteos

import time
from taller.clase05_imagenes import imagenes, rasgos_guardados, guardar_rasgos
'''),
("md", r'''
### Lo que Claude necesita saber

Copien este bloque al comienzo de su conversación con Claude.

> Trabajo en un cuaderno de Colab sin GPU. Ya están importados `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt` y `time`, y definidas las constantes `COSTO_FALSO_NEGATIVO` y `SEMILLA`.
>
> `tabla = descargar("rocole")` devuelve un DataFrame con una fila por foto y estas columnas: `foto` (nombre original, por ejemplo `C10P10E1.jpg`), `planta` (prefijo del nombre, por ejemplo `C10P10`; hay 390 plantas con 4 fotos cada una), `c` y `p` (los dos números del nombre), `letras` (las letras que siguen, casi siempre `E` o `H`), `estado` (`healthy` o `unhealthy`), `clase` (`healthy`, `rust_level_1` a `rust_level_4`, `red_spider_mite`), `bytes_original`, `sha256_original` y `archivo`.
>
> `carpeta = imagenes("rocole", tabla["archivo"])` devuelve la carpeta (un `pathlib.Path`) donde están las fotos en JPEG, ya reducidas a 256 píxeles en su lado corto. La foto de una fila es `carpeta / fila["archivo"]`.
>
> `rasgos_guardados(nombre, filas)` devuelve una matriz de numpy ya calculada o `None`, y `guardar_rasgos(nombre, matriz)` la guarda. Sirven para no repetir la extracción de características.
>
> Quiero usar ResNet18 de torchvision, preentrenada en ImageNet, como extractor fijo de características (sin entrenar la red), y encima modelos de scikit-learn. Todo debe correr en CPU en pocos minutos. Escribe código corto, con comentarios en español, y no uses datos que no sean los de la tabla.
'''),
("md", r'''
## 1. Formulación

Imagínense una finca de café robusta. La roya es un hongo que mancha las hojas, las tumba y deja a la planta sin con qué llenar el grano. La finca tiene un agrónomo y miles de plantas. Los trabajadores toman fotos de hojas con el teléfono mientras recorren los lotes, y el agrónomo quiere que un filtro le diga a cuáles plantas ir primero.

Las tres preguntas:

1. **Predecir.** Dada la foto de una hoja, ¿qué tan probable es que esté enferma?
2. **Explicar.** ¿En qué se apoya el modelo: en las manchas de la hoja o en algo que acompaña a la foto, como el fondo, la luz o el día en que se tomó?
3. **Decidir.** ¿A partir de qué probabilidad se manda al agrónomo a revisar la planta?

**El dato.** RoCoLe tiene 1.560 fotos de hojas de café robusta tomadas con un teléfono en el campus agropecuario de la ESPAM MFL en Calceta, Manabí, Ecuador. Lo publicaron Parraga-Alava, Cusme, Loor y Santander en 2019 (*Data in Brief* 25, 104414) con licencia CC BY 4.0. El artículo dice que hay cuatro fotos por cada una de 390 plantas y que cada hoja está etiquetada como sana o enferma. Las enfermas se clasifican en roya de nivel 1 a 4, según el área de la hoja con manchas (de 1 a 5 %, de 6 a 20 %, de 21 a 50 % y más de 50 %), o en arañita roja.

**Lo que no está documentado.** Los archivos se llaman `C10P10E1.jpg`, y el artículo no explica ese código. Las columnas `c`, `p` y `letras` guardan sus partes sin interpretarlas. Hay una sola inferencia mía: el prefijo `C…P…` forma 390 grupos de 4 fotos, lo mismo que el artículo dice de las plantas, así que lo trato como el identificador de la planta. Es una decisión de modelación y va a la bitácora.

**La métrica.** Positivo es «hoja enferma». Para comparar modelos usamos el área bajo la curva ROC,

$$\text{AUC}=P\bigl(s(X^+)>s(X^-)\bigr),$$

y para decidir, la sensibilidad y la especificidad con el umbral que salga de los costos,

$$\text{sensibilidad}=\frac{VP}{VP+FN},\qquad \text{especificidad}=\frac{VN}{VN+FP}.$$

**La decisión con costos.** Revisar una planta sana le cuesta al agrónomo una caminata, que tomamos como unidad. No revisar una planta con una hoja enferma cuesta $c$ veces eso, porque la roya se riega. Si el modelo dice que la hoja está enferma con probabilidad $p$, el costo esperado de no revisar es $p\,c$ y el de revisar es $1-p$. Conviene revisar cuando $p\,c\ge 1-p$, es decir, cuando

$$p\ \ge\ \tau^*=\frac{1}{1+c}.$$

El valor de $c$ es `COSTO_FALSO_NEGATIVO`. Es un supuesto que debe poner el agrónomo, porque el dato no lo trae, y por eso es la línea que van a cambiar al final.

### Predicciones escritas

Antes de correr nada:

- ¿Qué AUC alcanzará la red preentrenada con un clasificador encima? **Mi apuesta:**
- ¿Qué AUC alcanzará un modelo que solo conoce el color promedio de la foto? **Mi apuesta:**
- En *Lunares*, partir al azar infló el resultado. Aquí hay cuatro fotos por planta. ¿Se inflará más, menos o igual? ¿Por qué? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

### 2.1 Descargar y mirar

La descarga trae las 1.560 fotos una por una desde Mendeley Data y las reduce. Tarda cerca de tres minutos.
'''),
("tarea", r'''
# TAREA 1. Descarguen la tabla con descargar("rocole"), midan el tiempo y muestren cuántas fotos y
# cuántas plantas hay, y las primeras filas.
''', r'''
inicio = time.time()
tabla = descargar("rocole")
print(f"Tiempo: {time.time() - inicio:.0f} s")
print(len(tabla), "fotos |", tabla["planta"].nunique(), "plantas")
tabla.head()
'''),
("tarea", r'''
# TAREA 2. Cuenten: fotos por estado y por clase; fotos por planta; plantas según cuántas hojas
# enfermas tienen (0, 1, 2, 3 o 4); y las tablas cruzadas de `letras` contra `estado` y de `c` contra `estado`.
''', r'''
tabla["enferma"] = (tabla["estado"] == "unhealthy").astype(int)
y = tabla["enferma"].to_numpy()
planta = tabla["planta"].to_numpy()

print("Fotos por estado:"); print(tabla["estado"].value_counts().to_string())
print("\nFotos por clase:"); print(tabla["clase"].value_counts().to_string())
print("\nFotos por planta:", tabla.groupby("planta").size().value_counts().to_dict())
print("\nPlantas según cuántas de sus 4 hojas están enfermas:")
print(tabla.groupby("planta")["enferma"].sum().value_counts().sort_index().to_string())
display(pd.crosstab(tabla["letras"], tabla["estado"]))
display(pd.crosstab(tabla["c"], tabla["estado"]))
'''),
("md", r'''
**Preguntas.**

- ¿Cuántas plantas tienen a la vez hojas sanas y hojas enfermas? ¿Qué les dice eso sobre cómo se escogieron las hojas que se fotografiaron? **Mi respuesta:**
- Con ese diseño, ¿se puede estimar con este dato qué fracción de las plantas de la finca está enferma? ¿Y la fracción de hojas enfermas? **Mi respuesta:**
- La pregunta de decidir habla de revisar **plantas**, y la etiqueta es de **hojas**. Escriban en una línea cómo van a leer la decisión. **Mi respuesta:**
'''),
("tarea", r'''
# TAREA 3. Obtengan la carpeta de fotos con imagenes("rocole", tabla["archivo"]). Muestren las 4 fotos
# de tres plantas (una fila por planta, con la clase de cada foto en el título) y el tamaño de una foto.
''', r'''
from PIL import Image

carpeta = imagenes("rocole", tabla["archivo"])
rutas = [carpeta / a for a in tabla["archivo"]]

def abrir(ruta):
    return Image.open(ruta).convert("RGB")

fig, ejes = plt.subplots(3, 4, figsize=(9, 10))
for fila, identificador in enumerate(["C1P10", "C5P12", "C10P10"]):
    for eje, (_, foto) in zip(ejes[fila], tabla[tabla["planta"] == identificador].iterrows()):
        eje.imshow(abrir(carpeta / foto["archivo"])); eje.axis("off")
        eje.set_title(f"{foto['foto']}\n{foto['clase']}", fontsize=8)
plt.suptitle("Cada fila es una planta"); plt.tight_layout(); plt.show()
print("Tamaño de una foto guardada:", abrir(rutas[0]).size, "píxeles")
'''),
("md", r'''
**Pregunta.** En *Lunares* las fotos de una lesión eran casi la misma foto. Miren las cuatro fotos de una planta: ¿qué comparten y qué no? **Mi respuesta:**

### 2.2 Una línea base que no usa redes

Antes de la red, una descripción barata de cada foto: el promedio y la desviación estándar de cada canal de color (rojo, verde, azul). Son seis números por foto.
'''),
("tarea", r'''
# TAREA 4. Calculen para cada foto el promedio y la desviación estándar de cada canal RGB (matriz `colores`
# de 1.560 x 6). Dibujen la distribución del promedio de cada canal para hojas sanas y enfermas.
''', r'''
def seis_numeros(ruta):
    pixeles = np.asarray(abrir(ruta), dtype=float).reshape(-1, 3) / 255
    return np.concatenate([pixeles.mean(axis=0), pixeles.std(axis=0)])

colores = np.array([seis_numeros(r) for r in rutas])
fig, ejes = plt.subplots(1, 3, figsize=(13, 3.5))
for eje, canal, nombre in zip(ejes, range(3), ["rojo", "verde", "azul"]):
    for valor, color, etiqueta in [(0, "#1D4E89", "sana"), (1, "#D7263D", "enferma")]:
        eje.hist(colores[y == valor, canal], bins=30, alpha=0.6, color=color, density=True, label=etiqueta)
    eje.set(title=f"Promedio del canal {nombre}"); eje.legend()
plt.tight_layout(); plt.show()
'''),
("md", r'''
### 2.3 Características con una red preentrenada

Es el mismo paso de *Lunares*. La red $\phi$ convierte cada foto en un vector, $z_i=\phi(x_i)\in\mathbb{R}^{512}$, y sus pesos no se entrenan.
'''),
("tarea", r'''
# TAREA 5. Carguen ResNet18 preentrenada (torchvision), quítenle la última capa y pónganla en modo de
# evaluación. Extraigan las 512 características de cada foto, en el orden de la tabla, a la matriz X.
# Usen rasgos_guardados("rocole_resnet18", len(rutas)) y guardar_rasgos para no repetir el cálculo. Midan el tiempo.
''', r'''
import torch, torchvision

pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
red = torchvision.models.resnet18(weights=pesos, progress=False)
red.fc = torch.nn.Identity()
red.eval()
preparar = pesos.transforms()

def extraer(rutas, lote=64):
    partes = []
    with torch.no_grad():
        for i in range(0, len(rutas), lote):
            fotos = torch.stack([preparar(abrir(r)) for r in rutas[i:i + lote]])
            partes.append(red(fotos).numpy())
            if (i // lote) % 10 == 0:
                print(f"   {i} de {len(rutas)} fotos", flush=True)
    return np.concatenate(partes)

inicio = time.time()
X = rasgos_guardados("rocole_resnet18", len(rutas))
if X is None:
    X = extraer(rutas)
    guardar_rasgos("rocole_resnet18", X)
print(f"Matriz de características: {X.shape} | Tiempo: {time.time() - inicio:.0f} s")
'''),
("md", r'''
### 2.4 PCA y vecinos: ¿qué organiza el espacio?

Estandaricen las características y proyéctenlas con componentes principales, $\Sigma=\frac{1}{n-1}Z^\top Z=V\Lambda V^\top$. Después busquen, para cada foto, su vecina más cercana con la distancia $d(z_i,z_j)=\lVert z_i-z_j\rVert_2$.
'''),
("tarea", r'''
# TAREA 6. PCA de las características estandarizadas. Dibujen los dos primeros componentes dos veces:
# coloreados por estado y coloreados por `c`. Digan cuánta varianza explican.
''', r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(X)
pca = PCA(n_components=50, random_state=SEMILLA).fit(Z)
plano = pca.transform(Z)[:, :2]

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.5))
for valor, color, etiqueta in [(0, "#1D4E89", "sana"), (1, "#D7263D", "enferma")]:
    izq.scatter(plano[y == valor, 0], plano[y == valor, 1], s=6, alpha=0.5, color=color, label=etiqueta)
izq.set(title="Coloreado por estado", xlabel="CP1", ylabel="CP2"); izq.legend(markerscale=3)
puntos = der.scatter(plano[:, 0], plano[:, 1], s=6, alpha=0.6, c=tabla["c"], cmap="tab20")
der.set(title="Coloreado por el número C del nombre", xlabel="CP1", ylabel="CP2"); plt.colorbar(puntos, ax=der)
plt.tight_layout(); plt.show()
acumulada = pca.explained_variance_ratio_.cumsum() * 100
print(f"Dos componentes explican {acumulada[1]:.1f} % de la varianza; cincuenta, {acumulada[49]:.1f} %.")
'''),
("tarea", r'''
# TAREA 7. Para cada foto, encuentren su vecina más cercana en el espacio de características. Reporten en
# qué porcentaje de los casos la vecina es de la misma planta, tiene el mismo `c` y tiene el mismo estado.
# Comparen cada porcentaje con el que saldría si la vecina se escogiera al azar.
''', r'''
from sklearn.neighbors import NearestNeighbors

_, vecinos = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
vecina = vecinos[:, 1]
c = tabla["c"].to_numpy()
print(f"Vecina de la misma planta: {(planta[vecina] == planta).mean() * 100:.1f} %  (al azar: {100 * 3 / (len(tabla) - 1):.1f} %)")
al_azar_c = np.mean([((c == valor).sum() - 1) / (len(c) - 1) for valor in c]) * 100
print(f"Vecina con el mismo C:     {(c[vecina] == c).mean() * 100:.1f} %  (al azar: {al_azar_c:.1f} %)")
al_azar_y = np.mean([((y == valor).sum() - 1) / (len(y) - 1) for valor in y]) * 100
print(f"Vecina con el mismo estado: {(y[vecina] == y).mean() * 100:.1f} %  (al azar: {al_azar_y:.1f} %)")
'''),
("md", r'''
**Preguntas.**

- En *Lunares*, la vecina más cercana era de la misma lesión en cerca de la mitad de los casos. ¿Qué pasa aquí con la planta? **Mi respuesta:**
- ¿Qué pasa con el número `C`? No sabemos qué significa. ¿Qué podría ser, y cómo lo averiguarían? **Mi respuesta:**
- Con lo que acaban de ver, afinen su tercera apuesta: ¿partir al azar inflará el resultado? **Mi respuesta:**
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición

Tres tramos por **planta**: 60 % para entrenar, 20 % para validar y 20 % para la prueba. Todas las fotos de una planta van al mismo tramo,

$$\{g(i): i\in\text{entrenamiento}\}\ \cap\ \{g(j): j\in\text{prueba}\}=\varnothing .$$
'''),
("tarea", r'''
# TAREA 8. Partan los índices en entrenamiento (E), validación (V) y prueba (P) sorteando plantas, con SEMILLA.
# Impriman fotos, plantas y porcentaje de enfermas por tramo, y comprueben con un assert que ninguna
# planta aparece en dos tramos.
''', r'''
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def partir_por_grupo(indices, grupos, fraccion, semilla):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraccion, random_state=semilla).split(indices, groups=grupos[indices]))
    return indices[a], indices[b]

todas = np.arange(len(tabla))
resto, P = partir_por_grupo(todas, planta, 0.20, SEMILLA)
E, V = partir_por_grupo(resto, planta, 0.25, SEMILLA)
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {len(tramo):5d} fotos | {len(set(planta[tramo])):4d} plantas | {y[tramo].mean() * 100:.1f} % enfermas")
assert not set(planta[E]) & set(planta[V]) and not set(planta[E]) & set(planta[P]) and not set(planta[V]) & set(planta[P])
print("Ninguna planta aparece en dos tramos.")
'''),
("md", r'''
### 3.2 Línea base

Una regresión logística sobre los seis números de color, $P(Y=1\mid x)=\sigma(\beta_0+x^\top\beta)$ con $\sigma(t)=1/(1+e^{-t})$.
'''),
("tarea", r'''
# TAREA 9. Entrenen en E una regresión logística con los seis números de color (estandarizados) y reporten
# su AUC en validación y en prueba. Guarden el resultado en un diccionario `resultados`.
''', r'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline

base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(colores[E], y[E])
resultados = {"color (6 números)": {"validación": roc_auc_score(y[V], base.predict_proba(colores[V])[:, 1]),
                                    "prueba": roc_auc_score(y[P], base.predict_proba(colores[P])[:, 1])}}
display(pd.DataFrame(resultados).T.round(3))
'''),
("md", r'''
### 3.3 Cinco clasificadores

Los mismos cinco de la demostración, sobre las 512 características: regresión logística con castigo fuerte, máquina de soporte vectorial con núcleo gaussiano $K(z,z')=\exp(-\gamma\lVert z-z'\rVert^2)$, $k$ vecinos, bosque aleatorio y *gradient boosting* sobre 50 componentes principales.
'''),
("tarea", r'''
# TAREA 10. Definan los cinco modelos, entrénenlos en E y midan su AUC en validación y en prueba. Muestren
# una tabla ordenada por validación (con la línea base), una gráfica de barras y las curvas ROC en validación.
''', r'''
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def cinco_modelos():
    return {
        "logística": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "bosque": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEMILLA),
        "boosting": make_pipeline(StandardScaler(), PCA(n_components=50, random_state=SEMILLA),
                                  HistGradientBoostingClassifier(random_state=SEMILLA)),
    }

def puntaje(modelo, entradas):
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(entradas)[:, 1]
    return modelo.decision_function(entradas)

inicio = time.time()
entrenados, curvas = {}, {}
for nombre, modelo in cinco_modelos().items():
    entrenados[nombre] = modelo.fit(X[E], y[E])
    resultados[nombre] = {"validación": roc_auc_score(y[V], puntaje(modelo, X[V])),
                          "prueba": roc_auc_score(y[P], puntaje(modelo, X[P]))}
    curvas[nombre] = roc_curve(y[V], puntaje(modelo, X[V]))
comparacion = pd.DataFrame(resultados).T
print(f"Tiempo: {time.time() - inicio:.0f} s")

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.5))
comparacion.sort_values("validación").plot.barh(ax=izq, color=["#1D4E89", "#1B998B"])
izq.set(title="AUC: validación para escoger, prueba para reportar", xlabel="AUC", xlim=(0.45, 1))
for nombre, (fp, vp, _) in curvas.items():
    der.plot(fp, vp, label=f"{nombre} ({comparacion.loc[nombre, 'validación']:.3f})")
der.plot([0, 1], [0, 1], color="gray", ls=":")
der.set(title="Curvas ROC en validación", xlabel="1 − especificidad", ylabel="sensibilidad"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
display(comparacion.sort_values("validación", ascending=False).round(3))
'''),
("md", r'''
**Preguntas.**

- ¿Cuánto le gana la red a los seis números de color? Compárenlo con lo que pasó en *Lunares* entre la foto y la planilla. **Mi respuesta:**
- ¿Cuánto se diferencian los cinco modelos entre sí? **Mi respuesta:**

### 3.4 La trampa: ¿cuánto infla partir al azar?

Aquí hay tres maneras de partir y las tres se miden: al azar por foto, por planta y por el número `C` del nombre (todas las fotos con el mismo `C` al mismo lado). El tamaño de la prueba debe ser comparable en las tres.
'''),
("tarea", r'''
# TAREA 11. Repitan 30 veces, con semillas 0 a 29, y para la regresión logística y para un vecino (k = 1):
# partir al azar por foto, partir por planta y partir por `c`, siempre con cerca de 20 % en prueba.
# Guarden el AUC de la logística y la exactitud del vecino en cada partición. Muestren la media y la
# desviación estándar de cada columna y dos diagramas de caja.
''', r'''
inicio = time.time()
repeticiones = []
for semilla in range(30):
    particiones = {
        "por planta": partir_por_grupo(todas, planta, 0.20, semilla),
        "por C": partir_por_grupo(todas, c, 0.20, semilla),
    }
    particiones["al azar"] = train_test_split(todas, test_size=len(particiones["por planta"][1]), random_state=semilla, stratify=y)
    fila = {}
    for nombre in ["al azar", "por planta", "por C"]:
        entren, prueba = particiones[nombre]
        logistica = cinco_modelos()["logística"].fit(X[entren], y[entren])
        fila[f"logística (AUC), {nombre}"] = roc_auc_score(y[prueba], puntaje(logistica, X[prueba]))
        un_vecino = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)).fit(X[entren], y[entren])
        fila[f"1 vecino (exactitud), {nombre}"] = (un_vecino.predict(X[prueba]) == y[prueba]).mean()
    repeticiones.append(fila)
repeticiones = pd.DataFrame(repeticiones)
print(f"Tiempo: {time.time() - inicio:.0f} s")

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4))
logisticas = [k for k in repeticiones if k.startswith("logística")]
vecinos_1 = [k for k in repeticiones if k.startswith("1 vecino")]
repeticiones[logisticas].rename(columns=lambda k: k.split(", ")[1]).plot.box(ax=izq, color="#1D4E89"); izq.set(title="Regresión logística: AUC en 30 sorteos")
repeticiones[vecinos_1].rename(columns=lambda k: k.split(", ")[1]).plot.box(ax=der, color="#D7263D"); der.set(title="Un vecino: exactitud en 30 sorteos")
plt.tight_layout(); plt.show()
display(repeticiones.agg(["mean", "std"]).T.round(3))
'''),
("md", r'''
**Preguntas.** Respondan con los números de su tabla.

- ¿Cuánto infla el AUC partir al azar frente a partir por planta? Compárenlo con la desviación estándar entre sorteos. ¿La diferencia es clara o se confunde con el ruido? **Mi respuesta:**
- En *Lunares* el inflado fue mayor. Expliquen la diferencia con lo que vieron en la tarea 3 y en la tarea 7. **Mi respuesta:**
- ¿Qué pasa al partir por `C`? ¿Qué les dice eso sobre lo que el modelo aprendió? **Mi respuesta:**
- Si el inflado por planta es pequeño, ¿se justifica de todos modos partir por planta? **Mi respuesta:**

### 3.5 El veredicto
'''),
("tarea", r'''
# TAREA 12. Escojan como ganador el modelo con mayor AUC en validación entre los cinco y reporten su AUC en
# prueba junto al de la línea base de color.
''', r'''
mejor = comparacion.loc[list(cinco_modelos()), "validación"].idxmax()
ganador = entrenados[mejor]
print(f"Ganador en validación: {mejor}")
print(f"AUC en prueba: {comparacion.loc[mejor, 'prueba']:.3f} | color: {comparacion.loc['color (6 números)', 'prueba']:.3f}")
'''),
("md", r'''
### 3.6 Explicar

Tres caminos, como en la demostración: ver si las características delatan algo ajeno a la hoja, ver en qué clase de hoja falla el modelo, y tapar pedazos de la foto, con $I_k=s(x)-s\bigl(x^{(k)}\bigr)$.
'''),
("tarea", r'''
# TAREA 13. Entrenen en E una regresión logística que adivine el número `c` a partir de las características y
# midan su acierto en validación. Compárenlo con responder siempre el `c` más común.
''', r'''
detector = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[E], c[E])
print(f"Adivinar el número C con las características: {(detector.predict(X[V]) == c[V]).mean() * 100:.0f} % de acierto en validación")
print(f"Responder siempre el C más común: {pd.Series(c[E]).value_counts(normalize=True).iloc[0] * 100:.0f} %")
'''),
("tarea", r'''
# TAREA 14. Con el modelo ganador y el umbral de la sección 1 (si el ganador no entrega probabilidades,
# usen la regresión logística), calculen en validación qué fracción de las hojas de cada `clase` se manda
# a revisar. Muestren también cuántas hojas hay de cada clase.
''', r'''
umbral = 1 / (1 + COSTO_FALSO_NEGATIVO)
decisor_nombre = mejor if hasattr(ganador, "predict_proba") else "logística"
decisor = entrenados[decisor_nombre]
prob_V, prob_P = decisor.predict_proba(X[V])[:, 1], decisor.predict_proba(X[P])[:, 1]
revisar_V, revisar_P = prob_V >= umbral, prob_P >= umbral

por_clase = pd.DataFrame({"hojas": tabla["clase"].iloc[V].value_counts(),
                          "se mandan a revisar": pd.Series(revisar_V).groupby(tabla["clase"].iloc[V].to_numpy()).mean()})
print(f"Modelo para decidir: {decisor_nombre} | Umbral: {umbral:.3f}")
display(por_clase.round(2))
'''),
("tarea", r'''
# TAREA 15. Mapa de oclusión: tapen la foto por parches de 32 píxeles (sobre el recorte de 224 x 224 que ve
# la red), midan cuánto baja la probabilidad de «enferma» al tapar cada parche y dibújenlo sobre la foto.
# Háganlo para dos hojas enfermas con probabilidad alta y dos hojas sanas con probabilidad alta, de validación.
''', r'''
def mapa_de_oclusion(ruta, lado=32):
    original = preparar(abrir(ruta))
    variantes = [original]
    for fila in range(0, 224, lado):
        for columna in range(0, 224, lado):
            tapada = original.clone()
            tapada[:, fila:fila + lado, columna:columna + lado] = 0
            variantes.append(tapada)
    with torch.no_grad():
        rasgos = red(torch.stack(variantes)).numpy()
    s = decisor.predict_proba(rasgos)[:, 1]
    return (s[0] - s[1:]).reshape(224 // lado, 224 // lado), s[0]

def recorte(ruta):
    foto = abrir(ruta); ancho, alto = foto.size; escala = 256 / min(ancho, alto)
    foto = foto.resize((round(ancho * escala), round(alto * escala)))
    izquierda, arriba = (foto.width - 224) // 2, (foto.height - 224) // 2
    return foto.crop((izquierda, arriba, izquierda + 224, arriba + 224))

orden = np.argsort(-prob_V)
elegidas = [V[i] for i in orden if y[V][i] == 1][:2] + [V[i] for i in orden if y[V][i] == 0][:2]
fig, ejes = plt.subplots(2, 4, figsize=(13, 6.4))
for columna, i in enumerate(elegidas):
    mapa, s0 = mapa_de_oclusion(rutas[i])
    ejes[0, columna].imshow(recorte(rutas[i])); ejes[0, columna].axis("off")
    ejes[0, columna].set_title(f"{tabla['clase'][i]} · p = {s0:.2f}", fontsize=9)
    tope = np.abs(mapa).max()
    ejes[1, columna].imshow(recorte(rutas[i]))
    ejes[1, columna].imshow(mapa, cmap="RdBu_r", vmin=-tope, vmax=tope, alpha=0.6, extent=(0, 224, 224, 0), interpolation="nearest")
    ejes[1, columna].axis("off")
plt.suptitle("Rojo: al tapar esa región la probabilidad de «enferma» baja. Azul: al taparla, sube"); plt.tight_layout(); plt.show()
'''),
("md", r'''
**Preguntas.**

- ¿Las características delatan el número `C`? ¿Qué riesgo trae eso si la finca toma fotos en un lote nuevo? **Mi respuesta:**
- ¿Qué clase de hoja enferma se le escapa más al modelo? ¿Tiene sentido agronómico y visual? **Mi respuesta:**
- En los mapas, ¿el rojo cae sobre la hoja, sobre las manchas o sobre el fondo? **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir con costos

El umbral $\tau^*=1/(1+c)$ supone que $p$ es una probabilidad de verdad: que de las hojas con $p=0{,}3$, cerca de 30 % estén enfermas. Eso se llama calibración y hay que comprobarlo antes de usar el umbral. Después se mide el costo por cada 100 hojas en la prueba,

$$\text{costo}=\frac{100}{n}\,\bigl(FP+c\cdot FN\bigr),$$

y se compara con las dos políticas que no necesitan modelo: revisar todo y no revisar nada. La incertidumbre se mide con *bootstrap* por planta, $\text{IC}_{95\%}=\bigl[\hat\theta^{*}_{(0{,}025)},\ \hat\theta^{*}_{(0{,}975)}\bigr]$.
'''),
("tarea", r'''
# TAREA 16. Comprueben la calibración en validación: partan las probabilidades en 8 grupos de igual tamaño
# y dibujen la probabilidad promedio de cada grupo contra la fracción real de enfermas, con la diagonal.
''', r'''
grupos = pd.qcut(prob_V, 8, duplicates="drop")
calibracion = pd.DataFrame({"p": prob_V, "y": y[V]}).groupby(grupos, observed=True).agg(predicha=("p", "mean"), observada=("y", "mean"), hojas=("y", "size"))
plt.figure(figsize=(4.8, 4.5))
plt.plot(calibracion["predicha"], calibracion["observada"], "o-", color="#D7263D"); plt.plot([0, 1], [0, 1], color="gray", ls=":")
plt.axvline(umbral, color="#1B998B", ls="--", label=f"umbral {umbral:.2f}")
plt.xlabel("probabilidad predicha"); plt.ylabel("fracción observada de enfermas"); plt.title("Calibración en validación"); plt.legend(); plt.show()
display(calibracion.round(2))
'''),
("tarea", r'''
# TAREA 17. En la prueba, con el umbral 1/(1 + COSTO_FALSO_NEGATIVO): matriz de confusión, sensibilidad,
# especificidad, hojas revisadas por cada 100 y costo por cada 100 hojas, con intervalos de 95 % por
# bootstrap de plantas (1.000 réplicas). Comparen el costo con «revisar todo» y «no revisar nada».
''', r'''
def cifras(etiquetas, revisar):
    vp, fn = (revisar & (etiquetas == 1)).sum(), (~revisar & (etiquetas == 1)).sum()
    vn, fp = (~revisar & (etiquetas == 0)).sum(), (revisar & (etiquetas == 0)).sum()
    n = len(etiquetas)
    return {"sensibilidad": vp / (vp + fn), "especificidad": vn / (vn + fp), "revisadas por cada 100": 100 * revisar.mean(),
            "costo por cada 100 (modelo)": 100 * (fp + COSTO_FALSO_NEGATIVO * fn) / n,
            "costo por cada 100 (revisar todo)": 100 * (vn + fp) / n,
            "costo por cada 100 (no revisar nada)": 100 * COSTO_FALSO_NEGATIVO * (vp + fn) / n}

rng = np.random.default_rng(SEMILLA)
plantas_P = pd.unique(planta[P])
filas_de = {g: np.flatnonzero(planta[P] == g) for g in plantas_P}
replicas = []
for _ in range(1000):
    filas = np.concatenate([filas_de[g] for g in rng.choice(plantas_P, len(plantas_P))])
    replicas.append(cifras(y[P][filas], revisar_P[filas]))
replicas = pd.DataFrame(replicas)
informe = pd.DataFrame({"validación": cifras(y[V], revisar_V), "prueba": cifras(y[P], revisar_P),
                        "IC 95 % inferior": replicas.quantile(0.025), "IC 95 % superior": replicas.quantile(0.975)})
print(f"Modelo: {decisor_nombre} | c = {COSTO_FALSO_NEGATIVO} | umbral = {umbral:.3f} | AUC en prueba = {roc_auc_score(y[P], prob_P):.3f}")
display(informe.round(3))
display(pd.crosstab(pd.Series(np.where(y[P] == 1, "enferma", "sana"), name="etiqueta"),
                    pd.Series(np.where(revisar_P, "se revisa", "no se revisa"), name="filtro")))
'''),
("md", r'''
**Preguntas.**

- ¿El modelo está bien calibrado cerca del umbral? Si no lo está, ¿hacia qué lado se equivoca la regla? **Mi respuesta:**
- ¿Cuánto ahorra el modelo frente a la mejor de las dos políticas sin modelo? ¿El intervalo deja claro que ahorra? **Mi respuesta:**

### La decisión en tres frases

Para el agrónomo, que no sabe qué es una curva ROC.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Auditoría del código que les dio la IA

El código lo escribió Claude y la firma es de ustedes. Comprueben estas tres cosas y escriban qué encontraron y en qué celda.

1. **La partición no tiene fuga.** Ninguna planta aparece en dos tramos, y hay un `assert` que lo comprueba. Los pasos que aprenden del dato (estandarización, PCA) están dentro de un `Pipeline` y se ajustan solo con el entrenamiento. **Lo que encontré:**
2. **La prueba se usó una sola vez.** El modelo ganador y el umbral se escogieron sin mirar la prueba. Busquen en el código cualquier decisión que dependa de `P`. **Lo que encontré:**
3. **Las características corresponden a las fotos.** La red está en modo de evaluación (`eval()`), la foto se prepara con la transformación que acompaña a los pesos, y la fila $i$ de `X` es la fila $i$ de la tabla. Compruébenlo recalculando las características de tres fotos sueltas y comparándolas con sus filas de `X`. **Lo que encontré:**
'''),
("tarea", r'''
# TAREA 18. Auditoría 3: recalculen las características de las fotos de las filas 0, 500 y 1500 y comprueben
# que coinciden con las filas correspondientes de X (diferencia máxima cercana a cero).
''', r'''
for i in [0, 500, 1500]:
    with torch.no_grad():
        de_nuevo = red(preparar(abrir(rutas[i])).unsqueeze(0)).numpy()[0]
    print(f"Fila {i:4d} ({tabla['foto'][i]}): diferencia máxima {np.abs(de_nuevo - X[i]).max():.5f}")
print("¿La red está en modo de evaluación?", not red.training)
'''),
("md", r'''
### 4.3 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| El prefijo `C…P…` del nombre es la planta | Tratar cada foto como independiente | Forma 390 grupos de 4 fotos, como dice el artículo; el código no está documentado |
| Sana contra enferma | | |
| Fotos reducidas y recorte central de 224 píxeles | | |
| Red preentrenada como extractor fijo | | |
| Partición por planta en tres tramos | | |
| Línea base con seis números de color | | |
| Umbral $1/(1+c)$ | | |
| Valor de $c$ | | |

### 4.4 Riesgos y mantenimiento

- Las fotos son de café robusta en Manabí, Ecuador. En Caldas se siembra sobre todo café arábigo. ¿Qué habría que hacer antes de usar este modelo en una finca de Chinchiná?
- Casi todas las plantas tienen hojas sanas y enfermas porque así se diseñó la toma. ¿Qué le pasa al costo calculado si en la finca solo una de cada veinte hojas está enferma?
- El valor de $c$ lo pusimos nosotros. ¿Quién debería ponerlo y con qué información?
- Si las características delatan el número `C`, ¿qué prueba harían antes de confiar en el modelo en un lote nuevo?

### 4.5 Cambien una línea

Vuelvan a la celda del comienzo donde dice `COSTO_FALSO_NEGATIVO = 5`, pongan `COSTO_FALSO_NEGATIVO = 20` y corran todo de nuevo. Las características ya están guardadas. Llenen la tabla con sus dos corridas.

| En la prueba | $c=5$ | $c=20$ |
|---|---|---|
| Umbral | | |
| Sensibilidad | | |
| Especificidad | | |
| Hojas revisadas por cada 100 | | |
| Costo por cada 100 hojas (modelo) | | |
| Costo por cada 100 hojas (revisar todo) | | |

Con la tabla llena, respondan en tres líneas: ¿a partir de qué valor de $c$ deja de valer la pena el modelo frente a revisar todo? ¿Qué le dirían al agrónomo? **Mi respuesta:**

Dejen el cuaderno entregado con `COSTO_FALSO_NEGATIVO = 20` corrido y la tabla completa.

### 4.6 Procedencia y entrega
'''),
("tarea", r'''
# TAREA 19. Muestren la procedencia del dato con procedencia()["rocole"].
''', r'''
procedencia()["rocole"]
'''),
("md", r'''
**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase05/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]
