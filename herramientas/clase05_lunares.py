"""Celdas de la demostración de la clase 5: Lunares (HAM10000)."""
from molde import ARRANQUE

LUNARES = [
("md", r'''
# Clase 5 · Lunares

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. Ustedes lo siguen, lo corren y me interrumpen. En la segunda mitad cada uno resuelve el gemelo, *La roya*, con la misma técnica y otras fotos.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Clasificar imágenes con una red preentrenada usada como extractor fijo de características, comparar cinco clasificadores contra una línea base seria y medir con números cuánto se infla el resultado cuando dos fotos de la misma lesión caen a lados distintos de la partición. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular el problema y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que escriba una gráfica o que proponga otro clasificador. Deben poder explicar sin ayuda qué es una característica extraída por una red preentrenada, por qué se parte por lesión y no por imagen, cuánto cambió el resultado al partir mal y qué significan la sensibilidad y la especificidad que reportan. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con las predicciones escritas, la decisión en tres frases y la tabla de la variante llenas. |
'''),
("code", ARRANQUE),
("md", r'''
La celda siguiente tiene el único parámetro del cuaderno. Déjenlo como está. Al final de la clase lo cambian y comparan.
'''),
("code", r'''
SENSIBILIDAD = 0.90     # de cada 100 melanomas, cuántos exigimos que el filtro remita al dermatólogo
SEMILLA = 0             # fija los sorteos para que todos veamos los mismos números

import time
from taller.clase05_imagenes import imagenes, rasgos_guardados, guardar_rasgos
'''),
("md", r'''
## 1. Formulación

Imagínense un programa de teledermatología. En los puestos de salud alguien le toma una foto a un lunar con un dermatoscopio, que es una lupa con luz polarizada, y la foto viaja a la ciudad. Allá hay pocos dermatólogos y la fila es larga. El programa quiere un filtro que ordene la fila, de modo que los lunares sospechosos de melanoma se vean primero.

El programa necesita tres respuestas:

1. **Predecir.** Dada la foto de una lesión, ¿qué tan probable es que sea un melanoma?
2. **Explicar.** ¿En qué se apoya el modelo: en la lesión, o en algo que acompaña a la foto, como la cámara o el consultorio donde se tomó?
3. **Decidir.** ¿A partir de qué puntaje se remite la lesión al dermatólogo, si exigimos que el filtro deje pasar como máximo 10 de cada 100 melanomas?

**El dato.** HAM10000 reúne 10.015 imágenes dermatoscópicas de lesiones pigmentadas, recogidas durante veinte años en la Universidad Médica de Viena y en un consultorio de Queensland, Australia. Lo publicaron Tschandl, Rosendahl y Kittler en 2018 con licencia CC BY-NC 4.0. Cada imagen trae un diagnóstico y, lo que hoy nos importa más, el identificador de la lesión fotografiada.

**La métrica.** Llamo positivo al melanoma. El modelo le da a cada foto un puntaje $s(x)$ y se remite cuando $s(x)\ge\tau$. Con un umbral $\tau$ fijo se cuentan los verdaderos positivos (VP), los falsos negativos (FN), los verdaderos negativos (VN) y los falsos positivos (FP), y se definen

$$\text{sensibilidad}=\frac{VP}{VP+FN},\qquad \text{especificidad}=\frac{VN}{VN+FP}.$$

La sensibilidad cuenta los melanomas que el filtro atrapa. La especificidad cuenta los lunares benignos que deja tranquilos. Para comparar modelos sin casarnos todavía con un umbral usamos el área bajo la curva ROC, que tiene una lectura directa: es la probabilidad de que un melanoma tomado al azar reciba un puntaje mayor que un nevus tomado al azar,

$$\text{AUC}=P\bigl(s(X^+)>s(X^-)\bigr).$$

Un AUC de 0,5 es tirar una moneda y un AUC de 1 es ordenar perfecto.

### Predicción escrita

Escriban su apuesta antes de ver un solo dato. Después la comparamos con lo que salga.

- Si a un modelo solo le digo la **edad** del paciente, sin foto, ¿qué AUC alcanza? **Mi apuesta:**
- ¿Qué AUC alcanza un modelo que sí ve la foto, a través de una red que nunca vio un lunar? **Mi apuesta:**
- Hay lesiones con dos o tres fotos. Si parto al azar por imagen, sin fijarme en eso, ¿el AUC sube, baja o queda igual? ¿Cuánto? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

Una línea trae el dato desde Harvard Dataverse. La tabla llega completa, con las 10.015 filas. Las fotos no: los dos archivos de imágenes pesan 2,8 GB, así que la función baja solo el subconjunto que vamos a usar (más abajo lo defino y lo justifico) y guarda cada foto reducida a 256 píxeles en su lado corto. La primera vez tarda entre dos y tres minutos.
'''),
("code", r'''
inicio = time.time()
tabla = descargar("ham10000")
print(f"Tiempo: {time.time() - inicio:.0f} s")
print(len(tabla), "imágenes |", tabla["lesion_id"].nunique(), "lesiones |", int(tabla["en_muestra"].sum()), "fotos descargadas")
tabla.head()
'''),
("md", r'''
### 2.1 Qué dice cada columna

Los nombres son los del archivo original. El significado de los códigos sale del artículo de los autores (Tschandl, Rosendahl y Kittler, *Scientific Data* 5, 180161, 2018).

| Columna | Qué es |
|---|---|
| `lesion_id` | Identificador de la lesión. Varias imágenes pueden compartirlo |
| `image_id` | Identificador de la imagen |
| `dx` | Diagnóstico: `nv` nevus melanocítico (el lunar común), `mel` melanoma, `bkl` queratosis benignas, `bcc` carcinoma basocelular, `akiec` queratosis actínica y carcinoma intraepitelial, `vasc` lesiones vasculares, `df` dermatofibroma |
| `dx_type` | Cómo se confirmó el diagnóstico: `histo` histopatología (se extirpó y se miró al microscopio), `follow_up` seguimiento en el tiempo, `consensus` acuerdo de expertos, `confocal` microscopía confocal |
| `age`, `sex`, `localization` | Edad, sexo y sitio del cuerpo |
| `dataset` | De dónde viene la imagen: `rosendahl` es el consultorio de Queensland; `vidir_modern`, `vidir_molemax` y `vienna_dias` son tres épocas y equipos del grupo de Viena |
| `en_muestra`, `archivo` | Las agregué yo: si la foto está en el subconjunto del curso y cómo se llama su archivo |
'''),
("code", r'''
print("Imágenes por diagnóstico:")
print(tabla["dx"].value_counts().to_string())
print("\nDiagnóstico según cómo se confirmó:")
display(pd.crosstab(tabla["dx_type"], tabla["dx"]))
print("Diagnóstico según el origen de la imagen:")
display(pd.crosstab(tabla["dataset"], tabla["dx"]))
'''),
("md", r'''
Estas dos tablas cruzadas traen una trampa que conviene ver antes de entrenar nada.

Miren la fila `vidir_molemax`. Tiene 3.720 nevus y 24 melanomas. Casi todos esos nevus se confirmaron por seguimiento: son lunares que alguien vigiló en el tiempo y que nunca dieron motivo para extirparlos. Si entrenáramos con todo, el modelo podría aprender que una foto con el aspecto de ese equipo es un nevus, y acertaría sin mirar la lesión. Eso se llama un **atajo**.

Miren también la columna `mel`. Todos los melanomas tienen histopatología, porque un melanoma solo se confirma al microscopio. En cambio, buena parte de los nevus nunca se extirpó. Comparar melanomas extirpados con lunares que nadie consideró sospechosos responde una pregunta fácil, y el dermatólogo tiene la difícil: de las lesiones que alguien decidió extirpar, cuáles eran melanoma.

**Decisión de modelación.** Me quedo con los melanomas y los nevus confirmados por histopatología. Así los dos grupos pasaron por el mismo filtro clínico y el atajo de `vidir_molemax` casi desaparece. La decisión va a la bitácora de la sección 4, con su costo: el modelo queda entrenado en lesiones que un médico ya consideró dignas de biopsia.
'''),
("code", r'''
muestra = tabla[tabla["en_muestra"]].reset_index(drop=True)
muestra["melanoma"] = (muestra["dx"] == "mel").astype(int)
y = muestra["melanoma"].to_numpy()
lesion = muestra["lesion_id"].to_numpy()

print(len(muestra), "imágenes de", muestra["lesion_id"].nunique(), "lesiones")
print(f"Melanomas: {y.sum()} imágenes ({y.mean() * 100:.1f} %)")
display(pd.crosstab(muestra["dataset"], muestra["dx"]))
'''),
("md", r'''
### 2.2 La estructura que no se ve en la tabla: varias fotos por lesión

Una fila de la tabla es una imagen. El paciente, en cambio, llega con una lesión, y a muchas lesiones les tomaron dos o tres fotos, con otro encuadre o con otro aumento.
'''),
("code", r'''
fotos_por_lesion = muestra.groupby("lesion_id").size()
conteo = fotos_por_lesion.value_counts().sort_index()
print("Lesiones según cuántas fotos tienen:")
print(conteo.to_string())
print(f"\nLesiones con más de una foto: {(fotos_por_lesion > 1).mean() * 100:.0f} %")
print(f"Imágenes que tienen al menos una hermana de la misma lesión: {(fotos_por_lesion[muestra['lesion_id']] > 1).mean() * 100:.0f} %")
print("¿Alguna lesión cambia de diagnóstico entre fotos?", bool((muestra.groupby("lesion_id")["dx"].nunique() > 1).any()))
'''),
("code", r'''
from PIL import Image

carpeta = imagenes("ham10000", muestra["archivo"])
rutas = [carpeta / a for a in muestra["archivo"]]

def abrir(ruta):
    return Image.open(ruta).convert("RGB")

# Tres lesiones con tres fotos cada una: las de cada columna son la misma lesión
ejemplos = fotos_por_lesion[fotos_por_lesion == 3].index[:4]
fig, ejes = plt.subplots(3, 4, figsize=(11, 6.5))
for columna, identificador in enumerate(ejemplos):
    filas = muestra[muestra["lesion_id"] == identificador]
    for fila, (_, foto) in enumerate(filas.iterrows()):
        ejes[fila, columna].imshow(abrir(carpeta / foto["archivo"])); ejes[fila, columna].axis("off")
    ejes[0, columna].set_title(f"{identificador} ({filas['dx'].iloc[0]})", fontsize=9)
plt.suptitle("Cada columna es una sola lesión fotografiada tres veces"); plt.tight_layout(); plt.show()
'''),
("md", r'''
Guarden esta imagen para la sección 3. Las fotos de una columna no son observaciones independientes: comparten la lesión, la piel, el paciente y el diagnóstico. Si una cae en entrenamiento y otra en prueba, el modelo presenta un examen del que ya vio la respuesta.

### 2.3 Quiénes son los pacientes

Antes de mirar píxeles miro lo que la tabla dice de las personas. Si la edad o el sitio del cuerpo ya separan los dos grupos, un modelo de imágenes tiene que ganarle a eso para justificar su costo.
'''),
("code", r'''
print("Faltantes por columna:")
print(muestra[["age", "sex", "localization"]].isna().sum().to_string(), "| sexo 'unknown':", (muestra["sex"] == "unknown").sum(), "| sitio 'unknown':", (muestra["localization"] == "unknown").sum())

fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(14, 4))
for etiqueta, color in [("nv", "#1D4E89"), ("mel", "#D7263D")]:
    muestra.loc[muestra["dx"] == etiqueta, "age"].plot.hist(ax=izq, bins=range(0, 91, 5), alpha=0.6, color=color, density=True, label=etiqueta)
izq.set(title="Edad según el diagnóstico", xlabel="años", ylabel="densidad"); izq.legend()

por_sexo = muestra[muestra["sex"] != "unknown"].groupby("sex")["melanoma"].mean() * 100
por_sexo.plot.bar(ax=centro, color="#1B998B", rot=0); centro.set(title="Melanomas por sexo", xlabel="", ylabel="% de las imágenes")

por_sitio = muestra.groupby("localization")["melanoma"].agg(["mean", "size"]).query("size >= 40").sort_values("mean")
(por_sitio["mean"] * 100).plot.barh(ax=der, color="#1D4E89"); der.axvline(y.mean() * 100, color="gray", ls=":")
der.set(title="Melanomas por sitio del cuerpo (40 imágenes o más)", xlabel="% de las imágenes", ylabel="")
plt.tight_layout(); plt.show()

print(muestra.groupby("dx")["age"].describe()[["count", "mean", "25%", "50%", "75%"]].round(1))
'''),
("md", r'''
La edad separa bastante: los melanomas de este conjunto son de pacientes mayores. El sexo y el sitio también se asocian con el diagnóstico. Dos advertencias. La primera es que estas proporciones describen a los pacientes de dos clínicas, una de Austria y una de Australia, y no a la población de ningún país. La segunda es que el conjunto no trae el tono de piel, así que no podemos medir si el modelo funciona igual en pieles distintas. Esa ausencia vuelve en la sección de riesgos.

### 2.4 Las fotos
'''),
("code", r'''
rng = np.random.default_rng(SEMILLA)
fig, ejes = plt.subplots(2, 7, figsize=(14, 4.4))
for fila, (etiqueta, nombre) in enumerate([("mel", "melanoma"), ("nv", "nevus")]):
    elegidas = rng.choice(np.flatnonzero((muestra["dx"] == etiqueta).to_numpy()), 7, replace=False)
    for eje, i in zip(ejes[fila], elegidas):
        eje.imshow(abrir(rutas[i])); eje.axis("off"); eje.set_title(muestra["dataset"][i], fontsize=7)
    ejes[fila, 0].text(-0.1, 0.5, nombre, transform=ejes[fila, 0].transAxes, rotation=90, va="center", ha="right", fontsize=11)
plt.suptitle("Arriba, melanomas. Abajo, nevus. El título de cada foto es su origen"); plt.tight_layout(); plt.show()
print("Tamaño de una foto guardada:", abrir(rutas[0]).size, "píxeles")
'''),
("md", r'''
Intenten separarlos a ojo. A un dermatólogo le toma años de entrenamiento, y aun así manda a biopsia cuando duda. Fíjense además en que el fondo, el color y el encuadre cambian con el origen de la foto.

### 2.5 De píxeles a características: aprendizaje por transferencia

Una foto de $224\times224$ píxeles en tres colores son 150.528 números. Con 3.611 fotos no alcanza para entrenar desde cero una red que los entienda. La salida es pedir prestado. ResNet18 es una red convolucional que ya fue entrenada con 1,28 millones de fotos de ImageNet para distinguir mil clases de objetos. Nunca vio un lunar, pero para distinguir perros de teteras tuvo que aprender bordes, texturas, manchas y formas. Esas piezas sirven para otras fotos.

La red es una composición de capas, $f=g\circ\phi$. La parte $\phi$ convierte la imagen en un vector de 512 números y la última capa $g$ convierte ese vector en mil puntajes. Le quitamos $g$ y nos quedamos con

$$\phi:\ \mathbb{R}^{224\times224\times3}\longrightarrow\mathbb{R}^{512},\qquad z_i=\phi(x_i).$$

Los pesos de $\phi$ quedan congelados: no se entrena nada de la red. Cada foto pasa una sola vez y sale convertida en una fila de 512 columnas. Sobre esa tabla trabajan los modelos de siempre, y por eso todo corre sin GPU. A esto se le llama usar la red como **extractor fijo de características**. La alternativa, reentrenar también los pesos de $\phi$ con nuestras fotos, se llama ajuste fino. Suele rendir más y pide GPU y más cuidado.

La celda tarda entre dos y cuatro minutos en un computador sin GPU. Guarda el resultado, así que la segunda vez es inmediata.
'''),
("code", r'''
import torch, torchvision

pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
red = torchvision.models.resnet18(weights=pesos, progress=False)
red.fc = torch.nn.Identity()        # se quita la última capa: la red deja de clasificar y solo describe
red.eval()                          # modo de evaluación: los pesos no cambian
preparar = pesos.transforms()       # recorte central de 224 px y la normalización de colores de ImageNet

def extraer(rutas, lote=64):
    """Pasa cada foto por la red y devuelve una matriz con una fila de 512 números por foto."""
    partes = []
    with torch.no_grad():
        for i in range(0, len(rutas), lote):
            fotos = torch.stack([preparar(abrir(r)) for r in rutas[i:i + lote]])
            partes.append(red(fotos).numpy())
            if (i // lote) % 10 == 0:
                print(f"   {i} de {len(rutas)} fotos", flush=True)
    return np.concatenate(partes)

inicio = time.time()
X = rasgos_guardados("ham10000_resnet18", len(rutas))
if X is None:
    X = extraer(rutas)
    guardar_rasgos("ham10000_resnet18", X)
print(f"Matriz de características: {X.shape} | Tiempo: {time.time() - inicio:.0f} s")
'''),
("md", r'''
### 2.6 PCA: ¿qué organiza el espacio de características?

Las 512 columnas no tienen nombre, así que no se pueden explorar una por una. Las miro con componentes principales. Estandarizo cada columna y descompongo la matriz de covarianzas,

$$\Sigma=\frac{1}{n-1}Z^\top Z=V\Lambda V^\top,$$

y proyecto cada foto sobre las dos primeras columnas de $V$. La pregunta es qué separa a las fotos en ese plano: el diagnóstico o el origen de la imagen.

**Predicción escrita:** ¿qué se verá más separado, melanoma contra nevus o un origen contra otro? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(X)
pca = PCA(n_components=50, random_state=SEMILLA).fit(Z)
plano = pca.transform(Z)[:, :2]
acumulada = pca.explained_variance_ratio_.cumsum() * 100

fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(15, 4.3))
izq.plot(range(1, 51), acumulada, color="#1D4E89"); izq.set(title="Varianza acumulada", xlabel="componentes", ylabel="%")
for etiqueta, color in [("nv", "#1D4E89"), ("mel", "#D7263D")]:
    cuales = (muestra["dx"] == etiqueta).to_numpy()
    centro.scatter(plano[cuales, 0], plano[cuales, 1], s=5, alpha=0.4, color=color, label=etiqueta)
centro.set(title="Coloreado por diagnóstico", xlabel="CP1", ylabel="CP2"); centro.legend(markerscale=3)
for origen, color in zip(sorted(muestra["dataset"].unique()), ["#D7263D", "#1D4E89", "#1B998B", "#5B7DB1"]):
    cuales = (muestra["dataset"] == origen).to_numpy()
    der.scatter(plano[cuales, 0], plano[cuales, 1], s=5, alpha=0.4, color=color, label=origen)
der.set(title="Coloreado por origen de la imagen", xlabel="CP1", ylabel="CP2"); der.legend(markerscale=3, fontsize=8)
plt.tight_layout(); plt.show()
print(f"Los dos primeros componentes explican {acumulada[1]:.1f} % de la varianza; los 50 primeros, {acumulada[49]:.1f} %.")
'''),
("md", r'''
Comparen los dos paneles de colores con su apuesta. Ninguna de las dos variables se separa limpia en dos dimensiones, que apenas recogen una sexta parte de la varianza. Aun así hay estructura: los melanomas se cargan hacia la izquierda, y las fotos de `rosendahl` también. La red describe lo que hay en la foto, y en la foto están la lesión, la cámara, la iluminación y el encuadre. En la sección 3.7 medimos con un número cuánto del origen quedó guardado en las características.

Ahora la pregunta que prepara la trampa de hoy. Para cada foto busco su vecina más cercana en el espacio de características, con la distancia euclidiana $d(z_i,z_j)=\lVert z_i-z_j\rVert_2$, y pregunto si es otra foto de la misma lesión.
'''),
("code", r'''
from sklearn.neighbors import NearestNeighbors

_, vecinos = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)
vecina = vecinos[:, 1]                                   # la columna 0 es la propia foto
tiene_hermana = (fotos_por_lesion[muestra["lesion_id"]] > 1).to_numpy()
misma_lesion = lesion[vecina] == lesion

print(f"Fotos con hermanas cuya vecina más cercana es de su misma lesión: {misma_lesion[tiene_hermana].mean() * 100:.0f} %")
print(f"Si la vecina se escogiera al azar, eso pasaría en menos de {100 * 5 / len(muestra):.2f} % de los casos (cada foto tiene a lo sumo 5 hermanas entre {len(muestra)}).")
print(f"Fotos cuya vecina más cercana tiene su mismo diagnóstico: {(y[vecina] == y).mean() * 100:.0f} %")
'''),
("md", r'''
Entre más de tres mil fotos, la red encuentra a la hermana en cerca de la mitad de los casos. Para un modelo que decide por parecido, tener a la hermana en el entrenamiento es tener la respuesta.
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición: por lesión

La unidad que llega a consulta es la lesión. Por eso la partición sortea **lesiones**, y todas las fotos de una lesión van al mismo tramo. Uso tres tramos: 60 % de las lesiones para entrenar, 20 % para validar (ahí se escoge el modelo y el umbral) y 20 % para la prueba (ahí se reporta, una sola vez).
'''),
("code", r'''
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def partir_por_grupo(indices, grupos, fraccion, semilla):
    """Separa `indices` en dos partes sin que un mismo grupo quede en ambas."""
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraccion, random_state=semilla).split(indices, groups=grupos[indices]))
    return indices[a], indices[b]

todas = np.arange(len(muestra))
resto, P = partir_por_grupo(todas, lesion, 0.20, SEMILLA)          # prueba
E, V = partir_por_grupo(resto, lesion, 0.25, SEMILLA)              # entrenamiento y validación

for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {len(tramo):5d} imágenes | {len(set(lesion[tramo])):4d} lesiones | {y[tramo].mean() * 100:.1f} % melanoma")
assert not set(lesion[E]) & set(lesion[V]) and not set(lesion[E]) & set(lesion[P]) and not set(lesion[V]) & set(lesion[P])
print("Ninguna lesión aparece en dos tramos.")
'''),
("md", r'''
### 3.2 La línea base: un modelo que no ve la foto

La línea base de hoy es exigente a propósito. Es una regresión logística con lo que la tabla ya sabe del paciente: edad, sexo y sitio del cuerpo. El modelo es

$$P(Y=1\mid x)=\sigma(\beta_0+x^\top\beta),\qquad \sigma(t)=\frac{1}{1+e^{-t}}.$$

También mido la edad sola, usada directamente como puntaje. Si el modelo de imágenes no les gana, la red no está aportando nada que una planilla no tenga.
'''),
("code", r'''
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

planilla = muestra[["age", "sex", "localization"]]
base = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age"]),   # las edades que faltan se llenan con la mediana del entrenamiento
        (OneHotEncoder(handle_unknown="ignore"), ["sex", "localization"])),
    LogisticRegression(max_iter=1000))
base.fit(planilla.iloc[E], y[E])

edad = muestra["age"].fillna(muestra["age"].iloc[E].median()).to_numpy()
resultados = {
    "solo edad": {"validación": roc_auc_score(y[V], edad[V]), "prueba": roc_auc_score(y[P], edad[P])},
    "planilla (edad, sexo, sitio)": {"validación": roc_auc_score(y[V], base.predict_proba(planilla.iloc[V])[:, 1]),
                                     "prueba": roc_auc_score(y[P], base.predict_proba(planilla.iloc[P])[:, 1])},
}
display(pd.DataFrame(resultados).T.round(3))
'''),
("md", r'''
Comparen con su primera apuesta. La edad sola ordena bastante bien a estos pacientes. Esa es la cifra que la foto tiene que vencer.

### 3.3 Cinco clasificadores sobre las mismas características

Los cinco reciben la misma tabla de 512 columnas. Cambia la manera de trazar la frontera.

| Modelo | Idea en una línea |
|---|---|
| Regresión logística | Frontera lineal, $P(Y=1\mid z)=\sigma(\beta_0+z^\top\beta)$, con castigo $\lVert\beta\rVert_2^2$ fuerte porque hay 512 columnas y pocas fotos |
| Máquina de soporte vectorial | Frontera curva con núcleo gaussiano, $K(z,z')=\exp(-\gamma\lVert z-z'\rVert^2)$ |
| $k$ vecinos | El puntaje es la fracción de melanomas entre las $k=15$ fotos de entrenamiento más parecidas |
| Bosque aleatorio | Promedia 200 árboles entrenados sobre remuestreos del dato |
| *Gradient boosting* | Suma árboles pequeños, cada uno corrigiendo al anterior. Lo alimento con los 50 primeros componentes principales para que sea rápido |

Los hiperparámetros quedan fijos y a la vista. Afinarlos es legítimo si se hace mirando solo la validación.
'''),
("code", r'''
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
    """Puntaje continuo de cada foto: mientras más alto, más sospecha de melanoma."""
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
izq.axvline(0.5, color="gray", ls=":"); izq.set(title="AUC: validación para escoger, prueba para reportar", xlabel="AUC", xlim=(0.45, 1))
for nombre, (fp, vp, _) in curvas.items():
    der.plot(fp, vp, label=f"{nombre} ({comparacion.loc[nombre, 'validación']:.3f})")
fp, vp, _ = roc_curve(y[V], base.predict_proba(planilla.iloc[V])[:, 1])
der.plot(fp, vp, color="black", ls="--", label=f"planilla ({comparacion.loc['planilla (edad, sexo, sitio)', 'validación']:.3f})")
der.plot([0, 1], [0, 1], color="gray", ls=":")
der.set(title="Curvas ROC en validación", xlabel="1 − especificidad", ylabel="sensibilidad"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
display(comparacion.sort_values("validación", ascending=False).round(3))
'''),
("md", r'''
Lean la tabla con dos preguntas. La primera es cuánto le ganan las fotos a la planilla, y la respuesta honesta es que le ganan por poco: una red que nunca vio un lunar, sin reentrenar, queda apenas por encima de preguntar la edad. La segunda es cuánto se diferencian los cinco modelos entre sí. Cuando las características son las mismas, cambiar de clasificador mueve el resultado menos que cambiar de características.

### 3.4 La trampa de hoy: partir al azar por imagen

Ahora hago lo que hace `train_test_split` si uno no le dice nada: sortear **imágenes**. El tamaño de la prueba es el mismo y los modelos son los mismos. Lo único que cambia es que las hermanas de una lesión pueden quedar a lados distintos.

**Predicción escrita:** ¿cuál de los cinco modelos se infla más, y por qué? **Mi apuesta:**
'''),
("code", r'''
def al_azar(semilla, cuantas):
    """Partición por imagen, del mismo tamaño que la partición por lesión."""
    return train_test_split(todas, test_size=cuantas, random_state=semilla, stratify=y)

def por_lesion(semilla):
    return partir_por_grupo(todas, lesion, 0.20, semilla)

inicio = time.time()
entren_l, prueba_l = por_lesion(SEMILLA)
entren_a, prueba_a = al_azar(SEMILLA, len(prueba_l))
filtradas = np.isin(lesion[prueba_a], lesion[entren_a])
print(f"Al partir al azar, {filtradas.mean() * 100:.0f} % de las fotos de prueba tienen una hermana en el entrenamiento.")

trampa = {}
for nombre, modelo in cinco_modelos().items():
    limpio = roc_auc_score(y[prueba_l], puntaje(modelo.fit(X[entren_l], y[entren_l]), X[prueba_l]))
    con_fuga = roc_auc_score(y[prueba_a], puntaje(modelo.fit(X[entren_a], y[entren_a]), X[prueba_a]))
    trampa[nombre] = {"por lesión": limpio, "al azar por imagen": con_fuga, "inflado": con_fuga - limpio}
trampa = pd.DataFrame(trampa).T
print(f"Tiempo: {time.time() - inicio:.0f} s")

trampa[["por lesión", "al azar por imagen"]].plot.bar(color=["#1D4E89", "#D7263D"], rot=0, figsize=(9, 4))
plt.ylim(0.5, 1); plt.ylabel("AUC"); plt.title("El mismo modelo, dos exámenes"); plt.show()
display(trampa.round(3))
'''),
("md", r'''
Todos suben, y ninguno aprendió nada nuevo sobre melanomas. Un solo sorteo puede tener suerte, así que repito el experimento 30 veces con los dos modelos más rápidos. Agrego el caso extremo, el vecino más cercano con $k=1$, que clasifica cada foto con la etiqueta de la foto de entrenamiento que más se le parece. Para ese modelo mido la exactitud, que es la fracción de fotos bien clasificadas, y la separo según la foto tenga o no una hermana en el entrenamiento.
'''),
("code", r'''
inicio = time.time()
repeticiones = []
for semilla in range(30):
    entren_l, prueba_l = por_lesion(semilla)
    entren_a, prueba_a = al_azar(semilla, len(prueba_l))
    fila = {}
    logistica = cinco_modelos()["logística"]
    fila["logística, por lesión"] = roc_auc_score(y[prueba_l], puntaje(logistica.fit(X[entren_l], y[entren_l]), X[prueba_l]))
    fila["logística, al azar"] = roc_auc_score(y[prueba_a], puntaje(logistica.fit(X[entren_a], y[entren_a]), X[prueba_a]))
    un_vecino = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
    fila["1 vecino, por lesión"] = (un_vecino.fit(X[entren_l], y[entren_l]).predict(X[prueba_l]) == y[prueba_l]).mean()
    acierta = un_vecino.fit(X[entren_a], y[entren_a]).predict(X[prueba_a]) == y[prueba_a]
    con_hermana = np.isin(lesion[prueba_a], lesion[entren_a])
    fila["1 vecino, al azar"] = acierta.mean()
    fila["1 vecino, al azar, con hermana en entrenamiento"] = acierta[con_hermana].mean()
    fila["1 vecino, al azar, sin hermana"] = acierta[~con_hermana].mean()
    repeticiones.append(fila)
repeticiones = pd.DataFrame(repeticiones)
print(f"Tiempo: {time.time() - inicio:.0f} s")

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4))
repeticiones[["logística, por lesión", "logística, al azar"]].plot.box(ax=izq, color="#1D4E89")
izq.set(title="Regresión logística: AUC en 30 sorteos", ylabel="AUC")
repeticiones[["1 vecino, por lesión", "1 vecino, al azar"]].plot.box(ax=der, color="#D7263D")
der.set(title="Un vecino: exactitud en 30 sorteos", ylabel="exactitud")
plt.tight_layout(); plt.show()

resumen_fuga = repeticiones.agg(["mean", "std"]).T.round(3)
display(resumen_fuga)
print(f"Exactitud de responder siempre 'nevus': {1 - y.mean():.3f}")
'''),
("md", r'''
Lean las tres últimas filas de la tabla. Cuando la foto de prueba tiene una hermana en el entrenamiento, el vecino más cercano acierta mucho más que cuando no la tiene. Ahí está el mecanismo de la fuga: el modelo reconoce la lesión y repite su etiqueta. Cuando no hay hermana, vuelve a su nivel de verdad, que queda cerca del de responder siempre «nevus».

Esto tiene nombre. La validación supone que las observaciones de prueba son independientes de las de entrenamiento. Aquí las observaciones vienen en grupos, y dentro de un grupo están correlacionadas. Si $g(i)$ es la lesión de la foto $i$, la partición correcta exige

$$\{g(i): i\in\text{entrenamiento}\}\ \cap\ \{g(j): j\in\text{prueba}\}=\varnothing .$$

La regla general es partir por la unidad sobre la que se va a decidir. Aquí es la lesión. En el gemelo será la planta. En una historia clínica es el paciente.

El número inflado es el que uno querría publicar, y por eso esta fuga aparece tanto en artículos de imágenes médicas. De aquí en adelante solo vale la partición por lesión.

### 3.5 El veredicto

La regla quedó fijada antes de mirar: gana el mayor AUC en **validación** y se reporta su AUC en **prueba**.
'''),
("code", r'''
solo_imagen = comparacion.loc[list(cinco_modelos())]
mejor = solo_imagen["validación"].idxmax()
ganador = entrenados[mejor]
print(f"Ganador en validación: {mejor}")
print(f"AUC en prueba: {comparacion.loc[mejor, 'prueba']:.3f} | planilla: {comparacion.loc['planilla (edad, sexo, sitio)', 'prueba']:.3f} | solo edad: {comparacion.loc['solo edad', 'prueba']:.3f}")
'''),
("md", r'''
### 3.6 La foto y la planilla juntas

La foto y la planilla quedaron cerca, y eso deja una pregunta abierta: ¿saben lo mismo o saben cosas distintas? Si saben cosas distintas, juntarlas debe mejorar. Entreno una regresión logística con las 512 características y con la edad, el sexo y el sitio.
'''),
("code", r'''
columnas_z = [f"z{j}" for j in range(X.shape[1])]
todo = pd.concat([planilla, pd.DataFrame(X, columns=columnas_z)], axis=1)
combinado = make_pipeline(
    make_column_transformer(
        (make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age"]),
        (OneHotEncoder(handle_unknown="ignore"), ["sex", "localization"]),
        (StandardScaler(), columnas_z)),
    LogisticRegression(C=0.01, max_iter=3000)).fit(todo.iloc[E], y[E])
comparacion.loc["foto + planilla"] = {"validación": roc_auc_score(y[V], combinado.predict_proba(todo.iloc[V])[:, 1]),
                                      "prueba": roc_auc_score(y[P], combinado.predict_proba(todo.iloc[P])[:, 1])}
display(comparacion.loc[["planilla (edad, sexo, sitio)", mejor, "foto + planilla"]].round(3))
'''),
("md", r'''
Comparen las tres filas. Si la combinación supera a las dos por separado, la foto aporta información que la planilla no tiene, y al revés. Para lo que sigue trabajo con el modelo que solo ve la foto, porque así podemos preguntarle en qué parte de la imagen se apoya. Repetir la decisión de la sección 4 con el modelo combinado queda como ejercicio.

### 3.7 Explicar: ¿en qué se apoya el modelo?

Las 512 características no tienen nombre, así que los coeficientes no se pueden leer como en una regresión sobre edad y sexo. Uso tres caminos que sí dan evidencia.

**Primero, una prueba directa del atajo.** Si las características permiten adivinar de qué clínica viene la foto, el modelo tiene a la mano una señal que no es la lesión. Entreno una regresión logística para predecir el origen.
'''),
("code", r'''
origen = muestra["dataset"].to_numpy()
detector = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[E], origen[E])
acierto_origen = (detector.predict(X[V]) == origen[V]).mean()
print(f"Adivinar el origen de la foto con las características: {acierto_origen * 100:.0f} % de acierto en validación")
print(f"Responder siempre el origen más común: {pd.Series(origen[E]).value_counts(normalize=True).iloc[0] * 100:.0f} %")
print("\nProporción de melanomas según el origen (entrenamiento):")
print((muestra.iloc[E].groupby("dataset")["melanoma"].agg(["mean", "size"]).rename(columns={"mean": "melanoma", "size": "imágenes"})).round(2).to_string())
'''),
("md", r'''
Las características delatan el origen con facilidad, y la proporción de melanomas cambia de un origen a otro. El modelo puede usar esa señal. No sabemos cuánto la usa, pero sabemos que la tiene disponible.

**Segundo, mirar dónde falla.** Uso el umbral que más adelante le vamos a recomendar al cliente y cuento aciertos por subgrupo en la validación. Con pocos casos por grupo, las diferencias pequeñas son ruido.
'''),
("code", r'''
puntaje_V, puntaje_P = puntaje(ganador, X[V]), puntaje(ganador, X[P])

def umbral_para(puntajes, etiquetas, sensibilidad):
    """El umbral más alto que todavía remite al menos esa fracción de los melanomas."""
    positivos = np.sort(puntajes[etiquetas == 1])
    return positivos[int(np.floor((1 - sensibilidad) * len(positivos)))]

umbral = umbral_para(puntaje_V, y[V], SENSIBILIDAD)
remite_V = puntaje_V >= umbral

def por_subgrupo(columna):
    grupo = muestra[columna].iloc[V].to_numpy()
    filas = {}
    for valor in pd.unique(grupo):
        cuales = grupo == valor
        positivos, negativos = cuales & (y[V] == 1), cuales & (y[V] == 0)
        if positivos.sum() >= 15 and negativos.sum() >= 15:
            filas[valor] = {"melanomas": positivos.sum(), "sensibilidad": remite_V[positivos].mean(),
                            "nevus": negativos.sum(), "especificidad": (~remite_V[negativos]).mean()}
    return pd.DataFrame(filas).T

muestra["grupo_edad"] = pd.cut(muestra["age"], [0, 40, 60, 100], labels=["hasta 40", "41 a 60", "más de 60"], include_lowest=True).astype(str)
for columna in ["dataset", "sex", "grupo_edad"]:
    print(f"\nPor {columna} (grupos con al menos 15 melanomas y 15 nevus en validación):")
    print(por_subgrupo(columna).round(2).to_string())
'''),
("md", r'''
Busquen el subgrupo donde la especificidad se cae o donde la sensibilidad queda lejos de la que pedimos. Un filtro que trabaja distinto según la clínica de origen o según la edad del paciente no falla al azar, y eso hay que decírselo al cliente.

**Tercero, tapar pedazos de la foto.** Deslizo un parche gris sobre la imagen y mido cuánto cambia el puntaje. Si $x^{(k)}$ es la foto con la región $k$ tapada, la importancia de esa región es

$$I_k=s(x)-s\bigl(x^{(k)}\bigr).$$

Una región con $I_k$ grande y positivo es una que empujaba el puntaje hacia melanoma. El método solo necesita volver a pasar fotos por la red, así que sirve para cualquier modelo.
'''),
("code", r'''
def mapa_de_oclusion(ruta, lado=32):
    """Tapa la foto por parches de `lado` píxeles y devuelve el cambio de puntaje de cada parche."""
    original = preparar(abrir(ruta))                       # tensor de 3 x 224 x 224, ya normalizado
    variantes = [original]
    for fila in range(0, 224, lado):
        for columna in range(0, 224, lado):
            tapada = original.clone()
            tapada[:, fila:fila + lado, columna:columna + lado] = 0   # cero es el gris medio tras normalizar
            variantes.append(tapada)
    with torch.no_grad():
        rasgos = red(torch.stack(variantes)).numpy()
    s = puntaje(ganador, rasgos)
    return (s[0] - s[1:]).reshape(224 // lado, 224 // lado), s[0]

def recorte(ruta):
    """La misma región de 224 x 224 que ve la red, sin normalizar, para dibujarla."""
    foto = abrir(ruta); ancho, alto = foto.size; escala = 256 / min(ancho, alto)
    foto = foto.resize((round(ancho * escala), round(alto * escala)))
    izquierda, arriba = (foto.width - 224) // 2, (foto.height - 224) // 2
    return foto.crop((izquierda, arriba, izquierda + 224, arriba + 224))

# Dos melanomas con puntaje alto y dos nevus con puntaje alto (falsas alarmas), todos de validación
orden = np.argsort(-puntaje_V)
elegidas = [V[i] for i in orden if y[V][i] == 1][:2] + [V[i] for i in orden if y[V][i] == 0][:2]

fig, ejes = plt.subplots(2, 4, figsize=(13, 6.4))
for columna, i in enumerate(elegidas):
    mapa, s0 = mapa_de_oclusion(rutas[i])
    ejes[0, columna].imshow(recorte(rutas[i])); ejes[0, columna].axis("off")
    ejes[0, columna].set_title(f"{muestra['dx'][i]} · {muestra['dataset'][i]} · puntaje {s0:.2f}", fontsize=9)
    tope = np.abs(mapa).max()
    ejes[1, columna].imshow(recorte(rutas[i]))
    ejes[1, columna].imshow(mapa, cmap="RdBu_r", vmin=-tope, vmax=tope, alpha=0.6, extent=(0, 224, 224, 0), interpolation="nearest")
    ejes[1, columna].axis("off")
plt.suptitle("Rojo: al tapar esa región el puntaje de melanoma baja. Azul: al taparla, sube"); plt.tight_layout(); plt.show()
'''),
("md", r'''
Miren si el rojo cae sobre la lesión o sobre los bordes de la foto. Cuando una esquina, una regla o el viñeteado del lente mueven el puntaje, el modelo está leyendo el equipo. Cuatro fotos no prueban nada por sí solas. Sirven para generar sospechas que después se miden, como hicimos con el origen.

Ninguno de los tres caminos dice por qué una lesión es melanoma. Dicen en qué se apoya el modelo, que es lo que el cliente necesita saber antes de confiar en él.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: el umbral

El cliente necesita una regla, y un AUC todavía no lo es. La exigencia quedó escrita arriba en `SENSIBILIDAD`: de cada 100 melanomas, el filtro debe remitir al menos esa cantidad. Escojo en **validación** el umbral más alto que cumple,

$$\hat\tau=\max\bigl\{\tau:\ \widehat{\text{sens}}_V(\tau)\ge S\bigr\},$$

y lo aplico tal cual en la **prueba**. La sensibilidad de la prueba puede quedar por debajo de la pedida, porque el umbral se afinó con otras lesiones. Esa diferencia también es resultado.

Para la incertidumbre uso *bootstrap* por lesión: remuestreo lesiones de la prueba con reemplazo, $B=1000$ veces, recalculo cada cifra y tomo los percentiles 2,5 y 97,5,

$$\text{IC}_{95\%}=\bigl[\hat\theta^{*}_{(0{,}025)},\ \hat\theta^{*}_{(0{,}975)}\bigr].$$

Se remuestrean lesiones y no imágenes por la misma razón por la que se partió por lesión.
'''),
("code", r'''
remite_P = puntaje_P >= umbral

def cifras(etiquetas, remite, puntajes):
    vp, fn = (remite & (etiquetas == 1)).sum(), (~remite & (etiquetas == 1)).sum()
    vn, fp = (~remite & (etiquetas == 0)).sum(), (remite & (etiquetas == 0)).sum()
    return {"sensibilidad": vp / (vp + fn), "especificidad": vn / (vn + fp),
            "remitidas por cada 100": 100 * remite.mean(), "melanomas entre las remitidas (%)": 100 * vp / max(vp + fp, 1),
            "AUC": roc_auc_score(etiquetas, puntajes)}

en_prueba = cifras(y[P], remite_P, puntaje_P)

rng = np.random.default_rng(SEMILLA)
lesiones_P = pd.unique(lesion[P])
filas_de = {l: np.flatnonzero(lesion[P] == l) for l in lesiones_P}
replicas = []
for _ in range(1000):
    filas = np.concatenate([filas_de[l] for l in rng.choice(lesiones_P, len(lesiones_P))])
    replicas.append(cifras(y[P][filas], remite_P[filas], puntaje_P[filas]))
replicas = pd.DataFrame(replicas)

informe = pd.DataFrame({"validación": cifras(y[V], remite_V, puntaje_V), "prueba": en_prueba,
                        "IC 95 % inferior": replicas.quantile(0.025), "IC 95 % superior": replicas.quantile(0.975)})
print(f"Modelo: {mejor} | Sensibilidad exigida: {SENSIBILIDAD:.2f} | Umbral escogido en validación: {umbral:.3f}")
display(informe.round(3))

matriz = pd.crosstab(pd.Series(np.where(y[P] == 1, "melanoma", "nevus"), name="histopatología"),
                     pd.Series(np.where(remite_P, "se remite", "no se remite"), name="filtro"))
display(matriz)
escapan = int(matriz.loc["melanoma", "no se remite"]) if "no se remite" in matriz.columns else 0
print(f"En la prueba, el filtro dejó sin remitir {escapan} imágenes de melanoma de {int((y[P] == 1).sum())}, y le ahorró al dermatólogo {int((~remite_P & (y[P] == 0)).sum())} de {int((y[P] == 0).sum())} nevus.")
'''),
("md", r'''
Lean la tabla como la leería el coordinador del programa. La sensibilidad dice cuántos melanomas atrapa el filtro. La especificidad dice cuánta fila le quita al dermatólogo, y con una sensibilidad alta esa cifra es modesta. Miren el ancho de los intervalos antes de prometer nada.

**Una advertencia sobre la prevalencia.** En este conjunto cerca de un tercio de las imágenes son melanomas, porque todas las lesiones se extirparon por sospecha. En una fila general de teledermatología el melanoma es mucho más raro. La sensibilidad y la especificidad no dependen de la prevalencia $\pi$, pero la fracción de melanomas entre las lesiones remitidas sí:

$$\text{VPP}=\frac{S\,\pi}{S\,\pi+(1-E)(1-\pi)}.$$

La celda siguiente evalúa la fórmula con la sensibilidad y la especificidad de la prueba para varias prevalencias. Las prevalencias de 1 % y 5 % son **supuestos** para ver cómo se mueve la cifra. No salen de ningún dato de este cuaderno.
'''),
("code", r'''
S, Esp = en_prueba["sensibilidad"], en_prueba["especificidad"]
for prevalencia, origen_dato in [(y[P].mean(), "la de este conjunto"), (0.05, "supuesto"), (0.01, "supuesto")]:
    vpp = S * prevalencia / (S * prevalencia + (1 - Esp) * (1 - prevalencia))
    print(f"Prevalencia {prevalencia * 100:4.1f} % ({origen_dato}): de cada 100 lesiones remitidas, {vpp * 100:.0f} serían melanoma")
'''),
("md", r'''
### La decisión en tres frases

Escríbanla para el coordinador del programa, que no sabe qué es una curva ROC. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Melanoma contra nevus | Las siete clases | Es la pregunta clínica difícil y da una decisión de dos salidas |
| Solo lesiones con histopatología (3.611 imágenes) | Todas las imágenes de melanoma y nevus | Los nevus sin biopsia vienen casi todos de un solo equipo y nunca fueron sospechosos: el modelo aprendería un atajo |
| Fotos reducidas a 256 píxeles y recorte central de 224 | Resolución original (600 × 450) | Es la entrada que espera ResNet18 y cabe en un computador sin GPU |
| Red preentrenada como extractor fijo | Ajuste fino de la red | Corre en CPU en minutos; el costo es un AUC menor |
| Partición por lesión en tres tramos | Partición al azar por imagen | Las fotos de una lesión no son independientes; al azar el resultado se infla |
| Hiperparámetros fijos | Búsqueda en validación | Alcanza el tiempo de clase; queda como mejora |
| Decidir con el modelo que solo ve la foto | Modelo de foto y planilla | Permite preguntar en qué región de la imagen se apoya; el combinado queda como ejercicio |
| Línea base con edad, sexo y sitio | Línea base de azar | Obliga a la foto a aportar algo que la planilla no tenga |
| Umbral por sensibilidad exigida, escogido en validación | Umbral de 0,5 | El costo de un melanoma perdido no es el de una remisión de más |
| *Bootstrap* por lesión | *Bootstrap* por imagen | La unidad independiente es la lesión |

### 4.3 Riesgos y mantenimiento

El dato es de personas. Respondan en dos o tres líneas cada una.

- El conjunto viene de una clínica de Viena y un consultorio de Queensland y no registra el tono de piel. ¿Qué habría que medir antes de usar este filtro en un puesto de salud de Caldas?
- El modelo se entrenó con lesiones que un médico ya había decidido extirpar. ¿Qué le pasa a la fracción de melanomas entre las remitidas cuando el filtro recibe todos los lunares que llegan?
- Las características delatan la cámara. ¿Qué harían el día que el programa cambie de dermatoscopio?
- Un filtro con sensibilidad de 0,90 deja pasar uno de cada diez melanomas. ¿Quién responde por ese paciente, y cómo se le explica a quien usa el filtro que un «no se remite» no es un diagnóstico?
- La licencia del dato es CC BY-NC 4.0. ¿Qué les permite y qué les prohíbe?

### 4.4 Cambien una línea

Vuelvan a la celda del comienzo donde dice `SENSIBILIDAD = 0.90`, pongan `SENSIBILIDAD = 0.95` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Las características ya están guardadas, así que la segunda corrida es rápida. Llenen la tabla con sus dos corridas.

| En la prueba | $S=0{,}90$ | $S=0{,}95$ |
|---|---|---|
| Umbral escogido en validación | | |
| Sensibilidad | | |
| Especificidad | | |
| Lesiones remitidas por cada 100 | | |
| Imágenes de melanoma sin remitir | | |

Con la tabla llena, respondan en tres líneas: ¿cuántas remisiones adicionales cuesta cada melanoma adicional que se atrapa? ¿Se lo recomendarían al programa? **Mi respuesta:**

### 4.5 Procedencia

Lo último que hace un cuaderno serio es decir de dónde salió su dato.
'''),
("code", r'''
procedencia()["ham10000"]
'''),
]
