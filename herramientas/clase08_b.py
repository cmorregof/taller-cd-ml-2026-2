"""Celdas del gemelo de la clase 8, «Los sonidos del Magdalena».

Cada celda de código es una terna: ("code", solución, encargo). El cuaderno del estudiante
lleva solo el encargo, como comentario; la solución completa va al cuaderno del profesor.
Las celdas marcadas con "dado" salen iguales en los dos.
"""
from molde import ARRANQUE

AVES = [
("md", r'''
# Clase 8 · Los sonidos del Magdalena (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno en la segunda mitad de la clase. Es el gemelo de *El electrocardiograma*: la misma técnica, de la señal a la tabla y de la tabla al modelo, con otra señal.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Dirigir a una IA para que escriba el flujo completo de un clasificador de audio (ventanas, espectrograma, MFCC, un catálogo de modelos, partición por grabación) y auditar lo que escribió. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular), 2 (preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | El código lo escribe Claude, a pedido de ustedes, celda por celda. Ustedes deben poder explicar sin ayuda qué es un espectrograma, qué resume un MFCC, por qué se parte por grabación y no por ventana, y cómo se escogieron las cien grabaciones del experto. |
| **Entregable** | El cuaderno completo y corrido en su portafolio, con las predicciones escritas, la sección de auditoría llena, la decisión en tres frases y la tabla de la variante. |

**Cómo se trabaja.** Las celdas de código están vacías. Cada una trae un comentario que dice qué debe lograr y qué nombres de variable debe dejar. Copien el comentario en Claude, pidan el código, léanlo antes de pegarlo y córranlo. Si falla, lean el error antes de pedir ayuda. Lo único que ya está escrito es el arranque, los parámetros, la carga del dato y el catálogo de modelos de la sección 3.2.

**Cómputo.** Todo corre en CPU. La descarga tarda menos de un minuto. Sacar las características de las mil grabaciones tarda entre uno y dos minutos.
'''),
("md", r'''
> **Antes de empezar: conozcan la fuente.** Las grabaciones de hoy salen de la [muestra pública de BirdCLEF+ 2025 en Hugging Face](https://huggingface.co/datasets/mteb/birdclef25-mini). Abran la página y explórenla cinco minutos con el visor de datos. Cada grabación trae además, en la columna `url`, el enlace a su página original en xeno-canto o en iNaturalist. Abran una. Respondan mirando la fuente:
>
> - ¿Qué es una fila de esta tabla y qué columnas la describen?
> - ¿Quién grabó el audio de la fila que abrieron, y quién publica la muestra?
> - ¿Qué licencia tiene esa grabación? ¿La muestra de Hugging Face declara una licencia propia?
'''),
("dado", ARRANQUE),
("dado", r'''
# Los dos parámetros del cuaderno. Al final se cambia la primera línea y se corre todo de nuevo.
SEGUNDOS_OIDOS = 30       # cuántos segundos de cada grabación oye el modelo, contados desde el comienzo
REVISIONES = 100          # cuántas grabaciones alcanza a oír el experto

import time
import librosa
from taller.clase08_senales import grabaciones
'''),
("md", r'''
## 1. Formulación

Imagínense al equipo que cuida una reserva en el Magdalena Medio. Dejan grabadoras en el bosque y recogen horas de sonido. Saber qué especies cantan, y dónde, es su manera de medir si el bosque se recupera. El equipo tiene un solo ornitólogo capaz de reconocer los cantos de oído, y su tiempo alcanza para oír cien grabaciones.

Ese equipo necesita tres respuestas:

1. **Predecir.** ¿Qué especie canta en esta grabación?
2. **Explicar.** ¿Qué características del sonido separan a las especies, y qué especies confunde el modelo entre sí?
3. **Decidir.** ¿Qué grabaciones revisa primero el experto, si solo alcanza a oír cien?

**El dato.** BirdCLEF+ 2025 es una competencia de reconocimiento de especies por su sonido, organizada alrededor de la Reserva Natural El Silencio, en el Magdalena Medio. Sus grabaciones de entrenamiento vienen de xeno-canto y de iNaturalist, donde las suben aficionados y científicos de muchos países. El conjunto completo de la competencia vive en Kaggle, exige cuenta y trae mucho más audio del que alcanzamos a procesar en una clase. Por eso aquí usamos un subconjunto: una muestra pública de mil de esas grabaciones, 20 por cada una de 50 especies de aves.

**La métrica.** Con 50 especies igual de frecuentes, quien adivina al azar acierta 1 de cada 50. Medimos la exactitud por grabación y la exactitud entre las tres primeras:

$$\text{exactitud}=\frac{1}{n}\sum_{i=1}^{n}\mathbb 1\{\hat y_i=y_i\},\qquad \text{exactitud}_{\text{top-3}}=\frac{1}{n}\sum_{i=1}^{n}\mathbb 1\{y_i\in\{\hat y_i^{(1)},\hat y_i^{(2)},\hat y_i^{(3)}\}\}.$$

La segunda importa porque al experto le sirve una lista corta de candidatas.

### Predicción escrita

- Con 12 grabaciones de entrenamiento por especie, ¿qué exactitud alcanzará el mejor modelo? **Mi apuesta:**
- Si corto las grabaciones en ventanas de cinco segundos y separo entrenamiento y prueba barajando ventanas, ¿la exactitud sube, baja o queda igual? **Mi apuesta:**
- Si el experto oye las cien grabaciones en que el modelo está menos seguro, ¿qué fracción de los errores del modelo alcanza a corregir? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

La tabla trae una fila por grabación: especie, colección de origen, autor, licencia, coordenadas y duración. El sonido se pide aparte con `grabaciones(tabla)`, que entrega las grabaciones de a una como `(fila, onda, fs)`: la fila de la tabla, la onda como arreglo de numpy en una sola pista y la frecuencia de muestreo.
'''),
("dado", r'''
aves = descargar("aves_magdalena")
print("Grabaciones:", len(aves), "| Especies:", aves["primary_label"].nunique(), "| Horas de sonido:", round(aves["segundos"].sum() / 3600, 1))
aves.head(3)
'''),
("md", r'''
### 2.1 Conocer el dato antes de modelarlo

Respondan con código y escriban la lectura debajo de cada celda.
'''),
("code", r'''
print("Grabaciones por especie:", aves["primary_label"].value_counts().describe()[["min", "max"]].to_dict())
print("\nColección de origen:"); print(aves["collection"].value_counts())
print("\nLicencias:"); print(aves["license"].value_counts())
print("\nAutores distintos:", aves["author"].nunique(), "| los cinco con más grabaciones:")
print(aves["author"].value_counts().head(5))

# Rectángulo aproximado que encierra a Colombia y parte de sus vecinos
en_la_region = aves["latitude"].between(-4.3, 13.5) & aves["longitude"].between(-79.1, -66.8)
print(f"\nGrabaciones dentro del rectángulo de Colombia: {int(en_la_region.sum())} de {len(aves)} | sin coordenadas: {int(aves['latitude'].isna().sum())}")

fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(14, 3.6))
izq.hist(aves["segundos"].clip(upper=120), bins=40, color="#1D4E89")
izq.set(title="Duración (recortada en 120 s)", xlabel="segundos")
aves["rating"].value_counts().sort_index().plot.bar(ax=centro, color="#1B998B", rot=0)
centro.set(title="Calificación de calidad", xlabel="rating")
der.scatter(aves["longitude"], aves["latitude"], s=6, color="#D7263D", alpha=0.5)
der.add_patch(plt.Rectangle((-79.1, -4.3), 12.3, 17.8, fill=False, color="black", ls="--"))
der.set(title="Dónde se grabó", xlabel="longitud", ylabel="latitud")
plt.tight_layout(); plt.show()
print("Duración: mediana", aves["segundos"].median(), "s | menos de 5 s:", int((aves["segundos"] < 5).sum()), "| más de 60 s:", int((aves["segundos"] > 60).sum()))
''', r'''
# Esta celda debe explorar la tabla `aves` y mostrar, con números y tres gráficas:
#   - cuántas grabaciones hay por especie (primary_label) y de qué colección vienen (collection);
#   - qué licencias tienen (license) y cuántos autores distintos hay (author);
#   - la distribución de la duración (segundos) y de la calificación de calidad (rating);
#   - dónde se grabaron (latitude, longitude) y cuántas caen dentro de un rectángulo
#     aproximado alrededor de Colombia: latitud entre -4,3 y 13,5, longitud entre -79,1 y -66,8.
'''),
("md", r'''
**Preguntas de lectura.**

- ¿Cuántas de las mil grabaciones se hicieron en Colombia o cerca? ¿Qué le dice eso sobre un modelo que se va a usar en el Magdalena Medio?
- La calificación 0 no significa «mala». ¿De qué colección vienen esas grabaciones? Compruébenlo con una tabla cruzada antes de usar `rating` para algo.
- Un mismo autor aporta varias grabaciones. ¿Por qué eso puede inflar el resultado, igual que el paciente repetido de la demostración?

**Mi lectura:**
'''),
("code", r'''
from IPython.display import Audio, display

fila, onda, fs = next(grabaciones(aves[(aves["rating"] >= 4.5) & aves["segundos"].between(8, 30)].head(1)))
print(fila["common_name"], "|", fila["scientific_name"], "|", fila["author"], "|", fila["license"], "|", fila["url"])
print("Frecuencia de muestreo:", fs, "Hz | muestras:", len(onda), "| segundos:", round(len(onda) / fs, 1))
display(Audio(onda[:5 * fs], rate=fs))

S = librosa.feature.melspectrogram(y=onda, sr=fs, n_fft=1024, hop_length=512, n_mels=48, fmin=300, fmax=14000)
fig, (arriba, abajo) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
arriba.plot(np.arange(len(onda)) / fs, onda, color="#1D4E89", lw=0.4); arriba.set(title="La onda", ylabel="amplitud")
imagen = abajo.imshow(10 * np.log10(S + 1e-10), origin="lower", aspect="auto", cmap="magma", extent=[0, len(onda) / fs, 0, 48])
abajo.set(title="Espectrograma en escala mel", xlabel="segundos", ylabel="banda mel")
plt.tight_layout(); plt.show()
''', r'''
# Esta celda debe tomar UNA grabación de buena calidad (rating de 4,5 o más, entre 8 y 30 segundos),
# usando `next(grabaciones(tabla_de_una_fila))`, que devuelve (fila, onda, fs). Debe:
#   - imprimir la especie, el autor, la licencia y la frecuencia de muestreo fs;
#   - dejar oír los primeros 5 segundos con IPython.display.Audio;
#   - dibujar la onda en el tiempo y, debajo, su espectrograma en escala mel
#     (librosa.feature.melspectrogram con n_fft=1024, hop_length=512, n_mels=48, fmin=300, fmax=14000), en decibeles.
'''),
("md", r'''
### 2.2 De la onda a la tabla

Una grabación de 30 segundos a 32.000 Hz son 960.000 números. Para llevarla a una tabla seguimos el mismo camino de la demostración, con las herramientas del audio.

**Ventanas.** Cada grabación se corta en ventanas de 5 segundos, sin traslape, hasta completar `SEGUNDOS_OIDOS`. Las grabaciones más cortas que una ventana se completan con silencio. Cada ventana hereda la especie de su grabación.

**Espectrograma.** La transformada de Fourier de tiempo corto mira la señal por tramos de $N$ muestras con una ventana $w$ que se desliza de a $H$ muestras:

$$X[m,k]=\sum_{n=0}^{N-1}x[n+mH]\,w[n]\,e^{-2\pi i\,kn/N}.$$

**Escala mel.** El oído distingue mejor las frecuencias bajas que las altas. La escala mel imita esa percepción, $\text{mel}(f)=2595\,\log_{10}(1+f/700)$, y la energía se suma en $M=48$ bandas repartidas de manera uniforme en mel entre 300 y 14.000 Hz. Sea $L[b,m]$ la energía de la banda $b$ en el tramo $m$, en decibeles.

**Piso de ruido.** El río, el viento y el micrófono ponen un ruido de fondo distinto en cada grabación. A cada banda se le resta su mediana en el tiempo, $\tilde L[b,m]=L[b,m]-\operatorname{mediana}_m L[b,m]$, para que quede lo que sobresale del fondo.

**MFCC.** Los coeficientes cepstrales resumen la forma del espectro de cada tramo con una transformada de coseno:

$$c_k[m]=\sum_{b=0}^{M-1}\tilde L[b,m]\,\cos\!\Bigl[\frac{\pi k}{M}\bigl(b+\tfrac12\bigr)\Bigr],\qquad k=0,\dots,19.$$

De cada ventana salen dos familias de columnas. La familia `mfcc_` trae la media y la desviación de cada coeficiente en el tiempo, y la desviación de su cambio entre tramos vecinos. La familia `mel_` trae, por banda, la desviación, el percentil 90 y el máximo.
'''),
("code", r'''
VENTANA = 5                                                   # segundos por ventana
MAX_VENTANAS = max(1, SEGUNDOS_OIDOS // VENTANA)

def caracteristicas(ventana, fs):
    """Una ventana de audio -> 204 números: 60 de MFCC y 144 de bandas mel."""
    S = librosa.feature.melspectrogram(y=ventana, sr=fs, n_fft=1024, hop_length=512, n_mels=48, fmin=300, fmax=14000)
    L = 10 * np.log10(S + 1e-10)
    L = L - np.median(L, axis=1, keepdims=True)               # quitar el piso de ruido de cada banda
    M = librosa.feature.mfcc(S=L, n_mfcc=20)
    return np.r_[M.mean(axis=1), M.std(axis=1), np.diff(M, axis=1).std(axis=1),
                 L.std(axis=1), np.percentile(L, 90, axis=1), L.max(axis=1)]

nombres = ([f"mfcc_{tipo}_{k}" for tipo in ["media", "desv", "cambio"] for k in range(20)]
           + [f"mel_{tipo}_{b}" for tipo in ["desv", "p90", "max"] for b in range(48)])

inicio = time.time()
filas, de_grabacion, frecuencias = [], [], set()
for fila, onda, fs in grabaciones(aves):
    frecuencias.add(fs)
    n = VENTANA * fs
    if len(onda) < n:
        onda = np.pad(onda, (0, n - len(onda)))               # grabación corta: se completa con silencio
    for v in range(min(MAX_VENTANAS, len(onda) // n)):
        filas.append(caracteristicas(onda[v * n:(v + 1) * n], fs))
        de_grabacion.append(fila["indice"])
C = pd.DataFrame(filas, columns=nombres)
de_grabacion = np.array(de_grabacion)                         # a qué grabación pertenece cada ventana
print(f"Ventanas: {len(C)} | columnas: {C.shape[1]} | grabaciones: {len(set(de_grabacion))} | {time.time() - inicio:.0f} s")
print("Frecuencias de muestreo encontradas:", frecuencias, "| ventanas por grabación:", pd.Series(de_grabacion).value_counts().describe()[["min", "50%", "max"]].to_dict())
''', r'''
# Esta celda debe convertir las mil grabaciones en una tabla con una fila por VENTANA de 5 segundos.
#   - Recorrer `for fila, onda, fs in grabaciones(aves):`.
#   - Cortar cada onda en ventanas de 5 s sin traslape, máximo SEGUNDOS_OIDOS // 5 ventanas por grabación,
#     desde el comienzo. Si la grabación dura menos de 5 s, completarla con ceros.
#   - Para cada ventana: espectrograma mel (n_fft=1024, hop_length=512, n_mels=48, fmin=300, fmax=14000),
#     pasarlo a decibeles, restarle a cada banda su mediana en el tiempo (piso de ruido),
#     sacar 20 MFCC con librosa.feature.mfcc(S=..., n_mfcc=20) y resumir:
#       media, desviación y desviación del cambio de cada MFCC (60 columnas `mfcc_...`);
#       desviación, percentil 90 y máximo de cada banda mel (144 columnas `mel_...`).
#   - Dejar: `C` (DataFrame de características) y `de_grabacion` (arreglo con el `indice` de la
#     grabación de cada ventana). Imprimir cuántas ventanas salieron, el tiempo y la frecuencia de muestreo.
'''),
("md", r'''
### 2.3 PCA: ¿se ven las especies en la tabla?

Estandaricen las columnas y proyecten las ventanas sobre los dos primeros componentes principales. Si $Z$ es la tabla estandarizada, el PCA descompone $R=\tfrac{1}{n-1}Z^\top Z=V\Lambda V^\top$ y la fracción de varianza del componente $j$ es $\lambda_j/\sum_k\lambda_k$.

**Predicción escrita:** con 204 columnas, ¿cuánta varianza explicarán los dos primeros componentes? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=10, random_state=0).fit(StandardScaler().fit_transform(C))
Z = pca.transform(StandardScaler().fit_transform(C))
especie_ventana = aves["common_name"].values[de_grabacion]
cuatro = list(aves["common_name"].drop_duplicates().iloc[[0, 12, 25, 40]])

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4))
izq.bar(range(1, 11), pca.explained_variance_ratio_ * 100, color="#1D4E89")
izq.set(title="Varianza explicada", xlabel="componente", ylabel="%")
der.scatter(Z[:, 0], Z[:, 1], s=4, color="lightgray")
for nombre, color in zip(cuatro, ["#D7263D", "#1D4E89", "#1B998B", "#3B7EA1"]):
    der.scatter(Z[especie_ventana == nombre, 0], Z[especie_ventana == nombre, 1], s=12, color=color, label=nombre)
der.set(title="Ventanas en los dos primeros componentes", xlabel="CP1", ylabel="CP2"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("Varianza de los dos primeros componentes (%):", (pca.explained_variance_ratio_[:2] * 100).round(1), "| de los diez:", round(float(pca.explained_variance_ratio_.sum() * 100), 1))
''', r'''
# Esta celda debe estandarizar `C`, ajustar un PCA de 10 componentes y dibujar:
#   - a la izquierda, la varianza explicada por componente;
#   - a la derecha, las ventanas en CP1 contra CP2, en gris, y resaltadas en color las de cuatro especies.
# Debe imprimir la varianza que explican los dos primeros componentes. Es un análisis descriptivo:
# este PCA no entra al modelo.
'''),
("md", r'''
**Mi lectura** (¿las cuatro especies forman grupos separados o se traslapan?):
'''),
("md", r'''
## 3. Modelado

### 3.1 El protocolo y la línea base

Las grabaciones se reparten en tres grupos, estratificando por especie: 60 % para entrenar, 20 % para validar y 20 % para probar. Eso deja 12, 4 y 4 grabaciones por especie. **Se reparten grabaciones, no ventanas**: todas las ventanas de una grabación quedan del mismo lado.

El modelo predice por ventana y la grabación se decide promediando. Si la grabación $i$ tiene ventanas $v=1,\dots,n_i$ y el modelo da $\hat p_{v}(k)$ para la especie $k$,

$$\bar p_i(k)=\frac{1}{n_i}\sum_{v=1}^{n_i}\hat p_{v}(k),\qquad \hat y_i=\arg\max_k\ \bar p_i(k).$$

Hay dos líneas base. La primera es adivinar: 1 de cada 50, o 2 %. La segunda es una regresión logística con una sola familia pobre, las tres primeras columnas de MFCC.
'''),
("code", r'''
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

especies = sorted(aves["primary_label"].unique())
y_grab = aves["primary_label"].map({e: i for i, e in enumerate(especies)}).values      # especie de cada grabación
y_vent = y_grab[de_grabacion]                                                          # especie de cada ventana

ids = aves["indice"].values
g_entrena, resto = train_test_split(ids, test_size=0.4, stratify=y_grab, random_state=0)
g_valida, g_prueba = train_test_split(resto, test_size=0.5, stratify=y_grab[resto], random_state=0)
E, V, P_ = np.isin(de_grabacion, g_entrena), np.isin(de_grabacion, g_valida), np.isin(de_grabacion, g_prueba)
assert not set(de_grabacion[E]) & set(de_grabacion[V]) and not set(de_grabacion[E]) & set(de_grabacion[P_])
print(f"Grabaciones: {len(g_entrena)} / {len(g_valida)} / {len(g_prueba)} | Ventanas: {E.sum()} / {V.sum()} / {P_.sum()}")

def por_grabacion(probabilidades, tramo, grabs):
    """Promedia las probabilidades de las ventanas de cada grabación. Devuelve una fila por grabación, en el orden de `grabs`."""
    return pd.DataFrame(probabilidades, index=de_grabacion[tramo]).groupby(level=0).mean().loc[grabs].values

def exactitudes(modelo, columnas, tramo, grabs):
    p = por_grabacion(modelo.predict_proba(C.loc[tramo, columnas].values), tramo, grabs)
    orden = np.argsort(-p, axis=1)
    return float((orden[:, 0] == y_grab[grabs]).mean()), float((orden[:, :3] == y_grab[grabs][:, None]).any(axis=1).mean())

todas = list(C.columns)
pobres = ["mfcc_media_0", "mfcc_media_1", "mfcc_media_2"]
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(C.loc[E, pobres].values, y_vent[E])
filas_tabla = {"azar": {"validación": 1 / len(especies), "prueba": 1 / len(especies), "top-3 en prueba": 3 / len(especies)},
               "tres MFCC": {"validación": exactitudes(base, pobres, V, g_valida)[0], "prueba": exactitudes(base, pobres, P_, g_prueba)[0],
                             "top-3 en prueba": exactitudes(base, pobres, P_, g_prueba)[1]}}
print({n: {k: round(v, 3) for k, v in f.items()} for n, f in filas_tabla.items()})
''', r'''
# Esta celda debe:
#   - codificar la especie (primary_label) como entero: `y_grab` por grabación y `y_vent` por ventana;
#   - repartir las GRABACIONES (aves["indice"]) en 60 % / 20 % / 20 %, estratificando por especie, con random_state=0,
#     y construir las máscaras de ventanas `E`, `V`, `P_` con np.isin(de_grabacion, ...);
#   - comprobar con un `assert` que ninguna grabación quedó en dos grupos;
#   - definir `por_grabacion(probabilidades, tramo, grabs)`, que promedia las probabilidades de las ventanas de cada grabación,
#     y `exactitudes(modelo, columnas, tramo, grabs)`, que devuelve la exactitud top-1 y top-3 POR GRABACIÓN;
#   - calcular las dos líneas base: el azar (1/50) y una regresión logística con solo
#     las columnas mfcc_media_0, mfcc_media_1 y mfcc_media_2.
'''),
("md", r'''
### 3.2 El catálogo de modelos

| Familia | Idea en una línea |
|---|---|
| Regresión logística, con dos niveles de regularización | $P(y=k\mid x)=e^{\beta_k^\top x}/\sum_j e^{\beta_j^\top x}$ |
| Análisis discriminante lineal | Supone una normal por especie con la misma covarianza $\Sigma$: $\delta_k(x)=x^\top\Sigma^{-1}\mu_k-\tfrac12\mu_k^\top\Sigma^{-1}\mu_k+\log\pi_k$ |
| Bayes ingenuo | Supone que las columnas son independientes dentro de cada especie |
| $k$ vecinos, con dos valores de $k$ | Vota con las ventanas de entrenamiento más parecidas |
| Árbol de decisión | Parte la tabla con preguntas de sí o no sobre una columna a la vez |
| Bosque aleatorio y *ExtraTrees* | Promedian muchos árboles, entrenados sobre remuestreos o con cortes al azar |
| *Gradient boosting* por histogramas | Suma árboles pequeños, cada uno corrigiendo el error del anterior |
| Máquina de soporte vectorial | Busca la frontera con más margen, curva gracias a un núcleo gaussiano |
| Red neuronal pequeña | Una capa oculta de 128 unidades |

La regla: gana el de mayor exactitud por grabación en **validación**, y su exactitud en **prueba** es la que se reporta.

**Cómo leer la celda del catálogo.** Esta celda ya está escrita. El diccionario `MODELOS` tiene una línea por candidato, con su nombre y un `make_pipeline` que encadena la preparación de la tabla con el clasificador. `escalar()` pone las columnas en la misma escala, que es lo que necesitan los modelos que miden distancias o usan gradientes. Para agregar un modelo se importa su clase y se escribe una línea más en el diccionario. La grabación se decide promediando probabilidades, así que solo sirven clasificadores que tengan `predict_proba`. El código que le pidan a Claude en las celdas siguientes debe recorrer lo que haya en `MODELOS`, sin nombrar ningún modelo, para que no haya que tocar nada más.

**Ejercicio.** Cuando el cuaderno corra completo, agreguen al menos tres modelos más de scikit-learn y vuelvan a correr desde esa celda.
'''),
("dado", r'''
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

escalar = lambda: StandardScaler()                     # pone todas las columnas en la misma escala

# El catálogo: una línea por modelo. Para agregar uno, escriban otra línea con su nombre y su pipeline.
MODELOS = {
    "logística":              make_pipeline(escalar(), LogisticRegression(C=0.3, max_iter=2000)),
    "logística (C = 0,01)":   make_pipeline(escalar(), LogisticRegression(C=0.01, max_iter=2000)),
    "discriminante lineal":   make_pipeline(escalar(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "Bayes ingenuo":          make_pipeline(GaussianNB()),
    "k vecinos (k = 5)":      make_pipeline(escalar(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
    "k vecinos (k = 15)":     make_pipeline(escalar(), KNeighborsClassifier(n_neighbors=15, weights="distance")),
    "árbol":                  make_pipeline(DecisionTreeClassifier(min_samples_leaf=5, random_state=0)),
    "bosque":                 make_pipeline(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
    "ExtraTrees":             make_pipeline(ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=0)),
    "boosting":               make_pipeline(HistGradientBoostingClassifier(max_iter=30, max_depth=3, random_state=0)),
    "SVM":                    make_pipeline(escalar(), SVC(C=3, probability=True, random_state=0)),
    "red neuronal":           make_pipeline(escalar(), MLPClassifier(hidden_layer_sizes=(128,), alpha=1.0, max_iter=400, random_state=0)),
}
print(len(MODELOS), "modelos en el catálogo:", ", ".join(MODELOS))
'''),
("code", r'''
entrenados = {}
for nombre, modelo in MODELOS.items():                 # recorre lo que haya en el catálogo
    inicio = time.time()
    entrenados[nombre] = clone(modelo).fit(C.loc[E, todas].values, y_vent[E])
    en_prueba = exactitudes(entrenados[nombre], todas, P_, g_prueba)
    filas_tabla[nombre] = {"validación": exactitudes(entrenados[nombre], todas, V, g_valida)[0], "prueba": en_prueba[0],
                           "top-3 en prueba": en_prueba[1], "segundos": round(time.time() - inicio, 1)}
comparacion = pd.DataFrame(filas_tabla).T
display((comparacion[["validación", "prueba", "top-3 en prueba"]] * 100).round(1).join(comparacion["segundos"]))

(comparacion[["validación", "prueba"]] * 100).sort_values("validación").plot.barh(color=["#1D4E89", "#9AA5B1"], figsize=(9, 0.32 * len(comparacion) + 1.5))
plt.xlabel("exactitud por grabación (%)"); plt.title("Validación para escoger, prueba para reportar"); plt.show()

candidatos = comparacion.drop(index=["azar", "tres MFCC"])
mejor = candidatos["validación"].idxmax()
print(f"Ganador en validación: {mejor} | exactitud en prueba {comparacion.loc[mejor, 'prueba'] * 100:.1f} % | entre las tres primeras {comparacion.loc[mejor, 'top-3 en prueba'] * 100:.1f} %")
print("Modelos empatados en el primer puesto de validación:", int((candidatos["validación"] == candidatos["validación"].max()).sum()))
''', r'''
# Esta celda debe recorrer el diccionario `MODELOS` de la celda anterior, sin nombrar ningún modelo:
# `for nombre, modelo in MODELOS.items():`. Para cada uno debe entrenar una copia (sklearn.base.clone) con las
# ventanas de entrenamiento y medir con `exactitudes` la exactitud POR GRABACIÓN en validación y en prueba,
# y la top-3 en prueba. Debe mostrar una tabla con las líneas base y todos los modelos del catálogo,
# una gráfica de barras, y dejar:
#   `entrenados` (diccionario de modelos ya ajustados), `comparacion` (la tabla) y `mejor`
#   (el nombre del ganador EN VALIDACIÓN). Si hay empate en el primer puesto, debe decir cuántos empataron.
'''),
("md", r'''
Comparen con su primera apuesta. **Mi lectura** (¿cuántas veces mejor que el azar?, ¿sirve para algo una exactitud así?):

El ganador se escogió entre muchos candidatos, y eso tiene un costo que conviene mirar. Pidan una gráfica pequeña que ubique a cada modelo según su puesto en validación y su puesto en prueba.
'''),
("code", r'''
puestos = candidatos[["validación", "prueba"]].rank(ascending=False, method="min").astype(int)
n = len(puestos)
fig, eje = plt.subplots(figsize=(5.6, 5), facecolor="#FBFAF7"); eje.set_facecolor("#FBFAF7")
eje.plot([1, n], [1, n], color="#9AA5B1", ls="--", lw=1)             # sobre esta línea, el puesto se repite
eje.scatter(puestos["validación"], puestos["prueba"], s=45, zorder=3, color=["#D7263D" if m == mejor else "#1D4E89" for m in puestos.index])
for m, fila in puestos.iterrows():
    eje.annotate(m, (fila["validación"], fila["prueba"]), xytext=(6, 3), textcoords="offset points", fontsize=7.5)
eje.set(title="Puesto en validación contra puesto en prueba", xlabel="puesto en validación (1 es el mejor)", ylabel="puesto en prueba",
        xticks=range(1, n + 1), yticks=range(1, n + 1), xlim=(0.5, n + 2.5), ylim=(n + 0.5, 0.5))
plt.show()

en_orden = candidatos["validación"].sort_values(ascending=False)
print(f"El ganador en validación ({mejor}) quedó de puesto {puestos.loc[mejor, 'prueba']} en prueba, entre {n} candidatos.")
print(f"Ventaja del primero sobre el segundo en validación: {(en_orden.iloc[0] - en_orden.iloc[1]) * 100:.1f} puntos, con {len(g_valida)} grabaciones. Una grabación vale {100 / len(g_valida):.1f} puntos.")
print("Modelos que cambiaron de puesto entre validación y prueba:", int((puestos["validación"] != puestos["prueba"]).sum()), "de", n)
''', r'''
# Esta celda debe dibujar una gráfica pequeña con un punto por modelo del catálogo: en el eje horizontal su puesto
# en validación (1 es el mejor) y en el vertical su puesto en prueba, con el nombre de cada modelo, la diagonal
# punteada y el ganador de validación en rojo (#D7263D; los demás en azul #1D4E89). Debe imprimir en qué puesto
# quedó el ganador en prueba, la ventaja del primero sobre el segundo en validación y cuánto vale una sola grabación.
'''),
("md", r'''
Si el orden de validación se repitiera en la prueba, todos los puntos caerían sobre la diagonal. Miren cuánto se alejan y comparen la ventaja del ganador con lo que vale una sola grabación. Con tantos candidatos, ¿cuánto del ganador es azar? **Mi respuesta:**

### 3.3 La trampa: barajar ventanas

Repitan el experimento de la demostración. Tomen dos de los modelos y sepárenlos de dos maneras: barajando ventanas con `train_test_split`, y por grabación como en 3.1. Midan la exactitud **por ventana** en ambos casos para que la comparación sea pareja.
'''),
("code", r'''
A = C[todas].values
a, b = train_test_split(np.arange(len(A)), test_size=0.2, stratify=y_vent, random_state=0)       # barajando ventanas
trampa = {}
for nombre in ["k vecinos (k = 15)", "bosque"]:
    al_azar = float((clone(MODELOS[nombre]).fit(A[a], y_vent[a]).predict(A[b]) == y_vent[b]).mean())
    por_grab = float((entrenados[nombre].predict(A[P_]) == y_vent[P_]).mean())
    trampa[nombre] = {"ventanas al azar": al_azar, "por grabación": por_grab, "inflado": al_azar - por_grab}
compartidas = len(set(de_grabacion[a]) & set(de_grabacion[b]))
display((pd.DataFrame(trampa).T * 100).round(1))
print(f"Al barajar, {compartidas} de las {len(set(de_grabacion[b]))} grabaciones de la prueba tienen otra ventana suya en el entrenamiento.")
''', r'''
# Esta celda debe mostrar la trampa con números. Para dos modelos del catálogo, "k vecinos (k = 15)" y "bosque"
# (una copia nueva de cada uno, con sklearn.base.clone):
#   - separar las VENTANAS al azar (train_test_split, 20 % de prueba, estratificado, random_state=0),
#     entrenar y medir la exactitud por ventana;
#   - medir la exactitud por ventana del mismo modelo con la separación por grabación de 3.1.
# Debe mostrar las dos exactitudes lado a lado, la diferencia, y contar cuántas grabaciones de la prueba
# barajada tienen otra ventana suya en el entrenamiento.
'''),
("md", r'''
Comparen con su segunda apuesta. **Mi explicación** (¿qué aprendió el modelo cuando se barajaron las ventanas?, ¿cuál de las dos cifras le pasará al equipo de la reserva con una grabación nueva?):

### 3.4 Explicar

Dos preguntas. ¿Qué familia de columnas sostiene el resultado? ¿Qué especies confunde el modelo entre sí?
'''),
("code", r'''
familias = {"mfcc": [c for c in todas if c.startswith("mfcc_")], "mel": [c for c in todas if c.startswith("mel_")], "todas": todas}
por_familia = {}
for nombre, columnas in familias.items():
    modelo = clone(MODELOS[mejor]).fit(C.loc[E, columnas].values, y_vent[E])
    por_familia[nombre] = exactitudes(modelo, columnas, V, g_valida)[0] * 100
print("Exactitud en validación por familia (%):", {n: round(v, 1) for n, v in por_familia.items()})

p_valida = por_grabacion(entrenados[mejor].predict_proba(C.loc[V, todas].values), V, g_valida)
confusion = pd.crosstab(pd.Series(y_grab[g_valida], name="real"), pd.Series(p_valida.argmax(axis=1), name="predicha")).reindex(index=range(50), columns=range(50), fill_value=0)
nombre_comun = aves.drop_duplicates("primary_label").set_index("primary_label")["common_name"].reindex(especies).values
pares = confusion.where(~np.eye(50, dtype=bool)).stack().sort_values(ascending=False).head(6)
print("\nConfusiones más frecuentes en validación (real -> predicha: veces):")
for (real, predicha), veces in pares.items():
    print(f"  {nombre_comun[real]} -> {nombre_comun[predicha]}: {int(veces)}")
aciertos = pd.Series(np.diag(confusion), index=nombre_comun)
print("\nEspecies con 4 de 4 aciertos:", list(aciertos[aciertos == 4].index))
print("Especies con 0 de 4 aciertos:", len(aciertos[aciertos == 0]))

plt.figure(figsize=(6, 5)); plt.imshow(confusion, cmap="Greens"); plt.colorbar(label="grabaciones")
plt.title(f"Matriz de confusión en validación ({mejor})"); plt.xlabel("especie predicha"); plt.ylabel("especie real"); plt.show()
''', r'''
# Esta celda debe responder las dos preguntas con el modelo ganador, sobre la VALIDACIÓN:
#   - entrenarlo con solo las columnas `mfcc_`, con solo las `mel_` y con todas, y comparar la exactitud por grabación;
#   - construir la matriz de confusión por grabación (50 x 50), dibujarla, e imprimir las seis
#     confusiones más frecuentes con el nombre común de las especies, las especies que acierta siempre
#     y cuántas no acierta nunca.
'''),
("md", r'''
**Mi lectura** (¿las especies que se confunden tienen algo en común?, ¿qué le preguntarían al ornitólogo?):
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: las cien grabaciones del experto

El modelo se equivoca en muchas grabaciones y el experto solo puede oír `REVISIONES`. La propuesta es mandarle primero aquellas en que el modelo está menos seguro. La confianza de la grabación $i$ es la probabilidad promedio de la especie ganadora,

$$c_i=\max_k\ \bar p_i(k),$$

y el experto recibe las `REVISIONES` grabaciones de prueba con menor $c_i$. Supongan que el experto siempre acierta. Comparen tres políticas sobre la prueba: que el experto no oiga nada, que oiga cien escogidas al azar y que oiga las cien de menor confianza.

La política al azar tiene una exactitud esperada que se calcula sin simular. Si el modelo tiene exactitud $a$ y el experto revisa una fracción $r$ de las grabaciones, el sistema acierta $a+r\,(1-a)$.
'''),
("code", r'''
p_prueba = por_grabacion(entrenados[mejor].predict_proba(C.loc[P_, todas].values), P_, g_prueba)
acierta = p_prueba.argmax(axis=1) == y_grab[g_prueba]
confianza = p_prueba.max(axis=1)
orden = np.argsort(confianza)                                  # de menor a mayor confianza
revisadas, sin_revisar = orden[:REVISIONES], orden[REVISIONES:]

n, a, r = len(acierta), acierta.mean(), REVISIONES / len(acierta)
sistema_confianza = (REVISIONES + acierta[sin_revisar].sum()) / n
print(f"Grabaciones de prueba: {n} | errores del modelo: {int((~acierta).sum())}")
print(f"Exactitud del modelo solo:                         {a * 100:.1f} %")
print(f"Experto oye {REVISIONES} al azar (valor esperado):         {(a + r * (1 - a)) * 100:.1f} %")
print(f"Experto oye las {REVISIONES} de menor confianza:           {sistema_confianza * 100:.1f} %")
print(f"Errores del modelo entre las {REVISIONES} revisadas: {int((~acierta[revisadas]).sum())} | exactitud del modelo en las que no se revisan: {acierta[sin_revisar].mean() * 100:.1f} %")

azar = np.random.default_rng(0)
simulado = [(REVISIONES + acierta[azar.permutation(n)[REVISIONES:]].sum()) / n for _ in range(2000)]
print(f"Al azar, 95 % de los sorteos caen entre {np.percentile(simulado, 2.5) * 100:.1f} % y {np.percentile(simulado, 97.5) * 100:.1f} %")

cobertura = np.arange(1, n + 1)
exactitud_acumulada = np.cumsum(acierta[orden[::-1]]) / cobertura
plt.figure(figsize=(9, 3.6))
plt.plot(cobertura, exactitud_acumulada * 100, color="#1D4E89")
plt.axvline(n - REVISIONES, color="#D7263D", ls="--"); plt.axhline(a * 100, color="gray", ls=":")
plt.xlabel("grabaciones que se le dejan al modelo, de la más segura a la menos segura"); plt.ylabel("exactitud del modelo en ellas (%)")
plt.title("Entre más seguro dice estar el modelo, ¿más acierta?"); plt.show()

lista = aves.set_index("indice").loc[g_prueba[revisadas], ["archivo", "common_name", "url"]].assign(confianza=confianza[revisadas].round(3))
lista.head(10)
''', r'''
# Esta celda debe, con el modelo ganador y SOLO sobre la prueba:
#   - calcular por grabación la especie predicha y la confianza (la mayor probabilidad promedio);
#   - ordenar las grabaciones de menor a mayor confianza y marcar las primeras REVISIONES para el experto;
#   - comparar tres exactitudes del sistema completo, suponiendo que el experto no se equivoca:
#     modelo solo; experto oye REVISIONES al azar (fórmula a + r(1 - a), y además el rango de 2.000 sorteos);
#     experto oye las REVISIONES de menor confianza;
#   - dibujar la exactitud del modelo sobre las grabaciones más seguras a medida que se le dejan más;
#   - mostrar las diez primeras grabaciones de la lista del experto, con archivo, especie y url.
'''),
("md", r'''
Comparen con su tercera apuesta. Miren si la política de menor confianza cae dentro o fuera del rango de los sorteos al azar. Esa comparación dice si la confianza del modelo sirve para ordenar el trabajo del experto. **Mi lectura:**

### La decisión en tres frases

Para la directora de la reserva, que no sabe qué es un MFCC.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Auditoría del código que escribió la IA

Antes de creerle a sus números, comprueben tres cosas en el código que pegaron. Escriban qué encontraron y, si algo estaba mal, cómo lo corrigieron.

1. **¿La partición es por grabación?** Busquen dónde se separan entrenamiento, validación y prueba. Comprueben con código que ninguna grabación tiene ventanas en dos grupos. Una IA tiende a escribir `train_test_split(C, y)` sobre las ventanas, que es la trampa de 3.3. **Lo que encontré:**
2. **¿Se respetó la frecuencia de muestreo y se oyó lo que se dijo?** Si la IA usó `librosa.load`, revisen si puso `sr=None`: por defecto esa función remuestrea a 22.050 Hz y corta todo lo que suena por encima de 11.025 Hz. Comprueben también que cada ventana tiene `5 * fs` muestras y que no se usaron más de `SEGUNDOS_OIDOS` por grabación. **Lo que encontré:**
3. **¿Algo se ajustó con datos de validación o de prueba?** Revisen que el `StandardScaler` esté dentro del pipeline de cada modelo, que el ganador se haya escogido con la columna de validación y que la exactitud reportada sea por grabación y no por ventana. **Lo que encontré:**

### 4.3 Bitácora de decisiones

Llenen las filas vacías.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Muestra de 1.000 grabaciones de Hugging Face | Las grabaciones completas de Kaggle | No exige cuenta y se alcanza a procesar en una clase |
| Ventanas de 5 segundos desde el comienzo | | |
| Restar el piso de ruido por banda | | |
| Partición por grabación, 60 / 20 / 20 | | |
| Promediar probabilidades de las ventanas | | |
| Escoger el modelo en validación | | |
| Mandar al experto las de menor confianza | | |

### 4.4 Riesgos y mantenimiento

- Las grabaciones de entrenamiento son de alguien que apuntó el micrófono a un ave. Las de una grabadora fija en el bosque traen varias especies a la vez, lejos y con lluvia. ¿Qué le pasará a la exactitud?
- Buena parte de las grabaciones se hicieron fuera de Colombia. ¿Por qué puede importar, si la especie es la misma?
- Cada grabación tiene autor y licencia, casi todas de uso no comercial. ¿Qué debe hacer el equipo antes de publicar un producto con este modelo, y cómo se le da crédito a quien grabó?
- Publicar dónde canta una especie amenazada puede ponerla en riesgo. ¿Qué parte de los resultados no pondrían en un mapa abierto?
- El modelo solo conoce 50 especies. ¿Qué responde cuando canta la número 51, y cómo lo detectarían?

### 4.5 Cambien una línea

Vuelvan a la celda de parámetros, cambien `SEGUNDOS_OIDOS = 30` por `SEGUNDOS_OIDOS = 5` y corran todo de nuevo. Con 5 segundos el modelo oye una sola ventana por grabación.

| | 30 segundos | 5 segundos |
|---|---|---|
| Ventanas en total | | |
| Ganador en validación | | |
| Exactitud del ganador en prueba | | |
| Exactitud entre las tres primeras, en prueba | | |
| Exactitud del sistema con el experto oyendo las de menor confianza | | |

Con la tabla llena, respondan en tres líneas: ¿cuánto se gana por oír más de cada grabación? ¿De dónde creen que viene la ganancia, de tener más ventanas para entrenar o de promediar más ventanas al predecir? **Mi respuesta:**

### 4.6 Entrega

**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase08/`. Antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*: un cuaderno que no corre de principio a fin desde cero no se califica.
'''),
]

AVES_SOLUCION = [("md", c[1]) if c[0] == "md" else ("code", c[1]) for c in AVES]
AVES_ESTUDIANTE = [("md", c[1]) if c[0] == "md" else ("code", c[1] if c[0] == "dado" else c[2]) for c in AVES]
