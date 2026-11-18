"""Celdas de la demostración de la clase 7: Lo que dicen los turistas (Rest-Mex 2023)."""
from molde import ARRANQUE

TURISTAS = [
("md", r'''
# Clase 7 · Lo que dicen los turistas

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. En la segunda mitad cada uno resuelve el gemelo, *La plata pública*, con la misma técnica y otro texto.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Convertir texto en una tabla de números con TF-IDF, comparar cinco clasificadores contra una regla de palabras clave, leer en los coeficientes de qué se queja la gente y usar el clasificador para medir una proporción sin caer en el error de contar con él. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular el problema y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que proponga otra lista de palabras vacías o que escriba una gráfica. Deben poder explicar sin ayuda qué número guarda cada celda de una matriz TF-IDF, por qué la exactitud engaña cuando las quejas son pocas, qué dice un coeficiente de la regresión logística y por qué contar las predicciones no mide bien una proporción. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con las predicciones escritas, la decisión en tres frases y la tabla de la variante llenas. |
'''),
("code", ARRANQUE),
("md", r'''
La celda siguiente tiene el único parámetro del cuaderno. Al final lo cambian y comparan.
'''),
("code", r'''
UMBRAL_QUEJA = 2        # una reseña cuenta como queja si tiene esta cantidad de estrellas o menos
SEMILLA = 0

import time
'''),
("md", r'''
## 1. Formulación

Imagínense una mesa de turismo que reúne a tres destinos: México, Cuba y Colombia. Le llegan miles de comentarios escritos por visitantes sobre hoteles, restaurantes y atractivos. Tiene presupuesto para intervenir un solo frente este año. Nadie alcanza a leer todo, y muchos comentarios llegan sin calificación, por encuestas abiertas o por redes.

1. **Predecir.** Dado el texto de una reseña, ¿es una queja?
2. **Explicar.** ¿Qué palabras distinguen una queja? ¿De qué se queja la gente?
3. **Decidir.** ¿En qué país y tipo de servicio conviene intervenir primero, y sobre qué queja?

**El dato.** Rest-Mex 2023 es el conjunto de una competencia académica de análisis de sentimiento en español: reseñas de turismo con su calificación. Trabajamos con una muestra de 60.000 reseñas, 20.000 por país. El conjunto es para uso académico y no se redistribuye, así que cada uno lo descarga de la fuente.

**La métrica.** Positivo es «queja». Las quejas son pocas, y con clases desbalanceadas la exactitud no sirve. Uso la precisión y la sensibilidad,

$$\text{precisión}=\frac{VP}{VP+FP},\qquad \text{sensibilidad}=\frac{VP}{VP+FN},$$

que responden dos preguntas distintas: de lo que el modelo marcó como queja, cuánto lo era; y de las quejas que había, cuántas encontró. Su media armónica es

$$F_1=\frac{2\cdot\text{precisión}\cdot\text{sensibilidad}}{\text{precisión}+\text{sensibilidad}}.$$

Para comparar modelos sin fijar un umbral uso la **precisión promedio** (AP), que es el área bajo la curva de precisión contra sensibilidad. Un modelo que ordena al azar tiene una AP igual a la proporción de quejas.

### Predicción escrita

- ¿Qué fracción de las reseñas tiene una o dos estrellas? **Mi apuesta:**
- ¿Qué exactitud tiene el «modelo» que responde siempre «no es queja»? **Mi apuesta:**
- ¿Cuál será la palabra que más empuja una reseña hacia queja? **Mi apuesta:**
- ¿Qué combinación de país y servicio tendrá la mayor proporción de quejas? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos
'''),
("code", r'''
inicio = time.time()
crudo = descargar("restmex")
print(f"Tiempo: {time.time() - inicio:.0f} s | {len(crudo)} reseñas")
crudo.head(4)
'''),
("md", r'''
### 2.1 Un diccionario que nadie publicó

Las columnas `Polarity`, `Country` y `Type` vienen como números y la copia que usamos no trae diccionario. No voy a suponer qué significa cada código. Lo deduzco del contenido y lo dejo dicho como **inferencia**.

Para `Country` cuento en cuántas reseñas de cada código aparecen nombres de lugares. Para `Type`, palabras propias de cada servicio. Para `Polarity`, miro los títulos más frecuentes de cada valor.
'''),
("code", r'''
texto_crudo = (crudo["Title"].fillna("") + " " + crudo["Review"].fillna("")).str.lower()

pistas_pais = {"méxico|mexico|cancún|cancun|oaxaca": "México", "cuba|habana|varadero": "Cuba", "colombia|cartagena|bogotá|medellín": "Colombia"}
print("Porcentaje de reseñas de cada código de Country que menciona cada grupo de lugares:")
display(pd.DataFrame({nombre: texto_crudo.str.contains(patron).groupby(crudo["Country"]).mean() * 100 for patron, nombre in pistas_pais.items()}).round(1))

pistas_tipo = {"habitación|habitacion|hotel|recepción": "palabras de hotel", "comida|mesero|platillo|plato|restaurante": "palabras de restaurante",
               "museo|playa|parque|recorrido|guía|visita": "palabras de atractivo"}
print("Porcentaje de reseñas de cada código de Type que usa cada grupo de palabras:")
display(pd.DataFrame({nombre: texto_crudo.str.contains(patron).groupby(crudo["Type"]).mean() * 100 for patron, nombre in pistas_tipo.items()}).round(1))

print("Títulos más frecuentes por valor de Polarity:")
for valor, grupo in crudo.groupby("Polarity"):
    print(f"  {valor}: {grupo['Title'].str.strip().str.lower().value_counts().head(5).index.tolist()}")
'''),
("md", r'''
Las tres tablas dejan poco margen. Cada código de `Country` menciona sobre todo los lugares de un país, cada código de `Type` usa sobre todo las palabras de un servicio, y los títulos pasan de «pésimo» a «excelente» al subir `Polarity`. Mi lectura es

- `Country`: 0 es México, 1 es Cuba, 2 es Colombia;
- `Type`: 0 es hotel, 1 es restaurante, 2 es atractivo;
- `Polarity`: de 0 a 4, que corresponde a una calificación de 1 a 5 estrellas.

Es una inferencia sostenida por evidencia, no un dato publicado. Va a la bitácora.

### 2.2 Limpieza a la vista
'''),
("code", r'''
datos = crudo.copy()
datos["pais"] = datos["Country"].map({0: "México", 1: "Cuba", 2: "Colombia"})
datos["tipo"] = datos["Type"].map({0: "hotel", 1: "restaurante", 2: "atractivo"})
datos["estrellas"] = datos["Polarity"] + 1

print("Reseñas sin texto:", int(datos["Review"].isna().sum()), "| sin título:", int(datos["Title"].isna().sum()))
print("Textos de reseña más repetidos:")
print(datos["Review"].value_counts().head(4).to_string())
repetida = datos.duplicated(["Title", "Review"], keep="first")
print(f"\nReseñas que repiten título y texto de otra anterior: {repetida.sum()} ({repetida.mean() * 100:.1f} %)")
print("¿Las copias tienen las mismas estrellas que su original?",
      f"{(datos[datos.duplicated(['Title', 'Review'], keep=False)].groupby(['Title', 'Review'])['estrellas'].nunique() == 1).mean() * 100:.0f} % de los grupos")
'''),
("md", r'''
Dos trampas de limpieza. La primera es `#ERROR!`: hay reseñas cuyo texto es el mensaje de error de una hoja de cálculo. Alguien armó el conjunto pasando por Excel y esas celdas se dañaron. No tienen texto que analizar y las quito. La segunda son las reseñas repetidas, con el mismo título y el mismo texto. Si una copia cae en entrenamiento y otra en prueba, el modelo se examina con algo que ya vio, que es la fuga de la clase 5 en versión de texto. Me quedo con la primera aparición de cada una. En la sección 3 mido cuánto habría inflado el resultado dejarlas.
'''),
("code", r'''
con_copias = datos[datos["Review"].fillna("").str.strip() != "#ERROR!"].reset_index(drop=True)      # se guarda para medir la fuga
datos = con_copias[~con_copias.duplicated(["Title", "Review"], keep="first")].reset_index(drop=True)
datos["texto"] = datos["Title"].fillna("") + ". " + datos["Review"].fillna("")
datos["queja"] = (datos["estrellas"] <= UMBRAL_QUEJA).astype(int)
y = datos["queja"].to_numpy()
print(f"Quedan {len(datos)} reseñas | quejas ({UMBRAL_QUEJA} estrellas o menos): {y.sum()} ({y.mean() * 100:.1f} %)")
'''),
("md", r'''
### 2.3 Conocer el dato
'''),
("code", r'''
fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(15, 4))
(datos["estrellas"].value_counts(normalize=True).sort_index() * 100).plot.bar(ax=izq, color="#16302B", rot=0)
izq.set(title="Calificaciones", xlabel="estrellas", ylabel="% de las reseñas")

tasa = datos.pivot_table(index="pais", columns="tipo", values="queja", aggfunc="mean") * 100
imagen = centro.imshow(tasa, cmap="YlOrBr")
centro.set_xticks(range(3)); centro.set_xticklabels(tasa.columns); centro.set_yticks(range(3)); centro.set_yticklabels(tasa.index)
for i in range(3):
    for j in range(3):
        centro.text(j, i, f"{tasa.iloc[i, j]:.1f} %", ha="center", va="center")
centro.set_title("Proporción de quejas por país y servicio")

datos["palabras"] = datos["texto"].str.split().str.len()
datos.boxplot(column="palabras", by="estrellas", ax=der, showfliers=False, color="#16302B")
der.set(title="Largo de la reseña según las estrellas", xlabel="estrellas", ylabel="palabras"); plt.suptitle("")
plt.tight_layout(); plt.show()

print("Reseñas por país y servicio:"); display(pd.crosstab(datos["pais"], datos["tipo"]))
'''),
("md", r'''
Tres lecturas. La primera es el desbalance: la mayoría de las reseñas son de cuatro y cinco estrellas, y las quejas son una fracción pequeña. La segunda es que la proporción de quejas cambia con el país y con el servicio, y esa tabla ya responde parte de la pregunta de decidir. La tercera es que quien se queja escribe más.

Un sesgo que no se arregla con código: una reseña la escribe quien quiso escribirla, en una plataforma, en español. Estas proporciones describen a quienes reseñan, no a todos los visitantes.

### 2.4 De texto a números: TF-IDF

Un modelo necesita columnas. La manera clásica de dárselas es contar palabras. Cada reseña $d$ se vuelve un vector con una posición por cada término $t$ del vocabulario. El valor combina dos ideas. La frecuencia del término en la reseña, $\text{tf}(t,d)$, dice cuánto habla la reseña de eso. La frecuencia inversa de documento castiga a los términos que aparecen en todas partes,

$$\text{idf}(t)=\ln\frac{1+n}{1+\text{df}(t)}+1,$$

donde $n$ es el número de reseñas y $\text{df}(t)$ el número de reseñas que contienen el término. El peso es el producto,

$$w_{t,d}=\bigl(1+\ln \text{tf}(t,d)\bigr)\cdot\text{idf}(t),$$

y cada vector se divide por su norma para que una reseña larga no pese más que una corta. Uso palabras sueltas y pares de palabras seguidas, porque «no volvería» dice algo que «no» y «volvería» por separado no dicen. Quito las tildes para que «pésimo» y «pesimo» sean el mismo término.

El vocabulario se aprende **solo con el entrenamiento**. Por eso parto antes de vectorizar: 60 % para entrenar, 20 % para validar y 20 % para la prueba, con la misma proporción de quejas en los tres.
'''),
("code", r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

indices = np.arange(len(datos))
resto, P = train_test_split(indices, test_size=0.20, random_state=SEMILLA, stratify=y)
E, V = train_test_split(resto, test_size=0.25, random_state=SEMILLA, stratify=y[resto])
print(f"Entrenamiento: {len(E)} | Validación: {len(V)} | Prueba: {len(P)} | quejas en cada uno: {y[E].mean():.3f}, {y[V].mean():.3f}, {y[P].mean():.3f}")

def vectorizador():
    return TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")

inicio = time.time()
tfidf = vectorizador().fit(datos["texto"].iloc[E])
X_E, X_V, X_P = (tfidf.transform(datos["texto"].iloc[tramo]) for tramo in (E, V, P))
terminos = np.array(tfidf.get_feature_names_out())
print(f"Tiempo: {time.time() - inicio:.0f} s")
print(f"Matriz de entrenamiento: {X_E.shape[0]} reseñas por {X_E.shape[1]} términos")
print(f"Celdas distintas de cero: {X_E.nnz / (X_E.shape[0] * X_E.shape[1]) * 100:.3f} % | términos por reseña, en promedio: {X_E.nnz / X_E.shape[0]:.0f}")
ejemplo = X_E[0]
print("\nUna reseña:", datos["texto"].iloc[E[0]][:150], "…")
print("Sus términos de mayor peso:", [(terminos[j], round(float(v), 2)) for j, v in sorted(zip(ejemplo.indices, ejemplo.data), key=lambda par: -par[1])[:6]])
'''),
("md", r'''
La matriz tiene más columnas que filas y casi todas sus celdas son cero. Por eso se guarda en formato disperso, que solo anota las celdas ocupadas, y por eso los modelos lineales funcionan tan bien con texto: en un espacio con tantas dimensiones casi todo se puede separar con un plano.

### 2.5 Sin supervisión: ¿de qué hablan las quejas?

Antes de entrenar un clasificador le pregunto al texto de qué temas se compone. Tomo solo las quejas del entrenamiento y factorizo su matriz TF-IDF, $A\in\mathbb{R}^{n\times m}$ con entradas no negativas, en dos matrices no negativas más pequeñas,

$$A\approx WH,\qquad W\in\mathbb{R}_{\ge0}^{n\times k},\quad H\in\mathbb{R}_{\ge0}^{k\times m}.$$

Cada fila de $H$ es un **tema**, una lista de términos con pesos. Cada fila de $W$ dice cuánto de cada tema tiene una reseña. El método se llama factorización no negativa de matrices (NMF). El número de temas, $k=8$, lo fijo yo.
'''),
("code", r'''
from sklearn.decomposition import NMF

VACIAS = ("de la el en y a los las que un una por con para del al lo se es su no muy me nos mi le te ya o pero mas como "
          "fue era son ha han hay este esta estos estas eso esto ese esa si sin sobre todo toda todos todas tambien porque "
          "cuando donde nos les fueron ser estar esta estaba tiene tienen tenia e ni solo hasta desde entre asi yo tu el").split()

quejas_E = E[y[E] == 1]
tfidf_temas = TfidfVectorizer(min_df=5, max_df=0.5, strip_accents="unicode", stop_words=VACIAS).fit(datos["texto"].iloc[quejas_E])
A = tfidf_temas.transform(datos["texto"].iloc[quejas_E])
inicio = time.time()
nmf = NMF(n_components=8, init="nndsvd", random_state=SEMILLA, max_iter=400).fit(A)
W = nmf.transform(A)
terminos_temas = np.array(tfidf_temas.get_feature_names_out())
print(f"Tiempo: {time.time() - inicio:.0f} s | {A.shape[0]} quejas de entrenamiento")

tema_de = W.argmax(axis=1)
resumen_temas = []
for k in range(8):
    del_tema = datos.iloc[quejas_E[tema_de == k]]
    resumen_temas.append({"tema": k, "términos de mayor peso": ", ".join(terminos_temas[np.argsort(-nmf.components_[k])[:9]]),
                          "% de las quejas": 100 * (tema_de == k).mean(), "servicio más común": del_tema["tipo"].mode().iloc[0]})
resumen_temas = pd.DataFrame(resumen_temas).set_index("tema")
pd.set_option("display.max_colwidth", 120)
display(resumen_temas.round(1))
'''),
("md", r'''
Lean los términos de cada tema e intenten ponerle nombre. Nadie le dijo al método qué es un hotel ni qué es un restaurante, y aun así los temas se ordenan por servicio y, dentro de cada servicio, por motivo de queja. Los nombres los ponemos nosotros y son interpretación. Ojo con los temas que traen el nombre de una ciudad entre sus términos: esos separan un lugar más que un motivo. Estos temas vuelven en la sección 4, cuando haya que decir **sobre qué queja** intervenir.
'''),
("md", r'''
## 3. Modelado

### 3.1 Dos líneas base, y la primera trampa

La primera línea base responde siempre «no es queja». La segunda es la regla que escribiría cualquiera sin saber estadística: es queja si la reseña contiene alguna palabra de una lista corta de palabras negativas.
'''),
("code", r'''
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

def cifras(real, marcada, puntaje=None):
    fila = {"exactitud": float(np.mean(real == marcada)), "precisión": precision_score(real, marcada, zero_division=0),
            "sensibilidad": recall_score(real, marcada), "F1": f1_score(real, marcada)}
    if puntaje is not None:
        fila["AP"] = average_precision_score(real, puntaje)
    return fila

NEGATIVAS = ["pesimo", "pesima", "malo", "mala", "horrible", "terrible", "sucio", "sucia", "decepcion", "nunca mas", "no recomiendo", "no volveria"]
sin_tildes = datos["texto"].str.lower().str.normalize("NFD").str.replace(r"[̀-ͯ]", "", regex=True)
cuenta_negativas = sum(sin_tildes.str.count(palabra) for palabra in NEGATIVAS).to_numpy()

resultados = {
    "siempre «no es queja»": cifras(y[V], np.zeros(len(V), int), np.zeros(len(V))),
    "regla de palabras clave": cifras(y[V], (cuenta_negativas[V] > 0).astype(int), cuenta_negativas[V]),
}
display(pd.DataFrame(resultados).T.round(3))
'''),
("md", r'''
Comparen la primera fila con su segunda apuesta. El modelo que no hace nada acierta casi siempre, porque casi nada es queja, y no encuentra una sola. Esa es la **primera trampa** de hoy: con clases desbalanceadas, la exactitud premia la pereza. Cualquier modelo que reportemos con exactitud va a parecer excelente. De aquí en adelante la exactitud queda en la tabla solo para recordarlo.

La regla de palabras clave ya es un rival serio: encuentra una parte de las quejas con una lista de doce palabras.

### 3.2 Cinco modelos sobre la misma matriz

| Modelo | Idea en una línea |
|---|---|
| Bayes ingenuo | Supone que las palabras aparecen de forma independiente dada la clase: $P(y\mid d)\propto P(y)\prod_t P(t\mid y)^{w_{t,d}}$ |
| Regresión logística | $P(y=1\mid d)=\sigma(\beta_0+\sum_t\beta_t\,w_{t,d})$: un coeficiente por término |
| Máquina de soporte vectorial lineal | Busca el plano que separa las clases con el mayor margen |
| Regresión *ridge* | Mínimos cuadrados sobre la etiqueta 0 o 1, con castigo $\alpha\lVert\beta\rVert_2^2$ |
| LSA + *boosting* | Reduce la matriz a 100 dimensiones con una descomposición en valores singulares y entrena árboles encima |

Los cuatro primeros son lineales y trabajan sobre la matriz dispersa completa. El quinto es la alternativa no lineal. Como las quejas son pocas, a los modelos que lo permiten les pido que pesen las dos clases por igual.
'''),
("code", r'''
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

def cinco_modelos():
    return {
        "Bayes ingenuo": ComplementNB(alpha=0.3),
        "logística": LogisticRegression(C=5, max_iter=2000, class_weight="balanced"),
        "SVM lineal": LinearSVC(C=0.3, class_weight="balanced"),
        "ridge": RidgeClassifier(alpha=3.0, class_weight="balanced"),
        "LSA + boosting": make_pipeline(TruncatedSVD(n_components=100, random_state=SEMILLA),
                                        HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=SEMILLA)),
    }

def puntaje(modelo, matriz):
    """Puntaje continuo: mientras más alto, más parece una queja."""
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(matriz)
    return modelo.predict_proba(matriz)[:, 1]

entrenados, curvas = {}, {}
for nombre, modelo in cinco_modelos().items():
    inicio = time.time()
    entrenados[nombre] = modelo.fit(X_E, y[E])
    resultados[nombre] = cifras(y[V], modelo.predict(X_V), puntaje(modelo, X_V))
    resultados[nombre]["segundos"] = time.time() - inicio
    curvas[nombre] = precision_recall_curve(y[V], puntaje(modelo, X_V))
comparacion = pd.DataFrame(resultados).T

fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 4.5))
comparacion["AP"].sort_values().plot.barh(ax=izq, color="#16302B"); izq.axvline(y[V].mean(), color="gray", ls=":")
izq.set(title="Precisión promedio en validación (punteada: ordenar al azar)", xlabel="AP")
for nombre, (precision, sensibilidad, _) in curvas.items():
    der.plot(sensibilidad, precision, label=f"{nombre} ({comparacion.loc[nombre, 'AP']:.3f})")
regla = resultados["regla de palabras clave"]
der.scatter(regla["sensibilidad"], regla["precisión"], color="black", zorder=5, label="regla de palabras clave")
der.set(title="Precisión contra sensibilidad, validación", xlabel="sensibilidad", ylabel="precisión"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
display(comparacion.sort_values("AP", ascending=False).round(3))
'''),
("md", r'''
Miren primero la columna de exactitud: el modelo que no hace nada queda a un punto de los mejores y por encima de alguno que sí encuentra quejas. La exactitud no los ordena bien. La precisión promedio sí. Miren después dónde queda el punto negro de la regla de palabras clave respecto de las curvas: a la misma sensibilidad, los modelos tienen mucha más precisión, porque aprendieron miles de términos con su peso y no doce.

Los modelos lineales quedan juntos arriba, y la alternativa no lineal, con más cómputo, no les gana. Con texto y TF-IDF eso es lo habitual.

### 3.3 La segunda trampa, medida: las copias

En la limpieza quité las reseñas repetidas. ¿Cuánto habría cambiado el resultado si las dejo? Repito la partición y el entrenamiento con la tabla que conserva las copias y comparo con la tabla limpia. Son cuatro sorteos, y uso la máquina de soporte vectorial lineal porque es la más rápida.
'''),
("code", r'''
def examen_al_azar(tabla, semilla):
    """Parte al azar 80/20, vectoriza con el entrenamiento y devuelve la AP y la fracción de la prueba que ya estaba en el entrenamiento."""
    textos = (tabla["Title"].fillna("") + ". " + tabla["Review"].fillna("")).to_numpy()
    etiquetas = (tabla["estrellas"] <= UMBRAL_QUEJA).astype(int).to_numpy()
    entren, prueba = train_test_split(np.arange(len(tabla)), test_size=0.20, random_state=semilla, stratify=etiquetas)
    vec = vectorizador().fit(textos[entren])
    modelo = LinearSVC(C=0.3, class_weight="balanced").fit(vec.transform(textos[entren]), etiquetas[entren])
    ap = average_precision_score(etiquetas[prueba], modelo.decision_function(vec.transform(textos[prueba])))
    return ap, np.isin(textos[prueba], textos[entren]).mean()

inicio = time.time()
filas = []
for semilla in range(4):
    ap_limpia, vistas_limpia = examen_al_azar(datos, semilla)
    ap_copias, vistas_copias = examen_al_azar(con_copias, semilla)
    filas.append({"AP, tabla limpia": ap_limpia, "AP, con copias": ap_copias,
                  "% de la prueba ya visto, limpia": 100 * vistas_limpia, "% de la prueba ya visto, con copias": 100 * vistas_copias})
copias = pd.DataFrame(filas)
print(f"Tiempo: {time.time() - inicio:.0f} s")
display(copias.agg(["mean", "std"]).T.round(4))
'''),
("md", r'''
Lean la tabla con honestidad. Con copias, una parte pequeña de la prueba ya estaba en el entrenamiento. Comparen la diferencia entre las dos AP con la desviación entre sorteos: si la diferencia es del tamaño del ruido, en **este** conjunto la fuga por copias es pequeña, porque las copias son pocas. La limpieza sigue siendo correcta, y el tamaño del daño depende del dato. En el gemelo van a medir lo mismo en un texto donde una de cada cuatro filas repite a otra, y el número va a ser otro.

### 3.4 El veredicto
'''),
("code", r'''
mejor = comparacion.loc[list(cinco_modelos()), "AP"].idxmax()
ganador = entrenados[mejor]
en_prueba = cifras(y[P], ganador.predict(X_P), puntaje(ganador, X_P))
print(f"Ganador en validación: {mejor}")
print(f"En la prueba: AP {en_prueba['AP']:.3f} | precisión {en_prueba['precisión']:.3f} | sensibilidad {en_prueba['sensibilidad']:.3f} | F1 {en_prueba['F1']:.3f}")
print(f"Regla de palabras clave en la prueba: precisión {precision_score(y[P], cuenta_negativas[P] > 0):.3f} | sensibilidad {recall_score(y[P], cuenta_negativas[P] > 0):.3f}")
'''),
("md", r'''
### 3.5 Explicar: los coeficientes se leen

Aquí el texto tiene una ventaja sobre las imágenes de la semana pasada: cada columna es una palabra. En la regresión logística, el coeficiente $\beta_t$ dice cuánto sube el logaritmo de la razón de probabilidades cuando el peso del término $t$ aumenta en una unidad,

$$\ln\frac{P(y=1\mid d)}{1-P(y=1\mid d)}=\beta_0+\sum_t\beta_t\,w_{t,d}.$$

Un coeficiente grande y positivo es un término que empuja hacia queja.
'''),
("code", r'''
logistica = entrenados["logística"]
coeficientes = pd.Series(logistica.coef_[0], index=terminos)
orden = coeficientes.sort_values()

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 5.5))
orden.tail(20).plot.barh(ax=izq, color="#A8432C"); izq.set(title="Empujan hacia «queja»", xlabel="coeficiente")
orden.head(20).iloc[::-1].plot.barh(ax=der, color="#16302B"); der.set(title="Empujan hacia «no es queja»", xlabel="coeficiente")
plt.tight_layout(); plt.show()

def explicar(i, cuantos=6):
    """Los términos que más aportaron al puntaje de la reseña i."""
    fila = tfidf.transform([datos["texto"].iloc[i]])
    aportes = pd.Series(fila.data * logistica.coef_[0][fila.indices], index=terminos[fila.indices]).sort_values()
    return aportes.tail(cuantos).round(2).to_dict(), aportes.head(3).round(2).to_dict()

puntajes_V = logistica.decision_function(X_V)
falsa_alarma = V[np.argmax(np.where(y[V] == 0, puntajes_V, -np.inf))]
queja_perdida = V[np.argmin(np.where(y[V] == 1, puntajes_V, np.inf))]
for titulo, i in [("La falsa alarma más segura", falsa_alarma), ("La queja que el modelo menos vio", queja_perdida)]:
    hacia_queja, en_contra = explicar(i)
    print(f"\n{titulo} ({datos['estrellas'].iloc[i]} estrellas, {datos['tipo'].iloc[i]}, {datos['pais'].iloc[i]}):")
    print("  ", datos["texto"].iloc[i][:330].replace("\n", " "), "…")
    print("   empujan hacia queja:", hacia_queja); print("   empujan en contra:", en_contra)
'''),
("md", r'''
Comparen la gráfica con su tercera apuesta. Las palabras de la izquierda son las que uno esperaría, y aparecen pares como «no volveria» o «no recomiendo», que una bolsa de palabras sueltas no habría visto.

Lean después los dos errores. Son más instructivos que los aciertos. Una reseña de tres estrellas que cuenta un mal rato suena a queja aunque su autor no la haya calificado así. Y lean con cuidado la «queja» que el modelo no vio: si el texto es un elogio y la calificación es de una estrella, lo más probable es que quien escribió haya marcado mal las estrellas. Las etiquetas también tienen errores, y un modelo que «falla» en esos casos está leyendo mejor que la etiqueta. El modelo, de todos modos, no entiende la reseña. Suma pesos de palabras, y la ironía se le escapa.

Tampoco es una explicación causal. Que «sucio» empuje hacia queja no dice que limpiar suba las estrellas. Dice de qué hablan quienes califican mal.
'''),
("md", r'''
## 4. Comunicación

### 4.1 La tercera trampa: contar con el clasificador

La mesa de turismo quiere una proporción: qué fracción de los comentarios de cada país y servicio son quejas. La tentación es pasar todos los comentarios por el clasificador y contar cuántos marcó. Eso se llama **clasificar y contar**, y está sesgado. Si la proporción verdadera es $\pi$, el clasificador marca en promedio

$$\hat\pi_{\text{CC}}=\pi\cdot\text{sens}+(1-\pi)\cdot(1-\text{espec}),$$

que solo coincide con $\pi$ por casualidad. Un clasificador puede ser bueno reseña por reseña y malo para contar.

Hay un arreglo que sale de la misma fórmula. Si la sensibilidad y la especificidad se estiman en validación, se despeja $\pi$:

$$\hat\pi_{\text{aj}}=\frac{\hat\pi_{\text{CC}}-(1-\widehat{\text{espec}})}{\widehat{\text{sens}}-(1-\widehat{\text{espec}})}.$$

En la prueba conocemos las estrellas, así que puedo comparar las tres cifras con la verdad en cada combinación de país y servicio.
'''),
("code", r'''
marcada_V, marcada_P = ganador.predict(X_V), ganador.predict(X_P)
sens_V = recall_score(y[V], marcada_V)
espec_V = float(np.mean(marcada_V[y[V] == 0] == 0))
print(f"Estimadas en validación: sensibilidad {sens_V:.3f} | especificidad {espec_V:.3f}")

prueba = datos.iloc[P].assign(marcada=marcada_P)
grupos = prueba.groupby(["pais", "tipo"])
conteo = pd.DataFrame({"reseñas": grupos.size(), "verdad (%)": grupos["queja"].mean() * 100, "clasificar y contar (%)": grupos["marcada"].mean() * 100})
conteo["ajustado (%)"] = ((conteo["clasificar y contar (%)"] / 100 - (1 - espec_V)) / (sens_V - (1 - espec_V))).clip(0, 1) * 100

rng = np.random.default_rng(SEMILLA)
bajos, altos = [], []
for (pais, tipo), grupo in grupos:
    replicas = [grupo["queja"].to_numpy()[rng.integers(0, len(grupo), len(grupo))].mean() * 100 for _ in range(1000)]
    bajos.append(np.quantile(replicas, 0.025)); altos.append(np.quantile(replicas, 0.975))
conteo["verdad, IC 95 % inferior"], conteo["verdad, IC 95 % superior"] = bajos, altos
conteo = conteo.sort_values("verdad (%)", ascending=False)
display(conteo.round(1))

ax = conteo[["verdad (%)", "clasificar y contar (%)", "ajustado (%)"]].plot.bar(color=["black", "#A8432C", "#E0A526"], figsize=(11, 4), rot=25)
ax.set(title="Proporción de quejas en la prueba: la verdad y dos maneras de estimarla con el modelo", ylabel="%", xlabel=""); plt.show()
print(f"Error absoluto medio, clasificar y contar: {(conteo['clasificar y contar (%)'] - conteo['verdad (%)']).abs().mean():.2f} puntos")
print(f"Error absoluto medio, ajustado:            {(conteo['ajustado (%)'] - conteo['verdad (%)']).abs().mean():.2f} puntos")
print("¿El primer lugar coincide con la verdad?", "clasificar y contar:", conteo["clasificar y contar (%)"].idxmax() == conteo.index[0],
      "| ajustado:", conteo["ajustado (%)"].idxmax() == conteo.index[0])
print("¿El orden completo coincide con la verdad?",
      "clasificar y contar:", list(conteo["clasificar y contar (%)"].rank(ascending=False)) == list(conteo["verdad (%)"].rank(ascending=False)),
      "| ajustado:", list(conteo["ajustado (%)"].rank(ascending=False)) == list(conteo["verdad (%)"].rank(ascending=False)))
'''),
("md", r'''
Comparen las barras rojas con las negras. Clasificar y contar se pasa en casi todas las combinaciones, y en algunas por varios puntos. Tiene explicación: le pedimos al modelo que pesara las dos clases por igual, y un modelo así marca de más.

Miren ahora las barras amarillas y los dos errores medios. El ajuste de libro sale de suponer que la sensibilidad y la especificidad medidas en validación valen igual en cada grupo. No valen: el modelo da más falsas alarmas en unos servicios que en otros. Por eso el ajuste puede quedar peor que la cuenta cruda y mandar a cero las combinaciones con pocas quejas. Es un resultado incómodo y lo dejo como salió.

La conclusión práctica es que donde hay estrellas se cuentan las estrellas. Y donde solo hay texto, antes de creerle a la cuenta del modelo hay que etiquetar a mano una muestra de cada grupo y medir el sesgo ahí.

Para la decisión importa sobre todo quién va primero. Miren si las estimaciones conservan el primer lugar, y miren el intervalo de la verdad: dos combinaciones cuyos intervalos se cruzan no se pueden ordenar con estos datos.

### 4.2 Decidir: dónde y sobre qué

La primera mitad de la respuesta es la primera fila de la tabla. La segunda mitad, sobre qué queja, sale de los temas de la sección 2.5. Tomo las quejas de la prueba en la combinación de país y servicio con mayor proporción, y miro cómo se reparten entre los ocho temas, frente a las quejas del mismo servicio en los otros dos países.
'''),
("code", r'''
peor_pais, peor_tipo = conteo.index[0]
def reparto(tabla):
    quejas = tabla[tabla["queja"] == 1]
    temas = nmf.transform(tfidf_temas.transform(quejas["texto"])).argmax(axis=1)
    return pd.Series(temas).value_counts(normalize=True).reindex(range(8), fill_value=0) * 100, len(quejas)

foco, n_foco = reparto(prueba[(prueba["pais"] == peor_pais) & (prueba["tipo"] == peor_tipo)])
otros, n_otros = reparto(prueba[(prueba["pais"] != peor_pais) & (prueba["tipo"] == peor_tipo)])
reparto_temas = pd.DataFrame({f"{peor_tipo}, {peor_pais} (%)": foco, f"{peor_tipo}, otros países (%)": otros,
                              "términos": resumen_temas["términos de mayor peso"]})
print(f"Foco: {peor_tipo} en {peor_pais}, {n_foco} quejas en la prueba | comparación: {n_otros} quejas de {peor_tipo} en los otros dos países")
display(reparto_temas.sort_values(reparto_temas.columns[0], ascending=False).round(1))

tema_foco = int(foco.idxmax())
ejemplos = prueba[(prueba["pais"] == peor_pais) & (prueba["tipo"] == peor_tipo) & (prueba["queja"] == 1)]
del_tema = ejemplos[nmf.transform(tfidf_temas.transform(ejemplos["texto"])).argmax(axis=1) == tema_foco]
print(f"\nTres quejas del tema {tema_foco}:")
for texto in del_tema["texto"].head(3):
    print("  ·", texto[:260].replace("\n", " "), "…")
'''),
("md", r'''
Lean la tabla por la primera columna y compárenla con la segunda. Un tema que pesa mucho más en el foco que en los otros países es un candidato a intervención. Uno que pesa igual en todas partes es una queja del servicio en general. Lean las tres reseñas antes de ponerle nombre al tema: el nombre es de ustedes y tiene que sobrevivir a la lectura.

### La decisión en tres frases

Para la mesa de turismo, que no sabe qué es TF-IDF.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.3 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Diccionario de `Country`, `Type` y `Polarity` deducido del contenido | Esperar un diccionario oficial | La copia no lo publica; la evidencia de las tres tablas es consistente |
| Muestra de 20.000 reseñas por país | Las 251.291 | Corre en minutos sin GPU y deja a los tres países con el mismo peso |
| Quitar las reseñas `#ERROR!` y las repetidas | Dejarlas | Las primeras no tienen texto; las segundas pueden caer a los dos lados de la partición |
| Queja es dos estrellas o menos | Tres o menos | Tres estrellas mezcla quejas con reseñas tibias; es el parámetro de la variante |
| Título y texto juntos | Solo el texto | El título resume la opinión y también lo escribe el visitante |
| TF-IDF con palabras y pares, sin tildes, mínimo tres reseñas por término | Lematizar; quitar palabras vacías | Los pares guardan la negación; las palabras vacías reciben poco peso por el idf |
| Vocabulario aprendido solo con el entrenamiento | Vectorizar todo y después partir | El examen no debe aportar ni el vocabulario |
| Precisión promedio para escoger | Exactitud | Con pocas quejas, la exactitud premia al modelo que no hace nada |
| Ocho temas con NMF, solo sobre quejas | Otro número de temas | Es legible; el número es arbitrario y cambia los temas |
| Proporciones con las estrellas, y el modelo solo donde no hay estrellas | Clasificar y contar | Contar predicciones está sesgado |

### 4.4 Riesgos y mantenimiento

Las reseñas las escribieron personas. Respondan en dos o tres líneas cada una.

- Las reseñas pueden nombrar a un empleado. ¿Qué harían con esos nombres antes de mostrarle ejemplos a la mesa de turismo?
- Escribe reseñas quien quiere escribirlas. ¿En qué se equivoca la mesa si lee estas proporciones como la satisfacción de todos los visitantes?
- El modelo aprendió con reseñas de una plataforma. ¿Qué le pasa si lo usan con comentarios de redes sociales, que son más cortos y con otro vocabulario? Relaciónenlo con la clase 6.
- El español de México, el de Cuba y el de Colombia no usan las mismas palabras. ¿Cómo comprobarían que el modelo no encuentra peor las quejas de un país?
- El conjunto es de uso académico. ¿Qué pueden y qué no pueden hacer con él?

### 4.5 Cambien una línea

Vuelvan a la celda del comienzo, cambien `UMBRAL_QUEJA = 2` por `UMBRAL_QUEJA = 3` y corran todo de nuevo. Ahora una reseña de tres estrellas también cuenta como queja. Llenen la tabla con sus dos corridas.

| | Queja: 2 estrellas o menos | Queja: 3 estrellas o menos |
|---|---|---|
| Proporción de quejas | | |
| Exactitud de «siempre no es queja» | | |
| AP del ganador en la prueba | | |
| Sensibilidad del ganador en la prueba | | |
| País y servicio con más quejas | | |
| Error de clasificar y contar (puntos) | | |

Con la tabla llena, respondan en tres líneas: ¿cambió la recomendación? ¿Qué definición de queja le propondrían a la mesa y por qué? **Mi respuesta:**

### 4.6 Procedencia
'''),
("code", r'''
procedencia()["restmex"]
'''),
]
