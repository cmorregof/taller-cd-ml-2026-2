"""Celdas de la demostración de la clase 6: La tierra desde arriba (EuroSAT y Caldas)."""
from molde import ARRANQUE

TIERRA = [
("md", r'''
# Clase 6 · La tierra desde arriba

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Hoy hay parcial, así que esta demostración dura hora y media. La resuelvo yo en vivo y ustedes la siguen. El gemelo, *Fauna de la Orinoquía*, lo resuelve cada uno en su trabajo autónomo.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Medir qué le pasa a un modelo cuando se usa en un lugar distinto de aquel donde se entrenó: entrenar un clasificador de coberturas con parches de satélite de Europa, examinarlo en Europa y en Caldas, detectar el cambio de dominio con y sin etiquetas, y decidir cuántas etiquetas locales hacen falta. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular el problema y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar) y 6 (reflexión técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda o que escriba una gráfica. Deben poder explicar sin ayuda qué es un cambio de dominio, por qué la exactitud medida en Europa no vale en Caldas, cómo se detecta el cambio sin etiquetas y de dónde sale la referencia con la que se juzga al modelo en Caldas. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con las predicciones escritas, la decisión en tres frases y la tabla de la variante llenas. |
'''),
("code", ARRANQUE),
("md", r'''
La celda siguiente tiene el único parámetro del cuaderno. Al final lo cambian y comparan.
'''),
("code", r'''
ZONA = "cafetera"       # zona de Caldas donde se examina el modelo: "cafetera" o "nevados"
SEMILLA = 0

import time
from taller.clase05_imagenes import imagenes, rasgos_guardados, guardar_rasgos
'''),
("md", r'''
## 1. Formulación

Imagínense una oficina de planeación que quiere saber cada año qué cubre el suelo de Caldas: cuánto bosque, cuánto pasto, cuánta ciudad. Las imágenes del satélite Sentinel-2 son gratis y pasan cada cinco días. Lo que la oficina no tiene es parches de Caldas etiquetados a mano para entrenar un modelo. Lo que sí existe es EuroSAT, un conjunto público de 27.000 parches de Sentinel-2 ya etiquetados, todos de Europa.

La tentación es obvia: entrenar con Europa y usar en Caldas. Hoy medimos cuánto cuesta.

1. **Predecir.** Dado un parche de 640 metros de lado, ¿qué cobertura tiene?
2. **Explicar.** Si el modelo europeo falla en Caldas, ¿qué cambió entre un lugar y otro?
3. **Decidir.** ¿Se usa el modelo europeo tal cual, o hay que etiquetar parches de Caldas? ¿Cuántos?

**Los datos.** EuroSAT (Helber, Bischke, Dengel y Borth, 2019) trae parches de $64\times64$ píxeles, a 10 metros por píxel, en diez clases. Para Caldas preparé parches del mismo tamaño, cortados de una imagen real de Sentinel-2 del 10 de agosto de 2026. Caldas no tiene etiquetas hechas a mano, así que uso como **referencia** el mapa ESA WorldCover 2021, que asigna una cobertura a cada píxel de 10 metros del planeta. Esa referencia es a su vez el resultado de un modelo y tiene cinco años más que la imagen. Lo vamos a tener presente todo el tiempo.

**Las palabras.** El lugar donde se entrena se llama **dominio de origen** y el lugar donde se usa, **dominio de destino**. Si $P_o(x,y)$ es la distribución de parches y etiquetas en el origen y $P_d(x,y)$ en el destino, hay **cambio de dominio** cuando

$$P_o(x,y)\neq P_d(x,y).$$

Puede cambiar el aspecto de las imágenes, $P(x)$, puede cambiar la mezcla de clases, $P(y)$, y puede cambiar lo que una etiqueta significa, $P(y\mid x)$. En Caldas vamos a encontrar las tres.

**La métrica.** La exactitud es la fracción de parches bien clasificados,

$$\text{exactitud}=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{\hat y_i=y_i\}.$$

Cuando una clase domina, la exactitud premia al modelo que siempre responde esa clase. Por eso reporto también la exactitud balanceada, que promedia el acierto de cada una de las $K$ clases,

$$\text{exactitud balanceada}=\frac{1}{K}\sum_{k=1}^{K}\frac{\#\{i:\ \hat y_i=y_i=k\}}{\#\{i:\ y_i=k\}}.$$

### Predicción escrita

- ¿Qué exactitud tendrá en Europa un modelo entrenado en Europa? **Mi apuesta:**
- ¿Qué fracción de los parches de Caldas clasificará de acuerdo con la referencia? **Mi apuesta:**
- ¿Cuántos parches de Caldas etiquetados harán falta para superar al modelo entrenado con 2.400 parches europeos? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

Dos descargas. De EuroSAT se baja el archivo completo (94 MB) y se guardan 400 parches por clase, los numerados de 1 a 400. De Caldas se bajan desde Planetary Computer tres capas de una zona de 25,6 km de lado: el color verdadero de la imagen, la máscara de nubes de la propia escena y el mapa WorldCover. Entre las dos tardan menos de un minuto.
'''),
("code", r'''
inicio = time.time()
europa = descargar("eurosat")
caldas = descargar("caldas", zona=ZONA)
print(f"Tiempo: {time.time() - inicio:.0f} s")
print("\nEuroSAT, parches por clase:"); print(europa["clase"].value_counts().to_string())
caldas.head(3)
'''),
("md", r'''
Cada fila de `caldas` es un parche. Las columnas `frac_…` dicen qué fracción de sus 4.096 píxeles tiene cada cobertura según WorldCover. `referencia` es la cobertura más frecuente del parche, `pureza` es la fracción que ocupa y `nubes` es la fracción del parche que la escena marcó como nube o sombra de nube.

Los nombres de las coberturas son mi traducción de la leyenda que WorldCover publica en sus metadatos: *tree cover* es arbolado, *grassland* es pastizal, *built-up* es construido, *cropland* es cultivo, *bare / sparse vegetation* es suelo desnudo, *snow and ice* es nieve y hielo.
'''),
("code", r'''
from PIL import Image

carpeta_e = imagenes("eurosat", europa["archivo"])
carpeta_c = imagenes(f"caldas_{ZONA}", caldas["archivo"])

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 6.5))
izq.imshow(Image.open(carpeta_c / "mosaico.jpg")); izq.set_title(f"Sentinel-2, 10 de agosto de 2026, zona {ZONA}"); izq.axis("off")
der.imshow(Image.open(carpeta_c / "referencia.png")); der.set_title("ESA WorldCover 2021: verde oscuro arbolado, amarillo pastizal, rojo construido"); der.axis("off")
plt.tight_layout(); plt.show()

print("Parches según su cobertura de referencia:")
print(caldas["referencia"].value_counts().to_string())
print("\nFracción de todos los píxeles de la zona, por cobertura (%):")
print((caldas.filter(like="frac_").mean() * 100).round(2).to_string())
'''),
("md", r'''
Miren la última tabla y busquen el renglón `frac_cultivo`. Si la zona es la cafetera, esa cifra es casi cero: según WorldCover, alrededor de Manizales y Chinchiná prácticamente no hay cultivos. El café está ahí, y la referencia lo reparte entre arbolado y pastizal. Un cafetal es un arbusto leñoso bajo sombra, y desde arriba se parece más a un bosque ralo que a un trigal europeo.

Esto es un cambio en $P(y\mid x)$: la misma etiqueta nombra cosas distintas en lugares distintos. Va a la bitácora como límite de la referencia.

**Decisiones de limpieza.** Descarto los parches con más de 5 % de nube. Para juzgar al modelo uso solo los parches **puros**, donde una sola cobertura ocupa al menos 70 % del parche, porque en un parche mitad bosque y mitad potrero no hay una respuesta correcta.
'''),
("code", r'''
despejado = caldas["nubes"] <= 0.05
puro = despejado & (caldas["pureza"] >= 0.70)
print(f"Parches: {len(caldas)} | con nube: {(~despejado).sum()} | despejados y puros: {puro.sum()}")
print("\nParches puros por cobertura de referencia:")
print(caldas.loc[puro, "referencia"].value_counts().to_string())

fig, ejes = plt.subplots(2, 10, figsize=(15, 4.3))
for eje, clase in zip(ejes[0], sorted(europa["clase"].unique())):
    eje.imshow(Image.open(carpeta_e / f"{clase}_1.jpg")); eje.set_title(clase, fontsize=7); eje.axis("off")
ejemplos = caldas[puro].groupby("referencia").head(4).sort_values("referencia").head(10)
for eje, (_, fila) in zip(ejes[1], ejemplos.iterrows()):
    eje.imshow(Image.open(carpeta_c / fila["archivo"])); eje.set_title(fila["referencia"], fontsize=7); eje.axis("off")
for eje in ejes[1][len(ejemplos):]:
    eje.axis("off")
plt.suptitle("Arriba, un parche de cada clase de EuroSAT. Abajo, parches puros de Caldas con su referencia"); plt.tight_layout(); plt.show()
'''),
("md", r'''
Comparen el bosque de arriba con el arbolado de abajo. Antes de cualquier modelo ya se ve que los colores no son los mismos. Lo mido con el promedio de cada canal.
'''),
("code", r'''
def pixeles(carpeta, archivos):
    return np.concatenate([np.asarray(Image.open(carpeta / a).convert("RGB")).reshape(-1, 3) for a in archivos])

comparacion_color = pd.DataFrame({
    "Europa, todas las clases": pixeles(carpeta_e, europa["archivo"][::10]).mean(axis=0),
    "Europa, Forest": pixeles(carpeta_e, europa.loc[europa["clase"] == "Forest", "archivo"][:100]).mean(axis=0),
    f"Caldas ({ZONA}), todos": pixeles(carpeta_c, caldas.loc[despejado, "archivo"][::4]).mean(axis=0),
    f"Caldas ({ZONA}), arbolado puro": pixeles(carpeta_c, caldas.loc[puro & (caldas["referencia"] == "arbolado"), "archivo"][:100]).mean(axis=0),
}, index=["rojo", "verde", "azul"]).T
comparacion_color.plot.bar(color=["#D7263D", "#3E7D4E", "#5B7DB1"], rot=10, figsize=(10, 3.6))
plt.ylabel("valor promedio (0 a 255)"); plt.title("Color promedio por canal"); plt.show()
display(comparacion_color.round(0))
'''),
("md", r'''
Fíjense en el canal azul. En los parches europeos es el más alto de los tres, incluso en el bosque. En Caldas es el más bajo. Un bosque europeo de EuroSAT es azulado y un bosque de Caldas es verde.

Mi hipótesis es que la diferencia viene sobre todo del procesamiento. La imagen de Caldas es un producto de nivel 2A, al que ya se le corrigió el efecto de la atmósfera, y la bruma atmosférica azula las imágenes que no tienen esa corrección. No puedo comprobarlo con estos datos, así que queda como hipótesis. Lo que sí puedo medir es si igualar los colores arregla el problema, y lo hago en la sección 3.

### 2.1 Características

Es el paso de la clase 5: ResNet18 preentrenada como extractor fijo, $z=\phi(x)\in\mathbb{R}^{512}$. Hay un cambio. Los parches son de 64 píxeles y la red espera fotos más grandes, así que los amplío a 128 píxeles. A 224 la exactitud sube un punto y el cálculo tarda más del doble.
'''),
("code", r'''
import torch, torchvision

pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
red = torchvision.models.resnet18(weights=pesos, progress=False)
red.fc = torch.nn.Identity(); red.eval()
T = torchvision.transforms
preparar = T.Compose([T.Resize(128, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(),
                      T.Normalize(pesos.transforms().mean, pesos.transforms().std)])

def extraer(fotos, lote=100):
    """`fotos` es una lista de imágenes PIL. Devuelve una fila de 512 números por foto."""
    partes = []
    with torch.no_grad():
        for i in range(0, len(fotos), lote):
            partes.append(red(torch.stack([preparar(f.convert("RGB")) for f in fotos[i:i + lote]])).numpy())
    return np.concatenate(partes)

def rasgos(nombre, carpeta, archivos):
    X = rasgos_guardados(nombre, len(archivos))
    if X is None:
        X = extraer([Image.open(carpeta / a) for a in archivos])
        guardar_rasgos(nombre, X)
    return X

inicio = time.time()
Xe = rasgos("eurosat_resnet18", carpeta_e, europa["archivo"])
Xc = rasgos(f"caldas_{ZONA}_resnet18", carpeta_c, caldas["archivo"])
print(f"Europa: {Xe.shape} | Caldas: {Xc.shape} | Tiempo: {time.time() - inicio:.0f} s")
'''),
("md", r'''
### 2.2 PCA: los dos dominios en el mismo plano

Junto las características de los dos lugares, las estandarizo y las proyecto con componentes principales. Si los dos dominios fueran intercambiables, las dos nubes de puntos se mezclarían.

**Predicción escrita:** ¿se mezclarán? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

juntas = np.vstack([Xe, Xc[despejado]])
plano = PCA(n_components=2, random_state=SEMILLA).fit_transform(StandardScaler().fit_transform(juntas))
plano_e, plano_c = plano[:len(Xe)], plano[len(Xe):]

fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 4.8))
izq.scatter(plano_e[:, 0], plano_e[:, 1], s=4, alpha=0.3, color="#5B7DB1", label="Europa (EuroSAT)")
izq.scatter(plano_c[:, 0], plano_c[:, 1], s=4, alpha=0.3, color="#D7263D", label=f"Caldas ({ZONA})")
izq.set(title="Los dos dominios", xlabel="CP1", ylabel="CP2"); izq.legend(markerscale=4)
for clase, color in zip(sorted(europa["clase"].unique()), plt.cm.tab10.colors):
    cuales = (europa["clase"] == clase).to_numpy()
    der.scatter(plano_e[cuales, 0], plano_e[cuales, 1], s=4, alpha=0.4, color=color, label=clase)
der.scatter(plano_c[:, 0], plano_c[:, 1], s=4, alpha=0.25, color="black", label="Caldas")
der.set(title="Las diez clases europeas, y Caldas en negro", xlabel="CP1", ylabel="CP2"); der.legend(markerscale=4, fontsize=7, ncol=2)
plt.tight_layout(); plt.show()
'''),
("md", r'''
Miren dónde cae Caldas respecto de las clases europeas. Todos los parches de Caldas, sean bosque, potrero o ciudad, quedan apretados en una región pequeña del plano. Para el modelo europeo, Caldas entero se parece a una o dos de sus clases.
'''),
("md", r'''
## 3. Modelado

### 3.1 El examen en casa

Parto EuroSAT al azar en 60 % para entrenar, 20 % para validar y 20 % para la prueba, con las clases en la misma proporción. Hay una reserva. La versión RGB de EuroSAT no dice de qué imagen ni de qué lugar salió cada parche, así que no puedo partir por lugar como hicimos la clase pasada con las lesiones. Si dos parches vecinos cayeron a lados distintos, la exactitud europea está algo inflada. No lo puedo medir con este archivo y lo dejo anotado.

La línea base son seis números por parche, el promedio y la desviación de cada canal de color. Encima de las características compiten los mismos cinco clasificadores de la clase 5.
'''),
("code", r'''
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

ye = europa["clase"].to_numpy()
indices = np.arange(len(europa))
resto, P = train_test_split(indices, test_size=0.20, random_state=SEMILLA, stratify=ye)
E, V = train_test_split(resto, test_size=0.25, random_state=SEMILLA, stratify=ye[resto])
print(f"Entrenamiento: {len(E)} | Validación: {len(V)} | Prueba: {len(P)}")

def seis_numeros(carpeta, archivos):
    filas = []
    for a in archivos:
        p = np.asarray(Image.open(carpeta / a).convert("RGB"), dtype=float).reshape(-1, 3) / 255
        filas.append(np.concatenate([p.mean(axis=0), p.std(axis=0)]))
    return np.array(filas)

color_e = seis_numeros(carpeta_e, europa["archivo"])
base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(color_e[E], ye[E])

def cinco_modelos():
    return {
        "logística": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "bosque": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEMILLA),
        "boosting": make_pipeline(StandardScaler(), PCA(n_components=30, random_state=SEMILLA),
                                  HistGradientBoostingClassifier(max_iter=60, random_state=SEMILLA)),
    }

def exactitud(real, predicho):
    return float(np.mean(np.asarray(real) == np.asarray(predicho)))

inicio = time.time()
resultados = {"color (6 números)": {"validación": exactitud(ye[V], base.predict(color_e[V])), "prueba": exactitud(ye[P], base.predict(color_e[P]))}}
entrenados = {}
for nombre, modelo in cinco_modelos().items():
    entrenados[nombre] = modelo.fit(Xe[E], ye[E])
    resultados[nombre] = {"validación": exactitud(ye[V], modelo.predict(Xe[V])), "prueba": exactitud(ye[P], modelo.predict(Xe[P]))}
comparacion = pd.DataFrame(resultados).T
print(f"Tiempo: {time.time() - inicio:.0f} s")

comparacion.sort_values("validación").plot.barh(color=["#1D4E89", "#1B998B"], figsize=(8, 3.6))
plt.xlabel("exactitud en diez clases"); plt.title("Europa: validación para escoger, prueba para reportar"); plt.show()
mejor = comparacion.loc[list(cinco_modelos()), "validación"].idxmax()
ganador = entrenados[mejor]
display(comparacion.sort_values("validación", ascending=False).round(3))
print(f"Ganador en validación: {mejor} | exactitud en la prueba europea: {comparacion.loc[mejor, 'prueba']:.3f}")
'''),
("md", r'''
En casa el modelo funciona, y le saca mucha ventaja a la línea base de color. Con este número en la mano, cualquiera firmaría el contrato con la oficina de planeación.

### 3.2 El examen afuera

Ahora le paso al ganador los parches de Caldas. Para comparar necesito un idioma común entre las diez clases de EuroSAT y las coberturas de WorldCover. La tabla de equivalencias es una **decisión mía** y va a la bitácora.

| Clases de EuroSAT | Cobertura de referencia |
|---|---|
| Forest | arbolado |
| HerbaceousVegetation, Pasture | pastizal |
| AnnualCrop, PermanentCrop | cultivo |
| Residential, Industrial, Highway | construido |
| River, SeaLake | agua |

WorldCover tiene coberturas sin equivalente en EuroSAT, como suelo desnudo o nieve y hielo. El modelo europeo no puede responderlas, así que en esos parches falla por construcción.

Para que la comparación sea justa, también traduzco a este idioma la prueba europea.
'''),
("code", r'''
A_REFERENCIA = {"Forest": "arbolado", "HerbaceousVegetation": "pastizal", "Pasture": "pastizal",
                "AnnualCrop": "cultivo", "PermanentCrop": "cultivo",
                "Residential": "construido", "Industrial": "construido", "Highway": "construido",
                "River": "agua", "SeaLake": "agua"}
traducir = np.vectorize(A_REFERENCIA.get)

def balanceada(real, predicho):
    real, predicho = np.asarray(real), np.asarray(predicho)
    return float(np.mean([np.mean(predicho[real == k] == k) for k in np.unique(real)]))

ref = caldas["referencia"].to_numpy()
pred_caldas = ganador.predict(Xc)                       # una de las diez clases de EuroSAT
pred_ref = traducir(pred_caldas)                         # traducida al idioma de la referencia
p = puro.to_numpy()

examenes = pd.DataFrame({
    "Europa (prueba)": {"parches": len(P), "exactitud": exactitud(traducir(ye[P]), traducir(ganador.predict(Xe[P]))),
                        "exactitud balanceada": balanceada(traducir(ye[P]), traducir(ganador.predict(Xe[P])))},
    f"Caldas ({ZONA}), parches puros": {"parches": int(p.sum()), "exactitud": exactitud(ref[p], pred_ref[p]),
                                        "exactitud balanceada": balanceada(ref[p], pred_ref[p])},
}).T
display(examenes.round(3))

print("Qué responde el modelo europeo en los parches puros de Caldas (filas: referencia; columnas: respuesta):")
display(pd.crosstab(pd.Series(ref[p], name="referencia"), pd.Series(pred_caldas[p], name="modelo")))
print("Acierto por cobertura de referencia:")
print(pd.Series(pred_ref[p] == ref[p]).groupby(ref[p]).agg(["mean", "size"]).rename(columns={"mean": "acierto", "size": "parches"}).round(3).to_string())
'''),
("md", r'''
Comparen con su segunda apuesta. El mismo modelo, con los mismos pesos, pasa de acertar nueve de cada diez parches a acertar cerca de uno de cada siete. Miren la tabla cruzada de la zona cafetera para ver cómo falla: a la mayor parte de lo que la referencia llama arbolado, el modelo le responde `HerbaceousVegetation` o `Residential`. Acierta en lo construido, que se parece en todas partes, y acierta en buena parte del pastizal porque vegetación baja es lo que responde para casi todo.

Fíjense también en la diferencia entre las dos métricas. La exactitud balanceada de Caldas sale mucho mejor que la exactitud simple, porque dos de las tres coberturas salen bien y pesan lo mismo que la tercera, que es casi toda la zona. Ninguna de las dos métricas miente. Responden preguntas distintas, y a la oficina le importa el mapa completo.

La trampa de hoy tiene esta forma: la partición de la sección 3.1 estaba bien hecha y el número era honesto, pero respondía otra pregunta. Decía qué tan bien clasifica el modelo parches **como los de EuroSAT**. El cliente necesita saber qué tan bien clasifica parches **de Caldas**. Un conjunto de prueba solo certifica el desempeño sobre datos que vengan de su misma distribución.

Antes de seguir, una cautela sobre la referencia. Parte del desacuerdo puede ser culpa de WorldCover y no del modelo: ya vimos que llama arbolado al cafetal. Pero un modelo que no reconoce como arbolado casi ningún parche de una zona montañosa y boscosa no está en una discusión de matices.

### 3.3 Detectar el cambio sin etiquetas

En la vida real uno no tiene la referencia. ¿Se podía saber que algo andaba mal sin ella? Tres señales, ninguna usa etiquetas de Caldas.

**Un clasificador de dominio.** Entreno un modelo para distinguir si un parche es de Europa o de Caldas. Si los dominios fueran iguales, no podría, y su AUC sería 0,5. Lo mido con validación cruzada.

**La mezcla de respuestas.** Comparo qué fracción de parches manda el modelo a cada clase en Europa y en Caldas.

**La confianza.** Comparo la probabilidad que el modelo le da a su propia respuesta, $\max_k \hat p_k(x)$, en un lugar y en el otro.
'''),
("code", r'''
from sklearn.model_selection import cross_val_score

dominio = np.r_[np.zeros(len(Xe)), np.ones(int(despejado.sum()))]
auc_dominio = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
                              juntas, dominio, cv=5, scoring="roc_auc").mean()
print(f"AUC del clasificador de dominio (Europa contra Caldas), validación cruzada de 5 pliegues: {auc_dominio:.4f}")

logistica = entrenados["logística"]                      # este modelo entrega probabilidades
confianza_e = logistica.predict_proba(Xe[P]).max(axis=1)
confianza_c = logistica.predict_proba(Xc[despejado]).max(axis=1)
mezcla = pd.DataFrame({"Europa (prueba)": pd.Series(logistica.predict(Xe[P])).value_counts(normalize=True),
                       f"Caldas ({ZONA})": pd.Series(logistica.predict(Xc[despejado])).value_counts(normalize=True)}).fillna(0) * 100

fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 4))
mezcla.plot.bar(ax=izq, color=["#5B7DB1", "#D7263D"], rot=40); izq.set(title="Qué responde el modelo en cada lugar", ylabel="% de los parches")
der.hist(confianza_e, bins=25, alpha=0.6, color="#5B7DB1", density=True, label="Europa (prueba)")
der.hist(confianza_c, bins=25, alpha=0.6, color="#D7263D", density=True, label=f"Caldas ({ZONA})")
der.set(title="Confianza del modelo en su respuesta", xlabel="probabilidad de la clase elegida"); der.legend()
plt.tight_layout(); plt.show()
print(f"Confianza mediana: Europa {np.median(confianza_e):.2f} | Caldas {np.median(confianza_c):.2f}")
print(f"Parches de Caldas con confianza mayor de 0,90: {(confianza_c > 0.9).mean() * 100:.0f} %")
'''),
("md", r'''
Las tres señales avisan. El clasificador de dominio separa los dos lugares casi sin error, de modo que para la red un parche de Caldas no se confunde con uno europeo. La mezcla de respuestas es inverosímil para quien conozca la región. La confianza cae: en Europa el modelo suele estar seguro de su respuesta y en Caldas duda. Aquí la confianza sirvió de alarma. No conviene que sea la única, porque un modelo también puede equivocarse con la confianza alta, y los parches de la cola derecha del histograma de Caldas son de esos.

Ninguna de estas señales dice cuánto se equivoca el modelo. Dicen que el examen europeo dejó de valer, y eso basta para no firmar el contrato sin medir en el destino.

### 3.4 Explicar: ¿es solo el color?

En la sección 2 vimos que los colores cambian. Si el problema fuera solo ese, bastaría con llevar los colores de Caldas a los de Europa. Lo pruebo con el ajuste más simple: a cada canal de la imagen de Caldas le resto su media, lo divido por su desviación y le pongo la media y la desviación del mismo canal en EuroSAT,

$$x'_c=\frac{x_c-\mu^{d}_c}{\sigma^{d}_c}\,\sigma^{o}_c+\mu^{o}_c,\qquad c\in\{\text{rojo},\text{verde},\text{azul}\}.$$

Después vuelvo a extraer características y a clasificar.
'''),
("code", r'''
inicio = time.time()
px_e = pixeles(carpeta_e, europa["archivo"][::5]).astype(float)
px_c = pixeles(carpeta_c, caldas.loc[despejado, "archivo"]).astype(float)
mu_o, sigma_o, mu_d, sigma_d = px_e.mean(axis=0), px_e.std(axis=0), px_c.mean(axis=0), px_c.std(axis=0)

def igualar(archivo):
    x = np.asarray(Image.open(carpeta_c / archivo).convert("RGB"), dtype=float)
    return Image.fromarray(np.clip((x - mu_d) / sigma_d * sigma_o + mu_o, 0, 255).astype(np.uint8))

archivos_puros = caldas.loc[puro, "archivo"].tolist()
X_igualado = extraer([igualar(a) for a in archivos_puros])
pred_igualado = traducir(ganador.predict(X_igualado))
print(f"Tiempo: {time.time() - inicio:.0f} s")
print(f"Caldas, parches puros, sin tocar el color:  exactitud {exactitud(ref[p], pred_ref[p]):.3f} | balanceada {balanceada(ref[p], pred_ref[p]):.3f}")
print(f"Caldas, parches puros, con el color igualado: exactitud {exactitud(ref[p], pred_igualado):.3f} | balanceada {balanceada(ref[p], pred_igualado):.3f}")

fig, ejes = plt.subplots(1, 4, figsize=(9, 2.6))
for eje, archivo in zip(ejes[:2], archivos_puros[:2]):
    eje.imshow(Image.open(carpeta_c / archivo)); eje.set_title("original", fontsize=8); eje.axis("off")
for eje, archivo in zip(ejes[2:], archivos_puros[:2]):
    eje.imshow(igualar(archivo)); eje.set_title("color igualado", fontsize=8); eje.axis("off")
plt.tight_layout(); plt.show()
'''),
("md", r'''
El ajuste de color no rescata al modelo. La diferencia entre Europa y Caldas no se deja reducir a tres medias y tres desviaciones: cambian la textura del relieve, el tamaño de los lotes, la vegetación misma. Es un resultado negativo y vale tanto como uno positivo, porque descarta el arreglo barato.

### 3.5 El remedio: etiquetas del destino

Queda el arreglo caro, que es conseguir etiquetas de Caldas. La pregunta del cliente es cuántas. Hago el experimento con la referencia de WorldCover en el papel de etiqueta local.

Para no repetir la fuga de la clase pasada, separo por **lugar**: los parches etiquetados salen de la mitad norte de la zona y el examen se hace en la mitad sur. Dos parches vecinos se parecen, y si uno entrena y el otro examina, el resultado se infla.

Comparo dos recetas con $k$ parches locales por cobertura:

- **Europa + locales**: los 2.400 parches europeos de entrenamiento, traducidos al idioma de la referencia, más los $k$ locales.
- **Solo locales**: nada más los $k$ parches de Caldas por cobertura.

Repito cada combinación 20 veces, sorteando cuáles parches se etiquetan.
'''),
("code", r'''
norte = p & (caldas["fila"] < 20).to_numpy()
sur = p & (caldas["fila"] >= 20).to_numpy()
coberturas = [c for c in pd.unique(ref[sur]) if (ref[norte] == c).sum() >= 5 and (ref[sur] == c).sum() >= 5]
examen = np.flatnonzero(sur & np.isin(ref, coberturas))
print("Coberturas con al menos 5 parches puros en cada mitad:", coberturas)
print("Parches disponibles para etiquetar (norte):", {c: int((norte & (ref == c)).sum()) for c in coberturas})
print("Parches de examen (sur):", {c: int((ref[examen] == c).sum()) for c in coberturas})

ye_ref = traducir(ye)
def receta():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))

inicio = time.time()
filas = []
for k in [0, 5, 10, 20, 40]:
    for semilla in range(20):
        rng = np.random.default_rng(semilla)
        locales = np.concatenate([rng.choice(np.flatnonzero(norte & (ref == c)), min(k, int((norte & (ref == c)).sum())), replace=False)
                                  for c in coberturas]).astype(int)
        mezcla_pred = receta().fit(np.vstack([Xe[E], Xc[locales]]), np.r_[ye_ref[E], ref[locales]]).predict(Xc[examen])
        fila = {"k": k, "etiquetas locales": len(locales), "Europa + locales": balanceada(ref[examen], mezcla_pred)}
        if k > 0:
            fila["Solo locales"] = balanceada(ref[examen], receta().fit(Xc[locales], ref[locales]).predict(Xc[examen]))
        filas.append(fila)
        if k == 0:
            break                                        # sin locales no hay sorteo que repetir
curva = pd.DataFrame(filas)
resumen = curva.groupby("k").agg(etiquetas=("etiquetas locales", "mean"), europa_mas_locales=("Europa + locales", "mean"),
                                 solo_locales=("Solo locales", "mean"), desviacion_solo_locales=("Solo locales", "std"))
print(f"Tiempo: {time.time() - inicio:.0f} s")

plt.figure(figsize=(8, 4))
plt.plot(resumen.index, resumen["europa_mas_locales"], "o-", color="#5B7DB1", label="Europa + locales")
plt.errorbar(resumen.index[1:], resumen["solo_locales"][1:], yerr=resumen["desviacion_solo_locales"][1:], fmt="o-", color="#D7263D", capsize=3, label="Solo locales")
plt.axhline(1 / len(coberturas), color="gray", ls=":"); plt.text(0, 1 / len(coberturas) + 0.01, "responder siempre lo mismo", fontsize=8, color="gray")
plt.xlabel("parches locales etiquetados por cobertura (k)"); plt.ylabel("exactitud balanceada en la mitad sur")
plt.title("¿Cuántas etiquetas de Caldas hacen falta?"); plt.ylim(0, 1); plt.legend(); plt.show()
display(resumen.round(3))
'''),
("md", r'''
Comparen con su tercera apuesta. El punto de $k=0$ es el modelo europeo sin ayuda. Con cinco parches por cobertura, que son quince etiquetas, la receta local ya lo supera, y después de veinte por cobertura la curva casi no sube. Miren además si la línea azul queda por encima de la roja en algún punto: los 2.400 parches europeos, sumados a los locales, no mejoran la exactitud balanceada. En la sección 4.1 van a ver que con la exactitud simple incluso estorban, porque son muchos más que los locales y arrastran la frontera hacia un mundo que no es el del examen.

Dos reservas sobre este experimento. Las etiquetas locales son de WorldCover, que es un modelo, así que lo que medimos es qué tan bien se aprende a imitar a WorldCover. Y la mitad sur queda al lado de la mitad norte, de modo que esto todavía es un examen dentro de la misma zona y el mismo día. La variante del final lo pone a prueba en otra zona.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir

Reúno en una tabla las tres opciones que tiene la oficina, medidas en el mismo examen (la mitad sur), con un intervalo de 95 % por *bootstrap* sobre los parches del examen,

$$\text{IC}_{95\%}=\bigl[\hat\theta^{*}_{(0{,}025)},\ \hat\theta^{*}_{(0{,}975)}\bigr].$$

Parches vecinos no son independientes, así que el intervalo verdadero es más ancho que el que sale aquí. Lo digo en los riesgos.
'''),
("code", r'''
K_FINAL = 20
rng = np.random.default_rng(SEMILLA)
locales = np.concatenate([rng.choice(np.flatnonzero(norte & (ref == c)), min(K_FINAL, int((norte & (ref == c)).sum())), replace=False)
                          for c in coberturas]).astype(int)
opciones = {
    "Modelo europeo tal cual": pred_ref[examen],
    f"Europa + {len(locales)} parches locales": receta().fit(np.vstack([Xe[E], Xc[locales]]), np.r_[ye_ref[E], ref[locales]]).predict(Xc[examen]),
    f"Solo {len(locales)} parches locales": receta().fit(Xc[locales], ref[locales]).predict(Xc[examen]),
}
decision = {}
for nombre, predicho in opciones.items():
    replicas = []
    for _ in range(1000):
        b = rng.integers(0, len(examen), len(examen))
        replicas.append(balanceada(ref[examen][b], predicho[b]))
    decision[nombre] = {"exactitud": exactitud(ref[examen], predicho), "exactitud balanceada": balanceada(ref[examen], predicho),
                        "IC 95 % inferior": np.quantile(replicas, 0.025), "IC 95 % superior": np.quantile(replicas, 0.975)}
decision = pd.DataFrame(decision).T
print(f"Zona: {ZONA} | Examen: {len(examen)} parches puros de la mitad sur | Etiquetas locales usadas: {len(locales)}")
display(decision.round(3))
print(f"Para comparar, exactitud balanceada del modelo europeo en la prueba europea: {examenes.iloc[0]['exactitud balanceada']:.3f}")
'''),
("md", r'''
### La decisión en tres frases

Para la persona que dirige la oficina de planeación, que no sabe qué es un dominio.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| 400 parches por clase de EuroSAT (4.000) | Los 27.000 | Alcanza para una exactitud cercana a 0,9 y corre en un minuto sin GPU |
| Parches ampliados a 128 píxeles | 64 (original) o 224 | A 64 la exactitud baja; a 224 sube un punto y tarda más del doble |
| Imagen de Caldas del 10 de agosto de 2026, color verdadero de nivel 2A | Otras fechas; bandas originales | Es una escena casi sin nubes; EuroSAT RGB solo tiene tres canales |
| WorldCover 2021 como referencia | Etiquetar a mano | No hay etiquetas manuales de Caldas; la referencia es un modelo y es de 2021 |
| Solo parches con 70 % o más de una cobertura y menos de 5 % de nube | Todos los parches | En un parche mezclado no hay una respuesta correcta |
| Tabla de equivalencias entre EuroSAT y WorldCover | Otras agrupaciones | Es la más directa; es discutible y cambia el resultado |
| Partición al azar dentro de EuroSAT | Partir por lugar | El archivo RGB no dice de dónde salió cada parche |
| Etiquetas locales del norte, examen en el sur | Sortear parches locales en toda la zona | Parches vecinos se parecen; al sortearlos se infla el resultado |

### 4.3 Riesgos y mantenimiento

- La referencia llama arbolado al cafetal. ¿Qué le pasa a un mapa de cultivos de Caldas hecho con cualquiera de los modelos de hoy?
- El examen local se hizo el mismo día y en la misma zona de donde salieron las etiquetas. ¿Qué cambio de dominio falta por medir si la oficina quiere un mapa cada año?
- La confianza del modelo europeo bajó en Caldas. ¿Bastaría como única alarma en producción? ¿Qué otra pondrían que no dependa de la confianza?
- WorldCover es de 2021 y la imagen de 2026. ¿En qué parches esperarían que la referencia esté vieja?
- El intervalo de la sección 4.1 trata a parches vecinos como independientes. ¿Hacia qué lado se equivoca?

### 4.4 Cambien una línea

Vuelvan a la celda del comienzo donde dice `ZONA = "cafetera"`, pongan `ZONA = "nevados"` y corran todo de nuevo. La zona nueva es el Nevado del Ruiz y el páramo que lo rodea, dentro de la misma imagen de satélite. Tiene nubes, nieve y suelo desnudo, que son coberturas que EuroSAT no conoce. Llenen la tabla con sus dos corridas.

| | cafetera | nevados |
|---|---|---|
| Parches despejados y puros | | |
| Coberturas de referencia presentes | | |
| Exactitud del modelo europeo en parches puros | | |
| AUC del clasificador de dominio | | |
| Exactitud balanceada del modelo europeo en el examen del sur | | |
| Exactitud balanceada con solo parches locales | | |

Con la tabla llena, respondan en tres líneas: ¿la receta que sirvió en la zona cafetera sirve en el páramo? ¿Qué le dice eso a la oficina sobre un solo modelo para todo Caldas? **Mi respuesta:**

### 4.5 Procedencia
'''),
("code", r'''
{nombre: ficha for nombre, ficha in procedencia().items() if nombre in ("eurosat", "caldas")}
'''),
]
