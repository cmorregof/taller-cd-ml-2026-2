"""Celdas del gemelo de la clase 6: Fauna de la Orinoquía (cámaras trampa de LILA).

Cada ("tarea", comentario, solución) sale en el cuaderno del estudiante solo con el
comentario, y en la solución del profesor con el comentario y el código.
"""
from molde import ARRANQUE

FAUNA = [
("md", r'''
# Clase 6 · Fauna de la Orinoquía (gemelo, trabajo autónomo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno en su trabajo autónomo, porque el día de la clase hay parcial. Es el gemelo de *La tierra desde arriba*: la misma pregunta sobre el cambio de dominio, con fotos de cámaras trampa.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Medir por cuenta propia cuánto pierde un clasificador de especies cuando se usa en cámaras que nunca vio, detectar ese cambio de dominio y convertirlo en una regla sobre cuáles fotos etiqueta el modelo y cuáles revisa una persona. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular), 2 (seleccionar y preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | El código lo escribe Claude y ustedes lo auditan. Deben poder explicar sin ayuda por qué se parte por cámara, cuánto cambió el resultado frente a partir al azar, qué significa la cobertura de una regla de confianza y qué promesa le pueden hacer al cliente. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, las respuestas, la auditoría, la decisión en tres frases y la tabla de la variante llenas. |

**Cómo se trabaja.** Igual que en *La roya*: cada celda de código trae solo un comentario que dice qué debe lograr. Péguenle a Claude el bloque «Lo que Claude necesita saber», pídanle el código por secciones, léanlo, córranlo y respondan las preguntas con sus palabras.
'''),
("code", ARRANQUE),
("code", r'''
CONFIANZA = 0.80        # el modelo etiqueta solo las fotos en las que su confianza llega a este valor. La cambiarán al final.
SEMILLA = 0

import time
from taller.clase05_imagenes import imagenes, rasgos_guardados, guardar_rasgos
'''),
("md", r'''
### Lo que Claude necesita saber

> Trabajo en un cuaderno de Colab sin GPU. Ya están importados `numpy as np`, `pandas as pd`, `matplotlib.pyplot as plt` y `time`, y definidas las constantes `CONFIANZA` y `SEMILLA`.
>
> `tabla = descargar("orinoquia")` devuelve un DataFrame con una fila por foto de cámara trampa y estas columnas: `archivo`, `especie` (ocho valores, en inglés, por ejemplo `lowland_tapir`), `camara` (48 cámaras, por ejemplo `N25`), `seq_id` (identificador de la ráfaga; hay una sola foto por ráfaga), `frame_num`, `seq_num_frames`, `fecha` y `ruta_lila`.
>
> `carpeta = imagenes("orinoquia", tabla["archivo"])` devuelve la carpeta (`pathlib.Path`) con las fotos en JPEG, reducidas a 256 píxeles de lado corto. La foto de una fila es `carpeta / fila["archivo"]`.
>
> `rasgos_guardados(nombre, filas)` devuelve una matriz de numpy ya calculada o `None`, y `guardar_rasgos(nombre, matriz)` la guarda.
>
> Quiero usar ResNet18 de torchvision, preentrenada en ImageNet, como extractor fijo de características, y encima modelos de scikit-learn. Todo debe correr en CPU en pocos minutos. Escribe código corto, con comentarios en español, y no uses datos que no sean los de la tabla.
'''),
("md", r'''
## 1. Formulación

Imagínense un grupo de biólogos que estudia la fauna de dos reservas naturales privadas del Meta, El Rey Zamuro y Las Unamas. Pusieron cincuenta cámaras trampa, separadas un kilómetro entre sí, de enero a julio de 2020. Una cámara trampa dispara una ráfaga de fotos cuando algo se mueve. El resultado son más de cien mil fotos que alguien tiene que mirar.

El grupo quiere un modelo que diga qué especie hay en cada foto. Y quiere usarlo la próxima temporada, cuando las cámaras estén en **otros sitios**.

1. **Predecir.** Dada una foto, ¿qué especie aparece?
2. **Explicar.** ¿El modelo reconoce al animal o reconoce el sitio donde está la cámara?
3. **Decidir.** En cámaras nuevas, ¿cuáles fotos etiqueta el modelo solo y cuáles pasan a un biólogo?

**El dato.** *Orinoquía Camera Traps* está publicado en LILA BC (Vélez y colaboradores, *Methods in Ecology and Evolution*, 2023), con la licencia Community Data License Agreement en su variante permisiva. Para el curso preparé un subconjunto: ocho especies, **una sola foto por ráfaga** (la del medio) y hasta 250 ráfagas por especie. Son 1.876 fotos de 48 cámaras. Quienes publicaron el conjunto retiraron las fotos en las que aparecen personas.

**El dominio.** Aquí cada cámara es un dominio: tiene su fondo, su altura, su luz. El cambio de dominio es pasar de las cámaras de entrenamiento a cámaras nuevas, $P_o(x,y)\neq P_d(x,y)$.

**La métrica.** Exactitud y exactitud balanceada sobre las $K=8$ especies,

$$\text{exactitud}=\frac{1}{n}\sum_{i=1}^{n}\mathbb{1}\{\hat y_i=y_i\},\qquad \text{exactitud balanceada}=\frac{1}{K}\sum_{k=1}^{K}\frac{\#\{i:\ \hat y_i=y_i=k\}}{\#\{i:\ y_i=k\}}.$$

**La regla de decisión.** El modelo le da a cada foto una confianza $c(x)=\max_k\hat p_k(x)$. Con un umbral $\tau$ (`CONFIANZA`), el modelo etiqueta solo las fotos con $c(x)\ge\tau$ y manda el resto al biólogo. Dos números describen la regla:

$$\text{cobertura}(\tau)=\frac{1}{n}\sum_{i}\mathbb{1}\{c(x_i)\ge\tau\},\qquad \text{exactitud selectiva}(\tau)=\frac{\sum_i \mathbb{1}\{c(x_i)\ge\tau\}\,\mathbb{1}\{\hat y_i=y_i\}}{\sum_i \mathbb{1}\{c(x_i)\ge\tau\}}.$$

La cobertura es el trabajo que el modelo le quita al biólogo. La exactitud selectiva es la calidad de lo que el modelo etiqueta solo.

### Predicciones escritas

- ¿Qué exactitud tendrá el modelo si se parte al azar por foto? **Mi apuesta:**
- ¿Y en cámaras que nunca vio? **Mi apuesta:**
- Con ocho especies, ¿con qué acierto se podrá adivinar la **cámara** (entre 48) a partir de las características? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

### 2.1 Descargar y contar

La descarga trae 1.876 fotos de Google Cloud y tarda cerca de un minuto.
'''),
("tarea", r'''
# TAREA 1. Descarguen la tabla con descargar("orinoquia") y midan el tiempo. Cuenten fotos por especie,
# número de cámaras y fotos por cámara (mínimo, mediana, máximo). Comprueben que cada seq_id aparece una sola vez.
''', r'''
inicio = time.time()
tabla = descargar("orinoquia")
print(f"Tiempo: {time.time() - inicio:.0f} s")
y = tabla["especie"].to_numpy()
camara = tabla["camara"].to_numpy()
print(len(tabla), "fotos |", tabla["camara"].nunique(), "cámaras |", tabla["especie"].nunique(), "especies")
print(tabla["especie"].value_counts().to_string())
por_camara = tabla.groupby("camara").size()
print(f"\nFotos por cámara: mínimo {por_camara.min()}, mediana {por_camara.median():.0f}, máximo {por_camara.max()}")
print("¿Cada ráfaga aparece una sola vez?", tabla["seq_id"].is_unique)
'''),
("tarea", r'''
# TAREA 2. Tabla cruzada de cámara contra especie, dibujada como mapa de calor (cámaras en filas, ordenadas
# por nombre). Calculen además, para cada cámara, qué fracción de sus fotos es de su especie más frecuente.
''', r'''
cruce = pd.crosstab(tabla["camara"], tabla["especie"])
plt.figure(figsize=(7, 10))
plt.imshow(cruce, aspect="auto", cmap="YlOrBr")
plt.xticks(range(cruce.shape[1]), cruce.columns, rotation=60, ha="right", fontsize=8)
plt.yticks(range(cruce.shape[0]), cruce.index, fontsize=6); plt.colorbar(label="fotos")
plt.title("Fotos por cámara y especie"); plt.tight_layout(); plt.show()
dominante = (cruce.max(axis=1) / cruce.sum(axis=1))
print(f"Fracción de la especie más frecuente en cada cámara: mediana {dominante.median():.2f}, máximo {dominante.max():.2f}")
print(f"Cámaras donde una sola especie pone más de la mitad de las fotos: {(dominante > 0.5).sum()} de {len(dominante)}")
'''),
("md", r'''
**Preguntas.**

- ¿Las especies se reparten parejo entre cámaras? ¿Qué componente de $P(x,y)$ cambia entre una cámara y otra según esta tabla? **Mi respuesta:**
- Los nombres de las cámaras empiezan por `A`, `M` o `N`. La página del conjunto no dice qué significan esas letras. ¿Qué podrían ser, y por qué no conviene suponerlo? **Mi respuesta:**
- ¿Por qué el subconjunto tiene una sola foto por ráfaga? Relaciónenlo con las lesiones de la clase 5. **Mi respuesta:**
'''),
("tarea", r'''
# TAREA 3. Obtengan la carpeta con imagenes("orinoquia", tabla["archivo"]). Muestren una foto de cada especie
# (fila de 8) y, debajo, 8 fotos de la cámara con más fotos, con la especie en el título.
''', r'''
from PIL import Image

carpeta = imagenes("orinoquia", tabla["archivo"])
rutas = [carpeta / a for a in tabla["archivo"]]

def abrir(ruta):
    return Image.open(ruta).convert("RGB")

fig, ejes = plt.subplots(2, 8, figsize=(16, 4.2))
for eje, (especie, grupo) in zip(ejes[0], tabla.groupby("especie")):
    eje.imshow(abrir(carpeta / grupo["archivo"].iloc[len(grupo) // 2])); eje.set_title(especie, fontsize=7); eje.axis("off")
la_mayor = por_camara.idxmax()
for eje, (_, fila) in zip(ejes[1], tabla[tabla["camara"] == la_mayor].iloc[::max(1, por_camara.max() // 8)].iterrows()):
    eje.imshow(abrir(carpeta / fila["archivo"])); eje.set_title(f"{la_mayor}: {fila['especie']}", fontsize=7); eje.axis("off")
plt.suptitle("Arriba, una foto por especie. Abajo, ocho fotos de una misma cámara"); plt.tight_layout(); plt.show()
'''),
("tarea", r'''
# TAREA 4. Una foto nocturna de cámara trampa es infrarroja y sale en grises. Calculen para cada foto si es
# gris (los tres canales casi iguales) y reporten la fracción de fotos grises por especie.
''', r'''
def es_gris(ruta):
    pixeles = np.asarray(abrir(ruta).resize((64, 48)), dtype=float)
    return float(np.abs(pixeles - pixeles.mean(axis=2, keepdims=True)).mean()) < 2.0

tabla["gris"] = [es_gris(r) for r in rutas]
print(f"Fotos en grises: {tabla['gris'].mean() * 100:.0f} %")
print((tabla.groupby("especie")["gris"].mean() * 100).round(0).sort_values().to_string())
'''),
("md", r'''
**Pregunta.** ¿Qué especies salen casi siempre de noche y cuáles de día? Si el modelo aprendiera solo «gris o color», ¿cuánto adelantaría? **Mi respuesta:**

### 2.2 Características y exploración
'''),
("tarea", r'''
# TAREA 5. Extraigan las 512 características de cada foto con ResNet18 preentrenada (sin la última capa, en
# modo de evaluación, con la transformación que acompaña a los pesos), en el orden de la tabla. Usen
# rasgos_guardados("orinoquia_resnet18", len(rutas)) y guardar_rasgos. Midan el tiempo.
''', r'''
import torch, torchvision

pesos = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
red = torchvision.models.resnet18(weights=pesos, progress=False)
red.fc = torch.nn.Identity(); red.eval()
preparar = pesos.transforms()

def extraer(rutas, lote=64):
    partes = []
    with torch.no_grad():
        for i in range(0, len(rutas), lote):
            partes.append(red(torch.stack([preparar(abrir(r)) for r in rutas[i:i + lote]])).numpy())
    return np.concatenate(partes)

inicio = time.time()
X = rasgos_guardados("orinoquia_resnet18", len(rutas))
if X is None:
    X = extraer(rutas)
    guardar_rasgos("orinoquia_resnet18", X)
print(f"Matriz de características: {X.shape} | Tiempo: {time.time() - inicio:.0f} s")
'''),
("tarea", r'''
# TAREA 6. PCA de las características estandarizadas: dibujen los dos primeros componentes coloreados por
# especie y, al lado, coloreados por cámara (solo las 6 cámaras con más fotos). Después, para cada foto,
# busquen su vecina más cercana y reporten en qué porcentaje es de la misma cámara y de la misma especie,
# junto con lo que saldría al azar.
''', r'''
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(X)
plano = PCA(n_components=2, random_state=SEMILLA).fit_transform(Z)
fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 4.8))
for especie, color in zip(sorted(tabla["especie"].unique()), plt.cm.tab10.colors):
    izq.scatter(*plano[y == especie].T, s=5, alpha=0.5, color=color, label=especie)
izq.set(title="Coloreado por especie", xlabel="CP1", ylabel="CP2"); izq.legend(markerscale=3, fontsize=7)
der.scatter(*plano.T, s=3, alpha=0.15, color="lightgray")
for nombre, color in zip(por_camara.sort_values(ascending=False).index[:6], plt.cm.tab10.colors):
    der.scatter(*plano[camara == nombre].T, s=6, alpha=0.7, color=color, label=nombre)
der.set(title="Las seis cámaras con más fotos", xlabel="CP1", ylabel="CP2"); der.legend(markerscale=3, fontsize=7)
plt.tight_layout(); plt.show()

vecina = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[1][:, 1]
azar_camara = np.mean([((camara == v).sum() - 1) / (len(camara) - 1) for v in camara]) * 100
azar_especie = np.mean([((y == v).sum() - 1) / (len(y) - 1) for v in y]) * 100
print(f"Vecina de la misma cámara:  {(camara[vecina] == camara).mean() * 100:.0f} %  (al azar: {azar_camara:.0f} %)")
print(f"Vecina de la misma especie: {(y[vecina] == y).mean() * 100:.0f} %  (al azar: {azar_especie:.0f} %)")
'''),
("md", r'''
**Pregunta.** ¿Qué reconoce mejor la red, la cámara o la especie? ¿Qué anticipa eso para un modelo que se examine en cámaras nuevas? **Mi respuesta:**
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición por cámara

Tres tramos sorteando **cámaras**: 60 % para entrenar, 20 % para validar y 20 % para la prueba. Así la validación y la prueba son siempre cámaras nuevas para el modelo, que es la situación del cliente.
'''),
("tarea", r'''
# TAREA 7. Partan los índices en E, V y P sorteando cámaras (60/20/20) con SEMILLA. Impriman fotos y cámaras
# por tramo y comprueben con un assert que ninguna cámara aparece en dos tramos.
''', r'''
from sklearn.model_selection import GroupShuffleSplit, train_test_split

def partir_por_grupo(indices, grupos, fraccion, semilla):
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=fraccion, random_state=semilla).split(indices, groups=grupos[indices]))
    return indices[a], indices[b]

todas = np.arange(len(tabla))
resto, P = partir_por_grupo(todas, camara, 0.20, SEMILLA)
E, V = partir_por_grupo(resto, camara, 0.25, SEMILLA)
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {len(tramo):5d} fotos | {len(set(camara[tramo])):3d} cámaras")
assert not set(camara[E]) & set(camara[V]) and not set(camara[E]) & set(camara[P]) and not set(camara[V]) & set(camara[P])
print("Ninguna cámara aparece en dos tramos.")
'''),
("md", r'''
### 3.2 Línea base y cinco modelos

La línea base tiene dos versiones: responder siempre la especie más frecuente del entrenamiento, y una regresión logística sobre seis números de color por foto (promedio y desviación de cada canal). Encima de las características compiten los cinco clasificadores de la demostración.
'''),
("tarea", r'''
# TAREA 8. Calculen las dos líneas base y entrenen los cinco modelos (logística, SVM, k vecinos, bosque,
# boosting sobre componentes principales) en E. Reporten exactitud y exactitud balanceada en V y en P en
# una tabla ordenada por la exactitud balanceada de validación, con una gráfica de barras.
''', r'''
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

def exactitud(real, predicho):
    return float(np.mean(np.asarray(real) == np.asarray(predicho)))

def balanceada(real, predicho):
    real, predicho = np.asarray(real), np.asarray(predicho)
    return float(np.mean([np.mean(predicho[real == k] == k) for k in np.unique(real)]))

def cinco_modelos():
    return {
        "logística": make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
        "bosque": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEMILLA),
        "boosting": make_pipeline(StandardScaler(), PCA(n_components=30, random_state=SEMILLA),
                                  HistGradientBoostingClassifier(max_iter=60, random_state=SEMILLA)),
    }

def medir(predecir):
    return {"exactitud (val)": exactitud(y[V], predecir(V)), "balanceada (val)": balanceada(y[V], predecir(V)),
            "exactitud (prueba)": exactitud(y[P], predecir(P)), "balanceada (prueba)": balanceada(y[P], predecir(P))}

def seis_numeros(ruta):
    pixeles = np.asarray(abrir(ruta).resize((64, 48)), dtype=float).reshape(-1, 3) / 255
    return np.concatenate([pixeles.mean(axis=0), pixeles.std(axis=0)])

colores = np.array([seis_numeros(r) for r in rutas])
mas_frecuente = pd.Series(y[E]).value_counts().index[0]
base_color = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(colores[E], y[E])
resultados = {"siempre la más frecuente": medir(lambda tramo: np.full(len(tramo), mas_frecuente)),
              "color (6 números)": medir(lambda tramo: base_color.predict(colores[tramo]))}
inicio = time.time()
entrenados = {}
for nombre, modelo in cinco_modelos().items():
    entrenados[nombre] = modelo.fit(X[E], y[E])
    resultados[nombre] = medir(lambda tramo, m=modelo: m.predict(X[tramo]))
comparacion = pd.DataFrame(resultados).T.sort_values("balanceada (val)", ascending=False)
print(f"Tiempo: {time.time() - inicio:.0f} s")
comparacion[["balanceada (val)", "balanceada (prueba)"]].iloc[::-1].plot.barh(color=["#16302B", "#E0A526"], figsize=(8, 3.8))
plt.xlabel("exactitud balanceada en cámaras nuevas"); plt.show()
display(comparacion.round(3))
mejor = comparacion.loc[list(cinco_modelos()), "balanceada (val)"].idxmax()
print(f"Ganador en validación: {mejor}")
'''),
("md", r'''
### 3.3 La trampa: el examen en casa

Ahora midan lo que habría salido con una partición al azar por foto, que mezcla las cámaras. Ese es el examen en casa: el modelo ya conoce el fondo de cada cámara.
'''),
("tarea", r'''
# TAREA 9. Repitan 30 veces (semillas 0 a 29), para la regresión logística y para un vecino (k = 1): partir por
# cámara y partir al azar por foto, con 20 % en prueba. Guarden la exactitud en cada caso, muestren la media y
# la desviación estándar y un diagrama de caja.
''', r'''
inicio = time.time()
repeticiones = []
for semilla in range(30):
    entren_c, prueba_c = partir_por_grupo(todas, camara, 0.20, semilla)
    entren_a, prueba_a = train_test_split(todas, test_size=len(prueba_c), random_state=semilla, stratify=y)
    fila = {}
    for nombre, hacer in [("logística", lambda: cinco_modelos()["logística"]),
                          ("1 vecino", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)))]:
        fila[f"{nombre}, cámaras nuevas"] = exactitud(y[prueba_c], hacer().fit(X[entren_c], y[entren_c]).predict(X[prueba_c]))
        fila[f"{nombre}, al azar"] = exactitud(y[prueba_a], hacer().fit(X[entren_a], y[entren_a]).predict(X[prueba_a]))
    repeticiones.append(fila)
repeticiones = pd.DataFrame(repeticiones)
print(f"Tiempo: {time.time() - inicio:.0f} s")
repeticiones.plot.box(figsize=(10, 4), color="#16302B", rot=10); plt.ylabel("exactitud"); plt.title("30 sorteos"); plt.show()
display(repeticiones.agg(["mean", "std"]).T.round(3))
'''),
("md", r'''
**Preguntas.** Con los números de su tabla.

- ¿Cuántos puntos de exactitud separan el examen en casa del examen en cámaras nuevas? ¿La diferencia es mayor que el ruido entre sorteos? **Mi respuesta:**
- ¿Cuál de los dos números le pueden prometer al grupo de biólogos para la próxima temporada? **Mi respuesta:**
- La desviación estándar es mayor al partir por cámara. ¿Por qué? **Mi respuesta:**

### 3.4 Detectar y explicar

Como en la demostración, hay señales que no necesitan etiquetas de especie. Aquí el «clasificador de dominio» adivina la cámara.
'''),
("tarea", r'''
# TAREA 10. Entrenen una regresión logística que adivine la CÁMARA a partir de las características, con una
# partición al azar por foto (75/25). Reporten su exactitud y compárenla con responder siempre la cámara
# más frecuente.
''', r'''
entren_a, prueba_a = train_test_split(todas, test_size=0.25, random_state=SEMILLA)
detector = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000)).fit(X[entren_a], camara[entren_a])
print(f"Adivinar la cámara entre {tabla['camara'].nunique()}: {exactitud(camara[prueba_a], detector.predict(X[prueba_a])) * 100:.0f} % de acierto")
print(f"Responder siempre la cámara más frecuente: {pd.Series(camara[entren_a]).value_counts(normalize=True).iloc[0] * 100:.0f} %")
'''),
("tarea", r'''
# TAREA 11. Con el modelo ganador, en validación (cámaras nuevas): acierto por especie y matriz de confusión
# dibujada como mapa de calor, normalizada por filas.
''', r'''
ganador = entrenados[mejor]
pred_V = ganador.predict(X[V])
especies = sorted(tabla["especie"].unique())
print(pd.Series(pred_V == y[V]).groupby(y[V]).agg(["mean", "size"]).rename(columns={"mean": "acierto", "size": "fotos"}).round(2).sort_values("acierto").to_string())
confusion = pd.crosstab(pd.Series(y[V], name="real"), pd.Series(pred_V, name="modelo"), normalize="index").reindex(index=especies, columns=especies, fill_value=0)
plt.figure(figsize=(6.5, 5.5)); plt.imshow(confusion, cmap="YlOrBr", vmin=0, vmax=1)
plt.xticks(range(8), especies, rotation=60, ha="right", fontsize=8); plt.yticks(range(8), especies, fontsize=8)
for i in range(8):
    for j in range(8):
        plt.text(j, i, f"{confusion.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.title("Confusión en cámaras nuevas (validación)"); plt.colorbar(); plt.tight_layout(); plt.show()
'''),
("tarea", r'''
# TAREA 12. El remedio: ¿ayuda tener más cámaras en el entrenamiento? Con la regresión logística, entrenen
# con 4, 8, 16 y todas las cámaras de E (sorteadas, 10 repeticiones cada una) y midan la exactitud en V.
# Dibujen la exactitud contra el número de cámaras y anoten cuántas fotos de entrenamiento hubo en promedio.
''', r'''
camaras_E = pd.unique(camara[E])
curva = []
for cuantas in [4, 8, 16, len(camaras_E)]:
    for semilla in range(10 if cuantas < len(camaras_E) else 1):
        elegidas = np.random.default_rng(semilla).choice(camaras_E, cuantas, replace=False)
        sub = E[np.isin(camara[E], elegidas)]
        if len(np.unique(y[sub])) < 2:
            continue
        modelo = cinco_modelos()["logística"].fit(X[sub], y[sub])
        curva.append({"cámaras": cuantas, "fotos": len(sub), "exactitud": exactitud(y[V], modelo.predict(X[V]))})
curva = pd.DataFrame(curva).groupby("cámaras").agg(fotos=("fotos", "mean"), exactitud=("exactitud", "mean"), desviacion=("exactitud", "std"))
plt.figure(figsize=(7, 3.8)); plt.errorbar(curva.index, curva["exactitud"], yerr=curva["desviacion"].fillna(0), fmt="o-", color="#A8432C", capsize=3)
plt.xlabel("cámaras en el entrenamiento"); plt.ylabel("exactitud en cámaras nuevas (validación)"); plt.title("Más sitios distintos"); plt.show()
display(curva.round(3))
'''),
("md", r'''
**Preguntas.**

- ¿La red sabe en qué cámara está cada foto? ¿Qué significa eso para la pregunta de explicar? **Mi respuesta:**
- ¿Qué especies se confunden entre sí en cámaras nuevas? Miren fotos de esas especies antes de responder. **Mi respuesta:**
- Según la curva, ¿qué le recomendarían al grupo: más fotos por cámara o más cámaras? ¿La curva ya se aplanó? **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: qué fotos etiqueta el modelo solo

Apliquen la regla de confianza con $\tau=$ `CONFIANZA`. Midan cobertura y exactitud selectiva en tres exámenes: en casa (una partición al azar), en validación y en prueba (cámaras nuevas). La incertidumbre de la prueba se mide con *bootstrap* **por cámara**, $\text{IC}_{95\%}=[\hat\theta^*_{(0{,}025)},\hat\theta^*_{(0{,}975)}]$, porque las fotos de una cámara no son independientes.
'''),
("tarea", r'''
# TAREA 13. Con un modelo que entregue probabilidades (el ganador si tiene predict_proba; si no, la logística):
# cobertura y exactitud selectiva con el umbral CONFIANZA en (a) una partición al azar 80/20, (b) validación
# y (c) prueba. Para la prueba, intervalos de 95 % por bootstrap de cámaras (1.000 réplicas). Dibujen además
# la curva de exactitud selectiva contra cobertura para todos los umbrales, en casa y en cámaras nuevas.
''', r'''
decisor_nombre = mejor if hasattr(ganador, "predict_proba") else "logística"
decisor = entrenados[decisor_nombre]

def regla(modelo, tramo, umbral=None):
    umbral = CONFIANZA if umbral is None else umbral
    probabilidades = modelo.predict_proba(X[tramo])
    confianza, predicho = probabilidades.max(axis=1), modelo.classes_[probabilidades.argmax(axis=1)]
    acepta = confianza >= umbral
    return {"cobertura": acepta.mean(), "exactitud selectiva": exactitud(y[tramo][acepta], predicho[acepta]) if acepta.any() else np.nan,
            "exactitud sin regla": exactitud(y[tramo], predicho)}, confianza, predicho

entren_a, prueba_a = train_test_split(todas, test_size=0.20, random_state=SEMILLA, stratify=y)
en_casa = cinco_modelos()[decisor_nombre].fit(X[entren_a], y[entren_a])
informe = {"en casa (al azar)": regla(en_casa, prueba_a)[0], "validación (cámaras nuevas)": regla(decisor, V)[0]}
informe["prueba (cámaras nuevas)"], confianza_P, predicho_P = regla(decisor, P)

rng = np.random.default_rng(SEMILLA)
camaras_P = pd.unique(camara[P])
filas_de = {g: np.flatnonzero(camara[P] == g) for g in camaras_P}
replicas = []
for _ in range(1000):
    filas = np.concatenate([filas_de[g] for g in rng.choice(camaras_P, len(camaras_P))])
    acepta = confianza_P[filas] >= CONFIANZA
    replicas.append({"cobertura": acepta.mean(), "exactitud selectiva": exactitud(y[P][filas][acepta], predicho_P[filas][acepta]) if acepta.any() else np.nan,
                     "exactitud sin regla": exactitud(y[P][filas], predicho_P[filas])})
replicas = pd.DataFrame(replicas)
informe = pd.DataFrame(informe)
informe["prueba, IC 95 % inferior"], informe["prueba, IC 95 % superior"] = replicas.quantile(0.025), replicas.quantile(0.975)
print(f"Modelo: {decisor_nombre} | CONFIANZA = {CONFIANZA}")
display(informe.round(3))

plt.figure(figsize=(7, 4))
for nombre, modelo, tramo, color in [("en casa", en_casa, prueba_a, "#5B7DB1"), ("cámaras nuevas (prueba)", decisor, P, "#A8432C")]:
    puntos = [regla(modelo, tramo, u)[0] for u in np.linspace(0.2, 0.99, 40)]
    plt.plot([q["cobertura"] for q in puntos], [q["exactitud selectiva"] for q in puntos], color=color, label=nombre)
plt.xlabel("cobertura: fracción de fotos que etiqueta el modelo"); plt.ylabel("exactitud selectiva"); plt.legend(); plt.title("Lo que se promete y lo que se cumple"); plt.show()
'''),
("md", r'''
**Preguntas.**

- Con el mismo umbral, ¿qué cobertura y qué exactitud selectiva prometía el examen en casa y cuáles se cumplieron en cámaras nuevas? **Mi respuesta:**
- De cada 1.000 fotos de una cámara nueva, ¿cuántas etiqueta el modelo solo y cuántas de esas quedan mal etiquetadas? **Mi respuesta:**
- ¿El intervalo es ancho o angosto? ¿Cuántas cámaras tiene la prueba? **Mi respuesta:**

### La decisión en tres frases

Para quien dirige el grupo de biólogos.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Auditoría del código que les dio la IA

Comprueben estas tres cosas y escriban qué encontraron y en qué celda.

1. **Las cámaras no se mezclan.** Ninguna cámara aparece en dos tramos y hay un `assert` que lo comprueba. Revisen también que la partición por grupos no use `stratify`, que con grupos no hace lo que uno cree. **Lo que encontré:**
2. **El número que se promete es el de cámaras nuevas.** Busquen en cada tabla y en cada gráfica de la sección 4 con qué partición se calculó. El modelo y el umbral no se escogieron mirando la prueba. **Lo que encontré:**
3. **La confianza es lo que dice ser.** Comprueben en validación que, entre las fotos con confianza de 0,8 o más, la fracción de aciertos se parece a la confianza promedio. Si el modelo es demasiado seguro, la regla promete más de lo que cumple. **Lo que encontré:**
'''),
("tarea", r'''
# TAREA 14. Auditoría 3: en validación, partan las fotos en 5 grupos según la confianza del modelo y comparen
# la confianza promedio de cada grupo con su fracción de aciertos.
''', r'''
_, confianza_V, predicho_V = regla(decisor, V)
grupos = pd.qcut(confianza_V, 5, duplicates="drop")
print(pd.DataFrame({"confianza": confianza_V, "acierto": predicho_V == y[V]}).groupby(grupos, observed=True).agg(
    confianza_promedio=("confianza", "mean"), aciertos=("acierto", "mean"), fotos=("acierto", "size")).round(2).to_string())
'''),
("md", r'''
### 4.3 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Una foto por ráfaga | Todas las fotos de cada ráfaga | Las fotos de una ráfaga son casi iguales; con todas, el resultado se infla y el cálculo se multiplica |
| Ocho especies, sin fotos vacías | | |
| Hasta 250 ráfagas por especie | | |
| Partición por cámara | | |
| Exactitud balanceada para escoger el modelo | | |
| Regla de confianza con umbral fijo | | |
| *Bootstrap* por cámara | | |

### 4.4 Riesgos y mantenimiento

- El subconjunto no tiene fotos vacías ni especies raras, y en el conjunto completo una de cada cinco fotos está vacía. ¿Qué hará el modelo con una foto vacía o con un puma? ¿Qué hará la regla de confianza?
- Las reservas son privadas y las cámaras también fotografían personas. Quienes publicaron el dato retiraron esas fotos. ¿Qué deberían hacer ustedes con las fotos de personas si montan este sistema para la próxima temporada?
- La ubicación de especies amenazadas es información sensible. ¿Qué parte de una tabla como esta no publicarían?
- Las cámaras nuevas de la prueba están en las mismas dos reservas y en los mismos meses. ¿Qué cambio de dominio queda sin medir?

### 4.5 Cambien una línea

Vuelvan a la celda del comienzo, cambien `CONFIANZA = 0.80` por `CONFIANZA = 0.95` y corran todo de nuevo. Llenen la tabla.

| En la prueba (cámaras nuevas) | $\tau=0{,}80$ | $\tau=0{,}95$ |
|---|---|---|
| Cobertura | | |
| Exactitud selectiva | | |
| Intervalo de la exactitud selectiva | | |
| Fotos que revisa el biólogo, de cada 1.000 | | |
| Fotos mal etiquetadas por el modelo, de cada 1.000 | | |

Con la tabla llena, respondan en tres líneas: ¿qué umbral le recomiendan al grupo y qué gana y qué pierde con él? **Mi respuesta:**

Dejen el cuaderno entregado con `CONFIANZA = 0.95` corrido y la tabla completa.

### 4.6 Procedencia y entrega
'''),
("tarea", r'''
# TAREA 15. Muestren la procedencia del dato con procedencia()["orinoquia"].
''', r'''
procedencia()["orinoquia"]
'''),
("md", r'''
**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase06/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]
