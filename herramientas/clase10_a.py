"""Celdas de la demostración de la clase 10, «La cifra que falta»."""
from molde import ARRANQUE

CIFRA = [
("md", r'''
# Clase 10 · La cifra que falta

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. En la segunda mitad cada uno resuelve el gemelo, *El mapa de vuelos*, con la misma idea en un tema muy distinto.

**Antes de empezar.** El dato de hoy es sobre reclutamiento de niñas, niños y adolescentes en el conflicto armado colombiano. Cada fila es una persona. Vamos a contar con cuidado y a hablar de víctimas, que es lo que son. El tema de la clase es estadístico: qué se hace cuando al dato le faltan campos y cómo se dice la incertidumbre de una cifra. El respeto por el dato consiste hoy en no decir más de lo que el dato sostiene.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Calcular una cifra a partir de datos con campos imputados, combinar réplicas de imputación múltiple con su intervalo y escribir el resultado para quien no es técnico sin esconder la incertidumbre. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (preparar datos), 4 (implementar y evaluar), 5 (comunicar a audiencias técnicas y no técnicas) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una fórmula o que escriba una gráfica. Deben poder explicar sin ayuda la diferencia entre cifra documentada, imputada y estimada, por qué no se cuenta con una sola réplica, cómo se combinan las réplicas y qué supuestos tiene una imputación. |
| **Entregable** | Este cuaderno corrido en su portafolio, con la frase de la sección 4 escrita por ustedes y la tabla de la variante llena. |
'''),
("md", r'''
> **Antes de empezar: conozcan la fuente.** Los datos de hoy los publica el DANE en su catálogo de microdatos, a nombre de la JEP, la Comisión de la Verdad y HRDAG. Abran la [ficha del conjunto](https://microdatos.dane.gov.co/index.php/catalog/795) y explórenla cinco minutos. Miren también su [diccionario de datos](https://microdatos.dane.gov.co/index.php/catalog/795/data-dictionary). Respondan mirando la fuente:
>
> - ¿Quién produjo los datos y quién los publica?
> - ¿Qué es una fila de los archivos de reclutamiento, y qué periodo cubren?
> - ¿Qué licencia tiene el conjunto y a quién hay que citar?
'''),
("code", ARRANQUE),
("code", r'''
# El parámetro del cuaderno. Al final de la clase se cambia esta línea y se corre todo de nuevo.
N_REPLICAS = 10          # cuántas de las 100 réplicas publicadas se usan

import time
'''),
("md", r'''
## 1. Formulación

Imagínense al equipo que prepara un informe público sobre el reclutamiento de menores de edad en el conflicto. En algún párrafo tiene que ir un número. Detrás de ese número hay muchas listas: de entidades del Estado, de organizaciones de víctimas, de organizaciones sociales. Ninguna lista está completa, una misma persona aparece en varias y a muchos registros les faltan campos como la edad, el sexo o el presunto responsable.

La Jurisdicción Especial para la Paz, la Comisión de la Verdad y el Human Rights Data Analysis Group (HRDAG) integraron esas listas en un proyecto conjunto y publicaron el resultado en el DANE. Con ese material hay tres cifras distintas, y confundirlas es el error más común:

| Cifra | Qué cuenta | Qué le falta |
|---|---|---|
| **Documentada** | Las víctimas que aparecen en alguna lista y cuyos campos sí quedaron registrados | Deja por fuera los registros con campos vacíos |
| **Imputada** | Todas las víctimas documentadas, con los campos vacíos completados por un modelo | Sigue contando solo a quienes están en alguna lista |
| **Estimada** | Las documentadas más las que no están en ninguna lista (el subregistro) | Necesita un modelo adicional y tiene un intervalo ancho |

Hoy trabajamos la segunda. La tercera la calculó el proyecto con un método de sistemas múltiples que no cabe en esta clase, y la citaremos de su informe.

Las tres preguntas:

1. **Predecir.** De las víctimas documentadas a las que no les quedó registrada la edad, ¿cuántas eran menores de 18 años?
2. **Explicar.** ¿De qué depende que a un registro le falte la edad, y qué tanto se puede inferir con los demás campos?
3. **Decidir.** ¿Qué cifra se escribe en el informe y cómo se dice su incertidumbre?

**El método.** Cuando falta un campo, un modelo puede proponer un valor. Si se propone uno solo, ese valor inventado queda en la tabla con la misma autoridad que un dato registrado. La **imputación múltiple** propone $m$ versiones completas de la tabla, llamadas réplicas. Cada réplica sortea los valores faltantes de su distribución. Lo que cambia de una réplica a otra es lo que no sabemos.

Sea $\hat\theta_k$ la cifra calculada con la réplica $k$. Las reglas de combinación son

$$\bar\theta=\frac{1}{m}\sum_{k=1}^{m}\hat\theta_k,\qquad B=\frac{1}{m-1}\sum_{k=1}^{m}\bigl(\hat\theta_k-\bar\theta\bigr)^2,\qquad T=\bar W+\Bigl(1+\frac{1}{m}\Bigr)B,$$

donde $B$ es la varianza entre réplicas y $\bar W$ es la varianza de muestreo dentro de cada réplica. Aquí contamos víctimas documentadas, que no son una muestra, así que $\bar W=0$. El intervalo es

$$\bar\theta\ \pm\ 1{,}96\,\sqrt{T}.$$

Es la misma regla que usa el paquete `verdata` de HRDAG para estos datos.

### Predicción escrita

- ¿A qué porcentaje de las víctimas documentadas le falta la edad? **Mi apuesta:**
- Si cuento solo los registros con edad registrada, ¿en cuánto subestimo el número de menores de edad documentados? **Mi apuesta:**
- ¿Qué tan ancho será el intervalo de la cifra imputada, como porcentaje de la cifra? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

El DANE publica el resultado del proyecto en su catálogo de microdatos, número 795, con licencia Creative Commons Atribución 4.0. Para reclutamiento hay 100 réplicas, y cada una trae a todas las víctimas documentadas. Una llamada entrega las primeras `N_REPLICAS`, una debajo de otra. Empezamos con diez de las cien porque el informe del proyecto calcula su cifra con diez, y así podemos comparar. En la sección 4.5 se cambia ese número.

El DANE anonimizó los datos antes de publicarlos: las fuentes no tienen nombre, la edad viene en categorías y algunos municipios y meses se volvieron a imputar para que nadie pueda ser identificado.
'''),
("code", r'''
replicas = descargar("reclutamiento", replicas=N_REPLICAS)
fuentes = [c for c in replicas.columns if c.startswith("in_")]
print("Filas:", len(replicas), "| Réplicas:", replicas["replica"].nunique(), "| Columnas:", replicas.shape[1], f"(de ellas, {len(fuentes)} indicadoras de fuente)")
replicas.drop(columns=fuentes).head(3)
'''),
("md", r'''
### 2.1 Las trampas de este dato

**Primera trampa: las filas no son personas.** El archivo tiene una fila por víctima **y por réplica**. Quien cuente filas multiplica a las víctimas por el número de réplicas.
'''),
("code", r'''
por_replica = replicas.groupby("replica")["match_group_id"].agg(["size", "nunique"])
print("Víctimas por réplica:", por_replica["nunique"].unique(), "| filas en total:", len(replicas))
una, otra = [replicas[replicas["replica"] == r].reset_index(drop=True) for r in ["R1", "R2"]]
print("¿Las dos primeras réplicas traen las mismas víctimas, en el mismo orden?", bool((una["match_group_id"] == otra["match_group_id"]).all()))
print("Años de los hechos:", una["yy_hecho"].min(), "a", una["yy_hecho"].max(), "| departamentos:", una["dept_code_hecho"].nunique())

campos = ["edad_jep", "sexo", "etnia", "p_str", "muni_code_hecho"]
resumen = pd.DataFrame({
    "faltaba (%)": [una[c + "_imputed"].mean() * 100 for c in campos],
    "registrado: ¿igual en R1 y R2? (%)": [(una.loc[~una[c + "_imputed"], c] == otra.loc[~otra[c + "_imputed"], c]).mean() * 100 for c in campos],
    "imputado: ¿igual en R1 y R2? (%)": [(una.loc[una[c + "_imputed"], c] == otra.loc[otra[c + "_imputed"], c]).mean() * 100 for c in campos],
}, index=campos).round(1)
display(resumen)
'''),
("md", r'''
Comparen con su primera apuesta. La tabla muestra cómo está hecha una réplica. Lo que alguna fuente registró es idéntico en todas las réplicas. La única excepción es el municipio, que el DANE volvió a imputar en algunos registros para proteger a las víctimas. Lo que faltaba cambia de una réplica a otra, y cambia más cuando el modelo tenía menos de dónde agarrarse.

**Segunda trampa: la categoría de edad que cruza los 18 años.** La edad viene por quinquenios, y el quinquenio de 15 a 19 mezcla menores y mayores de edad. La columna que sirve es `edad_jep`, que separa infancia, adolescencia y adultez.

**Tercera trampa: los nombres de las categorías.** No encontré publicado un diccionario que diga qué significa cada código de `p_str` (presunto responsable) ni de `etnia`. El informe metodológico del proyecto usa en sus tablas los rótulos FARC-EP, ELN, paramilitares, agentes estatales, múltiple y otros grupos. Yo leo `GUE-FARC`, `GUE-ELN`, `PARA`, `EST`, `multiple` y `GUE-OTRO` u `OTRO` en ese sentido. Es mi lectura, y hoy no vamos a desagregar por responsable.
'''),
("code", r'''
print(pd.crosstab(una["edad_categoria"], una["edad_jep"]).loc[["0-4", "5-9", "10-14", "15-19", "20-24"]])
muy_mayores = una[una["edad_categoria"].isin(["90-94", "95+"])]
print(f"\nRegistros con edad de 90 años o más: {len(muy_mayores)} | de ellos con edad imputada: {int(muy_mayores['edad_categoria_imputed'].sum())}")
print("Categorías de presunto responsable, tal como vienen:", sorted(una["p_str"].unique()))
print("Categorías de etnia, tal como vienen:", sorted(una["etnia"].unique()))
'''),
("md", r'''
Aparece una rareza que no sé explicar: hay registros con edad de 90 años o más en un dato de reclutamiento, y esa edad no fue imputada, venía así de las fuentes. No tengo documentación que diga si es un error de digitación en las listas originales. No los corrijo. Quedan clasificados como adultos, y lo anoto en la bitácora.

**La definición de la cifra.** Una víctima cuenta como menor de edad si su `edad_jep` es `INFANCIA` o `ADOLESCENCIA`. El informe del proyecto analiza el reclutamiento de menores de 18 años entre 1990 y 2017.
'''),
("code", r'''
replicas["menor"] = replicas["edad_jep"].isin(["INFANCIA", "ADOLESCENCIA"])
una = replicas[replicas["replica"] == "R1"].reset_index(drop=True)
edad_registrada = ~una["edad_jep_imputed"].values

documentados = int((una["menor"] & edad_registrada).sum())
print(f"Víctimas documentadas: {len(una)}")
print(f"  con edad registrada: {edad_registrada.sum()}  ->  menores de 18 años: {documentados}")
print(f"  sin edad registrada: {(~edad_registrada).sum()}  ->  no se sabe")
'''),
("md", r'''
Esa primera cifra, la de menores de edad con edad registrada, es la **documentada**. El informe metodológico del proyecto reporta 14.669 en su tabla 3. Compárenla con la que acaba de salir.

### 2.2 Exploración

Miro tres cosas: cuándo ocurrieron los hechos, en cuántas fuentes aparece cada víctima y a quién le falta la edad.
'''),
("code", r'''
fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(15, 3.8))
por_anio = una.groupby("yy_hecho").agg(documentadas=("menor", "size"), sin_edad=("edad_jep_imputed", "sum"))
izq.bar(por_anio.index, por_anio["documentadas"], color="#1D4E89", label="con edad registrada")
izq.bar(por_anio.index, por_anio["sin_edad"], color="#1B998B", label="sin edad registrada")
izq.set(title="Víctimas documentadas por año del hecho", xlabel=""); izq.legend(fontsize=8)

en_fuentes = una["n_fuentes"].value_counts().sort_index()
centro.bar(en_fuentes.index, en_fuentes.values, color="#1D4E89")
centro.set(title="¿En cuántas fuentes aparece cada víctima?", xlabel="número de fuentes")

falta_por_fuentes = una.groupby(una["n_fuentes"].clip(upper=4))["edad_jep_imputed"].mean() * 100
der.bar(["1", "2", "3", "4 o más"], falta_por_fuentes.values, color="#D7263D")
der.set(title="Registros sin edad, según número de fuentes", xlabel="número de fuentes", ylabel="%")
plt.tight_layout(); plt.show()

print(f"Víctimas que aparecen en una sola fuente: {en_fuentes.loc[1] / len(una) * 100:.1f} %")
print("Registros sin edad (%), según número de fuentes:", falta_por_fuentes.round(1).to_dict())
print("Sexo, entre quienes lo tienen registrado (%):", (una.loc[~una["sexo_imputed"], "sexo"].value_counts(normalize=True) * 100).round(1).to_dict())
'''),
("md", r'''
Dos lecturas que sostienen el resto de la clase.

La primera es sobre el **subregistro**. Cerca de la mitad de las víctimas documentadas aparece en una sola fuente. Si tantas personas fueron vistas por una sola lista, es razonable pensar que hay otras que no fueron vistas por ninguna. Esa es la intuición detrás de la cifra estimada, que no calculamos hoy.

La segunda es sobre los **faltantes**. La edad no falta al azar. Falta sobre todo en los registros que vienen de una sola fuente, porque cuando dos listas registran a la misma persona, lo que una no anotó lo trae la otra. Esto importa para imputar: si los registros sin edad se parecen poco a los demás, rellenarlos con el promedio de los demás es una mala idea.

### 2.3 Un análisis no supervisado: ¿las fuentes ven a las mismas personas?

Cada víctima tiene un vector de ceros y unos que dice en qué fuentes aparece. Tomo las fuentes con más de 200 víctimas y calculo la correlación entre sus columnas. Después hago un PCA sobre esa matriz de correlaciones, $R=V\Lambda V^\top$. Dos fuentes con cargas parecidas tienden a documentar a las mismas personas.
'''),
("code", r'''
from sklearn.decomposition import PCA

grandes = [f for f in fuentes if una[f].sum() >= 200]
correlacion = una[grandes].corr()
pca = PCA(n_components=2).fit((una[grandes] - una[grandes].mean()) / una[grandes].std())
cargas = pd.DataFrame(pca.components_.T, index=grandes, columns=["CP1", "CP2"])

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.6))
imagen = izq.imshow(correlacion, cmap="RdBu_r", vmin=-0.6, vmax=0.6)
izq.set_xticks(range(len(grandes))); izq.set_yticks(range(len(grandes)))
izq.set_xticklabels(grandes, rotation=90, fontsize=7); izq.set_yticklabels(grandes, fontsize=7)
izq.set_title("Correlación entre fuentes"); plt.colorbar(imagen, ax=izq)
der.scatter(cargas["CP1"], cargas["CP2"], s=una[grandes].sum() / 20, color="#1D4E89", alpha=0.6)
for nombre, (a, b) in cargas.iterrows():
    der.text(a + 0.01, b, nombre, fontsize=8)
der.axhline(0, color="gray", lw=0.5); der.axvline(0, color="gray", lw=0.5)
der.set(title="Cargas de las fuentes (el tamaño es el número de víctimas)", xlabel="CP1", ylabel="CP2")
plt.tight_layout(); plt.show()
print("Varianza explicada (%):", (pca.explained_variance_ratio_ * 100).round(1), "| fuentes con 200 víctimas o más:", len(grandes))
sin_diagonal = correlacion.where(~np.eye(len(grandes), dtype=bool)).stack()
print("Par de fuentes que más coincide:", sin_diagonal.idxmax(), round(float(sin_diagonal.max()), 2), "| par que más se evita:", sin_diagonal.idxmin(), round(float(sin_diagonal.min()), 2))
'''),
("md", r'''
Las fuentes no son copias unas de otras ni son independientes. Algunas coinciden mucho, y otras casi nunca registran a la misma persona, lo que sugiere que miran poblaciones o momentos distintos. No sabemos cuál es cuál porque el DANE les quitó el nombre. Esa estructura es la razón por la que estimar el subregistro exige un modelo cuidadoso y no una regla de tres.
'''),
("md", r'''
## 3. Modelado

### 3.1 La cifra imputada: combinar las réplicas

Cada réplica da su propia cuenta de menores de edad, porque en cada una los registros sin edad recibieron un valor distinto. Aplico las reglas de la sección 1.
'''),
("code", r'''
def combinar(cifras):
    """Reglas de combinación para un conteo de víctimas documentadas (sin varianza de muestreo)."""
    cifras = np.asarray(cifras, dtype=float)
    m, media = len(cifras), cifras.mean()
    B = cifras.var(ddof=1)
    T = (1 + 1 / m) * B
    return {"media": media, "inferior": media - 1.96 * np.sqrt(T), "superior": media + 1.96 * np.sqrt(T), "entre réplicas (DE)": np.sqrt(B)}

por_replica = replicas.groupby("replica", sort=False)["menor"].sum()
imputada = combinar(por_replica.values)
print("Menores de edad en cada réplica:", por_replica.values)
print(f"\nCifra documentada: {documentados}")
print(f"Cifra imputada:    {imputada['media']:.0f}  (intervalo de 95 %: {imputada['inferior']:.0f} a {imputada['superior']:.0f}), con {len(por_replica)} réplicas")
print(f"Ancho del intervalo: {(imputada['superior'] - imputada['inferior']) / imputada['media'] * 100:.1f} % de la cifra")
print(f"Contar solo los registros con edad deja por fuera a {imputada['media'] - documentados:.0f} menores de edad, el {(1 - documentados / imputada['media']) * 100:.1f} %")
'''),
("md", r'''
Comparen con sus apuestas segunda y tercera. El informe metodológico del proyecto reporta para reclutamiento, con diez réplicas, una cifra imputada de 16.238, con intervalo de 16.027 a 16.449. Si corrieron este cuaderno con `N_REPLICAS = 10`, comparen con lo que salió: reproducir una cifra publicada con los datos publicados es la mejor prueba de que entendimos la definición.

El mismo informe estima que, contando el subregistro, las víctimas de reclutamiento fueron entre 27.101 y 40.828, con 32.806 como valor central. Esa es la cifra estimada. Fíjense en los tamaños: completar la edad mueve la cifra en unos 1.600, y el subregistro la mueve en más de 16.000. La incertidumbre que hoy aprendemos a calcular es la más pequeña de las dos.

La misma regla sirve para cualquier desagregación. Por año:
'''),
("code", r'''
por_anio_replica = replicas[replicas["menor"]].groupby(["yy_hecho", "replica"]).size().unstack(fill_value=0)
serie = pd.DataFrame([combinar(fila) for fila in por_anio_replica.values], index=por_anio_replica.index)
serie["documentada"] = una[una["menor"] & edad_registrada].groupby("yy_hecho").size()

plt.figure(figsize=(11, 3.8))
plt.fill_between(serie.index, serie["inferior"], serie["superior"], color="#1B998B", alpha=0.5, label="intervalo de 95 % de la imputada")
plt.plot(serie.index, serie["media"], color="#D7263D", label="imputada")
plt.plot(serie.index, serie["documentada"], color="#1D4E89", ls="--", label="documentada (solo con edad registrada)")
plt.title("Menores de edad reclutados, por año del hecho"); plt.ylabel("víctimas"); plt.legend(fontsize=8); plt.show()
serie["ancho relativo (%)"] = (serie["superior"] - serie["inferior"]) / serie["media"] * 100
print("Años con el intervalo más ancho en proporción a su cifra:")
print(serie.sort_values("ancho relativo (%)", ascending=False).head(4).round(1)[["documentada", "media", "inferior", "superior", "ancho relativo (%)"]])
'''),
("md", r'''
La banda es angosta cuando casi todos los registros del año traían la edad y se ensancha en los años con más faltantes. Una cifra desagregada tiene menos datos detrás y más incertidumbre relativa que el total.

### 3.2 ¿Se puede inferir el campo que falta? La línea base y un catálogo de modelos

Las réplicas las produjo HRDAG con un modelo que usó los datos completos, antes de la anonimización. No tenemos ese modelo, pero podemos hacernos la pregunta de fondo con lo publicado: entre los registros que **sí** tienen edad, ¿qué tan bien se adivina si la víctima era menor a partir de los demás campos?

Uso solo información registrada. Cuando el sexo, la etnia o el presunto responsable fueron imputados, los trato como «desconocido» para no entrenar con valores que otro modelo propuso. Las entradas son el año, el departamento, el número de fuentes, las indicadoras de fuente y esos tres campos.

Como el modelo se va a usar para sortear valores, importa que sus probabilidades sean honestas, y no solo que ordene bien. Por eso la métrica para escoger es la **pérdida logarítmica**,

$$\text{pérdida}=-\frac{1}{n}\sum_{i=1}^{n}\bigl[y_i\ln\hat p_i+(1-y_i)\ln(1-\hat p_i)\bigr],$$

que castiga con dureza a quien dice 0,99 y se equivoca. La línea base es decirle a todos la misma probabilidad, la proporción de menores del entrenamiento. Las víctimas con edad registrada se parten al azar en 60 % para entrenar, 20 % para validar y 20 % para probar. **Se parten víctimas**, usando una sola réplica.
'''),
("code", r'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

def entradas(tabla):
    """Solo información registrada: lo imputado se marca como desconocido."""
    X = pd.DataFrame({"anio": tabla["yy_hecho"].values, "n_fuentes": tabla["n_fuentes"].values}).join(tabla[fuentes].reset_index(drop=True))
    for campo in ["sexo", "etnia", "p_str"]:
        valores = tabla[campo].where(~tabla[campo + "_imputed"], "desconocido").reset_index(drop=True)
        X = X.join(pd.get_dummies(valores, prefix=campo).astype(int))
    return X.join(pd.get_dummies(tabla["dept_code_hecho"].reset_index(drop=True), prefix="dep").astype(int))

X = entradas(una)
A, y = X.values.astype(float), una["menor"].astype(int).values
con_edad = np.flatnonzero(edad_registrada)
entrena, resto = train_test_split(con_edad, test_size=0.4, stratify=y[con_edad], random_state=0)
valida, prueba = train_test_split(resto, test_size=0.5, stratify=y[resto], random_state=0)
print(f"Entradas: {X.shape[1]} columnas | entrenamiento {len(entrena)} víctimas | validación {len(valida)} | prueba {len(prueba)} | menores: {y[entrena].mean() * 100:.1f} %")
'''),
("md", r'''
**Cómo leer la celda del catálogo.** El diccionario `MODELOS` tiene una línea por candidato, con su nombre y un `make_pipeline` que encadena la preparación de la tabla con el clasificador. `escalar()` pone las columnas en la misma escala, que es lo que necesitan los modelos que miden distancias o usan gradientes. Para agregar un modelo se importa su clase y se escribe una línea más en el diccionario. La pérdida logarítmica se calcula con probabilidades, así que solo sirven clasificadores que tengan `predict_proba`. El resto del cuaderno recorre lo que haya en `MODELOS`, así que no hay que tocar nada más.

**Ejercicio.** Agreguen al menos tres modelos más de scikit-learn y vuelvan a correr desde esa celda.
'''),
("code", r'''
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

escalar = lambda: StandardScaler()                     # pone todas las columnas en la misma escala

# El catálogo: una línea por modelo. Para agregar uno, escriban otra línea con su nombre y su pipeline.
MODELOS = {
    "logística":              make_pipeline(escalar(), LogisticRegression(max_iter=2000)),
    "logística (C = 0,01)":   make_pipeline(escalar(), LogisticRegression(C=0.01, max_iter=2000)),
    "discriminante lineal":   make_pipeline(escalar(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    "Bayes ingenuo":          make_pipeline(BernoulliNB()),
    "k vecinos (k = 15)":     make_pipeline(escalar(), KNeighborsClassifier(n_neighbors=15)),
    "k vecinos (k = 50)":     make_pipeline(escalar(), KNeighborsClassifier(n_neighbors=50)),
    "árbol":                  make_pipeline(DecisionTreeClassifier(min_samples_leaf=20, random_state=0)),
    "bosque":                 make_pipeline(RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)),
    "ExtraTrees":             make_pipeline(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=0)),
    "boosting":               make_pipeline(HistGradientBoostingClassifier(random_state=0)),
    "AdaBoost":               make_pipeline(AdaBoostClassifier(n_estimators=50, random_state=0)),
    "red neuronal":           make_pipeline(escalar(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=500, random_state=0)),
}
print(len(MODELOS), "modelos en el catálogo:", ", ".join(MODELOS))
'''),
("code", r'''
proporcion = y[entrena].mean()
filas = {"misma probabilidad para todos": {"pérdida en validación": log_loss(y[valida], np.full(len(valida), proporcion)), "AUC en validación": 0.5,
                                          "pérdida en prueba": log_loss(y[prueba], np.full(len(prueba), proporcion)), "AUC en prueba": 0.5}}
entrenados = {}
for nombre, modelo in MODELOS.items():                 # recorre lo que haya en el catálogo
    entrenados[nombre] = clone(modelo).fit(A[entrena], y[entrena])
    pv, pp = entrenados[nombre].predict_proba(A[valida])[:, 1], entrenados[nombre].predict_proba(A[prueba])[:, 1]
    filas[nombre] = {"pérdida en validación": log_loss(y[valida], pv), "AUC en validación": roc_auc_score(y[valida], pv),
                     "pérdida en prueba": log_loss(y[prueba], pp), "AUC en prueba": roc_auc_score(y[prueba], pp)}
comparacion = pd.DataFrame(filas).T
display(comparacion.round(3))

comparacion[["pérdida en validación", "pérdida en prueba"]].sort_values("pérdida en validación", ascending=False).plot.barh(color=["#1D4E89", "#9AA5B1"], figsize=(9, 0.32 * len(comparacion) + 1.5))
plt.title("Pérdida logarítmica: menor es mejor"); plt.show()
mejor = comparacion.drop(index="misma probabilidad para todos")["pérdida en validación"].idxmin()
print(f"Ganador en validación: {mejor} | pérdida en prueba {comparacion.loc[mejor, 'pérdida en prueba']:.3f} (línea base {comparacion.iloc[0]['pérdida en prueba']:.3f}) | AUC en prueba {comparacion.loc[mejor, 'AUC en prueba']:.3f}")
'''),
("md", r'''
El ganador se escogió entre muchos candidatos, y eso tiene un costo que conviene mirar. La gráfica siguiente ubica a cada modelo según su puesto en validación y su puesto en prueba.
'''),
("code", r'''
candidatos = comparacion.drop(index="misma probabilidad para todos")
puestos = candidatos[["pérdida en validación", "pérdida en prueba"]].rank(method="min").astype(int)      # la menor pérdida es el puesto 1
puestos.columns = ["validación", "prueba"]
n = len(puestos)
fig, eje = plt.subplots(figsize=(5.6, 5), facecolor="#FBFAF7"); eje.set_facecolor("#FBFAF7")
eje.plot([1, n], [1, n], color="#9AA5B1", ls="--", lw=1)             # sobre esta línea, el puesto se repite
eje.scatter(puestos["validación"], puestos["prueba"], s=45, zorder=3, color=["#D7263D" if m == mejor else "#1D4E89" for m in puestos.index])
for m, fila in puestos.iterrows():
    eje.annotate(m, (fila["validación"], fila["prueba"]), xytext=(6, 3), textcoords="offset points", fontsize=7.5)
eje.set(title="Puesto en validación contra puesto en prueba", xlabel="puesto en validación (1 es el mejor)", ylabel="puesto en prueba",
        xticks=range(1, n + 1), yticks=range(1, n + 1), xlim=(0.5, n + 2.5), ylim=(n + 0.5, 0.5))
plt.show()

en_orden = candidatos["pérdida en validación"].sort_values()
print(f"El ganador en validación ({mejor}) quedó de puesto {puestos.loc[mejor, 'prueba']} en prueba, entre {n} candidatos.")
print(f"Ventaja del primero sobre el segundo en validación: {en_orden.iloc[1] - en_orden.iloc[0]:.4f} de pérdida, con {len(valida)} víctimas.")
print("Modelos que cambiaron de puesto entre validación y prueba:", int((puestos["validación"] != puestos["prueba"]).sum()), "de", n)
'''),
("md", r'''
Si el orden de validación se repitiera en la prueba, todos los puntos caerían sobre la diagonal. Miren cuánto se alejan y comparen esa distancia con la ventaja del primero sobre el segundo. Con tantos candidatos, ¿cuánto del ganador es azar? **Mi respuesta:**

Lean el resultado con calma. El ganador mejora a la línea base, así que los demás campos dicen algo sobre la edad. Pero su AUC está lejos de 1: con lo publicado no se puede saber con seguridad si un registro sin edad era de un menor. Esa es la razón de fondo para imputar varias veces. Cuando un campo se puede adivinar casi siempre, una sola imputación hace poco daño. Cuando solo se puede adivinar a medias, hay que cargar esa duda hasta la cifra final.

### 3.3 La trampa: tratar las réplicas como si fueran más datos

El archivo trae `N_REPLICAS` copias de cada víctima. Alguien que no leyó la documentación ve una tabla muchas veces más grande y hace lo de siempre: `train_test_split` sobre todas las filas. Midamos qué pasa con cinco réplicas apiladas.
'''),
("code", r'''
cinco = [f"R{k}" for k in range(1, 6)]                                              # bastan cinco réplicas para ver el efecto
apiladas = replicas[replicas["replica"].isin(cinco) & ~replicas["edad_jep_imputed"]].reset_index(drop=True)
A_ap, y_ap = entradas(apiladas).reindex(columns=X.columns, fill_value=0).values.astype(float), apiladas["menor"].astype(int).values
a, b = train_test_split(np.arange(len(y_ap)), test_size=0.2, stratify=y_ap, random_state=0)
victimas_compartidas = len(set(apiladas["match_group_id"].values[a]) & set(apiladas["match_group_id"].values[b]))

trampa = {}
for nombre in ["k vecinos (k = 50)", "bosque"]:
    modelo = clone(MODELOS[nombre]).fit(A_ap[a], y_ap[a])
    trampa[nombre] = {"AUC, filas apiladas al azar": roc_auc_score(y_ap[b], modelo.predict_proba(A_ap[b])[:, 1]),
                      "AUC, partiendo víctimas": comparacion.loc[nombre, "AUC en prueba"]}
trampa = pd.DataFrame(trampa).T; trampa["inflado"] = trampa.iloc[:, 0] - trampa.iloc[:, 1]
display(trampa.round(3))
print(f"Filas apiladas: {len(y_ap)} | víctimas distintas: {apiladas['match_group_id'].nunique()}")
print(f"Víctimas de la «prueba» que también están en el «entrenamiento»: {victimas_compartidas} de {len(set(apiladas['match_group_id'].values[b]))}")
'''),
("md", r'''
El modelo no mejoró: se encontró en la prueba con las mismas personas que ya había visto en el entrenamiento, copiadas en otra réplica. Es la misma fuga de la clase 8, donde el paciente repetido inflaba el resultado, con otro disfraz.

La trampa tiene una segunda cara, más silenciosa. Quien calcule un error estándar con las filas apiladas divide por la raíz de un $n$ varias veces mayor que el real, y sus intervalos salen angostos sin razón. Las réplicas no agregan información. Sirven para medir cuánta falta.

### 3.4 Nuestra propia imputación, para ver de qué depende la cifra

Ahora uso el modelo ganador para hacer lo que hace una imputación múltiple, con las herramientas del curso. Para cada una de $m$ réplicas propias:

1. Tomo una muestra *bootstrap* de las víctimas con edad registrada y reajusto el modelo. Así cada réplica lleva también la incertidumbre del modelo.
2. Calculo para cada registro sin edad su probabilidad $\hat p_i$ de ser menor.
3. Sorteo $z_i\sim\text{Bernoulli}(\hat p_i)$ y cuento.

La cifra de la réplica $k$ es $\hat\theta_k=\text{documentados}+\sum_i z_i^{(k)}$, y las $m$ cifras se combinan con las reglas de siempre.
'''),
("code", r'''
azar = np.random.default_rng(0)
sin_edad = np.flatnonzero(~edad_registrada)
propias = []
inicio = time.time()
for k in range(N_REPLICAS):
    remuestra = azar.choice(con_edad, len(con_edad), replace=True)
    modelo = clone(MODELOS[mejor]).fit(A[remuestra], y[remuestra])
    p = modelo.predict_proba(A[sin_edad])[:, 1]
    propias.append(documentados + int((azar.random(len(p)) < p).sum()))
nuestra = combinar(propias)
p_media = entrenados[mejor].predict_proba(A[sin_edad])[:, 1]

print(f"Registros sin edad: {len(sin_edad)} | {time.time() - inicio:.0f} s")
print(f"Proporción de menores entre quienes tienen edad registrada: {y[con_edad].mean() * 100:.1f} %")
print(f"Probabilidad media de ser menor que nuestro modelo les da a los registros sin edad: {p_media.mean() * 100:.1f} %")
print(f"Proporción de menores que las réplicas publicadas les asignan: {(imputada['media'] - documentados) / len(sin_edad) * 100:.1f} %")
print(f"\nCifra con las réplicas publicadas: {imputada['media']:.0f}  ({imputada['inferior']:.0f} a {imputada['superior']:.0f})")
print(f"Cifra con nuestra imputación:      {nuestra['media']:.0f}  ({nuestra['inferior']:.0f} a {nuestra['superior']:.0f})")
print(f"\nDesviación estándar entre réplicas: publicadas {imputada['entre réplicas (DE)']:.0f} | nuestras {nuestra['entre réplicas (DE)']:.0f}")
print(f"Si cada registro sin edad se sorteara por separado con su probabilidad, la suma variaría con desviación {np.sqrt((p_media * (1 - p_media)).sum()):.0f}")
'''),
("md", r'''
Las dos cifras responden la misma pregunta con el mismo método de combinación y pueden no coincidir. Si no coinciden, miren las tres proporciones de arriba: la diferencia está en qué tan parecidos a los demás considera cada modelo a los registros sin edad.

Miren también el ancho. Nuestras réplicas se mueven poco, cerca de lo que da sortear cada registro por separado. Las publicadas se mueven varias veces más. Una suma de sorteos independientes casi no varía, porque los errores de un registro se compensan con los de otro. Para que la suma varíe tanto como en las réplicas publicadas, la duda tiene que ser compartida por muchos registros a la vez, como ocurre cuando lo incierto es un parámetro del modelo y no el sorteo de cada persona. Esa es mi lectura de la diferencia, y no la puedo comprobar sin el modelo original. La consecuencia práctica sí es firme: un intervalo de imputación puede salir angosto por la manera como se sorteó y no porque sepamos mucho.

No lean esto como una corrección a la cifra oficial. El modelo de HRDAG trabajó con los datos completos, antes de anonimizar, y con campos que aquí no tenemos. El nuestro es un ejercicio de clase con lo publicado. Lo que el ejercicio sí enseña es que el intervalo de una imputación múltiple mide la duda **dentro** de un modelo de imputación. La duda **entre** modelos razonables no está en ese intervalo, y puede ser del mismo tamaño o mayor.

### 3.5 Explicar: ¿a quién le falta la edad, y qué usa el modelo?

Dos experimentos. El primero vuelve sobre la exploración: qué fuentes aportan los registros sin edad. El segundo es la importancia por permutación del modelo ganador sobre la validación, agrupando columnas por familia: desordeno entre víctimas todas las columnas de una familia y mido cuánto sube la pérdida.
'''),
("code", r'''
aporte = pd.DataFrame({"víctimas": una[grandes].sum(), "sin edad (%)": [una.loc[una[f] == 1, "edad_jep_imputed"].mean() * 100 for f in grandes]})
aporte["de los registros sin edad, están en esta fuente (%)"] = [una.loc[~edad_registrada, f].mean() * 100 for f in grandes]
display(aporte.sort_values("de los registros sin edad, están en esta fuente (%)", ascending=False).head(5).round(1))

familias = {"año": ["anio"], "número de fuentes": ["n_fuentes"], "qué fuentes": fuentes,
            "sexo": [c for c in X if c.startswith("sexo_")], "etnia": [c for c in X if c.startswith("etnia_")],
            "presunto responsable": [c for c in X if c.startswith("p_str_")], "departamento": [c for c in X if c.startswith("dep_")]}
referencia = log_loss(y[valida], entrenados[mejor].predict_proba(A[valida])[:, 1])
subida = {}
for nombre, columnas in familias.items():
    posiciones = [X.columns.get_loc(c) for c in columnas]
    subidas = []
    for _ in range(5):
        permutada = A[valida].copy()
        permutada[:, posiciones] = permutada[azar.permutation(len(valida))][:, posiciones]
        subidas.append(log_loss(y[valida], entrenados[mejor].predict_proba(permutada)[:, 1]) - referencia)
    subida[nombre] = np.mean(subidas)
pd.Series(subida).sort_values().plot.barh(color="#1D4E89", figsize=(8, 3.2))
plt.title(f"Cuánto sube la pérdida de «{mejor}» al desordenar cada familia de columnas"); plt.show()
print({n: round(float(v), 4) for n, v in sorted(subida.items(), key=lambda par: -par[1])})
'''),
("md", r'''
La tabla dice de dónde vienen los registros sin edad: unas pocas fuentes concentran casi todos. La gráfica dice en qué se apoya el modelo para inferir la edad. Las dos cosas describen el dato y el modelo. No dicen por qué una fuente no registró la edad, y eso solo lo sabe quien conoce cómo trabajó cada entidad.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: cuatro maneras de escribir la cifra

Pongo lado a lado las cuatro cuentas que alguien podría llevar al informe.
'''),
("code", r'''
moda = documentados + (len(sin_edad) if y[con_edad].mean() > 0.5 else 0)       # rellenar todos los faltantes con la categoría más común
opciones = pd.DataFrame([
    {"cómo se cuenta": "Solo registros con edad (casos completos)", "cifra": documentados, "intervalo": "no tiene", "qué supone": "Que los registros sin edad no existen"},
    {"cómo se cuenta": "Rellenar con la categoría más común", "cifra": moda, "intervalo": "no tiene", "qué supone": "Que todos los registros sin edad eran de menores"},
    {"cómo se cuenta": "Una sola réplica (R1)", "cifra": int(por_replica.iloc[0]), "intervalo": "no tiene", "qué supone": "Que el valor sorteado es un dato"},
    {"cómo se cuenta": f"Imputación múltiple, {len(por_replica)} réplicas publicadas", "cifra": round(imputada["media"]),
     "intervalo": f"{imputada['inferior']:.0f} a {imputada['superior']:.0f}", "qué supone": "El modelo de imputación del proyecto"},
])
display(opciones)
print(f"Distancia entre la cuenta más baja y la más alta: {opciones['cifra'].max() - opciones['cifra'].min()} víctimas")
print(f"Entre una réplica y otra, la cifra se mueve con una desviación estándar de {imputada['entre réplicas (DE)']:.0f}")
'''),
("md", r'''
Las tres primeras filas tienen el mismo defecto: entregan un número sin intervalo, y cada una esconde un supuesto que nadie escribió. La cuarta dice su supuesto y dice su duda.

Y sobre todas ellas está la advertencia de la sección 3.1: estas son cifras de víctimas **documentadas**. El informe del proyecto estima entre 27.101 y 40.828 víctimas de reclutamiento contando el subregistro. Una frase que diga «hubo 16.238 víctimas» está mal escrita aunque el número esté bien calculado, porque el lector entiende que fueron todas.

### 4.2 El resultado en una página

Un tablero es una página que responde una pregunta sin que el lector tenga que correr nada. Lleva la cifra con su intervalo, la gráfica que la sostiene y la letra pequeña que impide leerla mal. En el gemelo van a construir uno interactivo. Aquí va uno estático, hecho con lo que ya calculamos.
'''),
("code", r'''
fig = plt.figure(figsize=(13, 7))
rejilla = fig.add_gridspec(2, 3, height_ratios=[1, 1.5])

titular = fig.add_subplot(rejilla[0, 0]); titular.axis("off")
titular.text(0, 0.95, "Menores de edad reclutados\nen el conflicto, 1990 a 2017", fontsize=13, weight="bold", va="top")
titular.text(0, 0.42, f"{imputada['media']:,.0f}".replace(",", "."), fontsize=34, color="#D7263D", weight="bold")
titular.text(0, 0.22, f"víctimas documentadas\nintervalo de 95 %: {imputada['inferior']:,.0f} a {imputada['superior']:,.0f}".replace(",", "."), fontsize=10, va="top")

tres = fig.add_subplot(rejilla[0, 1:])
tres.barh(["Con edad registrada", "Documentadas, con edad imputada", "Estimadas con subregistro\n(informe del proyecto)"], [documentados, imputada["media"], 32806], color=["#1D4E89", "#D7263D", "#1B998B"])
tres.errorbar([imputada["media"], 32806], [1, 2], xerr=[[imputada["media"] - imputada["inferior"], 32806 - 27101], [imputada["superior"] - imputada["media"], 40828 - 32806]], fmt="none", color="black", capsize=4)
tres.set_title("Tres cifras que no son la misma", fontsize=10); tres.invert_yaxis()

anual = fig.add_subplot(rejilla[1, :2])
anual.fill_between(serie.index, serie["inferior"], serie["superior"], color="#1B998B", alpha=0.5)
anual.plot(serie.index, serie["media"], color="#D7263D"); anual.plot(serie.index, serie["documentada"], color="#1D4E89", ls="--")
anual.set_title("Por año del hecho: imputada con su intervalo (línea continua) y con edad registrada (punteada)", fontsize=10)

nota = fig.add_subplot(rejilla[1, 2]); nota.axis("off")
nota.text(0, 1, "Cómo leer esta página\n\n"
          f"• Son víctimas que aparecen en al menos\n  una de las listas integradas.\n"
          f"• A {len(sin_edad):,} de {len(una):,} no les quedó registrada\n  la edad. Se completó con imputación\n  múltiple ({len(por_replica)} réplicas).\n".replace(",", ".") +
          "• El intervalo mide la duda de esa\n  imputación, no el subregistro.\n"
          "• Fuente: JEP, Comisión de la Verdad y\n  HRDAG; DANE, catálogo 795. CC BY 4.0.", fontsize=9, va="top")
plt.tight_layout(); plt.show()
'''),
("md", r'''
### La frase del informe

Esta vez la decisión es una frase. Escriban la que pondrían en el informe, para un lector que no es estadístico. Debe decir qué se contó, cuántos, con qué margen y qué queda por fuera. Después escriban la decisión en tres frases para quien dirige el informe.

**La frase:**

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.3 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Contar como menor de edad con `edad_jep` | Usar el quinquenio `edad_categoria` | El quinquenio de 15 a 19 mezcla menores y mayores de 18 |
| Usar las réplicas publicadas para la cifra | Imputar por nuestra cuenta | El modelo del proyecto usó los datos completos, antes de anonimizar |
| Combinar con $\bar W=0$ | Agregar varianza de muestreo | Se cuentan víctimas documentadas, que no son una muestra |
| Dejar los registros con edad de 90 años o más | Corregirlos o borrarlos | No hay documentación que explique la rareza; quedan como adultos |
| No desagregar por presunto responsable | Mostrar esa tabla | El campo faltaba en uno de cada cinco registros y no tengo el diccionario oficial de los códigos |
| En el modelo, tratar lo imputado como «desconocido» | Usar los valores de una réplica como entradas | No se entrena un modelo con lo que otro modelo propuso |
| Pérdida logarítmica para escoger | Exactitud | El modelo se usa para sortear: importan las probabilidades |
| Partir víctimas, con una sola réplica | Partir filas del archivo apilado | La misma víctima está en todas las réplicas |
| Citar la cifra estimada del informe | Calcularla en clase | Estimar el subregistro exige un método que no cabe hoy |

### 4.4 Riesgos y cuidado del dato

Respondan en dos o tres líneas cada una.

- **Personas.** El DANE anonimizó estos datos. ¿Qué cruce con otra base podría volver a identificar a alguien, y por qué no se hace aunque se pueda?
- **Palabras.** ¿Qué cambia entre escribir «casos», «registros» y «víctimas»? ¿Cuál corresponde a cada una de las tres cifras?
- **Supuestos.** La imputación supone que, entre registros parecidos, los que no tienen edad se comportan como los que sí. Vimos que los registros sin edad vienen sobre todo de unas pocas fuentes. ¿Qué tan cómodos quedan con ese supuesto?
- **Lectura.** Alguien toma la cifra documentada de un departamento y concluye que allí «hubo menos reclutamiento». ¿Qué otra explicación tiene un número bajo en un dato hecho de listas?
- **Mantenimiento.** Si mañana aparece una fuente nueva con mil registros, ¿qué habría que rehacer: la cuenta, la imputación, la estimación?

### 4.5 Cambien una línea

Vuelvan a la celda de parámetros, cambien `N_REPLICAS = 10` por `N_REPLICAS = 30` y corran todo de nuevo. Con más réplicas el cuaderno tarda un poco más.

| | 10 réplicas | 30 réplicas |
|---|---|---|
| Cifra imputada | | |
| Intervalo de 95 % | | |
| Desviación estándar entre réplicas | | |
| Cifra con nuestra imputación | | |
| Ancho del intervalo, como porcentaje de la cifra | | |

Con la tabla llena, respondan en tres líneas: ¿la cifra cambió más de lo que su intervalo permitía? ¿Para qué sirven más réplicas, si no agregan información sobre las víctimas? **Mi respuesta:**
'''),
]
