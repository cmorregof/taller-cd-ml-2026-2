"""Genera los cuadernos de la clase 1. El de los estudiantes sale del mismo
texto que la solución: lo que va entre « » en el código se reemplaza por ____."""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno

# ---------------------------------------------------------------- EL SANCOCHO
SANCOCHO = [
("md", r'''
# Clase 1 · El sancocho

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. Ustedes lo siguen, lo corren y me interrumpen. En la segunda mitad cada uno resuelve el gemelo, *El precio de la luz*, con la misma técnica y otros datos.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Convertir una pregunta de la vida diaria en un problema de datos con métrica, explorarlo, comparar modelos de aprendizaje automático y de series de tiempo contra una línea base, y validar respetando el tiempo. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 3 (flujo reproducible y trazable), 4 (implementar y evaluar un modelo) y 5 (comunicar a quien no es técnico). |
| **Uso de IA** | Pueden pedirle a Claude que explique una línea de código o que escriba una gráfica. Deben poder explicar sin ayuda qué es una línea base, por qué no se barajan los meses, por qué el ganador se escoge en validación y qué significa el error que reportan. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con la decisión de la sección 4 escrita por ustedes. |
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

Imagínense a alguien que vive de vender sancocho en Manizales. Compra en la Galería, cocina para seis por olla y tiene que imprimir la carta con el precio del plato. La carta dura un trimestre. Si el mercado sube y el precio se quedó quieto, el trimestre se lo come el espinazo.

Esa persona no necesita un modelo. Necesita tres respuestas:

1. **Predecir.** ¿Cuánto costará la olla dentro de tres meses?
2. **Explicar.** ¿Qué ingrediente mueve el costo?
3. **Decidir.** ¿Con qué costo debe presupuestar la olla del próximo trimestre para no quedar corta?

**El dato.** El DANE publica cada mes, en el sistema SIPSA, el precio mayorista por kilo de cientos de productos en las centrales de abasto del país. Manizales reporta por la Galería.

**La receta, para seis:** 1,5 kg de espinazo de cerdo, 1 kg de yuca, 1 kg de papa, tres mazorcas (0,9 kg) y cuatro plátanos verdes (1,4 kg).

**La métrica.** Medimos el error porcentual absoluto medio (MAPE) del pronóstico a tres meses. Un MAPE de 6 % significa que, en promedio, el pronóstico se aleja 6 % del costo real. Elegimos un porcentaje porque el dueño del negocio piensa en márgenes, no en pesos sueltos.

### Predicción escrita

Antes de ver un solo dato, escriban aquí su apuesta. Después la comparamos con lo que salga.

- ¿Cuánto costaban los ingredientes de la olla en diciembre de 2024, al por mayor? **Mi apuesta:**
- ¿Qué ingrediente pesa más en el costo? **Mi apuesta:**
- Si yo pronostico «dentro de tres meses costará lo mismo que hoy», ¿de qué tamaño será mi error? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

Una sola línea trae el dato desde el DANE. La función descarga los ocho archivos oficiales, los unifica y revisa con una huella (MD5) que el resultado sea idéntico al que usé para preparar la clase. Tarda cerca de medio minuto.

El archivo no está guardado en el repositorio del curso porque el DANE no permite republicarlo. Por eso cada uno lo descarga de la fuente.
'''),
("code", r'''
sipsa = descargar("sipsa")
sipsa.head()
'''),
("code", r'''
print("Filas:", len(sipsa))
print("Meses:", sipsa["fecha"].min(), "a", sipsa["fecha"].max())
print("Ciudades:", sipsa["ciudad"].nunique(), "| Productos:", sipsa["producto"].nunique())
'''),
("md", r'''
Nos quedamos con Manizales. Antes de filtrar productos conviene mirar cómo se llaman, porque el DANE no dice «papa»: dice la variedad.
'''),
("code", r'''
manizales = sipsa[sipsa["ciudad"] == "Manizales"]
print(manizales["mercado"].unique())
for palabra in ["espinazo", "Yuca", "Papa", "mazorca", "Plátano"]:
    print(palabra, "->", sorted(p for p in manizales["producto"].unique() if palabra.lower() in p.lower()))
'''),
("md", r'''
Hay varias papas y varios plátanos. Escoger uno es una **decisión de modelación** y va a la bitácora de la sección 4. Tomo las variedades que se venden todos los meses en la Galería.
'''),
("code", r'''
receta = {                                  # kilos para seis personas
    "Carne de cerdo, espinazo": 1.5,
    "Yuca ICA": 1.0,
    "Papa capira": 1.0,
    "Chócolo mazorca": 0.9,
    "Plátano dominico hartón verde": 1.4,
}
precios = (manizales[manizales["producto"].isin(receta)]
           .pivot_table(index="fecha", columns="producto", values="precio_kg"))
precios.index = pd.to_datetime(precios.index)
print("Meses:", len(precios), "| Faltantes por producto:")
print(precios.isna().sum())
'''),
("md", r'''
Ningún mes falta. El costo de la olla es la suma de precio por cantidad.
'''),
("code", r'''
costo_por_ingrediente = precios * pd.Series(receta)
olla = costo_por_ingrediente.sum(axis=1).rename("olla")

ax = olla.plot(color="#D7263D", lw=2)
ax.set(title="Costo mayorista de una olla de sancocho para seis, Manizales", xlabel="", ylabel="pesos")
plt.show()
print(f"Enero de 2013: ${olla.iloc[0]:,.0f} | Diciembre de 2024: ${olla.iloc[-1]:,.0f} | Se multiplicó por {olla.iloc[-1] / olla.iloc[0]:.2f}")
'''),
("md", r'''
Comparen con su apuesta. Ojo con lo que mide esta cifra: es el costo **mayorista** de los ingredientes en pesos corrientes. No incluye gas, mano de obra ni el margen de la tienda.
'''),
("md", r'''
### 2.1 Conocer el dato antes de modelarlo

Antes de pronosticar nada conviene mirar el dato por cuatro lados: cuánto vale cada ingrediente, cuánto se mueve de un mes a otro, si tiene temporada y cuánta memoria tiene. Llamo $p_{i,t}$ al precio por kilo del ingrediente $i$ en el mes $t$ y $q_i$ a los kilos de la receta. El costo de la olla es

$$C_t=\sum_{i=1}^{5} q_i\,p_{i,t},$$

y la variación mensual de cada precio es

$$r_{i,t}=\frac{p_{i,t}-p_{i,t-1}}{p_{i,t-1}}.$$
'''),
("code", r'''
ficha = pd.DataFrame({
    "kilos en la receta": pd.Series(receta),
    "precio en ene 2013": precios.iloc[0],
    "precio en dic 2024": precios.iloc[-1],
    "veces que subió": precios.iloc[-1] / precios.iloc[0],
    "mínimo": precios.min(),
    "mediana": precios.median(),
    "máximo": precios.max(),
    "meses sin dato": precios.isna().sum(),
})
display(ficha.style.format("{:,.0f}").format("{:.2f}", subset=["veces que subió"]).format("{:.1f}", subset=["kilos en la receta"])
        .background_gradient(subset=["veces que subió"], cmap="Blues").set_caption("Ficha del dato: pesos por kilo, Galería de Manizales"))

ejes = precios.plot(subplots=True, layout=(2, 3), figsize=(12, 5), legend=False, color="#1D4E89", lw=1.2)
for eje, nombre in zip(ejes.ravel(), precios.columns):
    eje.set_title(nombre, fontsize=9); eje.set_xlabel("")
ejes.ravel()[-1].axis("off")
plt.suptitle("Precio mayorista por kilo, Manizales"); plt.tight_layout(); plt.show()
'''),
("code", r'''
cambios = precios.pct_change().dropna()            # variación mensual de cada ingrediente

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4))
(cambios * 100).plot.box(ax=izq, rot=20, color="#1D4E89")
izq.set(title="Variación mensual por ingrediente", ylabel="%"); izq.tick_params(axis="x", labelsize=7)

correlacion = cambios.corr()
imagen = der.imshow(correlacion, cmap="RdBu_r", vmin=-1, vmax=1)
der.set_xticks(range(5)); der.set_yticks(range(5))
der.set_xticklabels([c[:12] for c in correlacion.columns], rotation=30, fontsize=7); der.set_yticklabels([c[:12] for c in correlacion.columns], fontsize=7)
for a in range(5):
    for b in range(5):
        der.text(b, a, f"{correlacion.iloc[a, b]:.2f}", ha="center", va="center", fontsize=8)
der.set_title("Correlación entre variaciones mensuales"); plt.colorbar(imagen, ax=der); plt.tight_layout(); plt.show()

print("Desviación estándar de la variación mensual (%):")
print((cambios.std() * 100).round(1).sort_values(ascending=False))
'''),
("md", r'''
Dos lecturas. La primera es que hay ingredientes tranquilos e ingredientes nerviosos, y el más nervioso no tiene por qué ser el que más pesa en la olla. La segunda es que las variaciones casi no están correlacionadas entre sí: cada ingrediente tiene su propia historia. Eso lo retomamos con el PCA.

**Temporada y memoria.** Para ver la temporada comparo cada mes con el promedio móvil de doce meses centrado en él. Para ver la memoria uso la autocorrelación de orden $k$:

$$\hat\rho_k=\frac{\sum_{t=k+1}^{T}(C_t-\bar C)(C_{t-k}-\bar C)}{\sum_{t=1}^{T}(C_t-\bar C)^2}.$$
'''),
("code", r'''
tendencia = olla.rolling(12, center=True).mean()
indice_estacional = (olla / tendencia).groupby(olla.index.month).mean()

fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(13, 3.6))
((indice_estacional - 1) * 100).plot.bar(ax=izq, color="#1B998B")
izq.set(title="Temporada: desvío frente a la tendencia", xlabel="mes", ylabel="%")

rezagos = range(1, 25)
centro.bar(rezagos, [olla.autocorr(k) for k in rezagos], color="#1D4E89")
centro.set(title="Autocorrelación del costo", xlabel="rezago (meses)")
der.bar(rezagos, [olla.pct_change().autocorr(k) for k in rezagos], color="#D7263D")
der.axhline(2 / np.sqrt(len(olla)), color="gray", ls=":"); der.axhline(-2 / np.sqrt(len(olla)), color="gray", ls=":")
der.set(title="Autocorrelación de la variación mensual", xlabel="rezago (meses)")
plt.tight_layout(); plt.show()
'''),
("md", r'''
La gráfica del centro dice que el costo de hoy se parece mucho al de los meses anteriores: la serie tiene memoria larga porque tiene tendencia. La de la derecha quita esa tendencia y pregunta algo más fino: si este mes subió, ¿el próximo también? Las líneas punteadas marcan la banda $\pm 2/\sqrt{T}$, dentro de la cual una autocorrelación no se distingue del ruido.

Guarden esta imagen. Las variaciones tienen poca memoria: solo el primer rezago se asoma con claridad por fuera de la banda. Una serie cuyas variaciones no tienen memoria se llama **caminata aleatoria**, $C_{t+1}=C_t+\varepsilon_{t+1}$, y para una caminata aleatoria el mejor pronóstico es el último valor. La olla se le parece bastante, y eso anticipa lo que veremos en la sección 3.
'''),
("md", r'''
### 2.2 Una prueba de hipótesis: ¿es la olla una caminata aleatoria?

La gráfica sugiere que sí. Una prueba lo pone en números. La prueba de Dickey-Fuller aumentada ajusta la regresión

$$\Delta c_t=\alpha+\gamma\,c_{t-1}+\sum_{j=1}^{p}\phi_j\,\Delta c_{t-j}+\varepsilon_t,\qquad c_t=\log C_t,$$

y contrasta

$$H_0:\ \gamma=0\quad\text{(raíz unitaria: la serie no regresa a un nivel fijo)}\qquad\text{contra}\qquad H_1:\ \gamma<0.$$

Un valor $p$ pequeño rechaza $H_0$. La aplico dos veces: al nivel de la olla y a sus variaciones mensuales.

**Predicción escrita:** ¿en cuál de las dos se rechazará $H_0$? **Mi apuesta:**
'''),
("code", r'''
from statsmodels.tsa.stattools import adfuller

import warnings
log_olla = np.log(olla)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")              # la función avisa de un cambio futuro de formato que no nos afecta
    p_nivel = adfuller(log_olla)[1]
    p_variacion = adfuller(log_olla.diff().dropna())[1]
print(f"Valor p para el nivel de la olla:        {p_nivel:.3f}")
print(f"Valor p para las variaciones mensuales:  {p_variacion:.4f}")
'''),
("md", r'''
En el nivel no se rechaza la raíz unitaria: la olla no tiene un precio «normal» al que regrese. En las variaciones sí se rechaza, así que los cambios mes a mes son estables. Es la firma de una caminata aleatoria con deriva. No rechazar $H_0$ no la demuestra, pero la deja como la descripción más económica del dato.

La consecuencia práctica es que la cantidad correcta para modelar es el **cambio** y no el nivel. Lo vamos a usar dos veces en la sección 3.

### 2.3 PCA, primera mirada: cada punto es un mes

El análisis de componentes principales busca las direcciones en que más varían los datos. Aquí cada mes es un punto con cinco coordenadas, que son los precios de los cinco ingredientes. Estandarizo los logaritmos de los precios en una matriz $Z$ y descompongo su matriz de correlaciones:

$$R=\frac{1}{T-1}Z^\top Z=V\Lambda V^\top,\qquad \Lambda=\operatorname{diag}(\lambda_1\ge\dots\ge\lambda_5).$$

Las columnas de $V$ son los componentes, $\lambda_j/\sum_k\lambda_k$ es la fracción de la varianza que explica el componente $j$, y las nuevas coordenadas de cada mes son los **puntajes** $ZV$. Con los dos primeros puntajes se dibuja cada mes en un plano.

**Predicción escrita:** si dibujo los 144 meses en ese plano y los coloreo por año, ¿qué figura esperan ver? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(np.log(precios))
pca = PCA().fit(Z)
puntajes = pca.transform(Z)
explicada = pca.explained_variance_ratio_ * 100
cargas = pd.DataFrame(pca.components_.T, index=precios.columns, columns=[f"CP{j + 1}" for j in range(5)])

fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw={"width_ratios": [1, 1.5, 1.3]})
izq.bar(cargas.columns, explicada, color="#1D4E89")
izq.set(title="Varianza explicada", ylabel="%")
centro.plot(puntajes[:, 0], puntajes[:, 1], color="gray", lw=0.5)
puntos = centro.scatter(puntajes[:, 0], puntajes[:, 1], c=precios.index.year, cmap="viridis", s=28)
plt.colorbar(puntos, ax=centro, label="año")
centro.set(title="Cada punto es un mes", xlabel=f"CP1 ({explicada[0]:.0f} %)", ylabel=f"CP2 ({explicada[1]:.0f} %)")
cargas[["CP1", "CP2"]].plot.barh(ax=der, color=["#1D4E89", "#D7263D"])
der.set(title="Cargas: qué mezcla de ingredientes es cada componente"); der.tick_params(axis="y", labelsize=7)
plt.tight_layout(); plt.show()
print("Varianza explicada (%):", explicada.round(1))
'''),
("md", r'''
Lean las tres gráficas de izquierda a derecha. El primer componente explica la mayor parte, y sus cargas tienen todas el mismo signo: es el nivel general de precios, lo que uno llamaría inflación de la olla. En el plano, los meses forman un camino que avanza de izquierda a derecha con los años, y hay un tramo donde los puntos se separan de golpe. Búsquenlo y digan en qué año cae.

El segundo componente tiene cargas de signos opuestos. Opone unos ingredientes a otros, así que recoge los meses en que la olla cambia de composición sin cambiar de precio.

### 2.4 PCA, segunda mirada: cada punto es un producto

Ahora le doy la vuelta a la tabla. Tomo todos los productos que la Galería reporta casi todos los meses, y cada **producto** es un punto cuyas coordenadas son sus variaciones mensuales. Dos productos quedan cerca si suben y bajan en los mismos meses.

La pregunta es si los alimentos forman familias, y dónde caen los cinco ingredientes del sancocho. La agrupación por familia que uso para colorear la hice yo a partir del nombre del producto. El PCA no la conoce.
'''),
("code", r'''
todos_los_precios = manizales.pivot_table(index="fecha", columns="producto", values="precio_kg")
frecuentes = todos_los_precios.loc[:, todos_los_precios.notna().sum() >= 140].interpolate().bfill().ffill()
variaciones = np.log(frecuentes).diff().dropna()

FRUTAS = ["aguacate", "banano", "coco", "fresa", "guanábana", "guayaba", "kiwi", "limón", "lulo", "mandarina", "mango", "manzana",
          "maracuyá", "melón", "mora", "naranja", "papaya", "pera", "piña", "tangelo", "tomate de árbol", "uva"]
VERDURAS = ["ahuyama", "ajo", "apio", "arveja verde en vaina", "berenjena", "brócoli", "cebolla", "chócolo", "cilantro", "coliflor",
            "espinaca", "fríjol verde", "habichuela", "lechuga", "pepino", "perejil", "pimentón", "remolacha", "repollo", "tomate chonto", "zanahoria"]
TUBERCULOS = ["papa ", "yuca", "plátano", "arracacha"]

def familia(nombre):
    n = nombre.lower()
    if n.startswith("carne de res"): return "carne de res"
    if n.startswith("carne de cerdo"): return "carne de cerdo"
    if n.startswith("huevo"): return "huevos"
    if any(n.startswith(x) for x in FRUTAS): return "frutas"
    if any(n.startswith(x) for x in TUBERCULOS): return "tubérculos y plátanos"
    if any(n.startswith(x) for x in VERDURAS): return "verduras y hortalizas"
    return "abarrotes y otros"

Zp = StandardScaler().fit_transform(variaciones)            # una columna por producto
pca_productos = PCA(n_components=3).fit(Zp.T)               # al trasponer, cada producto es una fila
nube = pd.DataFrame(pca_productos.transform(Zp.T), index=frecuentes.columns, columns=["CP1", "CP2", "CP3"])
nube["familia"] = [familia(p) for p in nube.index]

colores = {"carne de res": "#D7263D", "carne de cerdo": "#F49097", "huevos": "#6C4AB6", "frutas": "#1B998B",
           "verduras y hortalizas": "#8BC34A", "tubérculos y plátanos": "#1D4E89", "abarrotes y otros": "#9AA5B1"}
fig, eje = plt.subplots(figsize=(10, 6.5))
for nombre, grupo in nube.groupby("familia"):
    eje.scatter(grupo["CP1"], grupo["CP2"], s=45, color=colores[nombre], label=f"{nombre} ({len(grupo)})", alpha=0.85)
for ingrediente in receta:
    x, y = nube.loc[ingrediente, ["CP1", "CP2"]]
    eje.scatter(x, y, s=190, facecolor="none", edgecolor="black", lw=1.6)
    eje.annotate(ingrediente, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=8, weight="bold")
var_p = pca_productos.explained_variance_ratio_ * 100
eje.set(title=f"Cada punto es un producto de la Galería ({len(nube)} productos)", xlabel=f"CP1 ({var_p[0]:.0f} %)", ylabel=f"CP2 ({var_p[1]:.0f} %)")
eje.legend(fontsize=8, loc="lower left"); plt.show()
print("Varianza explicada por los tres primeros componentes (%):", var_p.round(1))
'''),
("md", r'''
Nadie le dijo al PCA qué es carne y qué es fruta, y aun así las familias aparecen. Las carnes de res quedan juntas en una esquina, las de cerdo a su lado y los huevos en otra. Las frutas y las verduras se riegan por la parte baja. Los cinco ingredientes del sancocho, marcados con un círculo, caen en zonas distintas. Esa es la versión geométrica de lo que dijo la matriz de correlaciones: cada ingrediente tiene su propia historia, y por eso la olla promedia riesgos en lugar de sumarlos.

Dos advertencias. Los dos primeros componentes explican poca varianza, así que el plano es una sombra de una nube que vive en muchas dimensiones. Y la cercanía en el plano dice que dos productos se mueven parecido, sin decir por qué.

La celda siguiente dibuja la misma nube con el tercer componente. La gráfica se puede girar con el ratón.
'''),
("code", r'''
try:
    import plotly.express as px
    figura = px.scatter_3d(nube.reset_index(), x="CP1", y="CP2", z="CP3", color="familia", hover_name="producto",
                           color_discrete_map=colores, opacity=0.85, height=560, title="Los productos de la Galería en tres componentes")
    figura.update_traces(marker_size=4); figura.show()
except ImportError:
    eje = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
    eje.scatter(nube["CP1"], nube["CP2"], nube["CP3"], c=nube["familia"].map(colores)); plt.show()
'''),
("md", r'''
El PCA aquí sirve para describir. No lo vamos a usar como entrada del modelo.
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base

Todo modelo se mide contra el pronóstico más barato posible. Con horizonte $h=3$ meses hay dos candidatos que no cuestan nada:

$$\hat C^{\text{ing}}_{t+h}=C_t\qquad\text{y}\qquad \hat C^{\text{est}}_{t+h}=C_{t+h-12}.$$

El primero dice que dentro de tres meses costará lo mismo que hoy. El segundo, que costará lo mismo que ese mes del año pasado. El error se mide con

$$\text{MAPE}=\frac{100}{n}\sum_{t}\left|\frac{C_{t+h}-\hat C_{t+h}}{C_{t+h}}\right|.$$

Si un modelo sofisticado no les gana, no sirve, por elegante que sea.
'''),
("code", r'''
H = 3                                             # horizonte: tres meses
tabla = pd.DataFrame({"real": olla.shift(-H)})    # lo que queremos adivinar: el costo en t+3
tabla["ingenuo"] = olla                           # lo que sabemos hoy
tabla["estacional"] = olla.shift(12 - H)          # el mismo mes objetivo, un año antes
tabla = tabla.dropna()

def mape(real, pronostico):
    return float((abs(real - pronostico) / real).mean() * 100)

for metodo in ["ingenuo", "estacional"]:
    print(f"MAPE {metodo}: {mape(tabla['real'], tabla[metodo]):.1f} %")
'''),
("md", r'''
Ya hay una cifra que vencer. Comparen con su tercera apuesta.

### 3.2 La trampa de barajar

Empecemos por el modelo más simple, una regresión lineal sobre el costo de hoy, el de los dos meses anteriores y el del mismo mes objetivo un año atrás:

$$C_{t+h}=\beta_0+\beta_1C_t+\beta_2C_{t-1}+\beta_3C_{t-2}+\beta_4C_{t+h-12}+\varepsilon_{t+h}.$$
'''),
("code", r'''
X = pd.DataFrame({
    "hoy": olla,
    "hace_1": olla.shift(1),
    "hace_2": olla.shift(2),
    "anio_pasado": olla.shift(12 - H),
})
datos = X.join(olla.shift(-H).rename("real")).dropna()
variables = list(X.columns)
print(len(datos), "meses utilizables, de", datos.index.min().date(), "a", datos.index.max().date())
'''),
("md", r'''
Ahora viene la decisión que más errores causa en la práctica: **cómo separar entrenamiento y prueba**. Hagámoslo de las dos maneras y comparemos.

- **Barajando:** se toma al azar el 25 % de los meses para probar. Es lo que hace `train_test_split` por defecto.
- **Respetando el tiempo:** se entrena con todo hasta diciembre de 2021 y se prueba con 2022 en adelante.

**Predicción escrita:** ¿cuál de las dos dará un error más pequeño, y por qué? **Mi apuesta:**
'''),
("code", r'''
from sklearn.model_selection import train_test_split

# Manera 1: barajando. Se repite 200 veces para que el resultado no dependa de un sorteo con suerte.
barajados = []
for semilla in range(200):
    X_a, X_b, y_a, y_b = train_test_split(datos[variables], datos["real"], test_size=0.25, random_state=semilla)
    barajados.append(mape(y_b, LinearRegression().fit(X_a, y_a).predict(X_b)))
mape_barajado = float(np.mean(barajados))

# Manera 2: respetando el tiempo
antes, despues = datos.loc[:"2021-12-01"], datos.loc["2022-01-01":]
mape_temporal = mape(despues["real"], LinearRegression().fit(antes[variables], antes["real"]).predict(despues[variables]))

print(f"MAPE barajando (promedio de 200 sorteos): {mape_barajado:.1f} %")
print(f"MAPE respetando el tiempo:                 {mape_temporal:.1f} %")
'''),
("md", r'''
Los dos números miden el mismo modelo, así que la diferencia viene del examen. Al barajar, el modelo se entrena con meses de 2023 y lo examinan con meses de 2022: ya vio el futuro. Ese examen no se parece a lo que le pasará al dueño del negocio, que siempre pronostica hacia adelante. De aquí en adelante solo vale la validación que respeta el tiempo.

### 3.3 Cinco modelos y un protocolo para escoger

Una regresión es un candidato entre muchos. Vamos a poner a competir cinco familias:

| Modelo | Idea en una línea |
|---|---|
| Regresión lineal | $\hat y=\beta_0+x^\top\beta$, con $\beta$ que minimiza $\sum(y-\hat y)^2$ |
| Ridge | Lo mismo, con castigo a los coeficientes grandes: $\min_\beta\sum(y-\hat y)^2+\alpha\lVert\beta\rVert_2^2$ |
| $k$ vecinos | Promedia el resultado de los $k$ meses pasados más parecidos al de hoy |
| Bosque aleatorio | Promedia muchos árboles de decisión entrenados sobre remuestreos del dato |
| *Gradient boosting* | Suma árboles pequeños, cada uno corrigiendo el error del anterior |

Si escogemos al ganador mirando la prueba, la prueba deja de ser un examen limpio: de cinco candidatos, alguno saldrá bien por suerte. Por eso el tiempo se parte en **tres** tramos:

$$\underbrace{2013\ \dots\ 2019}_{\text{entrenamiento}}\ \Big|\ \underbrace{2020\ \dots\ 2021}_{\text{validación: se escoge}}\ \Big|\ \underbrace{2022\ \dots\ 2024}_{\text{prueba: se reporta}}$$

Hay un detalle fino. La fila del mes $t$ lleva como respuesta el costo de $t+h$. Para que el entrenamiento no use respuestas que caen dentro del tramo siguiente, solo entran las filas con $t+h$ anterior al corte.
'''),
("code", r'''
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def cinco_modelos():
    return {
        "lineal": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "bosque": RandomForestRegressor(n_estimators=300, random_state=0),
        "boosting": GradientBoostingRegressor(random_state=0),
    }

objetivo = datos.index + pd.DateOffset(months=H)                 # el mes que cada fila intenta adivinar
E = objetivo <= pd.Timestamp("2019-12-01")                        # entrenamiento
V = (datos.index >= "2020-01-01") & (objetivo <= pd.Timestamp("2021-12-01"))   # validación
P = datos.index >= "2022-01-01"                                   # prueba
print(f"Entrenamiento: {E.sum()} meses | Validación: {V.sum()} | Prueba: {P.sum()}")

def competir(entradas, respuesta, a_pesos):
    """Mide cada modelo en validación; luego lo reentrena con entrenamiento y validación y lo mide en prueba."""
    errores, pronosticos = {}, {}
    for nombre, modelo in cinco_modelos().items():
        modelo.fit(entradas[E], respuesta[E])
        en_validacion = mape(datos["real"][V], a_pesos(modelo.predict(entradas[V]), V))
        modelo.fit(entradas[E | V], respuesta[E | V])
        pronosticos[nombre] = pd.Series(a_pesos(modelo.predict(entradas[P]), P), index=datos.index[P])
        errores[nombre] = {"validación": en_validacion, "prueba": mape(datos["real"][P], pronosticos[nombre])}
    return pd.DataFrame(errores).T, pd.DataFrame(pronosticos)

errores_nivel, pron_nivel = competir(datos[variables], datos["real"], lambda pred, tramo: pred)
bases = pd.DataFrame({m: {"validación": mape(datos["real"][V], tabla.loc[datos.index[V], m]),
                          "prueba": mape(datos["real"][P], tabla.loc[datos.index[P], m])} for m in ["ingenuo", "estacional"]}).T
display(pd.concat([bases, errores_nivel]).round(1))
'''),
("code", r'''
fechas = datos.index[P] + pd.DateOffset(months=H)
plt.figure(figsize=(11, 4))
plt.plot(fechas, datos["real"][P], color="black", lw=2.5, label="real")
plt.plot(fechas, tabla.loc[datos.index[P], "ingenuo"], color="#1B998B", lw=1.5, label="ingenuo")
for nombre in pron_nivel:
    plt.plot(fechas, pron_nivel[nombre], lw=1, ls="--", label=nombre)
plt.axhline(datos["real"][E | V].max(), color="gray", ls=":")
plt.text(fechas[0], datos["real"][E | V].max() * 1.005, "máximo visto antes de 2022", fontsize=8, color="gray")
plt.title("Pronóstico a tres meses en la prueba"); plt.ylabel("pesos"); plt.legend(fontsize=8, ncol=4); plt.show()
'''),
("md", r'''
Miren las líneas del bosque, del *boosting* y de los vecinos contra la línea punteada. Un árbol predice promediando respuestas que ya vio, y un vecino también. Ninguno puede pronosticar un costo más alto que el máximo de su entrenamiento. La olla de 2022 a 2024 cuesta más que cualquier olla anterior, así que esos modelos se quedan pegados al techo. Los modelos lineales sí extrapolan.

Esto tiene arreglo, y el arreglo es una decisión de formulación. En vez de pedirles el **nivel** del costo, les pedimos el **cambio relativo**:

$$y_t=\frac{C_{t+h}}{C_t},\qquad \hat C_{t+h}=\hat y_t\,C_t.$$

El cambio relativo de 2023 se parece a los cambios relativos de 2015, aunque los precios sean otros. Las entradas también se dividen por el costo de hoy.
'''),
("code", r'''
relativas = datos[variables].div(datos["hoy"], axis=0).drop(columns="hoy")
relativas["mes_sen"] = np.sin(2 * np.pi * datos.index.month / 12)
relativas["mes_cos"] = np.cos(2 * np.pi * datos.index.month / 12)
razon = datos["real"] / datos["hoy"]

errores_cambio, pron_cambio = competir(relativas, razon, lambda pred, tramo: pred * datos["hoy"][tramo].values)

todos = pd.concat([bases,
                   errores_nivel.rename(index=lambda n: n + " (nivel)"),
                   errores_cambio.rename(index=lambda n: n + " (cambio)")])
todos.sort_values("validación").plot.barh(figsize=(9, 5), color=["#1D4E89", "#1B998B"])
plt.xlabel("MAPE (%)"); plt.title("Doce candidatos, y faltan dos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(1))
'''),
("md", r'''
### 3.4 Modelos hechos para series de tiempo

Los cinco modelos anteriores no saben que los datos están ordenados: reciben una fila y devuelven un número. Hay modelos que parten del orden. Probamos dos clásicos.

**Holt-Winters** descompone la serie en nivel $\ell_t$, tendencia $b_t$ y estación $s_t$, y actualiza cada parte con un suavizado exponencial. Su pronóstico a $h$ pasos es

$$\hat c_{t+h}=\ell_t+(\phi+\phi^2+\dots+\phi^h)\,b_t+s_{t+h-12},$$

donde $\phi<1$ amortigua la tendencia para que no se dispare.

**SARIMA** modela las variaciones. Con $B$ el operador de rezago, $Bc_t=c_{t-1}$, el modelo $(1,1,1)(0,1,1)_{12}$ es

$$(1-\varphi B)(1-B)(1-B^{12})\,c_t=(1+\theta B)(1+\Theta B^{12})\,\varepsilon_t.$$

El factor $(1-B)$ toma la diferencia mensual, que es lo que la prueba de la sección 2.2 pidió. El factor $(1-B^{12})$ quita la estación.

Estos modelos se evalúan con **origen móvil**: para pronosticar desde el mes $t$ se ajustan solo con los datos hasta $t$, y así mes a mes. Es más exigente con el computador y más fiel a como se usarían. La celda tarda cerca de un minuto.
'''),
("code", r'''
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def holt_winters(serie, pasos=H):
    ajuste = ExponentialSmoothing(np.log(serie), trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit()
    return float(np.exp(ajuste.forecast(pasos).iloc[-1]))

def sarima(serie, pasos=H):
    ajuste = SARIMAX(np.log(serie), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    return float(np.exp(ajuste.forecast(pasos).iloc[-1]))

series_de_tiempo = {"Holt-Winters": holt_winters, "SARIMA": sarima}
olla_m = olla.asfreq("MS")
pron_series, filas = {}, {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for nombre, metodo in series_de_tiempo.items():
        desde_cada_mes = pd.Series({t: metodo(olla_m[:t]) for t in datos.index[V | P]})   # origen móvil
        pron_series[nombre] = desde_cada_mes[datos.index[P]]
        filas[nombre] = {"validación": mape(datos["real"][V], desde_cada_mes[datos.index[V]]),
                         "prueba": mape(datos["real"][P], desde_cada_mes[datos.index[P]])}
pron_series = pd.DataFrame(pron_series)
todos = pd.concat([todos, pd.DataFrame(filas).T])
display(todos.loc[["ingenuo", "estacional", "Holt-Winters", "SARIMA"]].round(1))
'''),
("code", r'''
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    completo = SARIMAX(np.log(olla_m), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    futuro = completo.get_forecast(6)
centro = np.exp(futuro.predicted_mean)
banda = np.exp(futuro.conf_int(alpha=0.20))

plt.figure(figsize=(11, 4))
plt.plot(olla_m["2021":], color="#1D4E89", lw=2, label="costo observado")
plt.plot(np.exp(completo.fittedvalues)["2021":], color="#1B998B", lw=1, ls="--", label="SARIMA, un paso adelante")
plt.plot(centro, color="#D7263D", lw=2, label="pronóstico a seis meses")
plt.fill_between(banda.index, banda.iloc[:, 0], banda.iloc[:, 1], color="#D7263D", alpha=0.18, label="franja de 80 %")
plt.title("El modelo sigue la curva hacia atrás y abre un abanico hacia adelante"); plt.ylabel("pesos"); plt.legend(fontsize=8); plt.show()
'''),
("md", r'''
Esta gráfica tiene una trampa visual que conviene nombrar. La línea punteada va pegada a la curva observada, y uno diría que el modelo es excelente. Pero cada punto de esa línea se calculó conociendo el mes anterior. El pronóstico ingenuo dibujaría una línea igual de pegada, corrida un mes. **Seguir la curva hacia atrás es fácil. Lo que cuesta es el abanico rojo**, que es donde el modelo ya no ve el dato. Miren qué tan ancho se pone en seis meses.

### 3.5 ¿Y una red neuronal?

Ustedes vieron redes recurrentes en Aprendizaje Profundo, así que la pregunta es natural. Una LSTM recibe las doce últimas variaciones mensuales y predice el cambio de los tres meses siguientes.

**Predicción escrita:** ¿le ganará la red al pronóstico ingenuo? ¿Con cuántas filas se entrena? **Mi apuesta:**
'''),
("code", r'''
import torch
from torch import nn

variacion = np.log(olla).diff().dropna()
ventanas, respuestas, origenes = [], [], []
for i in range(12, len(variacion) - H + 1):
    ventanas.append(variacion.values[i - 12:i]); respuestas.append(variacion.values[i:i + H].sum()); origenes.append(variacion.index[i - 1])
ventanas = torch.tensor(np.array(ventanas), dtype=torch.float32).unsqueeze(-1) * 10     # se escala para que la red entrene mejor
respuestas = torch.tensor(np.array(respuestas), dtype=torch.float32) * 10
origenes = pd.DatetimeIndex(origenes); destino_red = origenes + pd.DateOffset(months=H)
E_red = destino_red <= pd.Timestamp("2019-12-01")
V_red = (origenes >= "2020-01-01") & (destino_red <= pd.Timestamp("2021-12-01"))
P_red = origenes >= "2022-01-01"

class Recurrente(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm, self.salida = nn.LSTM(1, 16, batch_first=True), nn.Linear(16, 1)
    def forward(self, x):
        return self.salida(self.lstm(x)[0][:, -1]).squeeze(-1)

resultados = []
for semilla in range(5):
    torch.manual_seed(semilla)
    red = Recurrente(); optimizador = torch.optim.Adam(red.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(300):
        optimizador.zero_grad(); perdida = ((red(ventanas[E_red]) - respuestas[E_red]) ** 2).mean(); perdida.backward(); optimizador.step()
    with torch.no_grad():
        pronostico_red = olla.reindex(origenes).values * np.exp(red(ventanas).numpy() / 10)
    real_red = olla.reindex(destino_red).values
    resultados.append({"semilla": semilla, "validación": mape(real_red[V_red], pronostico_red[V_red]), "prueba": mape(real_red[P_red], pronostico_red[P_red])})
resultados = pd.DataFrame(resultados).set_index("semilla")
print(f"Filas de entrenamiento: {int(E_red.sum())} | parámetros de la red: {sum(p.numel() for p in red.parameters())}")
display(resultados.round(1))
print(f"MAPE promedio de la red en prueba: {resultados['prueba'].mean():.1f} % | ingenuo: {todos.loc['ingenuo', 'prueba']:.1f} %")
'''),
("md", r'''
Comparen el número de filas de entrenamiento con el número de parámetros de la red. Hay muchos más parámetros que datos. La red pierde contra el pronóstico ingenuo, y además cambia de resultado con la semilla, que es otra señal de que no hay dato suficiente para fijarla.

La lección no es que las redes sean malas. Una red recurrente o un *transformer* necesitan miles de series o series muy largas. Con 144 meses de una sola serie, un modelo con tres parámetros como SARIMA tiene ventaja porque trae la estructura puesta de antemano. La red no entra a la competencia por eso: dejarla concursar con cinco semillas sería comprar cinco boletas.
'''),
("md", r'''
### 3.6 El veredicto

La regla quedó fijada antes de mirar: gana quien tenga menor MAPE en **validación**, y su error en **prueba** es el que se le reporta al cliente.
'''),
("code", r'''
mejor = todos["validación"].idxmin()
resumen = todos["prueba"].round(1)
print(f"Ganador en validación: {mejor}")
print(f"Su MAPE en prueba: {resumen[mejor]} %   |   MAPE del ingenuo en prueba: {resumen['ingenuo']} %")

pronosticos_prueba = pd.concat([tabla.loc[datos.index[P], ["ingenuo", "estacional"]],
                                pron_nivel.rename(columns=lambda n: n + " (nivel)"),
                                pron_cambio.rename(columns=lambda n: n + " (cambio)"), pron_series], axis=1)
plt.figure(figsize=(11, 3.8))
plt.plot(fechas, datos["real"][P], color="black", lw=2.5, label="real")
plt.plot(fechas, pronosticos_prueba["ingenuo"], color="#1B998B", label="ingenuo")
if mejor != "ingenuo":
    plt.plot(fechas, pronosticos_prueba[mejor], color="#D7263D", ls="--", label=mejor)
plt.title("El ganador contra la línea base"); plt.ylabel("pesos"); plt.legend(); plt.show()
'''),
("md", r'''
Lean el resultado con calma. Puede pasar que el ganador de validación pierda en prueba contra el pronóstico ingenuo. Eso no es un error del procedimiento: es el procedimiento diciendo la verdad sobre un dato con poca memoria en sus variaciones, como vimos en la sección 2.1. Un modelo que empata con la línea base solo agrega complejidad, y la respuesta honesta para el cliente es la más simple que no pierde.

**¿La diferencia es real?** Dos MAPE separados por medio punto pueden ser la misma cosa. Para cada mes de la prueba tengo el error porcentual absoluto de un candidato, $a_t$, y el del ingenuo, $b_t$. La prueba de rangos con signo de Wilcoxon contrasta

$$H_0:\ \text{la mediana de } a_t-b_t \text{ es } 0\qquad\text{contra}\qquad H_1:\ \text{no lo es}.$$
'''),
("code", r'''
from scipy.stats import wilcoxon

error = lambda metodo: (datos["real"][P] - pronosticos_prueba[metodo]).abs() / datos["real"][P] * 100
filas_prueba = {}
for candidato in dict.fromkeys([mejor, "SARIMA", "Holt-Winters", "lineal (cambio)"]):
    if candidato != "ingenuo":
        filas_prueba[candidato] = {"MAPE en prueba": error(candidato).mean(), "MAPE del ingenuo": error("ingenuo").mean(),
                                   "valor p": wilcoxon(error(candidato), error("ingenuo")).pvalue}
display(pd.DataFrame(filas_prueba).T.round(3))
print(f"Meses en la prueba: {int(P.sum())}")
'''),
("md", r'''
Lean la columna del valor $p$ con el número de meses al lado. Con tan pocos meses, una ventaja de medio punto no alcanza a distinguirse del azar. La prueba además es generosa, porque los errores de meses vecinos se parecen entre sí y eso hace ver más evidencia de la que hay. La versión cuidadosa de este contraste se llama prueba de Diebold-Mariano.

La conclusión honesta tiene dos partes. Varios modelos quedan por debajo del ingenuo en la prueba, y entre ellos están los dos de series de tiempo. Pero el dato no alcanza para declarar ganador a ninguno.

### 3.7 Explicar: ¿qué ingrediente mueve el costo?

Como el costo es una suma, el alza total se reparte de forma exacta entre ingredientes:

$$C_T-C_1=\sum_{i=1}^{5}q_i\,(p_{i,T}-p_{i,1}).$$
'''),
("code", r'''
participacion = (costo_por_ingrediente.iloc[-1] / olla.iloc[-1] * 100).round(1).sort_values(ascending=False)
print("Participación en el costo, diciembre de 2024 (%):")
print(participacion)

alza = costo_por_ingrediente.iloc[-1] - costo_por_ingrediente.iloc[0]
print("\nDe cada 100 pesos que subió la olla entre 2013 y 2024, cuántos puso cada ingrediente:")
print((alza / alza.sum() * 100).round(1).sort_values(ascending=False))

costo_por_ingrediente.plot.area(lw=0, color=PALETA[:5], alpha=0.9)
plt.title("Costo de la olla por ingrediente"); plt.xlabel(""); plt.ylabel("pesos"); plt.legend(fontsize=8); plt.show()
'''),
("md", r'''
Esta explicación es una descomposición contable: dice cuánto pesa cada ingrediente, no por qué subió su precio. Para el porqué harían falta otros datos, como el precio del maíz que come el cerdo o el clima de la cosecha.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿con cuánto se presupuesta la olla?

Un pronóstico de un solo número deja al cliente sin saber cuánto puede fallar. Usamos los errores de la prueba para ponerle un margen. Sea $u_t=C_{t+h}/\hat C_{t+h}-1$ lo que el costo real superó al pronóstico. El margen es el cuantil 0,90 de esos valores, y el costo para presupuestar es

$$\hat C_{T+h}\,\bigl(1+\hat Q_{0{,}90}(u)\bigr).$$

Escogemos el método más simple entre los que no pierden: si el ganador de validación no mejora al ingenuo en la prueba, nos quedamos con el ingenuo.
'''),
("code", r'''
elegido = mejor if resumen[mejor] < resumen["ingenuo"] else "ingenuo"
u = datos["real"][P] / pronosticos_prueba[elegido] - 1
margen = u.quantile(0.90)

hoy = olla.index[-1]
ultima = X.iloc[[-1]]
if elegido == "ingenuo":
    punto = olla.iloc[-1]
elif elegido == "estacional":
    punto = olla.shift(12 - H).iloc[-1]
elif elegido in series_de_tiempo:
    punto = series_de_tiempo[elegido](olla_m)
else:
    nombre, tipo = elegido.rsplit(" (", 1)
    modelo = cinco_modelos()[nombre]
    if tipo.startswith("nivel"):
        punto = float(modelo.fit(datos[variables], datos["real"]).predict(ultima[variables])[0])
    else:
        ultima_rel = ultima[variables].div(ultima["hoy"], axis=0).drop(columns="hoy")
        ultima_rel["mes_sen"] = np.sin(2 * np.pi * hoy.month / 12); ultima_rel["mes_cos"] = np.cos(2 * np.pi * hoy.month / 12)
        punto = float(modelo.fit(relativas, razon).predict(ultima_rel)[0]) * olla.iloc[-1]

meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
destino_mes = hoy + pd.DateOffset(months=H)
print(f"Método elegido: {elegido} (MAPE en prueba {resumen[elegido]} %)")
print(f"Pronóstico para {meses[destino_mes.month - 1]} de {destino_mes.year}: ${punto:,.0f}")
print(f"En 9 de cada 10 meses de prueba, el costo real no superó al pronóstico en más de {margen * 100:.1f} %")
print(f"Costo para presupuestar: ${punto * (1 + margen):,.0f}")
'''),
("md", r'''
### La decisión en tres frases

Escríbanla para alguien que no sabe qué es una regresión. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Variedades de la receta (papa capira, yuca ICA, plátano dominico hartón) | Otras variedades | Se reportan todos los meses en la Galería |
| Horizonte de tres meses | Un mes, seis meses | Es lo que dura la carta impresa |
| MAPE como métrica | Error en pesos | El cliente piensa en márgenes |
| Tres tramos temporales: hasta 2019, 2020 a 2021 y 2022 a 2024 | Separación al azar; escoger mirando la prueba | Al barajar, el modelo ve el futuro; si se escoge en la prueba, la prueba deja de ser examen |
| Pronosticar el cambio relativo además del nivel | Solo el nivel | Árboles y vecinos no extrapolan por encima de lo visto |
| Margen con el percentil 90 de los meses de prueba | Intervalo teórico | Sale de errores observados y se explica sin fórmulas |
| La red neuronal no concursa | Incluirla entre los candidatos | 69 filas de entrenamiento; cinco semillas serían cinco boletas |
| Familias de productos por nombre, para colorear el PCA | Sin colorear | Ayuda a leer la nube; el PCA no usa la familia |

### 4.3 Riesgos y mantenimiento

Respondan en dos o tres líneas cada una.

- El dato llega hasta diciembre de 2024. ¿Qué habría que hacer para usar esto hoy?
- El precio es mayorista. ¿En qué se equivoca el cliente si lo lee como el precio de la tienda?
- ¿Cada cuánto habría que revisar si el pronóstico ingenuo sigue siendo el mejor?

### 4.4 Procedencia

Lo último que hace un cuaderno serio es decir de dónde salió su dato.
'''),
("code", r'''
procedencia()
'''),
]

# ------------------------------------------------------------ EL PRECIO DE LA LUZ
LUZ = [
("md", r'''
# Clase 1 · El precio de la luz (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *El sancocho*: la misma técnica en otro mundo.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase: formular, limpiar, explorar, comparar cinco modelos contra una línea base con validación temporal y convertir el resultado en una decisión. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (seleccionar y preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar) y 5 (comunicar). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que les ayude a llenar un hueco o que interprete un error. Deben poder explicar sin ayuda la regla con que se escogió la versión del precio, por qué los tramos son temporales, por qué el ganador se escoge en validación y qué significa su MAPE. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, la decisión en tres frases y la bitácora llenas. |

**Cómo se trabaja.** El código está escrito casi todo. Donde vean `____` falta una pieza, y el comentario de al lado dice qué va. Lean la celda entera antes de llenar. Si una celda falla, lean el error antes de pedir ayuda.
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

En Colombia la energía se compra y se vende cada hora en un mercado que administra XM. El precio de ese mercado se llama **precio de bolsa** y se mide en pesos por kilovatio hora. La explicación que uno oye siempre es que el precio sube cuando los embalses bajan, porque hay que prender plantas térmicas, que son más caras. Hoy vamos a ver cuánto de esa explicación aguanta el dato.

Imagínense una empresa mediana de Manizales que gasta mucha energía. Puede comprarla al precio de bolsa, que cambia todos los días, o firmar un precio fijo. Su gerente necesita tres respuestas:

1. **Predecir.** ¿Cuál será el precio de bolsa promedio de mañana? ¿Y el de dentro de siete días?
2. **Explicar.** ¿Cuánto del precio se entiende mirando el nivel de los embalses?
3. **Decidir.** Si le ofrecen un precio fijo, ¿desde qué valor le conviene firmar? ¿Cambia la respuesta si el contrato es para mañana o para la otra semana?

**La métrica** es la misma de la clase: el MAPE del pronóstico, primero a un día y después a siete.

### Predicción escrita

Escriban su apuesta antes de correr nada.

- ¿Cuánto vale hoy un kilovatio hora en bolsa, en pesos? **Mi apuesta:**
- ¿Qué tan lejos queda el precio de mañana del precio de hoy, en porcentaje? ¿Y el de dentro de siete días? **Mi apuesta:**
- Cuando los embalses están llenos, ¿el precio es alto o bajo? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

Pedimos dos tablas a XM: el precio de bolsa hora por hora y la energía guardada en cada embalse día por día, de enero de 2023 a octubre de 2026. **La descarga tarda entre cinco y seis minutos** porque XM entrega un mes por consulta. Córranla y lean la sección mientras termina.
'''),
("code", r'''
precio_crudo = descargar("xm_precio")
embalses_crudo = descargar("xm_embalses")
precio_crudo.head(8)
'''),
("md", r'''
### 2.1 La primera trampa: cada hora aparece muchas veces

Miren la tabla de arriba. La misma hora sale repetida. Averigüemos por qué.
'''),
("code", r'''
print("Filas:", len(precio_crudo), "| Horas distintas:", precio_crudo["FechaHora"].nunique())
print(precio_crudo.groupby(["CodigoVariable", "Version"]).size().unstack())
'''),
("md", r'''
Hay tres variables y hasta diez versiones:

- `PB_Nal` es el precio de bolsa nacional. `PB_Int` y `PB_Tie` son precios para intercambios con otros países. A la empresa le interesa el nacional.
- XM liquida cada hora varias veces. Las versiones numeradas van de `TX1` a `TX8`: casi todas las horas tienen `TX1` y muy pocas llegan a `TX8`, así que el número indica cuántas veces se ha vuelto a liquidar esa hora. Además están `TXF` y `TXR`, que existen para todas las horas.

Si promediaran la tabla tal como viene, mezclarían precios internacionales con nacionales y contarían la misma hora varias veces. Además, las versiones no siempre coinciden. Miren un día concreto:
'''),
("code", r'''
un_dia = precio_crudo[(precio_crudo["CodigoVariable"] == "PB_Nal") & precio_crudo["FechaHora"].str.startswith("2024-09-30")]
print(un_dia.groupby("Version")["Valor"].mean().round(0))
'''),
("md", r'''
El mismo día vale más de 7.000 pesos según unas versiones y menos de 2.500 según otras. Quien escoja otra versión estará modelando otro precio.

**Regla que vamos a usar:** nos quedamos con el precio nacional y, para cada hora, con la liquidación numerada más reciente que exista (`TX8` si la hay, si no `TX7`, y así hasta `TX1`). Solo si una hora no tiene ninguna numerada usamos `TXF`. La razón es que una liquidación posterior corrige a las anteriores.

Esta regla es una decisión de modelación que tomé mirando el dato, sin tener a la mano la definición oficial de cada sigla. Anótenla en la bitácora y, en la sección de riesgos, digan qué habría que consultarle a XM.
'''),
("code", r'''
nacional = precio_crudo[precio_crudo["CodigoVariable"] == «"PB_Nal"»].copy()   # HUECO 1: el código del precio nacional, entre comillas

preferencia = ["TX8", "TX7", "TX6", "TX5", "TX4", "TX3", "TX2", "TX1", "TXF", "TXR"]
nacional["orden"] = nacional["Version"].map({v: i for i, v in enumerate(preferencia)})
horario = (nacional.sort_values(["FechaHora", "orden"])
           .drop_duplicates("FechaHora", keep=«"first"»))                       # HUECO 2: "first" o "last". ¿Cuál deja la versión preferida?

print("Horas después de limpiar:", len(horario), "| ¿Alguna hora repetida?", horario["FechaHora"].duplicated().any())
print(horario["Version"].value_counts().sort_index())
'''),
("md", r'''
**Compruébenlo ustedes mismos.** El número de horas después de limpiar debe ser igual al número de horas distintas que imprimieron antes. Si no coincide, revisen los huecos 1 y 2.

### 2.2 Del precio horario al precio diario
'''),
("code", r'''
horario["fecha"] = pd.to_datetime(horario["FechaHora"]).dt.normalize()
precio = horario.groupby("fecha")["Valor"].«mean»().rename("precio")           # HUECO 3: la operación que resume las 24 horas en un promedio

ax = precio.plot(color="#1D4E89", lw=1)
ax.set(title="Precio de bolsa nacional, promedio diario", xlabel="", ylabel="COP/kWh"); plt.show()
print(precio.describe().round(1))
'''),
("md", r'''
Comparen con su primera apuesta. ¿Qué pasó entre finales de 2023 y abril de 2024? Escríbanlo en una línea: **Mi explicación:**

### 2.3 El nivel de los embalses

La tabla trae, por embalse y por día, la energía útil guardada y la capacidad útil. El nivel agregado del país es la suma de lo guardado sobre la suma de la capacidad.

**Segunda trampa: filas que ya son sumas.** Miren los nombres de los embalses.
'''),
("code", r'''
print(sorted(embalses_crudo["CodigoEmbalse"].unique()))
'''),
("md", r'''
Tres nombres no son embalses. `AGREGADO_SIN` es el total del país, que XM publica desde julio de 2025. `AGREGADO_BOGOTA`, que desde enero de 2025 aparece como `AGREGADO`, es la suma de Neusa, Sisga y Tominé, que hasta junio de 2025 también vienen por separado. Si sumaran todas las filas contarían esa agua dos y hasta tres veces.

La celda siguiente quita las filas repetidas y calcula el nivel. Después lo compara con el total que publica XM, para saber si la regla quedó bien.
'''),
("code", r'''
embalses_crudo["fecha"] = pd.to_datetime(embalses_crudo["Fecha"])
repetidas = ["AGREGADO_SIN", "NEUSA", "SISGA", "TOMINE"]
sin_repetir = embalses_crudo[~embalses_crudo["CodigoEmbalse"].isin(repetidas)]
por_dia = sin_repetir.groupby("fecha")[["VolumenUtilDiarioEnergia", "CapacidadUtilEnergia"]].sum()
embalse = (por_dia["VolumenUtilDiarioEnergia"] / por_dia[«"CapacidadUtilEnergia"»] * 100).rename("embalse")   # HUECO 4: la columna que va en el denominador

oficial = embalses_crudo[embalses_crudo["CodigoEmbalse"] == "AGREGADO_SIN"].set_index("fecha")
oficial = oficial["VolumenUtilDiarioEnergia"] / oficial["CapacidadUtilEnergia"] * 100
diferencia = (embalse.reindex(oficial.index) - oficial).abs().max()
print(f"Diferencia máxima con el total de XM, en {len(oficial)} días comparables: {diferencia:.3f} puntos")

datos = pd.concat([precio, embalse], axis=1).dropna()
print(len(datos), "días, de", datos.index.min().date(), "a", datos.index.max().date())
datos["embalse"].plot(color="#5BC0EB"); plt.title("Nivel agregado de los embalses (%)"); plt.xlabel(""); plt.show()
'''),
("md", r'''
### 2.4 Conocer el dato antes de modelarlo

Cuatro miradas rápidas: a qué horas es cara la energía, si el día de la semana importa, cómo se reparten los precios y cuánta memoria tiene la serie. Llamo $P_t$ al precio promedio del día $t$.
'''),
("code", r'''
horario["hora"] = pd.to_datetime(horario["FechaHora"]).dt.hour
dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]

fig, ejes = plt.subplots(2, 2, figsize=(12, 7))
horario.groupby("hora")["Valor"].«median»().plot(ax=ejes[0, 0], color="#1D4E89", marker="o", ms=3)      # HUECO 5: un resumen que no se deje arrastrar por los días extremos
ejes[0, 0].set(title="Precio mediano por hora del día", xlabel="hora", ylabel="COP/kWh")

por_dia_semana = datos["precio"].groupby(datos.index.dayofweek).median()
ejes[0, 1].bar(dias, por_dia_semana.values, color="#1B998B"); ejes[0, 1].set(title="Precio mediano por día de la semana")

ejes[1, 0].hist(np.log10(datos["precio"]), bins=40, color="#D7263D")
ejes[1, 0].set(title="Distribución del precio diario (escala logarítmica)", xlabel="log10 del precio")

rezagos = range(1, 31)
ejes[1, 1].bar(rezagos, [datos["precio"].autocorr(k) for k in rezagos], color="#1D4E89")
ejes[1, 1].set(title="Autocorrelación del precio diario", xlabel="rezago (días)")
plt.tight_layout(); plt.show()
print(f"Media: {datos['precio'].mean():.0f} | Mediana: {datos['precio'].median():.0f} | Máximo: {datos['precio'].max():.0f} COP/kWh")
'''),
("md", r'''
Respondan en una línea cada una.

- ¿A qué hora es más cara la energía, y por qué creen? **Mi lectura:**
- La media y la mediana no coinciden. ¿Qué dice eso de la forma de la distribución? **Mi lectura:**
- La autocorrelación de orden $k$ es $\hat\rho_k=\dfrac{\sum_t (P_t-\bar P)(P_{t-k}-\bar P)}{\sum_t (P_t-\bar P)^2}$. ¿El precio de hoy se parece al de ayer? ¿Y al de hace un mes? **Mi lectura:**

### 2.5 PCA: ¿los embalses se llenan y se vacían juntos?

En clase usamos PCA para ver si los ingredientes se movían juntos. Aquí la pregunta es la misma con los embalses. Tomo el porcentaje de llenado de cada embalse, lo estandarizo y descompongo la matriz de correlaciones:

$$R=V\Lambda V^\top,\qquad \text{varianza explicada por el componente } j=\frac{\lambda_j}{\sum_k\lambda_k}.$$

**Predicción escrita:** hay más de veinte embalses. Si todos dependieran de la misma lluvia, ¿cuánto explicaría el primer componente? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

llenado = sin_repetir.assign(pct=sin_repetir["VolumenUtilDiarioEnergia"] / sin_repetir["CapacidadUtilEnergia"] * 100)
llenado = llenado[llenado["CapacidadUtilEnergia"] > 0].pivot_table(index="fecha", columns="CodigoEmbalse", values="pct")
llenado = llenado.drop(columns=[c for c in llenado if c.startswith("AGREGADO")]).ffill().dropna()

pca = PCA().fit(StandardScaler().fit_transform(llenado))
explicada = pca.explained_variance_ratio_ * 100
cargas = pd.DataFrame(pca.components_.T[:, :2], index=llenado.columns, columns=["CP1", "CP2"])
region = sin_repetir.drop_duplicates("CodigoEmbalse").set_index("CodigoEmbalse")["RegionHidrologica"]

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4.5))
izq.bar(range(1, 9), explicada[:«8»], color="#1D4E89")                                    # HUECO 6: cuántos componentes mostrar para que coincida con el eje
izq.set(title="Varianza explicada por componente", xlabel="componente", ylabel="%")
for nombre_region, grupo in cargas.groupby(region.reindex(cargas.index)):
    der.scatter(grupo["CP1"], grupo["CP2"], label=nombre_region, s=40)
for nombre, (x, y) in cargas.iterrows():
    der.text(x, y, " " + nombre, fontsize=6)
der.axhline(0, color="gray", lw=0.5); der.axvline(0, color="gray", lw=0.5)
der.set(title="Cada punto es un embalse", xlabel="CP1", ylabel="CP2"); der.legend(fontsize=7)
plt.tight_layout(); plt.show()
print(f"{llenado.shape[1]} embalses | Los dos primeros componentes explican {explicada[:2].sum():.0f} % de la varianza")
'''),
("md", r'''
- ¿Los embalses de una misma región hidrológica quedan cerca en la gráfica? **Mi lectura:**
- ¿Sirve resumir todos los embalses en un solo número, como hicimos con el nivel agregado? **Mi respuesta:**
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base

Con horizonte $h$ días, dos pronósticos que no cuestan nada:

$$\hat P^{\text{ing}}_{t+h}=P_t\qquad\text{y}\qquad \hat P^{\text{sem}}_{t+h}=\frac{1}{7}\sum_{j=0}^{6}P_{t-j}.$$

El error se mide igual que en clase: $\text{MAPE}=\dfrac{100}{n}\sum_t\left|\dfrac{P_{t+h}-\hat P_{t+h}}{P_{t+h}}\right|$.

Empezamos con $h=1$: el precio de mañana. Al final del cuaderno cambiarán esa línea por $h=7$.
'''),
("code", r'''
H = 1                                                         # horizonte en días. Al final lo cambiarán por 7.
tabla = pd.DataFrame({"real": datos["precio"].shift(«-H»)})   # HUECO 7: el desplazamiento que trae el precio de dentro de H días a la fila de hoy
tabla["ingenuo"] = datos["precio"]
tabla["promedio_7"] = datos["precio"].rolling(7).mean()
tabla = tabla.dropna()

def mape(real, pronostico):
    return float((abs(real - pronostico) / «real»).mean() * 100)   # HUECO 8: ¿contra qué se compara el error para volverlo porcentaje?

for metodo in ["ingenuo", "promedio_7"]:
    print(f"MAPE {metodo}: {mape(tabla['real'], tabla[metodo]):.1f} %")
'''),
("md", r'''
Comparen con su segunda apuesta.

### 3.2 Cinco modelos y el mismo protocolo de la clase

Los cinco candidatos son los de la clase: regresión lineal, Ridge, $k$ vecinos, bosque aleatorio y *gradient boosting*. Les pedimos el **cambio relativo** $y_t=P_{t+h}/P_t$ y no el nivel, por la razón que vimos con el sancocho: los árboles y los vecinos no pueden pronosticar por encima de lo que ya vieron.

El tiempo se parte en tres tramos:

$$\underbrace{2023\ \dots\ 2024}_{\text{entrenamiento}}\ \Big|\ \underbrace{\text{ene}\ \dots\ \text{jun } 2025}_{\text{validación: se escoge}}\ \Big|\ \underbrace{\text{jul } 2025\ \dots\ \text{oct } 2026}_{\text{prueba: se reporta}}$$

**Predicción escrita:** ¿algún modelo le ganará al pronóstico ingenuo en la prueba? **Mi apuesta:**
'''),
("code", r'''
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def cinco_modelos():
    return {
        "lineal": LinearRegression(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=«5»)),   # HUECO 0: cuántos vecinos promedia el modelo. En clase usamos 5
        "bosque": RandomForestRegressor(n_estimators=300, random_state=0),
        "boosting": GradientBoostingRegressor(random_state=0),
    }

X = pd.DataFrame({
    "ayer": datos["precio"].shift(1) / datos["precio"],
    "promedio_7": datos["precio"].rolling(7).mean() / datos["precio"],
    "promedio_30": datos["precio"].rolling(30).mean() / datos["precio"],
    "embalse": datos["embalse"],
    "embalse_cambio_7": datos["embalse"] - datos["embalse"].shift(7),
    "dia_semana": datos.index.dayofweek,
})
base = X.join(datos["precio"].rename("hoy")).join(datos["precio"].shift(-H).rename("real")).dropna()
variables = list(X.columns)
razon = base["real"] / base["hoy"]

objetivo = base.index + pd.Timedelta(days=H)                    # el día que cada fila intenta adivinar
E = objetivo <= pd.Timestamp("2024-12-31")
V = (base.index >= "2025-01-01") & (objetivo <= pd.Timestamp(«"2025-06-30"»))   # HUECO 10: la fecha en que termina la validación, entre comillas
P = base.index >= "2025-07-01"
print(f"Entrenamiento: {E.sum()} días | Validación: {V.sum()} | Prueba: {P.sum()}")

errores, pronosticos = {}, {}
for nombre, modelo in cinco_modelos().items():
    modelo.fit(base[variables][E], razon[E])                    # se entrena solo con el pasado
    en_validacion = mape(base["real"][V], modelo.predict(base[variables][V]) * base["hoy"][V])
    modelo.fit(base[variables][E | V], razon[E | V])            # se reentrena con todo lo anterior a la prueba
    pronosticos[nombre] = pd.Series(modelo.predict(base[variables][P]) * base["hoy"][P].values, index=base.index[P])
    errores[nombre] = {"validación": en_validacion, "prueba": mape(base["real"][P], pronosticos[nombre])}

for metodo in ["ingenuo", "promedio_7"]:
    errores[metodo] = {"validación": mape(base["real"][V], tabla.loc[base.index[V], metodo]),
                       "prueba": mape(base["real"][P], tabla.loc[base.index[P], metodo])}
    pronosticos[metodo] = tabla.loc[base.index[P], metodo]
todos = pd.DataFrame(errores).T
pronosticos_prueba = pd.DataFrame(pronosticos)

todos.sort_values("validación").plot.barh(figsize=(9, 4), color=["#1D4E89", "#1B998B"])
plt.xlabel("MAPE (%)"); plt.title("Siete candidatos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(1))
'''),
("md", r'''
**¿Por qué el ganador se escoge en validación y no en prueba?** Respondan en dos líneas, con sus palabras: **Mi respuesta:**
'''),
("code", r'''
mejor = todos[«"validación"»].idxmin()                          # HUECO 11: la columna con la que se escoge al ganador
resumen = todos["prueba"].round(1)
print(f"Ganador en validación: {mejor}")
print(f"Su MAPE en prueba: {resumen[mejor]} %   |   MAPE del ingenuo en prueba: {resumen['ingenuo']} %")

fechas = base.index[P] + pd.Timedelta(days=H)
plt.figure(figsize=(11, 3.8))
plt.plot(fechas, base["real"][P], color="black", lw=1.5, label="real")
plt.plot(fechas, pronosticos_prueba["ingenuo"], color="#1B998B", lw=1, label="ingenuo")
if mejor != "ingenuo":
    plt.plot(fechas, pronosticos_prueba[mejor], color="#D7263D", lw=1, ls="--", label=mejor)
plt.title(f"Pronóstico a {H} día(s) contra el precio real"); plt.ylabel("COP/kWh"); plt.legend(); plt.show()
'''),
("md", r'''
### 3.3 Explicar: ¿cuánto del precio se entiende con los embalses?

Dos medidas de asociación. La de Pearson mide relación lineal. La de Spearman mide si una variable sube cuando la otra baja, sin exigir una recta:

$$r=\frac{\sum_t (x_t-\bar x)(y_t-\bar y)}{\sqrt{\sum_t (x_t-\bar x)^2\sum_t (y_t-\bar y)^2}},\qquad r_s=r\ \text{calculado sobre los rangos}.$$
'''),
("code", r'''
pearson = datos["precio"].corr(datos[«"embalse"»])               # HUECO 9: la otra variable de la relación
spearman = datos["precio"].corr(datos["embalse"], method="spearman")
print(f"Pearson: {pearson:.2f} | Spearman: {spearman:.2f}")

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 4))
puntos = izq.scatter(datos["embalse"], datos["precio"], s=6, c=datos.index.year, cmap="viridis")
izq.set(xlabel="Nivel agregado de embalses (%)", ylabel="Precio de bolsa (COP/kWh)", yscale="log", title="Cada punto es un día, coloreado por año")
plt.colorbar(puntos, ax=izq, ticks=sorted(set(datos.index.year)))

modelo_bosque = cinco_modelos()["bosque"].fit(base[variables], razon)
pd.Series(modelo_bosque.feature_importances_, index=variables).sort_values().plot.barh(ax=der, color="#1D4E89")
der.set(title="Importancia de cada variable en el bosque aleatorio"); plt.tight_layout(); plt.show()

print("Precio mediano según el nivel de los embalses:")
print(datos.groupby(pd.cut(datos["embalse"], [0, 40, 50, 60, 70, 100]), observed=True)["precio"].agg(["median", "count"]).round(0))
'''),
("md", r'''
Comparen con su tercera apuesta. Luego respondan:

- ¿La relación es una línea recta? ¿Cambia de un año a otro? **Mi lectura:**
- Una correlación no prueba que los embalses causen el precio. ¿Qué otra cosa podría mover las dos variables a la vez? **Mi respuesta:**
- La importancia de una variable en el bosque dice cuánto la usa el modelo para pronosticar el cambio de mañana. ¿Es eso lo mismo que explicar el precio? **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿desde qué precio fijo conviene firmar?

Con los errores de la prueba armamos una franja alrededor del pronóstico. Sea $c_t=P_{t+h}/\hat P_{t+h}$ el cociente entre lo real y lo pronosticado. La franja que contiene 8 de cada 10 días es

$$\Bigl[\hat P_{T+h}\,\hat Q_{0{,}10}(c),\ \ \hat P_{T+h}\,\hat Q_{0{,}90}(c)\Bigr].$$

Usamos el método más simple entre los que no pierden: si el ganador de validación no mejora al ingenuo en la prueba, nos quedamos con el ingenuo.
'''),
("code", r'''
elegido = mejor if resumen[mejor] < resumen["ingenuo"] else "ingenuo"
cociente = base["real"][P] / pronosticos_prueba[elegido]
bajo, alto = cociente.quantile(0.10), cociente.quantile(«0.90»)      # HUECO 12: el cuantil que deja por debajo 9 de cada 10 días

hoy = datos.index[-1]
if elegido == "ingenuo":
    punto = datos["precio"].iloc[-1]
elif elegido == "promedio_7":
    punto = datos["precio"].rolling(7).mean().iloc[-1]
else:
    modelo = cinco_modelos()[elegido].fit(base[variables], razon)
    punto = float(modelo.predict(X.iloc[[-1]][variables])[0]) * datos["precio"].iloc[-1]

print(f"Horizonte: {H} día(s) | Método elegido: {elegido} (MAPE en prueba {resumen[elegido]} %)")
print(f"Pronóstico para el {(hoy + pd.Timedelta(days=H)).date()}: {punto:.0f} COP/kWh")
print(f"Franja de 8 de cada 10 días: entre {punto * bajo:.0f} y {punto * alto:.0f} COP/kWh")
print(f"Ancho de la franja: de {bajo:.2f} a {alto:.2f} veces el pronóstico")
'''),
("md", r'''
Lean la franja así. Si a la empresa le ofrecen un precio fijo por debajo del borde inferior, casi siempre le conviene firmar. Si se lo ofrecen por encima del borde superior, casi siempre le conviene la bolsa. En el medio, la decisión depende de cuánto le duele un susto.

### La decisión en tres frases

Para el gerente, que no sabe qué es un percentil.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Precio nacional (`PB_Nal`) | `PB_Int`, `PB_Tie` | La empresa compra en el mercado nacional |
| Versión del precio | | |
| Filas de embalses que se quitaron | | |
| Promedio diario simple | | |
| Tres tramos: entrenamiento, validación y prueba | | |
| Pronosticar el cambio relativo y no el nivel | | |
| Franja con percentiles 10 y 90 | | |

### 4.3 Riesgos y mantenimiento

- Los días más recientes solo tienen las primeras liquidaciones, que después se corrigen. ¿Qué le pasa a un pronóstico que se hace con datos provisionales?
- ¿Qué le preguntarían a XM sobre las versiones `TXF` y `TXR` antes de usar este análisis con plata de verdad?
- El promedio diario simple trata igual la hora de las 3 de la mañana y la de las 7 de la noche. ¿A qué empresa le sirve mal ese promedio?
- Si llega un fenómeno de El Niño, ¿en cuál de sus tres respuestas confiarían menos?

### 4.4 Cambien una línea

Vuelvan a la celda donde dice `H = 1`, pongan `H = 7` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Llenen la tabla con sus dos corridas.

| | $h=1$ | $h=7$ |
|---|---|---|
| MAPE del ingenuo en prueba | | |
| Ganador en validación | | |
| MAPE del ganador en prueba | | |
| Ancho de la franja | | |

Con la tabla llena, respondan en tres líneas: ¿cuánto le cuesta al gerente, en incertidumbre, cada día adicional de anticipación? ¿Qué le dice eso sobre el precio fijo? **Mi respuesta:**

Dejen el cuaderno entregado con `H = 7` corrido y la tabla completa.

### 4.5 Procedencia y entrega
'''),
("code", r'''
procedencia()
'''),
("md", r'''
**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase01/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]

destino = RAIZ / "clases" / "01_formular"
nbf.write(cuaderno(SANCOCHO, False), destino / "01a_el_sancocho.ipynb")
nbf.write(cuaderno(LUZ, True), destino / "01b_el_precio_de_la_luz.ipynb")
(RAIZ / "_profesor").mkdir(exist_ok=True)
nbf.write(cuaderno(LUZ, False), RAIZ / "_profesor" / "01b_el_precio_de_la_luz_SOLUCION.ipynb")
print("Cuadernos escritos.")
