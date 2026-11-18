"""Genera los cuadernos de la clase 1. El de los estudiantes sale del mismo
texto que la solución: lo que va entre « » en el código se reemplaza por ____."""
import re, sys
from pathlib import Path
import nbformat as nbf

REPO = "https://github.com/cmorregof/taller-cd-ml-2026-2.git"
RAIZ = Path(__file__).resolve().parent.parent

ARRANQUE = f'''# Celda de arranque. No hay que cambiarle nada: trae las herramientas del curso.
import pathlib, subprocess, sys
raiz = next((p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "taller" / "datos.py").exists()), None)
if raiz is None:
    subprocess.run(["git", "clone", "-q", "{REPO}"], check=True)
    raiz = pathlib.Path("taller-cd-ml-2026-2")
sys.path.insert(0, str(raiz))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from taller import descargar, procedencia

plt.rcParams.update({{"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False}})
print("Herramientas listas.")'''

def cuaderno(celdas, estudiante):
    nb = nbf.v4.new_notebook()
    for tipo, texto in celdas:
        texto = texto.strip("\n")
        if tipo == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(texto))
        else:
            texto = re.sub(r"«(.*?)»", "____" if estudiante else r"\1", texto)
            nb.cells.append(nbf.v4.new_code_cell(texto))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb

# ---------------------------------------------------------------- EL SANCOCHO
SANCOCHO = [
("md", r'''
# Clase 1 · El sancocho

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. Ustedes lo siguen, lo corren y me interrumpen. En la segunda mitad cada uno resuelve el gemelo, *El precio de la luz*, con la misma técnica y otros datos.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Convertir una pregunta de la vida diaria en un problema de datos con métrica, vencer una línea base y validar respetando el tiempo. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 3 (flujo reproducible y trazable), 4 (implementar y evaluar un modelo) y 5 (comunicar a quien no es técnico). |
| **Uso de IA** | Pueden pedirle a Claude que explique una línea de código o que escriba una gráfica. Deben poder explicar sin ayuda qué es una línea base, por qué no se barajan los meses y qué significa el error que reportan. |
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

ax = olla.plot(color="#A8432C", lw=2)
ax.set(title="Costo mayorista de una olla de sancocho para seis, Manizales", xlabel="", ylabel="pesos")
plt.show()
print(f"Enero de 2013: ${olla.iloc[0]:,.0f} | Diciembre de 2024: ${olla.iloc[-1]:,.0f} | Se multiplicó por {olla.iloc[-1] / olla.iloc[0]:.2f}")
'''),
("md", r'''
Comparen con su apuesta. Ojo con lo que mide esta cifra: es el costo **mayorista** de los ingredientes en pesos corrientes. No incluye gas, mano de obra ni el margen de la tienda.
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base

Todo modelo se mide contra el pronóstico más barato posible. Aquí hay dos candidatos que no cuestan nada:

- **Ingenuo:** dentro de tres meses costará lo mismo que hoy.
- **Estacional:** dentro de tres meses costará lo mismo que ese mes del año pasado.

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

### 3.2 Un modelo, y la trampa de barajar

Probemos una regresión lineal que mira el costo de hoy, el de los dos meses anteriores y el del mismo mes objetivo un año atrás.
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
corte = "2021-12-31"
entreno, prueba = datos.loc[:corte], datos.loc[corte:].iloc[1:]
modelo = LinearRegression().fit(entreno[variables], entreno["real"])
pronostico = pd.Series(modelo.predict(prueba[variables]), index=prueba.index)
mape_temporal = mape(prueba["real"], pronostico)

print(f"MAPE barajando (promedio de 200 sorteos): {mape_barajado:.1f} %")
print(f"MAPE respetando el tiempo:                 {mape_temporal:.1f} %")
'''),
("md", r'''
Los dos números miden el mismo modelo, así que la diferencia viene del examen. Al barajar, el modelo se entrena con meses de 2023 y lo examinan con meses de 2022: ya vio el futuro. Ese examen no se parece a lo que le pasará al dueño del negocio, que siempre pronostica hacia adelante. De aquí en adelante solo vale la validación que respeta el tiempo.

### 3.3 El veredicto, con la misma regla para todos
'''),
("code", r'''
comparacion = pd.DataFrame({
    "real": prueba["real"],
    "ingenuo": tabla.loc[prueba.index, "ingenuo"],
    "estacional": tabla.loc[prueba.index, "estacional"],
    "regresion": pronostico,
})
resumen = pd.Series({m: mape(comparacion["real"], comparacion[m]) for m in ["ingenuo", "estacional", "regresion"]}).round(1)
print("MAPE en la prueba (2022 a 2024), en %:")
print(resumen.sort_values())

comparacion.index = comparacion.index + pd.DateOffset(months=H)   # se grafica en el mes pronosticado
ax = comparacion["real"].plot(color="black", lw=2, label="real")
comparacion["ingenuo"].plot(ax=ax, color="#E0A526", label="ingenuo")
comparacion["regresion"].plot(ax=ax, color="#A8432C", ls="--", label="regresión")
ax.set(title="Pronóstico a tres meses contra el costo real", xlabel="", ylabel="pesos"); ax.legend(); plt.show()
'''),
("md", r'''
Lean la tabla con calma antes de opinar. Si la regresión no le gana con claridad al pronóstico ingenuo, la respuesta honesta para el cliente es el pronóstico ingenuo, que además se explica en una frase. Un modelo que empata con la línea base solo agrega complejidad.

### 3.4 Explicar: ¿qué ingrediente mueve el costo?
'''),
("code", r'''
participacion = (costo_por_ingrediente.iloc[-1] / olla.iloc[-1] * 100).round(1).sort_values(ascending=False)
print("Participación en el costo, diciembre de 2024 (%):")
print(participacion)

alza = costo_por_ingrediente.iloc[-1] - costo_por_ingrediente.iloc[0]
print("\nDe cada 100 pesos que subió la olla entre 2013 y 2024, cuántos puso cada ingrediente:")
print((alza / alza.sum() * 100).round(1).sort_values(ascending=False))

costo_por_ingrediente.plot.area(lw=0, cmap="YlOrBr_r")
plt.title("Costo de la olla por ingrediente"); plt.xlabel(""); plt.ylabel("pesos"); plt.legend(fontsize=8); plt.show()
'''),
("md", r'''
Esta explicación es una descomposición contable: dice cuánto pesa cada ingrediente, no por qué subió su precio. Para el porqué harían falta otros datos, como el precio del maíz que come el cerdo o el clima de la cosecha.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿con cuánto se presupuesta la olla?

Un pronóstico de un solo número deja al cliente sin saber cuánto puede fallar. Usamos los errores de la prueba para ponerle un margen. La pregunta del cliente es «con qué costo presupuesto para no quedar corto», así que nos interesa cuánto se queda corto el pronóstico en los meses malos.
'''),
("code", r'''
mejor = resumen.idxmin()
comparacion_original = comparacion.copy(); comparacion_original.index = prueba.index
subestimacion = (comparacion_original["real"] / comparacion_original[mejor] - 1)   # positivo: el costo real salió más alto
margen = subestimacion.quantile(0.90)

hoy = olla.index[-1]
if mejor == "ingenuo":
    punto = olla.iloc[-1]
elif mejor == "estacional":
    punto = olla.shift(12 - H).iloc[-1]
else:
    punto = float(modelo.predict(X.iloc[[-1]][variables])[0])

print(f"Método elegido: {mejor} (MAPE {resumen[mejor]} %)")
meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
objetivo = hoy + pd.DateOffset(months=H)
print(f"Pronóstico para {meses[objetivo.month - 1]} de {objetivo.year}: ${punto:,.0f}")
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
| Corte temporal en diciembre de 2021 | Separación al azar | Al barajar, el modelo ve el futuro |
| Margen con el percentil 90 de los meses de prueba | Intervalo teórico | Sale de errores observados y se explica sin fórmulas |

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
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase: formular, limpiar, vencer una línea base con validación temporal y convertir el resultado en una decisión. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (seleccionar y preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar) y 5 (comunicar). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que les ayude a llenar un hueco o que interprete un error. Deben poder explicar sin ayuda la regla con que se escogió la versión del precio, por qué el corte es temporal y qué significa su MAPE. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, la decisión en tres frases y la bitácora llenas. |

**Cómo se trabaja.** El código está escrito casi todo. Donde vean `____` falta una pieza, y el comentario de al lado dice qué va. Lean la celda entera antes de llenar. Si una celda falla, lean el error antes de pedir ayuda.
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

En Colombia la energía se compra y se vende cada hora en un mercado que administra XM. El precio de ese mercado se llama **precio de bolsa** y se mide en pesos por kilovatio hora. La explicación que uno oye siempre es que el precio sube cuando los embalses bajan, porque hay que prender plantas térmicas, que son más caras. Hoy vamos a ver cuánto de esa explicación aguanta el dato.

Imagínense una empresa mediana de Manizales que gasta mucha energía. Puede comprarla al precio de bolsa, que cambia todos los días, o firmar un precio fijo. Su gerente necesita tres respuestas:

1. **Predecir.** ¿Cuál será el precio de bolsa promedio dentro de siete días?
2. **Explicar.** ¿Cuánto del precio se entiende mirando el nivel de los embalses?
3. **Decidir.** Si le ofrecen un precio fijo para la semana entrante, ¿desde qué valor le conviene firmar?

**La métrica** es la misma de la clase: el MAPE del pronóstico, ahora a siete días.

### Predicción escrita

Escriban su apuesta antes de correr nada.

- ¿Cuánto vale hoy un kilovatio hora en bolsa, en pesos? **Mi apuesta:**
- ¿Qué tan lejos queda el precio de dentro de siete días del precio de hoy, en porcentaje? **Mi apuesta:**
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

ax = precio.plot(color="#16302B", lw=1)
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
datos["embalse"].plot(color="#3E7CB1"); plt.title("Nivel agregado de los embalses (%)"); plt.xlabel(""); plt.show()
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base

Dos pronósticos que no cuestan nada:

- **Ingenuo:** dentro de siete días el precio será el de hoy.
- **Promedio semanal:** dentro de siete días el precio será el promedio de los últimos siete días.
'''),
("code", r'''
H = 7                                                         # horizonte: siete días
tabla = pd.DataFrame({"real": datos["precio"].shift(«-H»)})   # HUECO 5: el desplazamiento que trae el precio de dentro de H días a la fila de hoy
tabla["ingenuo"] = datos["precio"]
tabla["promedio_7"] = datos["precio"].rolling(7).mean()
tabla = tabla.dropna()

def mape(real, pronostico):
    return float((abs(real - pronostico) / «real»).mean() * 100)   # HUECO 6: ¿contra qué se compara el error para volverlo porcentaje?

for metodo in ["ingenuo", "promedio_7"]:
    print(f"MAPE {metodo}: {mape(tabla['real'], tabla[metodo]):.1f} %")
'''),
("md", r'''
Comparen con su segunda apuesta.

### 3.2 Un modelo, con validación temporal

La regresión mira el precio de hoy, el promedio de la semana y el nivel de los embalses.

**Predicción escrita:** ¿le ganará la regresión al pronóstico ingenuo? **Mi apuesta:**
'''),
("code", r'''
X = pd.DataFrame({
    "hoy": datos["precio"],
    "promedio_7": datos["precio"].rolling(7).mean(),
    "embalse": datos["embalse"],
})
base = X.join(datos["precio"].shift(-H).rename("real")).dropna()
variables = list(X.columns)

corte = "2025-06-30"
entreno = base.loc[:corte]
prueba = base.loc[«corte»:].iloc[1:]                            # HUECO 7: la prueba empieza donde termina el entrenamiento
modelo = LinearRegression().fit(entreno[variables], entreno[«"real"»])   # HUECO 8: la columna que el modelo debe aprender a adivinar
pronostico = pd.Series(modelo.predict(prueba[variables]), index=prueba.index)

comparacion = pd.DataFrame({
    "real": prueba["real"],
    "ingenuo": tabla.loc[prueba.index, "ingenuo"],
    "promedio_7": tabla.loc[prueba.index, "promedio_7"],
    "regresion": pronostico,
})
resumen = pd.Series({m: mape(comparacion["real"], comparacion[m]) for m in ["ingenuo", "promedio_7", "regresion"]}).round(1)
print(f"Entrenamiento: {len(entreno)} días | Prueba: {len(prueba)} días")
print("MAPE en la prueba, en %:")
print(resumen.sort_values())
'''),
("md", r'''
**¿Por qué el corte es una fecha y no un sorteo?** Respondan en dos líneas, con sus palabras: **Mi respuesta:**
'''),
("code", r'''
grafica = comparacion.copy(); grafica.index = grafica.index + pd.Timedelta(days=H)
ax = grafica["real"].plot(color="black", lw=1.5, label="real")
grafica["ingenuo"].plot(ax=ax, color="#E0A526", lw=1, label="ingenuo")
grafica["regresion"].plot(ax=ax, color="#A8432C", lw=1, ls="--", label="regresión")
ax.set(title="Pronóstico a siete días contra el precio real", xlabel="", ylabel="COP/kWh"); ax.legend(); plt.show()
'''),
("md", r'''
### 3.3 Explicar: ¿cuánto del precio se entiende con los embalses?
'''),
("code", r'''
correlacion = datos["precio"].corr(datos[«"embalse"»])           # HUECO 9: la otra variable de la relación
print(f"Correlación entre precio y nivel de embalses: {correlacion:.2f}")

plt.scatter(datos["embalse"], datos["precio"], s=6, alpha=0.4, color="#16302B")
plt.xlabel("Nivel agregado de embalses (%)"); plt.ylabel("Precio de bolsa (COP/kWh)")
plt.title("Cada punto es un día"); plt.show()

print("Precio promedio según el nivel de los embalses:")
print(datos.groupby(pd.cut(datos["embalse"], [0, 40, 50, 60, 70, 100]), observed=True)["precio"].agg(["mean", "count"]).round(0))
'''),
("md", r'''
Comparen con su tercera apuesta. Luego respondan:

- ¿La relación es una línea recta? **Mi lectura:**
- Una correlación no prueba que los embalses causen el precio. ¿Qué otra cosa podría mover las dos variables a la vez? **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿desde qué precio fijo conviene firmar?

Con los errores de la prueba armamos una franja alrededor del pronóstico. Usamos el método que ganó en la sección 3.
'''),
("code", r'''
mejor = resumen.idxmin()
razon = comparacion["real"] / comparacion[mejor]               # real dividido entre pronosticado, día por día
bajo, alto = razon.quantile(0.10), razon.quantile(«0.90»)      # HUECO 10: el percentil que deja por debajo 9 de cada 10 días

hoy = datos.index[-1]
if mejor == "ingenuo":
    punto = datos["precio"].iloc[-1]
elif mejor == "promedio_7":
    punto = datos["precio"].rolling(7).mean().iloc[-1]
else:
    punto = float(modelo.predict(X.iloc[[-1]][variables])[0])

print(f"Método elegido: {mejor} (MAPE {resumen[mejor]} %)")
print(f"Pronóstico para el {(hoy + pd.Timedelta(days=H)).date()}: {punto:.0f} COP/kWh")
print(f"Franja de 8 de cada 10 días: entre {punto * bajo:.0f} y {punto * alto:.0f} COP/kWh")
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
| Corte temporal en junio de 2025 | | |
| Franja con percentiles 10 y 90 | | |

### 4.3 Riesgos y mantenimiento

- Los días más recientes solo tienen las primeras liquidaciones, que después se corrigen. ¿Qué le pasa a un pronóstico que se hace con datos provisionales?
- ¿Qué le preguntarían a XM sobre las versiones `TXF` y `TXR` antes de usar este análisis con plata de verdad?
- El promedio diario simple trata igual la hora de las 3 de la mañana y la de las 7 de la noche. ¿A qué empresa le sirve mal ese promedio?
- Si llega un fenómeno de El Niño, ¿en cuál de sus tres respuestas confiarían menos?

### 4.4 Procedencia y entrega
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
