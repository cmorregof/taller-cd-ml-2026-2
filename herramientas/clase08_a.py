"""Celdas de la demostración de la clase 8, «El electrocardiograma». Las usa construir_clase08.py."""
from molde import ARRANQUE

ECG = [
("md", r'''
# Clase 8 · El electrocardiograma

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. En la segunda mitad cada uno resuelve el gemelo, *Los sonidos del Magdalena*, con la misma técnica y otra señal.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Convertir una señal en una tabla de características (estadísticas, espectrales y de forma), comparar un catálogo de modelos de scikit-learn sobre esa tabla y partir los datos por paciente para que el examen sea limpio. |
| **Enfoque** | Machine Learning. |
| **Resultados de aprendizaje** | 1 (formular el problema y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique un filtro, una transformada o una línea de código. Deben poder explicar sin ayuda qué es una frecuencia de muestreo, qué resume cada familia de características, por qué se parte por paciente y cómo se escogió el umbral. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con la decisión de la sección 4 escrita por ustedes y la tabla de la variante llena. |

**Cómputo.** Todo corre en CPU. La descarga de los trazos tarda cerca de tres minutos y el resto del cuaderno, entre tres y seis minutos en Colab gratuito.
'''),
("md", r'''
> **Antes de empezar: conozcan la fuente.** Los electrocardiogramas de hoy son los de PTB-XL, que publica PhysioNet. Abran la [página del conjunto](https://physionet.org/content/ptb-xl/1.0.3/) y explórenla cinco minutos. El diccionario de los códigos diagnósticos es el archivo [`scp_statements.csv`](https://physionet.org/files/ptb-xl/1.0.3/scp_statements.csv), que está en la lista de archivos de esa misma página. Respondan mirando la fuente:
>
> - ¿Quién recogió los trazos, en qué años, y quién los publica hoy?
> - ¿Una fila de `ptbxl_database.csv` es un paciente o es un registro?
> - ¿Qué licencia tiene el conjunto y qué pide a cambio de usarlo?
'''),
("code", ARRANQUE),
("code", r'''
# Los dos parámetros del cuaderno. Al final de la clase se cambia la primera línea y se corre todo de nuevo.
DERIVACIONES_USADAS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SENSIBILIDAD_MINIMA = 0.90      # de cada 100 trazos con infarto, cuántos debe atrapar la alerta

import ast, time
from scipy import signal, stats
from taller.clase08_senales import trazos, DERIVACIONES, FS_ECG
print(len(DERIVACIONES_USADAS), "derivaciones en uso, a", FS_ECG, "muestras por segundo.")
'''),
("md", r'''
## 1. Formulación

Imagínense un hospital de primer nivel en un municipio que no tiene cardiólogo de turno. La enfermera toma el electrocardiograma, el trazo viaja por la red y un cardiólogo lo lee a distancia, en el orden en que va llegando. Si el trazo de un infarto queda de último en la fila, el orden de llegada le costó tiempo al paciente que menos tiempo tenía.

Ese hospital necesita tres respuestas:

1. **Predecir.** ¿Este trazo muestra signos de infarto de miocardio o es un trazo normal?
2. **Explicar.** ¿Qué parte de la señal sostiene la predicción: qué derivaciones y qué tipo de característica?
3. **Decidir.** ¿Con qué umbral se manda un trazo al principio de la fila del cardiólogo, y cuántas falsas alarmas cuesta ese umbral?

**El dato.** PTB-XL reúne 21.799 electrocardiogramas de 12 derivaciones y diez segundos, tomados a 18.869 pacientes en Alemania entre 1989 y 1996 y anotados por cardiólogos. Lo publica PhysioNet con licencia CC BY 4.0.

**Qué es una señal.** Un electrocardiograma es un voltaje que cambia en el tiempo, $x(t)$. El aparato lo mide a intervalos fijos y entrega una lista de números,

$$x[n]=x\!\left(\frac{n}{f_s}\right),\qquad n=0,1,\dots,N-1,$$

donde $f_s$ es la frecuencia de muestreo. Aquí $f_s=100$ Hz y $N=1000$, así que cada derivación trae mil números. Una derivación es la misma actividad eléctrica del corazón vista desde otro par de electrodos.

**La métrica.** El modelo entrega un puntaje $s$ por trazo y la alerta se prende cuando $s\ge\tau$. Con un umbral fijo se cuentan aciertos y errores:

$$\text{sensibilidad}=\frac{VP}{VP+FN},\qquad \text{especificidad}=\frac{VN}{VN+FP}.$$

Para comparar modelos sin casarnos con un umbral usamos el área bajo la curva ROC, que es la probabilidad de que un trazo con infarto reciba un puntaje más alto que uno normal:

$$\text{AUC}=P\bigl(s(X^{+})>s(X^{-})\bigr).$$

Un AUC de 0,5 equivale a ordenar al azar y uno de 1 equivale a un orden perfecto. El umbral se fija después, con una regla que escribimos antes de mirar: la alerta debe atrapar al menos 90 de cada 100 infartos.

### Predicción escrita

Antes de ver el dato, escriban su apuesta.

- Si solo conozco la edad y el sexo del paciente, sin mirar el trazo, ¿qué AUC alcanzo? **Mi apuesta:**
- ¿Qué AUC alcanzará el mejor modelo con las doce derivaciones? **Mi apuesta:**
- Con una alerta que atrapa 90 de cada 100 infartos, ¿cuántos trazos normales de cada 100 quedarán marcados por error? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

PTB-XL tiene dos partes. La primera es una tabla con un registro por fila: quién es el paciente, qué leyó el cardiólogo y dónde está el archivo de la señal. La segunda son las señales. Empezamos por la tabla y por el diccionario de códigos diagnósticos.
'''),
("code", r'''
meta = descargar("ptbxl")
codigos = descargar("ptbxl_codigos")
print("Registros:", len(meta), "| Pacientes:", meta["patient_id"].nunique(), "| Columnas:", meta.shape[1])
meta.head(3)
'''),
("md", r'''
### 2.1 Las trampas de la tabla

**Primera trampa: el diagnóstico llega como texto.** La columna `scp_codes` parece un diccionario de Python, pero pandas la lee como una cadena de caracteres. Cada entrada tiene la forma `código: verosimilitud`. La documentación de PTB-XL dice que la verosimilitud vale 0 cuando se desconoce, así que un 0 significa «sin dato» y se lee distinto de «improbable».
'''),
("code", r'''
print("Tal como llega:", repr(meta.loc[0, "scp_codes"]), type(meta.loc[0, "scp_codes"]))
meta["codigos"] = meta["scp_codes"].apply(ast.literal_eval)        # de texto a diccionario

# El diccionario oficial dice a qué superclase diagnóstica pertenece cada código
superclase = codigos[codigos["diagnostic"] == 1].set_index("codigo")["diagnostic_class"]
meta["superclases"] = meta["codigos"].apply(lambda d: frozenset(superclase[c] for c in d if c in superclase.index))

conteo = pd.Series({s: int(meta["superclases"].apply(lambda conjunto: s in conjunto).sum())
                    for s in ["NORM", "MI", "STTC", "CD", "HYP"]})
print("\nRegistros por superclase (un registro puede estar en varias):")
print(conteo)
print("\nRegistros con más de una superclase:", int((meta["superclases"].apply(len) > 1).sum()),
      "| sin ninguna:", int((meta["superclases"].apply(len) == 0).sum()))
'''),
("md", r'''
Los cinco conteos coinciden con los que publica PhysioNet en la página del conjunto: 9.514, 5.469, 5.235, 4.898 y 2.649. Esa coincidencia es la prueba de que leímos bien los códigos. NORM es un trazo normal, MI es infarto de miocardio, STTC son cambios del segmento ST y de la onda T, CD son trastornos de conducción y HYP es hipertrofia.

**Segunda trampa: la edad de 300 años.** Para proteger la identidad de los pacientes mayores de 89 años, PTB-XL les pone una edad cercana a 300. Quien calcule un promedio sin saberlo obtiene una población que no existe.

**Tercera trampa: el código de sexo.** La tabla trae `sex` con valores 0 y 1, y la página del conjunto no dice cuál es cuál. Sí dice que el 52 % son hombres. Miren la proporción de cada código y deduzcan.
'''),
("code", r'''
print("Edad máxima:", meta["age"].max(), "| registros con edad mayor que 120:", int((meta["age"] > 120).sum()))
print("Promedio de edad con el 300 adentro:", round(meta["age"].mean(), 1),
      "| sin esos registros:", round(meta.loc[meta["age"] <= 120, "age"].mean(), 1))

print("\nProporción de registros por código de sexo (%):")
print((meta["sex"].value_counts(normalize=True) * 100).round(1))

print("\nFaltantes (%):")
print((meta[["age", "sex", "height", "weight"]].isna().mean() * 100).round(1))
'''),
("md", r'''
Tres decisiones de limpieza, que van a la bitácora.

- **Edad.** A los mayores de 89 años les pongo 90. Es la cota inferior de lo que sabemos, y solo la usaré en la línea base.
- **Sexo.** El código 0 es el 52 % de los registros. Por eso leo 0 como hombre y 1 como mujer. Es una **inferencia** mía a partir de la proporción publicada y no un diccionario oficial.
- **Estatura y peso.** Faltan en más de la mitad de los registros. No los uso.

**Cuarta trampa: columnas que ya saben la respuesta.** `report` es el informe escrito por el cardiólogo, y `infarction_stadium1`, `infarction_stadium2` y `heart_axis` son anotaciones suyas. Si alguna entra al modelo, el modelo aprende a leer al cardiólogo y no al corazón. Ninguna entra.

### 2.2 El problema que sí vamos a resolver

Un registro puede tener varias superclases, y un infarto puede venir con un trastorno de conducción. Para tener una pregunta limpia comparo dos grupos: los registros cuya única superclase es NORM y los registros que tienen MI, solo o acompañado. Los demás quedan por fuera.
'''),
("code", r'''
meta["edad"] = meta["age"].where(meta["age"] <= 120, 90)
solo_normal = meta["superclases"] == frozenset({"NORM"})
con_infarto = meta["superclases"].apply(lambda conjunto: "MI" in conjunto)

datos = meta[solo_normal | con_infarto].reset_index(drop=True)
datos["infarto"] = datos["superclases"].apply(lambda conjunto: "MI" in conjunto).astype(int)
y = datos["infarto"].values
# Verosimilitud más alta que el cardiólogo le dio a un código de infarto en cada registro
datos["verosimilitud_mi"] = datos["codigos"].apply(lambda d: max([v for c, v in d.items() if superclase.get(c) == "MI"], default=np.nan))

print("Registros que entran:", len(datos), "| quedan por fuera:", len(meta) - len(datos))
print("Normales:", int((y == 0).sum()), "| Con infarto:", int((y == 1).sum()), f"| Proporción de infarto: {y.mean() * 100:.1f} %")
print("Pacientes:", datos["patient_id"].nunique(), "| pacientes con más de un registro:",
      int((datos.groupby("patient_id").size() > 1).sum()))
print("\nRegistros con infarto según la verosimilitud anotada (0 significa desconocida):")
print(datos.loc[y == 1, "verosimilitud_mi"].value_counts().sort_index().to_dict())
'''),
("md", r'''
Tres advertencias sobre este recorte. La primera es que dejar por fuera los demás diagnósticos vuelve el examen más fácil que la vida real, donde al hospital le llegan trazos de todo tipo. La segunda es que la etiqueta MI dice que el cardiólogo leyó signos de infarto en el trazo, reciente o antiguo. Este cuaderno no distingue un infarto que está ocurriendo de la cicatriz de uno viejo. La tercera es que la etiqueta misma tiene grados: el cardiólogo anotó qué tan seguro estaba, y no todos los registros del grupo con infarto llevan un 100. Yo los tomo todos como positivos, y esa decisión va a la bitácora.

### 2.3 Las señales

La función `trazos` entrega la señal de cada registro de la tabla, en milivoltios. Son todos los trazos de los dos grupos, sin muestreo. La primera vez tarda cerca de tres minutos.
'''),
("code", r'''
inicio = time.time()
X = trazos(datos)
if len(DERIVACIONES_USADAS) < len(DERIVACIONES):
    X = X[:, :, [DERIVACIONES.index(d) for d in DERIVACIONES_USADAS]]
print(f"Forma: {X.shape} (registros, muestras, derivaciones) | {X.nbytes / 1e6:.0f} MB en memoria | {time.time() - inicio:.0f} s")
'''),
("code", r'''
mostrar = [d for d in ["II", "AVF", "V1", "V2", "V3", "V5"] if d in DERIVACIONES_USADAS] or DERIVACIONES_USADAS[:1]
limpios = datos[(datos["strat_fold"] == 10) & (datos["validated_by_human"])]
ejemplo = {"normal": limpios.index[limpios["infarto"] == 0][0], "con infarto": limpios.index[limpios["verosimilitud_mi"] == 100][0]}
t = np.arange(X.shape[1]) / FS_ECG

fig, ejes = plt.subplots(len(mostrar), 2, figsize=(13, 1.5 * len(mostrar) + 1), sharex=True, squeeze=False)
for columna, (nombre, i) in enumerate(ejemplo.items()):
    for fila, d in enumerate(mostrar):
        ejes[fila, columna].plot(t, X[i, :, DERIVACIONES_USADAS.index(d)], color="#1D4E89" if columna == 0 else "#D7263D", lw=0.8)
        ejes[fila, columna].set_ylabel(d, rotation=0, labelpad=14)
    ejes[0, columna].set_title(f"Trazo {nombre} (registro {datos.loc[i, 'ecg_id']})")
    ejes[-1, columna].set_xlabel("segundos")
plt.tight_layout(); plt.show()
print("Lo que anotó el cardiólogo:", {n: datos.loc[i, "codigos"] for n, i in ejemplo.items()})
'''),
("md", r'''
Cada pico alto es un latido. A simple vista los dos trazos se parecen, y por eso leer electrocardiogramas es un oficio. La diferencia está en la forma de cada latido en ciertas derivaciones, y esa forma es lo que vamos a convertir en números.

### 2.4 Exploración

Antes de modelar miro quiénes son los pacientes de cada grupo, cómo quedaron repartidos en las diez particiones del conjunto y qué tan limpias vienen las señales.
'''),
("code", r'''
fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(14, 3.8))
for valor, nombre, color in [(0, "normal", "#1D4E89"), (1, "con infarto", "#D7263D")]:
    izq.hist(datos.loc[y == valor, "edad"], bins=range(0, 96, 5), alpha=0.6, label=nombre, color=color, density=True)
izq.set(title="Edad por grupo", xlabel="años"); izq.legend()

(datos.groupby("sex")["infarto"].mean() * 100).rename({0: "0 (hombre, inferido)", 1: "1 (mujer, inferido)"}).plot.bar(ax=centro, color="#1B998B", rot=0)
centro.set(title="Proporción con infarto según sexo", xlabel="", ylabel="%")

(datos.groupby("strat_fold")["infarto"].mean() * 100).plot.bar(ax=der, color="#1D4E89", rot=0)
der.set(title="Proporción con infarto por partición", xlabel="strat_fold", ylabel="%")
plt.tight_layout(); plt.show()

print("Edad mediana: normal", datos.loc[y == 0, "edad"].median(), "| con infarto", datos.loc[y == 1, "edad"].median())
print("Registros por partición:", datos["strat_fold"].value_counts().sort_index().to_dict())
print("Máximo de particiones distintas en que aparece un mismo paciente:", int(datos.groupby("patient_id")["strat_fold"].nunique().max()))
'''),
("md", r'''
Tres lecturas. Los pacientes con infarto son mayores, así que la edad sola ya dice algo y por eso será la línea base. Las diez particiones tienen casi la misma proporción de infarto, porque los autores las armaron estratificadas. Y ningún paciente aparece en dos particiones distintas, que es la propiedad que vamos a necesitar en la sección 3.

Falta mirar un sesgo menos visible. Los trazos se tomaron con varios aparatos. Si un aparato se usó sobre todo en la unidad donde llegan los infartos, un modelo puede aprender a reconocer el aparato.
'''),
("code", r'''
aparatos = datos.groupby("device")["infarto"].agg(registros="size", proporcion_infarto="mean")
aparatos["proporcion_infarto"] = (aparatos["proporcion_infarto"] * 100).round(1)
display(aparatos.sort_values("registros", ascending=False).head(8))

planas = (X.std(axis=1) == 0)
print("Registros con al menos una derivación plana (sin señal):", int(planas.any(axis=1).sum()))
print("Registros con ruido anotado (%):",
      {c: round(float(datos[c].notna().mean() * 100), 1) for c in ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]})
print("Registros validados por una persona (%):", round(float(datos["validated_by_human"].mean() * 100), 1))
'''),
("md", r'''
La proporción de infarto cambia mucho de un aparato a otro. Ese es un **sesgo de procedencia**: el aparato está asociado con la respuesta por la manera como se recogió el dato, sin que tenga que ver con el corazón. El modelo de hoy solo ve la señal y no el nombre del aparato, pero cada aparato puede dejar su huella en la señal. Lo anoto como riesgo en la sección 4.

### 2.5 De la señal a la tabla

Un modelo de scikit-learn recibe una tabla con una fila por registro. Hay que resumir doce mil números por trazo en unas decenas de columnas, y cada columna debe medir algo que un cardiólogo reconocería. Lo hago en cuatro pasos.

**Paso 1: filtrar.** La respiración mueve la línea de base muy despacio y la red eléctrica mete un zumbido de 50 Hz. Un filtro pasabanda deja pasar solo las frecuencias entre 0,5 y 40 Hz, donde vive el latido.

**Paso 2: estadísticas por derivación.** Desviación estándar, rango, asimetría, curtosis y cambio medio entre muestras vecinas:

$$\sigma=\sqrt{\tfrac{1}{N}\textstyle\sum_n (x[n]-\bar x)^2},\qquad \text{asimetría}=\frac{1}{N}\sum_n\Bigl(\frac{x[n]-\bar x}{\sigma}\Bigr)^{3},\qquad \text{curtosis}=\frac{1}{N}\sum_n\Bigl(\frac{x[n]-\bar x}{\sigma}\Bigr)^{4}-3.$$

**Paso 3: características espectrales.** La transformada de Fourier descompone la señal en frecuencias. La densidad espectral de potencia $\hat P(f)$ dice cuánta energía hay en cada una, y de ella tomo la fracción que cae en tres bandas:

$$b_{[f_1,f_2)}=\frac{\sum_{f_1\le f<f_2}\hat P(f)}{\sum_{0{,}5\le f\le 40}\hat P(f)},\qquad \text{bandas: }[0{,}5;\,5),\ [5;\,15),\ [15;\,40)\ \text{Hz}.$$

**Paso 4: ritmo y forma del latido.** Ubico los picos R, que son la punta de cada latido, en los instantes $r_1,\dots,r_m$. De ahí sale la frecuencia cardiaca, $\text{FC}=60/\operatorname{mediana}(r_{i+1}-r_i)$ con los intervalos en segundos. Después recorto una ventana alrededor de cada pico y calculo el **latido mediano** de cada derivación $j$:

$$m_j[k]=\operatorname{mediana}_{i}\;x_j[r_i+k],\qquad k=-25,\dots,44,$$

que son 0,25 segundos antes del pico y 0,45 después. El latido mediano borra el ruido de un latido suelto y deja la forma típica.
'''),
("code", r'''
b, a = signal.butter(2, [0.5, 40], btype="band", fs=FS_ECG)          # filtro pasabanda de 0,5 a 40 Hz
guia = DERIVACIONES_USADAS.index("II") if "II" in DERIVACIONES_USADAS else 0   # derivación donde se buscan los picos R

def latido_mediano(x):
    """x: (1000, d) ya filtrado. Devuelve el latido mediano (70, d), la frecuencia cardiaca y la variación del ritmo."""
    ii = x[:, guia]
    pendiente = np.convolve(np.abs(np.diff(ii, prepend=ii[0])), np.ones(8) / 8, "same")   # el pico R es donde la señal cambia más rápido
    picos, _ = signal.find_peaks(pendiente, distance=30, height=0.5 * np.percentile(pendiente, 98))
    picos = picos[(picos >= 25) & (picos <= len(ii) - 45)]
    if len(picos) < 2:
        return np.zeros((70, x.shape[1]), "float32"), np.nan, np.nan
    latidos = np.stack([x[p - 25:p + 45] for p in picos])
    mediano = np.median(latidos, axis=0)
    rr = np.diff(picos) / FS_ECG
    return mediano - mediano[:5].mean(axis=0), 60 / np.median(rr), rr.std()

def caracteristicas(lote):
    """lote: (n, 1000, d) ya filtrado. Devuelve la tabla de estadísticas, bandas y ritmo, y los latidos medianos."""
    columnas = {}
    medidas = {"desv": lote.std(axis=1), "rango": lote.max(axis=1) - lote.min(axis=1),
               "asim": stats.skew(lote, axis=1), "curt": stats.kurtosis(lote, axis=1),
               "cambio": np.abs(np.diff(lote, axis=1)).mean(axis=1)}
    for nombre, valores in medidas.items():
        for j, d in enumerate(DERIVACIONES_USADAS):
            columnas[f"est_{nombre}_{d}"] = valores[:, j]
    f, P = signal.welch(lote, fs=FS_ECG, nperseg=256, axis=1)
    total = P[:, (f >= 0.5) & (f <= 40)].sum(axis=1)
    total[total == 0] = np.nan                                       # derivación plana: no hay espectro que repartir
    for bajo, alto in [(0.5, 5), (5, 15), (15, 40)]:
        banda = P[:, (f >= bajo) & (f < alto)].sum(axis=1) / total
        for j, d in enumerate(DERIVACIONES_USADAS):
            columnas[f"esp_{bajo:g}a{alto:g}_{d}"] = banda[:, j]
    por_registro = [latido_mediano(x) for x in lote]
    columnas["rit_fc"] = [p[1] for p in por_registro]
    columnas["rit_variacion"] = [p[2] for p in por_registro]
    return pd.DataFrame(columnas), np.stack([p[0] for p in por_registro])

crudo = X[ejemplo["con infarto"], :, guia].copy()
inicio = time.time()
tablas, latidos = [], []
for i in range(0, len(X), 500):                                      # por lotes de 500 para no llenar la memoria
    X[i:i + 500] = signal.filtfilt(b, a, X[i:i + 500], axis=1)       # desde aquí X queda filtrado
    tabla_lote, latidos_lote = caracteristicas(X[i:i + 500])
    tablas.append(tabla_lote); latidos.append(latidos_lote)
C = pd.concat(tablas, ignore_index=True)
latidos = np.concatenate(latidos)
print(f"Características: {C.shape} | latidos medianos: {latidos.shape} | {time.time() - inicio:.0f} s")
print("Registros sin picos R suficientes:", int(C["rit_fc"].isna().sum()), "| celdas faltantes en la tabla:", int(C.isna().sum().sum()))
'''),
("code", r'''
fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 3.6))
izq.plot(t, crudo, color="gray", lw=0.8, label="crudo")
izq.plot(t, X[ejemplo["con infarto"], :, guia], color="#D7263D", lw=0.9, label="filtrado")
izq.set(title=f"El filtro, derivación {DERIVACIONES_USADAS[guia]}", xlabel="segundos", ylabel="mV"); izq.legend()

f, P = signal.welch(X[:, :, guia], fs=FS_ECG, nperseg=256, axis=1)
for valor, nombre, color in [(0, "normal", "#1D4E89"), (1, "con infarto", "#D7263D")]:
    der.semilogy(f, P[y == valor].mean(axis=0), color=color, label=nombre)
for corte in [5, 15, 40]:
    der.axvline(corte, color="gray", ls=":")
der.set(title="Espectro promedio de cada grupo", xlabel="Hz", ylabel="potencia"); der.legend()
plt.tight_layout(); plt.show()

print("Frecuencia cardiaca (latidos por minuto):")
print(C["rit_fc"].groupby(y).describe().rename(index={0: "normal", 1: "con infarto"}).round(1))
'''),
("md", r'''
La gráfica de la derecha muestra que casi toda la energía del electrocardiograma está por debajo de 15 Hz y que los dos grupos se separan poco en el espectro. La frecuencia cardiaca tampoco los separa mucho. Lo que queda por mirar es la forma del latido.
'''),
("code", r'''
k = (np.arange(70) - 25) / FS_ECG * 1000                             # milisegundos respecto al pico R
fig, ejes = plt.subplots(1, len(mostrar), figsize=(2.6 * len(mostrar) + 1, 3.2), squeeze=False)
for eje, d in zip(ejes[0], mostrar):
    j = DERIVACIONES_USADAS.index(d)
    for valor, nombre, color in [(0, "normal", "#1D4E89"), (1, "con infarto", "#D7263D")]:
        eje.plot(k, latidos[y == valor][:, :, j].mean(axis=0), color=color, label=nombre)
    eje.axhline(0, color="gray", lw=0.5); eje.set_title(d); eje.set_xlabel("ms desde el pico R")
ejes[0, 0].set_ylabel("mV"); ejes[0, 0].legend(fontsize=8)
plt.suptitle("Latido mediano, promedio de cada grupo"); plt.tight_layout(); plt.show()
'''),
("md", r'''
Aquí sí hay diferencia. En varias derivaciones el latido promedio del grupo con infarto tiene otra altura y otra forma después del pico. Eso es lo que el cardiólogo mira con el ojo y lo que el modelo tendrá que medir.

### 2.6 PCA: la forma del latido en pocos números

El latido mediano de una derivación son 70 números muy correlacionados entre sí, porque dos muestras vecinas valen casi lo mismo. El análisis de componentes principales los resume. Para cada derivación $j$ centro los latidos del entrenamiento, calculo su matriz de covarianzas $S_j$ y la descompongo:

$$S_j=V_j\Lambda_jV_j^\top,\qquad z_{j}=V_{j,[1:4]}^\top\,(m_j-\bar m_j).$$

Me quedo con los cuatro primeros componentes de cada derivación. Las direcciones $V_j$ se calculan solo con los registros de entrenamiento, porque calcularlas con todos sería dejar que la prueba opine sobre cómo se preparan los datos.

**Predicción escrita:** ¿qué fracción de la varianza de la forma del latido guardan cuatro componentes de 70? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA

E = (datos["strat_fold"] <= 8).values          # entrenamiento
V = (datos["strat_fold"] == 9).values          # validación: aquí se escoge
P_ = (datos["strat_fold"] == 10).values        # prueba: aquí se reporta

explicada, direcciones = {}, {}
for j, d in enumerate(DERIVACIONES_USADAS):
    pca = PCA(n_components=4, random_state=0).fit(latidos[E][:, :, j])
    Z = pca.transform(latidos[:, :, j])
    for c in range(4):
        C[f"lat_cp{c + 1}_{d}"] = Z[:, c]
    explicada[d] = pca.explained_variance_ratio_ * 100
    direcciones[d] = pca

d0 = mostrar[0]
fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(14, 3.8))
pd.DataFrame(explicada, index=["CP1", "CP2", "CP3", "CP4"]).T.plot.bar(stacked=True, ax=izq, cmap="Blues_r", width=0.8)
izq.set(title="Varianza explicada por derivación", ylabel="%"); izq.legend(fontsize=7, ncol=4)
for c in range(2):
    centro.plot(k, direcciones[d0].components_[c], label=f"CP{c + 1}")
centro.plot(k, direcciones[d0].mean_, color="gray", ls=":", label="latido medio")
centro.set(title=f"Los dos primeros componentes, derivación {d0}", xlabel="ms desde el pico R"); centro.legend(fontsize=8)
muestra = np.random.default_rng(0).choice(np.flatnonzero(E), 3000, replace=False)
der.scatter(C.loc[muestra, f"lat_cp1_{d0}"], C.loc[muestra, f"lat_cp2_{d0}"], c=np.where(y[muestra] == 1, "#D7263D", "#1D4E89"), s=5, alpha=0.4)
der.set(title=f"3.000 registros de entrenamiento, derivación {d0}\n(rojo: con infarto)", xlabel="CP1", ylabel="CP2")
plt.tight_layout(); plt.show()

print("Varianza que guardan 4 componentes, por derivación (%):", {d: round(float(v.sum()), 1) for d, v in explicada.items()})
print("Columnas de la tabla final:", C.shape[1])
'''),
("md", r'''
Cuatro números guardan la mayor parte de la forma de un latido de 70 muestras. En la nube de puntos los dos grupos se traslapan: ninguna derivación sola separa bien, y por eso el modelo va a necesitar varias columnas a la vez.

La tabla quedó con cuatro familias de columnas por derivación: `est_` (estadísticas), `esp_` (bandas del espectro), `lat_` (forma del latido) y dos columnas de ritmo.
'''),
("md", r'''
## 3. Modelado

### 3.1 El protocolo y la línea base

Los autores de PTB-XL repartieron los registros en diez particiones (`strat_fold`) y recomiendan usar las ocho primeras para entrenar, la novena para validar y la décima para probar. Las particiones 9 y 10 tienen además las etiquetas más revisadas. Sigo esa recomendación:

$$\underbrace{\text{particiones }1\dots 8}_{\text{entrenamiento}}\ \Big|\ \underbrace{\text{partición }9}_{\text{validación: se escoge}}\ \Big|\ \underbrace{\text{partición }10}_{\text{prueba: se reporta}}$$

La línea base es lo que se sabe del paciente antes de mirar el trazo: la edad y el sexo, en una regresión logística,

$$P(\text{infarto}\mid \text{edad},\text{sexo})=\frac{1}{1+e^{-(\beta_0+\beta_1\,\text{edad}+\beta_2\,\text{sexo})}}.$$

Un modelo que lee la señal y no le gana a esta cifra no está leyendo nada.
'''),
("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

print(f"Entrenamiento: {E.sum()} registros | Validación: {V.sum()} | Prueba: {P_.sum()}")
print(f"Proporción de infarto (%): {y[E].mean() * 100:.1f} | {y[V].mean() * 100:.1f} | {y[P_].mean() * 100:.1f}")

sin_senal = datos[["edad", "sex"]].values
base = LogisticRegression().fit(sin_senal[E], y[E])
auc_base = {"validación": roc_auc_score(y[V], base.predict_proba(sin_senal[V])[:, 1]),
            "prueba": roc_auc_score(y[P_], base.predict_proba(sin_senal[P_])[:, 1])}
print(f"Línea base (edad y sexo): AUC {auc_base['validación']:.3f} en validación, {auc_base['prueba']:.3f} en prueba")
'''),
("md", r'''
Comparen con su primera apuesta. Esa es la cifra que hay que vencer.

### 3.2 Un catálogo de modelos sobre la tabla de características

| Familia | Idea en una línea |
|---|---|
| Regresión logística, con dos niveles de regularización | Una frontera lineal: $P(y=1\mid x)=1/(1+e^{-(\beta_0+x^\top\beta)})$ |
| Análisis discriminante lineal | Supone una normal por grupo con la misma covarianza |
| Bayes ingenuo | Supone que las columnas son independientes dentro de cada grupo |
| $k$ vecinos, con dos valores de $k$ | Vota con los $k$ registros de entrenamiento más parecidos |
| Árbol de decisión | Parte la tabla con preguntas de sí o no sobre una columna a la vez |
| Bosque aleatorio y *ExtraTrees* | Promedian muchos árboles, entrenados sobre remuestreos o con cortes al azar |
| *Gradient boosting* por histogramas | Suma árboles pequeños, cada uno corrigiendo el error del anterior |
| Máquina de soporte vectorial | Busca la frontera con más margen, curva gracias a un núcleo gaussiano |
| Red neuronal pequeña | Una capa oculta de 32 unidades |

La regla queda escrita antes de correr: gana el de mayor AUC en **validación**, y su AUC en **prueba** es el que se reporta. Cada modelo se entrena una sola vez, con las particiones 1 a 8.

**Cómo leer la celda del catálogo.** El diccionario `MODELOS` tiene una línea por candidato, con su nombre y un `make_pipeline` que encadena la preparación de la tabla con el clasificador. `llenar()` reemplaza los faltantes por la mediana y `escalar()` pone las columnas en la misma escala, que es lo que necesitan los modelos que miden distancias o usan gradientes. Para agregar un modelo se importa su clase y se escribe una línea más en el diccionario. El resto del cuaderno recorre lo que haya en `MODELOS`, así que no hay que tocar nada más. La métrica es el AUC, que solo necesita un puntaje que ordene los trazos, y por eso sirve cualquier clasificador que tenga `predict_proba` o `decision_function`.

**Ejercicio.** Agreguen al menos tres modelos más de scikit-learn y vuelvan a correr desde esa celda.
'''),
("code", r'''
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

llenar = lambda: SimpleImputer(strategy="median")      # reemplaza los faltantes por la mediana de la columna
escalar = lambda: StandardScaler()                     # pone todas las columnas en la misma escala

# El catálogo: una línea por modelo. Para agregar uno, escriban otra línea con su nombre y su pipeline.
MODELOS = {
    "logística":              make_pipeline(llenar(), escalar(), LogisticRegression(max_iter=3000)),
    "logística (C = 0,01)":   make_pipeline(llenar(), escalar(), LogisticRegression(C=0.01, max_iter=3000)),
    "discriminante lineal":   make_pipeline(llenar(), escalar(), LinearDiscriminantAnalysis()),
    "Bayes ingenuo":          make_pipeline(llenar(), GaussianNB()),
    "k vecinos (k = 5)":      make_pipeline(llenar(), escalar(), KNeighborsClassifier(n_neighbors=5)),
    "k vecinos (k = 25)":     make_pipeline(llenar(), escalar(), KNeighborsClassifier(n_neighbors=25)),
    "árbol":                  make_pipeline(llenar(), DecisionTreeClassifier(min_samples_leaf=20, random_state=0)),
    "bosque":                 make_pipeline(llenar(), RandomForestClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    "ExtraTrees":             make_pipeline(llenar(), ExtraTreesClassifier(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=0)),
    "boosting":               make_pipeline(HistGradientBoostingClassifier(random_state=0)),
    "SVM":                    make_pipeline(llenar(), escalar(), SVC(C=3)),
    "red neuronal":           make_pipeline(llenar(), escalar(), MLPClassifier(hidden_layer_sizes=(32,), alpha=1.0, max_iter=300, random_state=0)),
}
print(len(MODELOS), "modelos en el catálogo:", ", ".join(MODELOS))
'''),
("code", r'''
def puntaje(modelo, tabla):
    """Un número por registro: más alto significa más sospecha de infarto."""
    return modelo.predict_proba(tabla)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(tabla)

A = C.values
resultados, entrenados, segundos = {}, {}, {}
for nombre, modelo in MODELOS.items():                 # recorre lo que haya en el catálogo
    inicio = time.time()
    entrenados[nombre] = clone(modelo).fit(A[E], y[E])
    resultados[nombre] = {"validación": roc_auc_score(y[V], puntaje(entrenados[nombre], A[V])),
                          "prueba": roc_auc_score(y[P_], puntaje(entrenados[nombre], A[P_]))}
    segundos[nombre] = time.time() - inicio
comparacion = pd.concat([pd.DataFrame({"edad y sexo": auc_base}).T, pd.DataFrame(resultados).T])
comparacion["segundos"] = pd.Series(segundos).round(1)
display(comparacion.round(3))
'''),
("code", r'''
mejor = comparacion.drop(index="edad y sexo")["validación"].idxmax()

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 0.32 * len(comparacion) + 1.5))
comparacion[["validación", "prueba"]].sort_values("validación").plot.barh(ax=izq, color=["#1D4E89", "#9AA5B1"])
izq.set(title="AUC: validación para escoger, prueba para reportar", xlim=(0.5, 1), xlabel="AUC")
izq.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)

fp, vp, _ = roc_curve(y[V], base.predict_proba(sin_senal[V])[:, 1]); der.plot(fp, vp, color="#6C4AB6", ls="--", label="edad y sexo")
for nombre, modelo in entrenados.items():                              # todos en gris y el ganador en rojo
    fp, vp, _ = roc_curve(y[V], puntaje(modelo, A[V]))
    der.plot(fp, vp, color="#D7263D" if nombre == mejor else "#9AA5B1", lw=1.8 if nombre == mejor else 0.8, zorder=3 if nombre == mejor else 2,
             label=nombre if nombre == mejor else None)
der.plot([], [], color="#9AA5B1", lw=0.8, label="los demás modelos")
der.plot([0, 1], [0, 1], color="gray", lw=0.5)
der.set(title="Curvas ROC en validación", xlabel="1 − especificidad", ylabel="sensibilidad"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()

print(f"Ganador en validación: {mejor} | AUC en validación {comparacion.loc[mejor, 'validación']:.3f} | AUC en prueba {comparacion.loc[mejor, 'prueba']:.3f}")
print(f"Línea base en prueba: {auc_base['prueba']:.3f}")
'''),
("md", r'''
El ganador se escogió entre muchos candidatos, y eso tiene un costo que conviene mirar. La gráfica siguiente ubica a cada modelo según su puesto en validación y su puesto en prueba.
'''),
("code", r'''
puestos = comparacion.drop(index="edad y sexo")[["validación", "prueba"]].rank(ascending=False, method="min").astype(int)
n = len(puestos)
fig, eje = plt.subplots(figsize=(5.6, 5), facecolor="#FBFAF7"); eje.set_facecolor("#FBFAF7")
eje.plot([1, n], [1, n], color="#9AA5B1", ls="--", lw=1)             # sobre esta línea, el puesto se repite
eje.scatter(puestos["validación"], puestos["prueba"], s=45, zorder=3, color=["#D7263D" if m == mejor else "#1D4E89" for m in puestos.index])
for m, fila in puestos.iterrows():
    eje.annotate(m, (fila["validación"], fila["prueba"]), xytext=(6, 3), textcoords="offset points", fontsize=7.5)
eje.set(title="Puesto en validación contra puesto en prueba", xlabel="puesto en validación (1 es el mejor)", ylabel="puesto en prueba",
        xticks=range(1, n + 1), yticks=range(1, n + 1), xlim=(0.5, n + 2.5), ylim=(n + 0.5, 0.5))
plt.show()

en_orden = comparacion.drop(index="edad y sexo")["validación"].sort_values(ascending=False)
print(f"El ganador en validación ({mejor}) quedó de puesto {puestos.loc[mejor, 'prueba']} en prueba, entre {n} candidatos.")
print(f"Ventaja del primero sobre el segundo en validación: {en_orden.iloc[0] - en_orden.iloc[1]:.4f} de AUC, con {int(V.sum())} trazos.")
print("Modelos que cambiaron de puesto entre validación y prueba:", int((puestos["validación"] != puestos["prueba"]).sum()), "de", n)
'''),
("md", r'''
Si el orden de validación se repitiera en la prueba, todos los puntos caerían sobre la diagonal. Miren cuánto se alejan y comparen esa distancia con la ventaja del primero sobre el segundo. Con tantos candidatos, ¿cuánto del ganador es azar? **Mi respuesta:**

Comparen con su segunda apuesta. Los modelos del catálogo leen la misma tabla y varios quedan cerca unos de otros. Esa cercanía es una lección sobre las señales: buena parte del resultado se decidió en la sección 2.5, cuando escogimos qué medirle al trazo.

### 3.3 La trampa: partir por pedazos en vez de por paciente

Con señales es común cortar cada registro en ventanas para tener más filas. Hagámoslo: cada trazo de diez segundos se parte en cuatro ventanas de 2,5 segundos, y cada ventana hereda la etiqueta de su trazo. Hay cuatro filas por trazo y dos maneras de separar entrenamiento y prueba.

- **Al azar por ventana**, que es lo que hace `train_test_split` por defecto. Tres ventanas de un trazo pueden quedar en entrenamiento y la cuarta en prueba.
- **Por paciente**: todas las ventanas de todos los trazos de un paciente quedan del mismo lado.

**Predicción escrita:** ¿cuál dará mejor resultado, y a cuál le creen? **Mi apuesta:**
'''),
("code", r'''
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def tabla_ventanas(lote):
    v = lote.reshape(len(lote) * 4, 250, lote.shape[2])              # cuatro ventanas de 2,5 s por trazo
    return np.concatenate([v.std(axis=1), v.max(axis=1) - v.min(axis=1), stats.skew(v, axis=1),
                           np.abs(np.diff(v, axis=1)).mean(axis=1)], axis=1)

inicio = time.time()
W = np.nan_to_num(np.concatenate([tabla_ventanas(X[i:i + 2000]) for i in range(0, len(X), 2000)]))
y_w, paciente_w = np.repeat(y, 4), np.repeat(datos["patient_id"].values, 4)

candidatos = {"k vecinos (k = 5)": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
              "boosting": lambda: HistGradientBoostingClassifier(random_state=0)}
trampa = {}
for nombre, crear in candidatos.items():
    a, b_ = train_test_split(np.arange(len(y_w)), test_size=0.2, random_state=0, stratify=y_w)
    al_azar = roc_auc_score(y_w[b_], crear().fit(W[a], y_w[a]).predict_proba(W[b_])[:, 1])
    a, b_ = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(W, y_w, paciente_w))
    por_paciente = roc_auc_score(y_w[b_], crear().fit(W[a], y_w[a]).predict_proba(W[b_])[:, 1])
    trampa[nombre] = {"AUC, ventanas al azar": al_azar, "AUC, por paciente": por_paciente, "inflado": al_azar - por_paciente}
print(f"{len(X)} trazos, {len(W)} ventanas, {W.shape[1]} columnas | {time.time() - inicio:.0f} s")
display(pd.DataFrame(trampa).T.round(3))
'''),
("md", r'''
Las dos columnas miden el mismo modelo sobre las mismas ventanas, así que la diferencia viene del examen. Al barajar ventanas, el modelo encuentra en el entrenamiento otros pedazos del mismo trazo que le preguntan en la prueba. Reconoce al paciente y parece que reconociera el infarto. Los vecinos más cercanos son los que más se inflan porque su oficio es buscar parecidos, y nada se parece más a un pedazo de un trazo que otro pedazo del mismo trazo.

En producción todos los pacientes son nuevos. Por eso el único examen que se parece a la realidad es el que separa por paciente, y por eso `strat_fold` vale tanto: los autores ya dejaron a cada paciente en una sola partición. Esta misma trampa los espera en el gemelo, con grabaciones en lugar de pacientes.

### 3.4 Explicar: ¿qué parte de la señal sostiene la predicción?

Respondo con dos experimentos sobre la validación. En el primero entreno el modelo ganador con una sola familia de columnas a la vez. En el segundo uso **importancia por permutación** agrupada por derivación: desordeno entre registros todas las columnas de una derivación $d$ y mido cuánto cae el AUC,

$$I_d=\text{AUC}(\text{validación})-\text{AUC}\bigl(\text{validación con las columnas de } d \text{ permutadas}\bigr).$$

Si el modelo no usaba esa derivación, desordenarla no cambia nada.
'''),
("code", r'''
familias = {"estadísticas": [c for c in C if c.startswith("est_")], "espectro": [c for c in C if c.startswith("esp_")],
            "ritmo": [c for c in C if c.startswith("rit_")], "forma del latido": [c for c in C if c.startswith("lat_")]}
por_familia = {}
for nombre, columnas in familias.items():
    modelo = clone(MODELOS[mejor]).fit(C.loc[E, columnas].values, y[E])
    por_familia[nombre] = roc_auc_score(y[V], puntaje(modelo, C.loc[V, columnas].values))
por_familia["todas"] = comparacion.loc[mejor, "validación"]

azar = np.random.default_rng(0)
referencia = roc_auc_score(y[V], puntaje(entrenados[mejor], A[V]))
caida = {}
for d in DERIVACIONES_USADAS:
    columnas = [i for i, c in enumerate(C.columns) if c.endswith("_" + d)]
    caidas = []
    for _ in range(5):
        permutada = A[V].copy()
        permutada[:, columnas] = permutada[azar.permutation(V.sum())][:, columnas]
        caidas.append(referencia - roc_auc_score(y[V], puntaje(entrenados[mejor], permutada)))
    caida[d] = np.mean(caidas)

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 3.8))
pd.Series(por_familia).sort_values().plot.barh(ax=izq, color="#1D4E89")
izq.axvline(auc_base["validación"], color="gray", ls=":"); izq.set(title=f"AUC en validación de «{mejor}» con una sola familia", xlim=(0.5, 1))
pd.Series(caida).plot.bar(ax=der, color="#D7263D", rot=0)
der.set(title="Caída del AUC al desordenar cada derivación", ylabel="puntos de AUC")
plt.tight_layout(); plt.show()
print("AUC por familia:", {n: round(float(v), 3) for n, v in por_familia.items()})
print("Derivaciones que más pesan:", pd.Series(caida).sort_values(ascending=False).head(4).round(4).to_dict())
'''),
("md", r'''
La línea punteada de la izquierda es la línea base de edad y sexo. Lean cuál familia llega más lejos sola y cuánto agrega juntarlas todas.

Dos cuidados al leer la gráfica de la derecha. Las derivaciones están correlacionadas entre sí, así que cuando se desordena una, el modelo todavía puede apoyarse en sus vecinas, y la caída subestima lo que esa derivación sabe. Además esta importancia describe en qué se apoya **este modelo** y nada afirma sobre la fisiología del infarto. Un cardiólogo es quien puede decir si el orden tiene sentido clínico.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: el umbral de la alerta

La regla quedó escrita en la sección 1: la alerta debe atrapar al menos 90 de cada 100 infartos. El umbral se fija en validación como el más alto que cumple la regla,

$$\tau^\ast=\max\{\tau:\ \text{sensibilidad}_{\text{validación}}(\tau)\ge s_0\},\qquad s_0=0{,}90,$$

y con ese umbral quieto se mide lo que pasa en la prueba. La prueba tiene un número finito de trazos, así que la sensibilidad medida tiene incertidumbre. La calculo por *bootstrap*: remuestreo los trazos de prueba con reemplazo $B=1000$ veces y tomo los percentiles 2,5 y 97,5 de las sensibilidades que salen,

$$IC_{95\,\%}=\bigl[\hat\theta^{\ast}_{(0{,}025)},\ \hat\theta^{\ast}_{(0{,}975)}\bigr].$$
'''),
("code", r'''
ganador = entrenados[mejor]
s_val, s_prueba = puntaje(ganador, A[V]), puntaje(ganador, A[P_])

fp, vp, umbrales = roc_curve(y[V], s_val)
umbral = umbrales[vp >= SENSIBILIDAD_MINIMA][0]                       # el umbral más alto que cumple la regla en validación

alerta = s_prueba >= umbral
yp = y[P_]
sens = (alerta & (yp == 1)).sum() / (yp == 1).sum()
espe = (~alerta & (yp == 0)).sum() / (yp == 0).sum()
vpp = (alerta & (yp == 1)).sum() / alerta.sum()

remuestreos = []
for _ in range(1000):
    i = azar.integers(0, len(yp), len(yp))
    remuestreos.append(((alerta[i] & (yp[i] == 1)).sum() / (yp[i] == 1).sum(), (~alerta[i] & (yp[i] == 0)).sum() / (yp[i] == 0).sum()))
ic = np.percentile(remuestreos, [2.5, 97.5], axis=0) * 100

print(f"Modelo: {mejor} | umbral fijado en validación: {umbral:.3f}")
print(f"En la prueba ({len(yp)} trazos, {int((yp == 1).sum())} con infarto):")
print(f"  Sensibilidad:   {sens * 100:.1f} %  (IC 95 %: {ic[0, 0]:.1f} a {ic[1, 0]:.1f})")
print(f"  Especificidad:  {espe * 100:.1f} %  (IC 95 %: {ic[0, 1]:.1f} a {ic[1, 1]:.1f})")
print(f"  De cada 100 trazos normales, quedan marcados por error: {(1 - espe) * 100:.0f}")
print(f"  De cada 100 trazos marcados, tienen infarto: {vpp * 100:.0f}")
print(f"  La alerta marca el {alerta.mean() * 100:.0f} % de los trazos que llegan; infartos que se le escapan: {int((~alerta & (yp == 1)).sum())} de {int((yp == 1).sum())}")

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 3.6))
izq.hist(s_prueba[yp == 0], bins=40, alpha=0.6, color="#1D4E89", label="normal", density=True)
izq.hist(s_prueba[yp == 1], bins=40, alpha=0.6, color="#D7263D", label="con infarto", density=True)
izq.axvline(umbral, color="black", ls="--"); izq.set(title="Puntajes en la prueba y umbral", xlabel="puntaje"); izq.legend()
fp_p, vp_p, _ = roc_curve(yp, s_prueba)
der.plot(1 - fp_p, vp_p, color="#1D4E89"); der.scatter([espe], [sens], color="#D7263D", zorder=3)
der.set(title="Lo que cuesta cada nivel de sensibilidad (prueba)", xlabel="especificidad", ylabel="sensibilidad")
plt.tight_layout(); plt.show()
'''),
("md", r'''
Comparen con su tercera apuesta. Lean la sensibilidad medida contra la regla del 90 %. El umbral se fijó con los trazos de validación y la prueba es otra muestra, así que la sensibilidad en prueba puede quedar por encima o por debajo de la meta. Si el intervalo deja por fuera el 90 %, la regla no se cumplió en la prueba, y eso se le dice al hospital con la misma claridad que el AUC. Quien quiera una alerta que cumpla la meta con holgura tiene dos caminos: fijar el umbral con más trazos, por ejemplo con validación cruzada dentro del entrenamiento, o pedir en validación una sensibilidad más alta que la meta.

Esta alerta ordena la fila del cardiólogo. No diagnostica ni reemplaza la lectura: un trazo sin alerta también se lee, solo que después.

### La decisión en tres frases

Escríbanla para la gerente del hospital, que no sabe qué es un AUC. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Comparar «solo NORM» contra «con MI» | Las cinco superclases a la vez | Una pregunta limpia para una clase; deja por fuera 7.261 registros y vuelve el examen más fácil que la realidad |
| Usar todos los códigos anotados, sin filtrar por verosimilitud | Exigir verosimilitud de 100 | La verosimilitud 0 significa «desconocida» según la documentación |
| Edad 90 para los mayores de 89 | Borrar esos registros | Es la cota inferior de lo que se sabe; solo entra en la línea base |
| Sexo: 0 hombre, 1 mujer | Dejarlo sin interpretar | Inferencia a partir del 52 % de hombres que reporta PhysioNet |
| No usar `report`, `infarction_stadium*` ni `heart_axis` | Usarlas como entradas | Son anotaciones del cardiólogo: filtran la respuesta |
| Versión de 100 Hz | Versión de 500 Hz | Cinco veces menos datos; el filtro deja pasar hasta 40 Hz, que caben en 100 Hz |
| Filtro pasabanda de 0,5 a 40 Hz | Señal cruda | Quita la deriva de la línea de base y el zumbido de la red |
| Cuatro componentes principales por derivación, ajustados en entrenamiento | Las 70 muestras del latido | Guardan la mayor parte de la varianza con 17 veces menos columnas |
| Particiones 1 a 8, 9 y 10 de `strat_fold` | Separación al azar | Recomendación de los autores; ningún paciente queda en dos lados |
| Umbral con sensibilidad mínima de 90 % en validación | Umbral de 0,5 | Un infarto que se escapa cuesta más que una falsa alarma |

### 4.3 Riesgos y mantenimiento

Este dato es de personas. Respondan en dos o tres líneas cada pregunta.

- Los trazos son de pacientes alemanes de 1989 a 1996, tomados con aparatos de un solo fabricante. ¿Qué habría que medir antes de usar este modelo en un hospital de Caldas en 2026?
- Vimos que la proporción de infarto cambia según el aparato. ¿Cómo comprobarían si el modelo aprendió algo del aparato y no solo del corazón?
- El modelo nunca vio trazos con otros diagnósticos. ¿Qué creen que hará con un trastorno de conducción, y cómo lo medirían con los 7.261 registros que dejamos por fuera?
- ¿Quién responde si la alerta no se prende y el trazo era un infarto? ¿Qué debe decir el protocolo del hospital para que la alerta ordene la fila sin reemplazar la lectura?
- Los datos son anónimos y públicos. ¿Qué cambiaría en el manejo del dato si el hospital quisiera reentrenar el modelo con sus propios pacientes?

### 4.4 Cambien una línea

Un reloj inteligente mide una sola derivación, parecida a la I. Vuelvan a la celda de parámetros, cambien la primera línea por

```python
DERIVACIONES_USADAS = ["I"]
```

y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Los trazos ya están descargados, así que esta vez no hay espera. Llenen la tabla con sus dos corridas.

| | 12 derivaciones | Solo la derivación I |
|---|---|---|
| Columnas de la tabla de características | | |
| Ganador en validación | | |
| AUC del ganador en prueba | | |
| Especificidad en prueba con la alerta del 90 % | | |
| Trazos normales marcados por error, de cada 100 | | |

Con la tabla llena, respondan en tres líneas: ¿cuánto se pierde al pasar de doce derivaciones a una? ¿Sirve un reloj para ordenar la fila del cardiólogo? **Mi respuesta:**
'''),
]
