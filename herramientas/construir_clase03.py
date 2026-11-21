"""Genera los cuadernos de la clase 3: La factura de Londres (demostración) y
El cielo de Caldas (gemelo).

El cuaderno de los estudiantes sale del mismo texto que la solución: lo que va
entre « » en el código se reemplaza por ____.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno

from clase03_londres import LONDRES      # el cuaderno compacto de Londres vive en su propio archivo

# ------------------------------------------------------------ EL CIELO DE CALDAS
CIELO = [
("md", r'''
# Clase 3 · El cielo de Caldas (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *La factura de Londres*: la misma técnica en otro mundo.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase con un dato que llega por pedazos desde una API: resumir cada pedazo, juntar los resúmenes, versionar cada tabla con su huella y amarrar el resultado a la versión del dato. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 2 (seleccionar y preparar datos), 3 (flujo reproducible y trazable), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que les ayude a llenar un hueco o que interprete un error. Deben poder explicar sin ayuda por qué el resumen guarda suma y cuenta, qué es un día completo, qué dice la huella de una tabla, por qué aquí no sirve el MAPE y cómo se lee la tabla de umbrales. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, la decisión en tres frases, la bitácora y la tabla de la sección 4.5 llenas. |

**Cómo se trabaja.** El código está escrito casi todo. Donde vean `____` falta una pieza, y el comentario de al lado dice qué va. Lean la celda entera antes de llenar. Si una celda falla, lean el error antes de pedir ayuda.
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

En Caldas la lluvia tumba montañas. La Unidad Nacional para la Gestión del Riesgo de Desastres (UNGRD) registra cada emergencia que atiende, y en el departamento la más frecuente es el movimiento en masa. El IDEAM, por su parte, publica la lluvia que miden sus estaciones automáticas cada pocos minutos.

Imagínense la oficina de gestión del riesgo del departamento. Tiene que decidir cuándo declara una alerta por lluvias. Si la declara tarde, la emergencia la encuentra sin preparar. Si la declara todos los días, nadie le vuelve a hacer caso. Necesita tres respuestas:

1. **Predecir.** ¿Cuánto lloverá mañana en la estación?
2. **Explicar.** ¿Qué lluvia acumulada antecede a los movimientos en masa?
3. **Decidir.** ¿Con qué umbral de lluvia acumulada se declara la alerta, y cuántas falsas alarmas cuesta?

**El dato.** Dos fuentes. La lluvia de una estación del IDEAM, que llega en pedazos de un año. Y las emergencias de la UNGRD en Caldas, de 2019 a 2025, que vienen repartidas en tres conjuntos.

**La métrica.** Para el pronóstico usamos el error absoluto medio, en milímetros. Llamo $L_t$ a la lluvia del día $t$:

$$\text{MAE}=\frac{1}{n}\sum_t\bigl|L_{t+1}-\hat L_{t+1}\bigr|.$$

No usamos el MAPE de la clase porque divide por el valor real, y aquí hay muchos días en que no llueve.

### Predicción escrita

Escriban su apuesta antes de correr nada.

- ¿En qué fracción de los días llueve en la estación? **Mi apuesta:**
- Si hoy llovió mucho, ¿mañana también? ¿Qué tanto ayuda la lluvia de hoy a adivinar la de mañana? **Mi apuesta:**
- ¿Qué antecede mejor a un movimiento en masa: la lluvia del día anterior, la de los tres días anteriores o la del último mes? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

### 2.1 Un pedazo, mirado con calma

La API de datos.gov.co se cae cuando se le pide mucho de una vez. Por eso la lluvia se pide por pedazos: una estación y un año cada vez. La estación de arranque es Cenicafé, que queda entre Manizales y Chinchiná.
'''),
("code", r'''
ESTACION = "0026155502"      # Cenicafé. Al final lo cambiarán por "0023025502", que es Almacafé Letras
ANIOS = range(2018, 2024)    # los pedazos que se piden: un año cada uno
'''),
("code", r'''
pedazo = descargar("lluvia_ideam", estacion=ESTACION, anio=2019)
memoria = pedazo.memory_usage(deep=True).sum() / 1e6
print(f"Filas: {len(pedazo)} | memoria: {memoria:.0f} MB")
pedazo.head()
'''),
("md", r'''
Como en clase, antes de procesar todo se mide un pedazo y se le buscan las trampas.
'''),
("code", r'''
print("Filas repetidas idénticas:", int(pedazo.duplicated().sum()))
momento = pd.to_datetime(pedazo["fechaobservacion"].drop_duplicates())
print("Minutos entre una lectura y la siguiente (los más frecuentes):")
print((momento.sort_values().diff().dt.total_seconds() / 60).value_counts().head(4).to_string())
por_dia = pedazo.drop_duplicates().groupby(pedazo["fechaobservacion"].str[:10]).size()
print("\nLecturas por día (las más frecuentes):"); print(por_dia.value_counts().head(4).to_string())
print(f"\nDías con alguna lectura: {len(por_dia)} de 365 | lecturas en cero: {(pedazo['valorobservado'] == 0).mean() * 100:.0f} %")
'''),
("md", r'''
Tres cosas que anotar. Puede haber filas repetidas, y como la lluvia de un día es una suma, cada repetida contaría doble. La estación no entrega una lectura cada diez minutos, como dice la descripción del conjunto: miren el intervalo más frecuente. Y hay días con menos lecturas de las que tocan, y días que no aparecen.

Respondan: ¿cuántas lecturas tiene un día completo en esta estación? **Mi respuesta:**

### 2.2 Resumir cada pedazo y juntar los resúmenes

La misma idea de la clase. Cada pedazo se resume a una fila por día con dos números, la suma y la cuenta,

$$s_d=\sum_{i\in d}x_i,\qquad n_d=\#\{i\in d\},$$

y se suelta. Además se guarda un segundo resumen, la lluvia por día y por hora, que sirve para la sección 2.8.
'''),
("code", r'''
def resumir(crudo):
    """De lecturas cada pocos minutos a una fila por día, y a una fila por día y hora."""
    crudo = crudo.«drop_duplicates»()                                            # HUECO 1: el método que deja una sola copia de cada fila repetida
    crudo = crudo.assign(mm=pd.to_numeric(crudo["valorobservado"], errors="coerce"),
                         fecha=crudo["fechaobservacion"].str[:10], hora=crudo["fechaobservacion"].str[11:13].astype(int))
    diario = crudo.groupby("fecha")["mm"].agg(mm=«"sum"», lecturas="count").reset_index()   # HUECO 2: la operación que convierte lecturas en lluvia del día, entre comillas
    por_hora = crudo.groupby(["fecha", "hora"])["mm"].sum().reset_index()
    return diario, por_hora

import time
resumenes, horarios, filas_leidas, inicio = [], [], 0, time.time()
for anio in ANIOS:
    crudo = descargar("lluvia_ideam", estacion=ESTACION, anio=anio)
    if len(crudo) == 0:
        print(f"  {anio}: la estación no tiene datos"); continue
    diario, por_hora = resumir(crudo)
    resumenes.append(diario); horarios.append(por_hora); filas_leidas += len(crudo)
    print(f"  {anio}: {len(crudo)} lecturas -> {len(diario)} días | {time.time() - inicio:4.0f} s")

lluvia_cruda = pd.concat(resumenes).groupby("fecha")[["mm", "lecturas"]].«sum»().reset_index()    # HUECO 3: al juntar resúmenes de suma y cuenta, ¿qué operación se usa?
lluvia_cruda["mm"] = lluvia_cruda["mm"].round(2)
por_hora = pd.concat(horarios, ignore_index=True)
print(f"\nDe {filas_leidas} lecturas quedaron {len(lluvia_cruda)} días.")
'''),
("md", r'''
**Compruébenlo ustedes mismos.** El número de días debe ser menor o igual que los días del calendario entre el primero y el último. Si les sale mayor, revisen el hueco 3.

Recuerden la regla de la clase: se juntan sumas y cuentas, nunca promedios, porque

$$\bar x=\frac{\sum_k s_k}{\sum_k n_k}\quad\text{en general no es}\quad\frac{1}{K}\sum_k\bar x_k.$$

### 2.3 La trampa de la clase: un pedazo no es una muestra

En Londres, el primer archivo traía treinta hogares. Aquí cada pedazo es un año. Miren cuánto llovió y cuántos días se midieron en cada uno.
'''),
("code", r'''
por_pedazo = pd.DataFrame({
    "días con datos": [len(r) for r in resumenes],
    "lluvia total (mm)": [r["mm"].sum().round(0) for r in resumenes],
    "mm por día medido": [round(r["mm"].sum() / len(r), 2) for r in resumenes],
}, index=[r["fecha"].str[:4].iloc[0] for r in resumenes])
display(por_pedazo)
print(f"Con todos los pedazos: {lluvia_cruda['mm'].sum() / len(lluvia_cruda):.2f} mm por día medido")
'''),
("md", r'''
- Si alguien hubiera bajado un solo año para «tener una idea», ¿entre qué valores habría quedado su estimación de la lluvia diaria? **Mi lectura:**
- La lluvia total de un año depende de cuánto llovió y de cuántos días se midieron. ¿Cuál de las tres columnas se puede comparar entre años y cuál no? **Mi respuesta:**

### 2.4 Versionar el dato

Cada tabla que se guarda lleva su huella, la regla que la produjo y su versión madre. La huella es SHA-256 sobre el contenido: si cambia una celda, cambia la huella.
'''),
("code", r'''
import hashlib, json
from datetime import datetime, timezone

carpeta = raiz / "datos" / "cache" / "versiones"
carpeta.mkdir(parents=True, exist_ok=True)

def huella(tabla):
    return hashlib.«sha256»(tabla.to_csv(index=False).encode("utf-8")).hexdigest()[:12]      # HUECO 4: el nombre de la función de huella que usamos en clase

def versionar(tabla, nombre, regla, padre=None):
    """Guarda la tabla en parquet con su huella en el nombre y la anota en el registro."""
    marca = huella(tabla)
    tabla.to_parquet(carpeta / f"{nombre}_{marca}.parquet", index=False)
    registro = carpeta / "registro.json"
    versiones = json.loads(registro.read_text(encoding="utf-8")) if registro.exists() else []
    versiones = [v for v in versiones if v["huella"] != marca]
    versiones.append({"nombre": nombre, "huella": marca, "filas": len(tabla), "regla": regla, "padre": «padre»,     # HUECO 5: la variable que dice de qué versión salió esta
                      "guardada": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
    registro.write_text(json.dumps(versiones, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Versión {marca}: {len(tabla)} filas | {regla}")
    return marca

nombre_tabla = f"lluvia_{ESTACION}"
v1 = versionar(lluvia_cruda, nombre_tabla, f"IDEAM s54a-sgyg, estación {ESTACION}, {ANIOS[0]} a {ANIOS[-1]}; suma y cuenta por día; sin filas repetidas")
'''),
("md", r'''
Comparen su huella con la de un compañero. El IDEAM corrige y actualiza esta serie, así que dos personas que bajen el dato en días distintos pueden tener tablas distintas sin saberlo. Si las huellas no coinciden, miren en la salida de `descargar` si alguno de los dos está leyendo la copia de respaldo. Anoten aquí su huella: **Mi huella:**

### 2.5 Limpieza: días completos

Un día al que le faltan lecturas subestima la lluvia, porque lo que no se midió entra a la suma como cero. **Decisión:** solo entran los días con al menos el 95 % de las lecturas que tiene un día completo en esta estación.
'''),
("code", r'''
esperadas = int(lluvia_cruda["lecturas"].mode().iloc[0])            # las lecturas de un día completo: el valor más frecuente
completo = lluvia_cruda["lecturas"] >= «0.95» * esperadas             # HUECO 6: la fracción mínima de lecturas para aceptar un día
print(f"Un día completo tiene {esperadas} lecturas. Días completos: {completo.sum()} de {len(lluvia_cruda)}")

lluvia_limpia = lluvia_cruda.loc[completo, ["fecha", "mm"]].reset_index(drop=True)
v2 = versionar(lluvia_limpia, nombre_tabla, "solo días con al menos 95 % de las lecturas", padre=«v1»)        # HUECO 7: la huella de la versión de la que sale esta

lluvia = lluvia_limpia.set_index(pd.to_datetime(lluvia_limpia["fecha"]))["mm"].asfreq("D")      # los días sin medición quedan como faltantes
print(f"Serie: del {lluvia.index.min().date()} al {lluvia.index.max().date()} | {lluvia.notna().sum()} días medidos y {lluvia.isna().sum()} faltantes")
display(pd.DataFrame(json.loads((carpeta / "registro.json").read_text(encoding="utf-8"))).query("nombre == @nombre_tabla").tail(4))

medidos_por_mes = lluvia.notna().groupby(lluvia.index.to_period("M")).mean() * 100
medidos_por_mes.plot(color="#1D4E89", figsize=(10, 2.8)); plt.title("% de días medidos en cada mes"); plt.xlabel(""); plt.show()
'''),
("md", r'''
A diferencia de Londres, aquí no interpolamos los días que faltan. Un día sin medición puede haber sido un aguacero, y rellenarlo con el promedio de sus vecinos inventaría una lluvia que nadie midió. Los faltantes se quedan como faltantes.

### 2.6 La etiqueta: las emergencias de la UNGRD

La UNGRD publica sus emergencias en tres conjuntos con columnas distintas, uno por periodo. La función de descarga pide las cuatro columnas que comparten: fecha, departamento, municipio y evento.
'''),
("code", r'''
emergencias = descargar("emergencias_ungrd", departamento="CALDAS")
emergencias["fecha"] = pd.to_datetime(emergencias["fecha"].str[:10])
emergencias["evento"] = emergencias["evento"].str.upper().str.strip()
print(emergencias["evento"].value_counts().head(6).to_string())

masa = emergencias[emergencias["evento"] == «"MOVIMIENTO EN MASA"»]                    # HUECO 8: el nombre del evento que nos interesa, como aparece arriba
print(f"\nMovimientos en masa en Caldas: {len(masa)}, en {masa['fecha'].nunique()} días distintos")
print("Por año:", masa.groupby(masa["fecha"].dt.year).size().to_dict())
print("Municipios con más reportes:", masa["municipio"].str.title().value_counts().head(5).to_dict())

hubo_evento = (masa.groupby("fecha").size().reindex(lluvia.index).fillna(0) > 0).astype(int)
print(f"\nDías de la serie de lluvia con al menos un movimiento en masa en Caldas: {hubo_evento.sum()} de {len(hubo_evento)}")
'''),
("md", r'''
Miren los reportes por año. Hay años con muy pocos y un año con muchísimos. Respondan: ¿eso describe cómo llovió o cómo se reportó? ¿Qué le hace a un modelo una etiqueta que depende de quién reporta? **Mi respuesta:**

Hay otra distancia que conviene decir en voz alta. La lluvia es de **una** estación y las emergencias son de **todo** el departamento. Un deslizamiento en Samaná, a tres horas de la estación, cuenta igual que uno en Chinchiná. Eso debilita la relación que vamos a medir, y va a los riesgos.

### 2.7 Conocer el dato antes de modelarlo

Cuatro miradas: cómo se reparte la lluvia diaria, en qué meses llueve, a qué horas y cuánta memoria tiene.
'''),
("code", r'''
meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
fig, ejes = plt.subplots(2, 2, figsize=(13, 7.5))

ejes[0, 0].hist(lluvia.dropna(), bins=60, color="#1D4E89"); ejes[0, 0].set_yscale("log")
ejes[0, 0].set(title="Lluvia diaria (eje vertical logarítmico)", xlabel="mm", ylabel="días")

por_mes = lluvia.groupby(lluvia.index.month).mean() * 30
ejes[0, 1].bar([meses[m - 1] for m in por_mes.index], por_mes.values, color="#5BC0EB"); ejes[0, 1].set(title="Lluvia de un mes típico", ylabel="mm")

fechas_completas = set(lluvia_limpia["fecha"])
horas = por_hora[por_hora["fecha"].isin(fechas_completas)]
ejes[1, 0].bar(range(24), horas.groupby("hora")["mm"].mean().reindex(range(24)).values, color="#1B998B")
ejes[1, 0].set(title="Lluvia promedio por hora del día", xlabel="hora", ylabel="mm")

rezagos = range(1, 15)
ejes[1, 1].bar(rezagos, [lluvia.autocorr(k) for k in rezagos], color="#D7263D")
ejes[1, 1].axhline(2 / np.sqrt(lluvia.notna().sum()), color="gray", ls=":")
ejes[1, 1].set(title="Autocorrelación de la lluvia diaria", xlabel="rezago (días)")
plt.tight_layout(); plt.show()

print(f"Días con lluvia (más de 0,1 mm): {(lluvia.dropna() > 0.1).mean() * 100:.0f} % | media: {lluvia.mean():.1f} mm | mediana: {lluvia.median():.1f} mm | máximo: {lluvia.max():.0f} mm")
print(f"El 10 % de los días más lluviosos aporta {lluvia.dropna().nlargest(int(lluvia.notna().sum() * 0.1)).sum() / lluvia.sum() * 100:.0f} % de toda la lluvia.")
print(f"Autocorrelación de un día: {lluvia.autocorr(1):.2f}")
'''),
("md", r'''
Comparen con sus dos primeras apuestas y respondan en una línea cada una.

- La media y la mediana quedan lejos. ¿Qué dice eso de la forma de la distribución, y de lo que significa «un día típico»? **Mi lectura:**
- ¿A qué horas llueve? El conjunto no dice si la hora es la de Colombia o la universal (UTC), que va cinco horas adelante. ¿Cómo cambiaría su lectura en cada caso? **Mi lectura:**
- La autocorrelación de un día para el consumo de Londres era casi uno. ¿Cuánto es aquí? ¿Qué anticipa eso sobre el pronóstico de mañana? **Mi lectura:**

### 2.8 PCA: ¿cuántas maneras hay de llover en un día?

Para cada día con al menos 5 mm de lluvia tomo las 24 horas y las divido por el total del día. Queda una matriz con una fila por día lluvioso y 24 columnas, que describe a qué horas cayó el agua. Descompongo su matriz de covarianzas:

$$S=V\Lambda V^\top,\qquad \text{varianza explicada por el componente } j=\frac{\lambda_j}{\sum_k\lambda_k}.$$

**Predicción escrita:** ¿qué horas opondrá el primer componente? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA

matriz = horas.pivot_table(index="fecha", columns="hora", values="mm", aggfunc="sum").reindex(columns=range(24)).fillna(0)
matriz = matriz[matriz.sum(axis=1) >= 5]
forma = matriz.div(matriz.sum(axis=1), axis=0)
pca = PCA(n_components=4).fit(forma - forma.mean())
explicada = pca.explained_variance_ratio_ * 100

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 3.8))
izq.bar(range(1, 5), explicada, color="#1D4E89"); izq.set(title="Varianza explicada", xlabel="componente", ylabel="%")
der.plot(range(24), pca.components_[0], color="#D7263D", marker="o", label="CP1"); der.plot(range(24), pca.components_[1], color="#5BC0EB", marker="o", label="CP2")
der.axhline(0, color="gray", lw=0.5); der.set(title="Cargas por hora del día", xlabel="hora"); der.legend()
plt.tight_layout(); plt.show()
print(f"Días con 5 mm o más: {len(forma)} | Los dos primeros componentes explican {explicada[:2].sum():.0f} % de la varianza.")
'''),
("md", r'''
- ¿Qué horas tienen carga positiva y cuáles negativa en el primer componente? Descríbanlo como dos tipos de día lluvioso. **Mi lectura:**
- En Londres el primer componente explicaba casi la mitad de la forma del año. ¿Aquí la lluvia es igual de ordenada? **Mi lectura:**
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base

Dos pronósticos que no cuestan nada:

$$\hat L^{\text{ing}}_{t+1}=L_t\qquad\text{y}\qquad \hat L^{\text{cli}}_{t+1}=\bar L_{\text{entrenamiento}}.$$

El primero dice que mañana llueve lo mismo que hoy. El segundo dice que mañana llueve el promedio histórico, sin mirar el día de hoy. A ese segundo se le llama **climatología**.

Las entradas de los modelos son la lluvia de hoy y de ayer, los acumulados de 3, 7 y 30 días, si hoy llovió y la época del año, escrita como $\sin(2\pi m/12)$ y $\cos(2\pi m/12)$ con $m$ el mes. Las filas se ordenan por fecha y se parten como siempre:

$$\underbrace{\text{primer } 60\,\%}_{\text{entrenamiento}}\ \Big|\ \underbrace{20\,\%\ \text{siguiente}}_{\text{validación: se escoge}}\ \Big|\ \underbrace{\text{último } 20\,\%}_{\text{prueba: se reporta}}$$
'''),
("code", r'''
def acumulado(dias):
    """Lluvia de los últimos `dias` días, contando hoy. Exige que al menos el 80 % de esos días tenga medición."""
    return lluvia.rolling(dias, min_periods=int(np.ceil(0.8 * dias))).sum()

manana = lluvia.index + pd.Timedelta(days=1)
X = pd.DataFrame({
    "hoy": lluvia,
    "ayer": lluvia.shift(1),
    "acumulado_3": acumulado(3),
    "acumulado_7": acumulado(7),
    "acumulado_30": acumulado(30),
    "hoy_llovio": (lluvia > 0.1).astype(float).where(lluvia.notna()),
    "epoca_sen": np.sin(2 * np.pi * manana.month / 12),
    "epoca_cos": np.cos(2 * np.pi * manana.month / 12),
}, index=lluvia.index)
base = X.join(lluvia.shift(«-1»).rename("real")).dropna()                    # HUECO 10: el desplazamiento que trae la lluvia de mañana a la fila de hoy
variables = list(X.columns)

n = len(base); posicion = np.arange(n)
E, V, P = posicion < int(0.6 * n), (posicion >= int(0.6 * n)) & (posicion < int(0.8 * n)), posicion >= int(0.8 * n)
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {tramo.sum():4d} días | {base.index[tramo].min().date()} a {base.index[tramo].max().date()} | {base['real'][tramo].mean():.1f} mm por día")

def mae(real, pronostico):
    return float(abs(real - «pronostico»).mean())                             # HUECO 11: lo que se le resta al valor real para medir el error

climatologia = base["real"][E].mean()
lineas_base = {"ingenuo": base["hoy"], "climatología": pd.Series(climatologia, index=base.index)}
errores = {nombre: {"validación": mae(base["real"][V], p[V]), "prueba": mae(base["real"][P], p[P])} for nombre, p in lineas_base.items()}
display(pd.DataFrame(errores).T.round(2))
'''),
("md", r'''
Antes de seguir, respondan: ¿cuál de las dos líneas base gana, y qué dice eso sobre la memoria de la lluvia que vieron en la sección 2.7? **Mi respuesta:**

### 3.2 Cinco modelos y el mismo protocolo de la clase

Los cinco candidatos son los de la clase. Aquí pronostican el nivel de lluvia y no un cambio relativo, porque no se puede dividir por un día sin lluvia.

**Predicción escrita:** ¿algún modelo le ganará a la climatología en la prueba? ¿Por cuánto? **Mi apuesta:**
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
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25)),
        "bosque": RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=0),
        "boosting": GradientBoostingRegressor(random_state=0),
    }

pronosticos = {nombre: p[P] for nombre, p in lineas_base.items()}
for nombre, modelo in cinco_modelos().items():
    modelo.fit(base[variables][E], base["real"][E])
    en_validacion = mae(base["real"][V], modelo.predict(base[variables][V]))
    modelo.fit(base[variables][E | V], base["real"][E | V])
    pronosticos[nombre] = pd.Series(modelo.predict(base[variables][P]), index=base.index[P])
    errores[nombre] = {"validación": en_validacion, "prueba": mae(base["real"][P], pronosticos[nombre])}
todos = pd.DataFrame(errores).T

todos.sort_values("validación").plot.barh(figsize=(9, 4), color=["#1D4E89", "#1B998B"])
plt.xlabel("MAE (mm)"); plt.title("Siete candidatos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(2))

mejor = todos[«"validación"»].idxmin()                                         # HUECO 12: la columna con la que se escoge al ganador
resumen = todos["prueba"].round(2)
resultado = {"dato": v2, "estacion": ESTACION, "modelo": mejor, "mae_prueba": float(resumen[mejor]), "mae_climatologia": float(resumen["climatología"])}
print(f"Ganador en validación: {mejor} | MAE en prueba: {resumen[mejor]} mm | climatología: {resumen['climatología']} mm | ingenuo: {resumen['ingenuo']} mm")
print(resultado)
'''),
("md", r'''
- La mediana de la lluvia diaria es muy baja. Un pronóstico que diga siempre «cero» tendría un MAE pequeño. ¿Le serviría a la oficina de gestión del riesgo? ¿Qué dice eso de la métrica? **Mi respuesta:**
- El resultado quedó guardado con la huella del dato. ¿Para qué sirve ese renglón dentro de seis meses? **Mi respuesta:**

### 3.3 El mismo cálculo sobre otra versión del dato

Repitan la línea base de climatología sobre la primera versión, la que incluía los días incompletos.
'''),
("code", r'''
primera = pd.read_parquet(carpeta / f"{nombre_tabla}_{«v1»}.parquet")           # HUECO 13: la huella de la primera versión
serie_v1 = primera.set_index(pd.to_datetime(primera["fecha"]))["mm"].asfreq("D")
comparacion = pd.DataFrame({
    f"versión {v1} (todos los días)": {"días": serie_v1.notna().sum(), "mm por día": serie_v1.mean(), "lluvia total (mm)": serie_v1.sum()},
    f"versión {v2} (días completos)": {"días": lluvia.notna().sum(), "mm por día": lluvia.mean(), "lluvia total (mm)": lluvia.sum()},
}).T
display(comparacion.round(2))
'''),
("md", r'''
Respondan: ¿en cuál de las tres columnas se nota más el cambio de versión? ¿Cuál de las dos versiones le darían a alguien que quiere saber cuánto llueve en un año? **Mi respuesta:**

### 3.4 Explicar: ¿qué acumulado antecede a los movimientos en masa?

Para cada ventana de $k$ días calculo la lluvia acumulada **hasta ayer**,

$$A^{(k)}_t=\sum_{j=1}^{k}L_{t-j},$$

y mido qué tan bien separa los días con movimiento en masa de los días sin él. La medida es el área bajo la curva ROC, que es la probabilidad de que un día con evento tenga más lluvia acumulada que un día sin evento:

$$\text{AUC}=P\bigl(A_i>A_j\ \big|\ \text{evento en } i,\ \text{sin evento en } j\bigr).$$

Un AUC de 0,5 es no separar nada. La ventana se escoge con los días anteriores a la prueba y se reporta en la prueba.
'''),
("code", r'''
from sklearn.metrics import roc_auc_score

corte = base.index[P].min()                                    # la misma fecha en que empieza la prueba del pronóstico
desde = max(lluvia.index.min(), pd.Timestamp("2019-01-01"))     # la UNGRD publica desde 2019
ventanas = {}
for k in [1, 3, 7, 15, 30]:
    antecedente = acumulado(k).shift(«1»)                       # HUECO 14: cuántos días se corre el acumulado para que termine ayer y no hoy
    tabla_k = pd.DataFrame({"acumulado": antecedente, "evento": hubo_evento})[desde:].dropna()
    antes, despues = tabla_k[:corte - pd.Timedelta(days=1)], tabla_k[corte:]
    ventanas[k] = {"días": len(tabla_k), "eventos antes de la prueba": int(antes["evento"].sum()),
                   "mm con evento": antes.loc[antes["evento"] == 1, "acumulado"].mean(), "mm sin evento": antes.loc[antes["evento"] == 0, "acumulado"].mean(),
                   "AUC antes de la prueba": roc_auc_score(antes["evento"], antes["acumulado"]) if antes["evento"].nunique() == 2 else np.nan,
                   "AUC en la prueba": roc_auc_score(despues["evento"], despues["acumulado"]) if despues["evento"].nunique() == 2 else np.nan}
ventanas = pd.DataFrame(ventanas).T
ventanas.index.name = "ventana (días)"
display(ventanas.round(2))
K = int(ventanas["AUC antes de la prueba"].idxmax())
print(f"Ventana escogida con los días anteriores a la prueba: {K} día(s)")
'''),
("md", r'''
Comparen con su tercera apuesta.

- ¿Qué ventana separa mejor? ¿La diferencia entre ventanas es grande o pequeña? **Mi lectura:**
- El AUC cambia entre antes de la prueba y la prueba. Miren cuántos eventos hay en cada tramo y recuerden lo que respondieron en la sección 2.6. **Mi lectura:**
- Esto es una asociación entre la lluvia de una estación y los reportes de un departamento. ¿Qué otra cosa, además de la lluvia, decide que un movimiento en masa quede reportado? **Mi respuesta:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿con qué umbral se declara la alerta?

La oficina declara alerta el día $t$ si el acumulado de los $K$ días anteriores supera un umbral $u$. Con cada umbral salen dos medidas, las mismas de la clase:

$$\text{aciertos}=\frac{\#\{A_t>u\ \text{y evento}\}}{\#\{\text{evento}\}},\qquad \text{falsas alarmas}=\frac{\#\{A_t>u\ \text{y sin evento}\}}{\#\{A_t>u\}}.$$

Los umbrales candidatos son percentiles del acumulado en los días anteriores a la prueba, para que ningún umbral conozca el futuro.
'''),
("code", r'''
antecedente = acumulado(K).shift(1)
tabla_k = pd.DataFrame({"acumulado": antecedente, "evento": hubo_evento})[desde:].dropna()
antes, despues = tabla_k[:corte - pd.Timedelta(days=1)], tabla_k[corte:]
filas = []
for percentil in [50, 70, 80, 90, 95]:
    u = antes["acumulado"].quantile(percentil / 100)
    alerta, evento = despues["acumulado"] > u, despues["evento"] == 1
    filas.append({"percentil": percentil, "umbral (mm)": u, "días en alerta": int(alerta.sum()), "eventos con alerta": int((alerta & evento).sum()),
                  "eventos sin alerta": int((~alerta & evento).sum()), "alertas en falso": int((alerta & ~evento).sum()),
                  "aciertos": (alerta & evento).sum() / max(evento.sum(), 1), "falsas alarmas": (alerta & ~evento).sum() / max(alerta.sum(), 1)})
decision = pd.DataFrame(filas)
print(f"Estación {ESTACION} | ventana de {K} día(s) | prueba: {len(despues)} días con {int(despues['evento'].sum())} días de evento ({despues['evento'].mean() * 100:.0f} %)")
display(decision.round(2))

fig, eje = plt.subplots(figsize=(11, 3.6))
eje.plot(despues.index, despues["acumulado"], color="#5BC0EB", lw=1, label=f"lluvia acumulada de {K} día(s), hasta ayer")
eje.scatter(despues.index[despues["evento"] == 1], despues.loc[despues["evento"] == 1, "acumulado"], color="#D7263D", s=18, zorder=3, label="día con movimiento en masa en Caldas")
eje.axhline(decision.loc[decision["percentil"] == 80, "umbral (mm)"].iloc[0], color="#1B998B", label="umbral del percentil 80")
eje.set(ylabel="mm", title="La alerta en la prueba"); eje.legend(fontsize=8); plt.show()
'''),
("md", r'''
Lean la tabla de arriba hacia abajo. Cada renglón es una política. La columna de falsas alarmas dice, de cada cien días en alerta, en cuántos no hubo ningún reporte. Fíjense también en la base: si los eventos ocurren en un porcentaje pequeño de los días, hasta una buena regla tendrá muchas alertas en falso.

### La decisión en tres frases

Para la directora de gestión del riesgo, que conoce el territorio y no sabe qué es un percentil.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Pedir la lluvia por estación y por año | Una sola consulta con todo | La API se cae con consultas grandes |
| Resumen con suma y cuenta por día | | |
| Filas repetidas fuera | | |
| Solo días con 95 % de las lecturas | | |
| Los días sin medición no se rellenan | | |
| MAE como métrica del pronóstico | | |
| Etiqueta: cualquier movimiento en masa en Caldas | | |
| Acumulado hasta ayer, no hasta hoy | | |
| Ventana y umbrales escogidos antes de la prueba | | |
| Cada tabla guardada lleva huella, regla y versión madre | | |

### 4.3 Riesgos, ética y mantenimiento

- Una estación no es el cielo de Caldas. ¿Qué habría que agregar para que la alerta sirviera en Samaná o en Marmato?
- La etiqueta depende de que alguien reporte. ¿Qué municipios o qué años pueden estar mal representados, y cómo se nota en sus resultados?
- Una alerta en falso le cuesta credibilidad a la oficina y una alerta que falta puede costar vidas. ¿Cómo cambia eso el renglón de la tabla que ustedes recomiendan?
- El IDEAM corrige la serie. Si dentro de un año alguien repite este cuaderno y le sale otro número, ¿qué dos cosas compararía primero?
- El conjunto no declara la zona horaria. Si las horas estuvieran en UTC, el «día» de este cuaderno iría de las 7 de la noche a las 7 de la noche. ¿Qué resultados habría que recalcular?
- La serie de la estación tiene meses casi sin datos. ¿Qué le pasa a una alerta que depende de un sensor que se apaga justo en invierno?

### 4.4 Procedencia
'''),
("code", r'''
print({nombre: {k: v for k, v in ficha.items() if k in ("fuente", "origen", "descargado", "filas")} for nombre, ficha in procedencia().items() if nombre in ("lluvia_ideam", "emergencias_ungrd")})
print(json.dumps([v for v in json.loads((carpeta / "registro.json").read_text(encoding="utf-8")) if v["huella"] in (v1, v2)], ensure_ascii=False, indent=2))
'''),
("md", r'''
### 4.5 Cambien una línea

Vuelvan a la celda donde dice `ESTACION = "0026155502"`, pongan `ESTACION = "0023025502"` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Esa es la estación Almacafé Letras, en la parte alta de la cordillera. Llenen la tabla con sus dos corridas.

| | Cenicafé | Almacafé Letras |
|---|---|---|
| Lecturas leídas y años con datos | | |
| Lecturas de un día completo | | |
| Huella de la versión limpia | | |
| Días medidos y mm por día | | |
| % de días con lluvia | | |
| Ganador en validación y su MAE en prueba | | |
| MAE de la climatología en prueba | | |
| Ventana escogida y su AUC en la prueba | | |

Con la tabla llena, respondan en tres líneas. ¿Las dos estaciones cuentan la misma historia sobre los movimientos en masa? Si tuvieran que montar la alerta del departamento con una sola, ¿cuál escogerían y qué le dirían a la directora sobre lo que esa estación no ve? **Mi respuesta:**

Dejen el cuaderno entregado con la segunda estación corrida y la tabla completa.

**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase03/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]

if __name__ == "__main__":
    destino = RAIZ / "clases" / "03_datos_grandes"
    destino.mkdir(parents=True, exist_ok=True)
    nbf.write(cuaderno(LONDRES, False), destino / "03a_la_factura_de_londres.ipynb")
    if CIELO:
        nbf.write(cuaderno(CIELO, True), destino / "03b_el_cielo_de_caldas.ipynb")
        (RAIZ / "_profesor").mkdir(exist_ok=True)
        nbf.write(cuaderno(CIELO, False), RAIZ / "_profesor" / "03b_el_cielo_de_caldas_SOLUCION.ipynb")
    print("Cuadernos de la clase 3 escritos.")
