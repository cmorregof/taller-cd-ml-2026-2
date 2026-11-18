"""Celdas del gemelo de la clase 9, «El parte de incendios».

El cuaderno del estudiante es casi vacío: enunciado, criterios de aceptación y la carga del dato.
Cada celda de código es ("code", solución, encargo); las marcadas "dado" salen iguales en los dos.
"""
from molde import ARRANQUE

PARTE = [
("md", r'''
# Clase 9 · El parte de incendios (gemelo, trabajo autónomo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este proyecto lo hacen por su cuenta, fuera del salón. Es el gemelo de *El noticiero de un minuto*: un análisis que termina convertido en un servicio. Tiene dos partes. En este cuaderno justifican con números a qué departamentos mandarían las brigadas. En la carpeta `parte_de_incendios/` construyen la API que entrega el parte.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Construir, probar y desplegar una API sobre datos reales, trabajando con una IA que escribe el código contra pruebas de aceptación que ustedes no escribieron. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Claude escribe la API y, si quieren, el análisis. Ustedes deben poder explicar sin ayuda qué cuenta cada ruta, por qué `/brigadas` solo puede mirar el pasado, cómo midieron su método y qué hace cada código de respuesta. |
| **Entregable** | La dirección de su repositorio, la dirección pública de su API en Railway y este cuaderno corrido, con la sección 3 llena. |
'''),
("dado", ARRANQUE),
("dado", r'''
# El parámetro del cuaderno. Al final se cambia esta línea y se corre todo de nuevo.
N_BRIGADAS = 5            # a cuántos departamentos alcanza a ir la entidad en una semana

import time
'''),
("md", r'''
## 1. El enunciado

Imagínense la sala de crisis de una entidad de gestión del riesgo. Cada lunes alguien pregunta cuántos focos de calor hubo en un departamento y a dónde conviene mandar las brigadas esta semana. Hay brigadas para `N_BRIGADAS` departamentos.

1. **Predecir.** ¿Cuántos focos de calor habrá la próxima semana en cada departamento?
2. **Explicar.** ¿Qué pesa más en esa cifra, la época del año o lo que pasó la semana anterior?
3. **Decidir.** ¿A qué departamentos mandas las brigadas la próxima semana?

**La métrica.** Una propuesta de brigadas es un conjunto $B_t$ de departamentos para la semana $t$. Si $y_{d,t}$ son los focos que de verdad hubo en el departamento $d$, la **cobertura** de la propuesta es la fracción de los focos del país que cayó donde había brigada:

$$\text{cobertura}_t=\frac{\sum_{d\in B_t}y_{d,t}}{\sum_{d}y_{d,t}}.$$

Nadie puede cubrirlo todo con cinco brigadas. El **techo** es la cobertura de quien conociera el futuro y escogiera los departamentos que de verdad tuvieron más focos, $B_t^\ast$. Un método se juzga por cuánto se acerca al techo y por cuánto le gana a la línea base, que es mandar las brigadas a donde hubo más focos la semana pasada.

### Criterios de aceptación

**De la API** (carpeta `clases/09_api/parte_de_incendios/`, léanlos completos en su `README.md`): la entrega se acepta cuando `python pruebas_aceptacion.py` termina con «10 de 10 criterios cumplidos». Las diez pruebas ya están escritas y no se cambian.

**De este cuaderno:**

1. Corre de principio a fin desde cero.
2. Muestra la exploración del dato: focos por año, por mes y por departamento, con su lectura.
3. Compara al menos tres métodos para `/brigadas` contra la línea base, con tramos de entrenamiento, validación y prueba **separados en el tiempo**, y reporta la cobertura y el techo.
4. Muestra con un número qué pasa cuando el método usa datos del futuro.
5. Termina con la propuesta para una semana concreta, su incertidumbre y la decisión en tres frases.

### Predicción escrita

- ¿En qué meses se concentran los focos de calor en Colombia? **Mi apuesta:**
- Con cinco brigadas y conociendo el futuro, ¿qué fracción de los focos de una semana se alcanza a cubrir? **Mi apuesta:**
- ¿Qué servirá más para acertar: saber qué pasó la semana pasada o saber qué pasa en esta época todos los años? **Mi apuesta:**
'''),
("md", r'''
## 2. El dato

NASA FIRMS publica los focos de calor que detecta el sensor MODIS de los satélites Terra y Aqua. Cada fila es un píxel de cerca de un kilómetro donde el satélite vio una anomalía térmica. La función trae los archivos anuales de 2016 a 2025 y le agrega a cada foco su departamento, calculado con los límites de geoBoundaries. Tarda unos veinte segundos.

Según la documentación de FIRMS, `confidence` va de 0 a 100, `frp` es la potencia radiativa del fuego en megavatios, `acq_time` es la hora de paso del satélite en tiempo universal y `type` vale 0 para presunto incendio de vegetación, 1 para volcán activo, 2 para otra fuente fija en tierra y 3 para mar adentro.
'''),
("dado", r'''
focos = descargar("firms_colombia")
focos["fecha"] = pd.to_datetime(focos["acq_date"])
print("Focos:", len(focos), "| de", focos["fecha"].min().date(), "a", focos["fecha"].max().date())
print("Por tipo:", focos["type"].value_counts().to_dict(), "| sin departamento:", int(focos["departamento"].isna().sum()))
focos.head(3)
'''),
("md", r'''
## 3. Su análisis

De aquí en adelante el cuaderno es suyo. Sigan los cuatro módulos del curso: datos y exploración, modelado, y comunicación. Los títulos son una guía.

### 3.1 Limpieza y exploración
'''),
("code", r'''
# Decisiones de limpieza: solo presuntos incendios de vegetación y solo focos que cayeron en un departamento
vegetacion = focos[(focos["type"] == 0) & focos["departamento"].notna()].copy()
print(f"Focos que entran: {len(vegetacion)} de {len(focos)}")
print("Tipo 2 (fuente fija), departamentos con más:", focos[focos["type"] == 2]["departamento"].value_counts().head(3).to_dict())

# La hora es universal: un foco de las 02:00 UTC ocurrió a las 9 de la noche del día anterior en Colombia
hora_utc = focos["acq_time"].astype(int) // 100
print("Focos cuya fecha en Colombia es el día anterior a la fecha del archivo (%):", round(float((hora_utc < 5).mean() * 100), 1))

fig, ejes = plt.subplots(1, 4, figsize=(16, 3.8))
vegetacion.groupby(vegetacion["fecha"].dt.year).size().plot.bar(ax=ejes[0], color="#16302B", rot=45); ejes[0].set(title="Focos por año", xlabel="")
vegetacion.groupby(vegetacion["fecha"].dt.month).size().plot.bar(ax=ejes[1], color="#A8432C", rot=0); ejes[1].set(title="Focos por mes (2016 a 2025)", xlabel="mes")
vegetacion["departamento"].value_counts().head(10).sort_values().plot.barh(ax=ejes[2], color="#E0A526"); ejes[2].set(title="Diez departamentos con más focos", ylabel="")
ejes[3].scatter(vegetacion["longitude"][::10], vegetacion["latitude"][::10], s=0.3, color="#A8432C", alpha=0.4); ejes[3].set(title="Uno de cada diez focos", xlabel="longitud", ylabel="latitud")
plt.tight_layout(); plt.show()

por_mes = vegetacion.groupby(vegetacion["fecha"].dt.month).size()
print("Fracción de los focos entre diciembre y marzo (%):", round(float(por_mes.loc[[12, 1, 2, 3]].sum() / por_mes.sum() * 100), 1))
print("Fracción en los dos departamentos con más focos (%):", round(float(vegetacion["departamento"].value_counts().head(2).sum() / len(vegetacion) * 100), 1))
print("Confianza: mediana", vegetacion["confidence"].median(), "| focos con confianza menor que 30 (%):", round(float((vegetacion["confidence"] < 30).mean() * 100), 1))
''', r'''
# Limpieza (qué focos entran y por qué) y exploración: por año, por mes, por departamento, en el mapa.
'''),
("md", r'''
**Mi lectura** (¿cuándo y dónde se concentran los focos?, ¿qué decisión de limpieza tomaron y por qué?):

### 3.2 Un análisis no supervisado: ¿todos los departamentos tienen la misma temporada?

El perfil estacional del departamento $d$ es el vector de doce números $s_d=(s_{d,1},\dots,s_{d,12})$, con $s_{d,m}$ igual a la fracción de sus focos que cae en el mes $m$. Un PCA sobre esos perfiles dice si hay una sola temporada nacional o varias.
'''),
("code", r'''
from sklearn.decomposition import PCA

mensual = vegetacion.groupby(["departamento", vegetacion["fecha"].dt.month]).size().unstack(fill_value=0)
mensual = mensual[mensual.sum(axis=1) >= 500]                        # departamentos con suficientes focos para tener perfil
perfil = mensual.div(mensual.sum(axis=1), axis=0)
pca = PCA(n_components=2).fit(perfil)
Z = pca.transform(perfil)

fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 4.2))
izq.scatter(Z[:, 0], Z[:, 1], color="#16302B")
for nombre, (a, b) in zip(perfil.index, Z):
    izq.text(a + 0.005, b, nombre, fontsize=7)
izq.set(title="Departamentos según su perfil estacional", xlabel="CP1", ylabel="CP2")
extremos = [perfil.index[Z[:, 0].argmin()], perfil.index[Z[:, 0].argmax()], perfil.index[Z[:, 1].argmax()]]
for nombre, color in zip(extremos, ["#A8432C", "#16302B", "#E0A526"]):
    der.plot(range(1, 13), perfil.loc[nombre] * 100, marker="o", color=color, label=nombre)
der.set(title="Perfil de tres departamentos extremos", xlabel="mes", ylabel="% de sus focos"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("Varianza explicada (%):", (pca.explained_variance_ratio_ * 100).round(1), "| departamentos con 500 focos o más:", len(perfil))
''', r'''
# Perfil estacional por departamento (fracción de focos en cada mes), PCA de los perfiles y su lectura.
'''),
("md", r'''
**Mi lectura** (¿hay una sola temporada de incendios en el país?):

### 3.3 Modelado: la tabla semanal, la línea base y los candidatos

La unidad es la semana. Sea $y_{d,t}$ el número de focos del departamento $d$ en la semana $t$. Todos los candidatos producen un valor esperado $\hat y_{d,t}$ usando solo semanas anteriores a $t$, y la propuesta de brigadas son los `N_BRIGADAS` departamentos con mayor $\hat y_{d,t}$.

| Candidato | $\hat y_{d,t}$ |
|---|---|
| Línea base: la semana pasada | $y_{d,t-1}$ |
| Promedio de cuatro semanas | $\tfrac14\sum_{k=1}^{4}y_{d,t-k}$ |
| Estacional | $c_{d,t}$: promedio de $y_d$ en la misma semana del año, y sus dos vecinas, en los años **anteriores** |
| Estacional con nivel | $c_{d,t}\cdot\rho_{d,t}$, con $\rho_{d,t}=\dfrac{1+\sum_{k=1}^{4}y_{d,t-k}}{1+\sum_{k=1}^{4}c_{d,t-k}}$ acotado entre $\tfrac14$ y 4 |
| Regresión de Poisson | $\log\hat y_{d,t}=\beta_0+\beta^\top x_{d,t}$, con $x$ hecho de rezagos, estacional y época del año |
| *Gradient boosting* de Poisson | Árboles sobre las mismas entradas |

Los tramos van en el tiempo: los modelos se ajustan con 2018 a 2022, se escoge con 2023 y se reporta con 2024 y 2025. Los años 2016 y 2017 solo sirven para que el estacional tenga pasado.
'''),
("code", r'''
semanal = vegetacion.groupby([pd.Grouper(key="fecha", freq="W-SUN"), "departamento"]).size().unstack(fill_value=0)
semanal = semanal.reindex(pd.date_range(semanal.index.min(), semanal.index.max(), freq="W-SUN"), fill_value=0)
semanal = semanal.iloc[:-1] if semanal.index[-1] > vegetacion["fecha"].max() else semanal      # la última semana puede estar incompleta
Y = semanal.values.astype(float)
T, D = Y.shape
semana_del_anio, anio = semanal.index.isocalendar().week.values.astype(int), semanal.index.year.values
print(f"Tabla semanal: {T} semanas x {D} departamentos | semanas sin ningún foco: {int((Y.sum(axis=1) == 0).sum())}")

def estacional(Y, hasta_anio=None):
    """Promedio de la misma semana del año (y sus dos vecinas) en años anteriores. Con hasta_anio=None respeta el tiempo."""
    c = np.full_like(Y, np.nan)
    for t in range(T):
        misma_epoca = np.abs(((semana_del_anio - semana_del_anio[t] + 26) % 52) - 26) <= 1
        permitidas = misma_epoca & ((anio < anio[t]) if hasta_anio is None else (anio != anio[t]))
        if permitidas.sum() >= 3:
            c[t] = Y[permitidas].mean(axis=0)
    return c

atras = lambda k: np.vstack([np.full((k, D), np.nan), Y[:-k]])       # la tabla corrida k semanas
suma4 = lambda M: sum(np.vstack([np.full((k, D), np.nan), M[:-k]]) for k in range(1, 5))
c = estacional(Y)
pronosticos = {
    "semana pasada (línea base)": atras(1),
    "promedio de 4 semanas": suma4(Y) / 4,
    "estacional": c,
    "estacional con nivel": c * np.clip((1 + suma4(Y)) / (1 + suma4(c)), 0.25, 4),
}

def cobertura(pronostico, semanas, n=N_BRIGADAS):
    """Cobertura semana a semana de mandar las brigadas a los n departamentos con mayor pronóstico."""
    valores = {}
    for t in np.flatnonzero(semanas):
        if np.isnan(pronostico[t]).any() or Y[t].sum() == 0:
            continue
        valores[semanal.index[t]] = Y[t][np.argsort(-pronostico[t])[:n]].sum() / Y[t].sum()
    return pd.Series(valores)

E, V, P_ = (anio >= 2018) & (anio <= 2022), anio == 2023, anio >= 2024
print(f"Semanas: ajuste {E.sum()} | validación {V.sum()} | prueba {P_.sum()}")
''', r'''
# Tabla semanal de focos por departamento, la métrica de cobertura, la línea base y los candidatos sin modelo.
'''),
("code", r'''
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

entradas = ["semana pasada", "hace dos semanas", "promedio de 4", "estacional", "seno de la época", "coseno de la época"]
def tabla_larga(semanas):
    t, d = np.repeat(semanas, D), np.tile(np.arange(D), len(semanas))
    X = np.column_stack([np.log1p(atras(1)[t, d]), np.log1p(atras(2)[t, d]), np.log1p((suma4(Y) / 4)[t, d]), np.log1p(c[t, d]),
                         np.sin(2 * np.pi * semana_del_anio[t] / 52), np.cos(2 * np.pi * semana_del_anio[t] / 52)])
    return X, Y[t, d]

completas = ~np.isnan(c).any(axis=1) & ~np.isnan(suma4(Y)).any(axis=1)
modelos = {"regresión de Poisson": make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500)),
           "boosting de Poisson": HistGradientBoostingRegressor(loss="poisson", random_state=0)}
for nombre, modelo in modelos.items():
    modelo.fit(*tabla_larga(np.flatnonzero(E & completas)))
    pronostico = np.full_like(Y, np.nan)
    semanas = np.flatnonzero((V | P_) & completas)
    pronostico[semanas] = modelo.predict(tabla_larga(semanas)[0]).reshape(len(semanas), D)
    pronosticos[nombre] = pronostico

techo = {"validación": cobertura(Y, V).mean(), "prueba": cobertura(Y, P_).mean()}
comparacion = pd.DataFrame({nombre: {"validación": cobertura(p, V).mean(), "prueba": cobertura(p, P_).mean(),
                                     "error absoluto medio (validación)": np.nanmean(np.abs(p[V] - Y[V]))} for nombre, p in pronosticos.items()}).T
display((comparacion[["validación", "prueba"]] * 100).round(1).join(comparacion["error absoluto medio (validación)"].round(1)))
print(f"Techo (quien conoce el futuro): {techo['validación'] * 100:.1f} % en validación | {techo['prueba'] * 100:.1f} % en prueba")

ax = (comparacion[["validación", "prueba"]] * 100).sort_values("validación").plot.barh(color=["#16302B", "#E0A526"], figsize=(9, 3.8))
ax.axvline(techo["validación"] * 100, color="gray", ls=":"); ax.set(title=f"Cobertura con {N_BRIGADAS} brigadas (línea punteada: techo en validación)", xlabel="% de los focos de la semana")
plt.show()

mejor = comparacion.drop(index="semana pasada (línea base)")["validación"].idxmax()
dif = (cobertura(pronosticos[mejor], P_) - cobertura(pronosticos["semana pasada (línea base)"], P_)).dropna().values
azar = np.random.default_rng(0)
remuestreos = [dif[azar.integers(0, len(dif), len(dif))].mean() for _ in range(2000)]
print(f"Ganador en validación: {mejor} | cobertura en prueba {comparacion.loc[mejor, 'prueba'] * 100:.1f} % | línea base {comparacion.loc['semana pasada (línea base)', 'prueba'] * 100:.1f} %")
print(f"Ventaja sobre la línea base en prueba: {dif.mean() * 100:+.1f} puntos (IC 95 %: {np.percentile(remuestreos, 2.5) * 100:+.1f} a {np.percentile(remuestreos, 97.5) * 100:+.1f}), con {len(dif)} semanas")
''', r'''
# Dos modelos ajustados (por ejemplo, regresión de Poisson y boosting), la tabla comparativa con el techo,
# el ganador escogido EN VALIDACIÓN y su ventaja sobre la línea base en prueba, con intervalo.
'''),
("md", r'''
**Mi lectura** (¿quién gana?, ¿la ventaja se distingue del ruido?, ¿qué tan lejos queda del techo?):

### 3.4 La trampa: mirar el futuro

Hay dos descuidos que dejan entrar el futuro, y los dos salen de escribir una línea de pandas sin pensar en el calendario.

- **El promedio móvil sin correr.** `rolling(4).mean()` promedia las cuatro semanas que terminan en $t$, incluida la semana $t$, que es la que se quiere adivinar. La versión correcta corre la tabla una semana antes de promediar.
- **El estacional con todos los años.** Si el promedio de «la misma época» se calcula una sola vez con toda la tabla, para pronosticar 2023 se están usando 2024 y 2025.

Midan cada descuido contra su versión correcta.
'''),
("code", r'''
movil_tramposo = (Y + atras(1) + atras(2) + atras(3)) / 4             # lo que da rolling(4).mean() sin correr: incluye la semana t
con_futuro = estacional(Y, hasta_anio="todos")
versiones = {"promedio de 4 semanas, corrido (correcto)": pronosticos["promedio de 4 semanas"],
             "promedio de 4 semanas, sin correr (ve la semana t)": movil_tramposo,
             "estacional, solo años anteriores (correcto)": c,
             "estacional, con años futuros": con_futuro}
trampa = pd.DataFrame({nombre: {"validación": cobertura(p, V).mean() * 100, "prueba": cobertura(p, P_).mean() * 100} for nombre, p in versiones.items()}).T
display(trampa.round(1))
print(f"Techo en validación: {techo['validación'] * 100:.1f} %")
print(f"Promedio móvil: el descuido infla la cobertura en {trampa.iloc[1, 0] - trampa.iloc[0, 0]:+.1f} puntos en validación")
print(f"Estacional: usar años futuros la cambia en {trampa.iloc[3, 0] - trampa.iloc[2, 0]:+.1f} puntos en validación")
''', r'''
# Los dos descuidos, cada uno contra su versión correcta: el promedio móvil que incluye la semana pronosticada
# y el estacional calculado con años futuros. Mostrar la cobertura de las cuatro versiones y las diferencias.
'''),
("md", r'''
**Mi explicación** (¿cuál de los dos descuidos infla más y por qué?, ¿qué cifras puede cumplir la API un lunes cualquiera?, ¿qué criterio de aceptación protege contra esta trampa?). Ojo con el segundo: que un descuido casi no cambie el número no lo vuelve correcto, solo dice que la temporada se repite mucho de un año a otro.

### 3.5 Explicar: ¿la época del año o la semana pasada?
'''),
("code", r'''
poisson = modelos["regresión de Poisson"]
coeficientes = pd.Series(poisson[-1].coef_, index=entradas).sort_values()
coeficientes.plot.barh(color="#16302B", figsize=(8, 3))
plt.title("Regresión de Poisson: coeficientes sobre entradas estandarizadas"); plt.xlabel("efecto sobre el logaritmo de los focos esperados"); plt.show()
print(coeficientes.round(3).to_dict())

solo = {}
for nombre, columnas in {"solo rezagos": [0, 1, 2], "solo época del año": [3, 4, 5]}.items():
    X_e, y_e = tabla_larga(np.flatnonzero(E & completas))
    modelo = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=500)).fit(X_e[:, columnas], y_e)
    pronostico = np.full_like(Y, np.nan); semanas = np.flatnonzero(V & completas)
    pronostico[semanas] = modelo.predict(tabla_larga(semanas)[0][:, columnas]).reshape(len(semanas), D)
    solo[nombre] = cobertura(pronostico, V).mean() * 100
print("Cobertura en validación (%):", {n: round(float(v), 1) for n, v in solo.items()}, "| con todo:", round(comparacion.loc["regresión de Poisson", "validación"] * 100, 1))
''', r'''
# Una respuesta con números a la pregunta de explicar: por ejemplo, los coeficientes de la regresión de Poisson
# y la cobertura de un modelo que solo ve rezagos contra uno que solo ve la época del año.
'''),
("md", r'''
**Mi lectura:**

### 3.6 Decidir: la propuesta de una semana y su incertidumbre

Usen el método más simple entre los que no pierden: si el ganador de validación no le saca una ventaja clara al estacional, la API se queda con el estacional, que cabe en una frase. Den la propuesta para la semana siguiente a la última completa y digan cuánto puede fallar. La cobertura cambia de una semana a otra, y cambia entre la temporada seca y el resto del año.
'''),
("code", r'''
simple = "estacional"                                               # se explica en una frase y se programa en diez líneas
ventaja = (cobertura(pronosticos[mejor], P_) - cobertura(pronosticos[simple], P_)).dropna().values
remuestreos_s = [ventaja[azar.integers(0, len(ventaja), len(ventaja))].mean() for _ in range(2000)]
ic = np.percentile(remuestreos_s, [2.5, 97.5]) * 100
elegido = mejor if ic[0] > 0 else simple
print(f"Ventaja de «{mejor}» sobre «{simple}» en prueba: {ventaja.mean() * 100:+.1f} puntos (IC 95 %: {ic[0]:+.1f} a {ic[1]:+.1f})")
print(f"Método elegido: {elegido}")

en_prueba = pd.DataFrame({nombre: cobertura(pronosticos[nombre], P_) for nombre in dict.fromkeys([elegido, simple, "semana pasada (línea base)"])}).dropna()
seca = en_prueba.index.month.isin([12, 1, 2, 3])
resumen = pd.DataFrame({"media": en_prueba.mean(), "percentil 10": en_prueba.quantile(0.10), "percentil 90": en_prueba.quantile(0.90),
                        "dic a mar": en_prueba[seca].mean(), "abr a nov": en_prueba[~seca].mean()})
display((resumen * 100).round(1))

plt.figure(figsize=(11, 3.2))
plt.plot(en_prueba.index, en_prueba[elegido].values * 100, color="#A8432C", label=elegido)
plt.plot(cobertura(Y, P_).index, cobertura(Y, P_).values * 100, color="gray", ls=":", label="techo")
plt.title("Cobertura semanal en la prueba"); plt.ylabel("% de los focos"); plt.legend(); plt.show()

# La propuesta para la semana siguiente a la última completa, con el método estacional y solo datos pasados
siguiente = semanal.index[-1] + pd.Timedelta(days=7)
epoca = np.abs(((semana_del_anio - int(siguiente.isocalendar().week) + 26) % 52) - 26) <= 1
propuesta = pd.Series(Y[epoca].mean(axis=0), index=semanal.columns).sort_values(ascending=False).head(N_BRIGADAS)
print(f"\nPropuesta para la semana del {(siguiente - pd.Timedelta(days=6)).date()} al {siguiente.date()} (focos esperados, método estacional):")
print(propuesta.round(1).to_string())
''', r'''
# La distribución de la cobertura semanal del método elegido en la prueba (media, percentiles 10 y 90,
# temporada seca contra el resto) y la propuesta de departamentos para la semana siguiente al último dato.
'''),
("md", r'''
### La decisión en tres frases

Para quien dirige la sala de crisis.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 3.7 La API

Corran aquí las pruebas de aceptación de su API y dejen el resultado a la vista.
'''),
("code", r'''
import shutil, subprocess, tempfile
plantilla = pathlib.Path(raiz).resolve() / "clases" / "09_api" / "parte_de_incendios"
solucion = pathlib.Path(raiz).resolve() / "_profesor" / "09b_parte_de_incendios_SOLUCION" / "main.py"
trabajo = pathlib.Path(tempfile.mkdtemp()) / "parte"
shutil.copytree(plantilla, trabajo, ignore=shutil.ignore_patterns("focos.csv.gz", "__pycache__"))
shutil.copy(solucion, trabajo / "main.py")                          # la solución del profesor sobre una copia de la plantilla
inicio = time.time()
corrida = subprocess.run([sys.executable, "pruebas_aceptacion.py"], cwd=trabajo, capture_output=True, text=True)
print(corrida.stdout[-1500:], corrida.stderr[-800:])
print(f"{time.time() - inicio:.0f} s, incluida la descarga de los datos por la plantilla")
''', r'''
# Corran `python pruebas_aceptacion.py` dentro de la carpeta de su API y muestren la salida.
# Por ejemplo:  import subprocess; print(subprocess.run(["python", "pruebas_aceptacion.py"], cwd="RUTA/A/SU/CARPETA", capture_output=True, text=True).stdout)
'''),
("md", r'''
Dirección pública de mi API: 

Dirección de mi repositorio: 

### 3.8 Bitácora, riesgos y variante

**Bitácora de decisiones.** Al menos cinco filas.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| | | |

**Riesgos y mantenimiento.** Respondan en dos o tres líneas cada una.

- Un foco no es un incendio, y bajo nubes el satélite no ve. ¿Qué error comete quien lee «cero focos» como «cero incendios»?
- Los departamentos con más focos son siempre los mismos. Si las brigadas van siempre allá, ¿quién queda sin atención? ¿Qué otra métrica propondrían para que la decisión no dependa solo del conteo?
- La fecha del archivo es universal y los límites vienen de OpenStreetMap. ¿En qué casos cambia eso el parte de un departamento?
- Su API calcula `/brigadas` con datos hasta 2025. ¿Qué habría que hacer para que sirva un lunes de 2027, y cómo se enteraría alguien de que los datos están viejos?

**Cambien una línea.** Pongan `N_BRIGADAS = 3`, corran todo y llenen la tabla.

| | 5 brigadas | 3 brigadas |
|---|---|---|
| Techo en prueba | | |
| Ganador en validación | | |
| Cobertura del ganador en prueba | | |
| Cobertura de la línea base en prueba | | |

¿Cuánto cuesta, en focos sin cubrir, tener dos brigadas menos? **Mi respuesta:**

### Procedencia y entrega
'''),
("dado", r'''
{nombre: ficha for nombre, ficha in procedencia().items() if nombre == "firms_colombia"}
'''),
("md", r'''
**Para entregar:** el cuaderno corrido en su portafolio, carpeta `clase09/`, con las dos direcciones escritas en la sección 3.7.
'''),
]

PARTE_SOLUCION = [("md", c[1]) if c[0] == "md" else ("code", c[1]) for c in PARTE]
PARTE_ESTUDIANTE = [("md", c[1]) if c[0] == "md" else ("code", c[1] if c[0] == "dado" else c[2]) for c in PARTE]
