"""Celdas del gemelo de la clase 10, «El mapa de vuelos».

El cuaderno del estudiante es casi vacío: enunciado, criterios de aceptación y la carga del dato.
Cada celda de código es ("code", solución, encargo); las marcadas "dado" salen iguales en los dos.
"""
from molde import ARRANQUE

VUELOS = [
("md", r'''
# Clase 10 · El mapa de vuelos (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno en la segunda mitad de la clase. Es el gemelo de *La cifra que falta*: otra vez hay que dar una cifra con su incertidumbre y contarla en una página, ahora con los vuelos de un aeropuerto.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Calcular una cifra con su intervalo a partir de réplicas, decidir qué hacer con lo que falta en la tabla y entregar el resultado como un tablero que se lee sin correr nada. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (preparar datos), 3 (flujo reproducible), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | El código lo escribe Claude a pedido de ustedes. Deben poder explicar sin ayuda qué decidieron hacer con los meses que no aparecen en la tabla, cómo se calcula el intervalo de la pérdida, por qué no se comparan años calendario incompletos y qué muestra cada panel del tablero. |
| **Entregable** | El cuaderno corrido en su portafolio y el archivo `tablero_vuelos.html` que el cuaderno genera. |
'''),
("dado", ARRANQUE),
("dado", r'''
# Los parámetros del cuaderno. Al final se cambia la primera línea y se corre todo de nuevo.
AEROPUERTO = "MZL"        # sigla IATA: MZL es La Nubia, en Manizales
REPLICAS = 500            # réplicas bootstrap para el intervalo

import time
'''),
("md", r'''
## 1. El enunciado

Imagínense que la Cámara de Comercio y la Alcaldía se van a sentar con las aerolíneas. Quieren llegar con una página que diga qué rutas perdió el aeropuerto, cuántos pasajeros se fueron con cada una y por cuál vale la pena pelear.

1. **Predecir.** Si nada hubiera cambiado, ¿cuántos pasajeros movería hoy cada ruta?
2. **Explicar.** ¿Qué pasó con la ruta que más perdió: se fue una aerolínea, cayó la demanda o cambió el reporte?
3. **Decidir.** ¿Qué ruta perdió más pasajeros y cuál vale la pena recuperar?

**El dato.** La Aeronáutica Civil publica el tráfico aéreo comercial por origen y destino, mes a mes, desde 2020. Cada fila es una empresa, un mes, un trayecto y un tipo de vuelo, con sus pasajeros. Según el diccionario del conjunto, `tipo_vuelo` vale R para regular, A para adicional, C para chárter, N para no regular y T para taxi aéreo. Las coordenadas de los aeropuertos no vienen en ese conjunto: las tomamos de OurAirports, que es un catálogo abierto y no es fuente oficial.

**La cifra.** La pérdida anual de una ruta es la diferencia entre lo que movía en su mejor año y lo que movió en los últimos doce meses:

$$\hat L=12\,(\bar x_{\text{ref}}-\bar x_{\text{rec}}),$$

donde $\bar x_{\text{ref}}$ es el promedio mensual de pasajeros en la ventana de doce meses seguidos con más pasajeros y $\bar x_{\text{rec}}$ es el promedio de los últimos doce meses. Los meses varían, así que $\hat L$ tiene incertidumbre. Se mide con réplicas *bootstrap*: en cada réplica $k$ se sortean con reemplazo doce meses de cada ventana y se recalcula $\hat L_k$. Las réplicas se combinan como en la demostración,

$$\bar L=\frac1m\sum_{k}\hat L_k,\qquad B=\frac{1}{m-1}\sum_k(\hat L_k-\bar L)^2,\qquad \bar L\pm1{,}96\sqrt{\bigl(1+\tfrac1m\bigr)B}.$$

### Criterios de aceptación

1. El cuaderno corre de principio a fin desde cero y funciona con otro aeropuerto cambiando solo la línea `AEROPUERTO`.
2. Hay una tabla mensual completa, con una fila por mes desde el primero hasta el último y una columna por ruta. El cuaderno dice cuántas celdas no venían en el dato y qué se decidió hacer con ellas.
3. Hay una tabla con la pérdida anual de cada ruta, su intervalo de 95 % calculado con al menos 500 réplicas y el porcentaje que representa.
4. El cuaderno muestra con números qué pasa si se comparan años calendario cuando el último año está incompleto.
5. El cuaderno mide qué tan bueno es «lo que movía antes» como pronóstico de «lo que movería hoy», y lo compara con el ancho del intervalo.
6. El cuaderno genera `tablero_vuelos.html`, que se abre en un navegador sin el cuaderno y contiene: un mapa de las rutas, la serie mensual por ruta, la pérdida por ruta con su intervalo y una nota de lectura con la fuente, la fecha del dato y lo que la cifra no dice.
7. Están escritas la decisión en tres frases, la bitácora y las respuestas de riesgos.

### Predicción escrita

- ¿Cuál es la ruta que más pasajeros perdió en el aeropuerto? **Mi apuesta:**
- ¿Qué fracción de las celdas mes por ruta no aparece en el dato? **Mi apuesta:**
- ¿Qué tan ancho será el intervalo de la pérdida de esa ruta, como porcentaje de la pérdida? **Mi apuesta:**
'''),
("md", r'''
## 2. El dato
'''),
("dado", r'''
vuelos = descargar("vuelos", aeropuerto=AEROPUERTO)
aeropuertos = descargar("aeropuertos")
print("Filas:", len(vuelos), "| de", f"{vuelos['anio'].min()}-{vuelos.loc[vuelos['anio'] == vuelos['anio'].min(), 'mes'].min():02d}",
      "a", f"{vuelos['anio'].max()}-{vuelos.loc[vuelos['anio'] == vuelos['anio'].max(), 'mes'].max():02d}")
print("Pasajeros por tipo de vuelo:", vuelos.groupby("tipo_vuelo")["pasajeros"].sum().to_dict())
vuelos.head(3)
'''),
("md", r'''
## 3. Su análisis

De aquí en adelante el cuaderno es suyo. Los títulos siguen los criterios de aceptación.

### 3.1 Limpieza, tabla mensual completa y exploración
'''),
("code", r'''
vuelos["fecha"] = pd.to_datetime(dict(year=vuelos["anio"], month=vuelos["mes"], day=1))
vuelos["ruta"] = np.where(vuelos["origen"] == AEROPUERTO, vuelos["destino"], vuelos["origen"])     # el otro extremo del trayecto
regulares = vuelos[vuelos["tipo_vuelo"] == "R"]                      # decisión: solo vuelos regulares
print(f"Pasajeros en vuelos regulares: {regulares['pasajeros'].sum() / vuelos['pasajeros'].sum() * 100:.1f} % del total")
print("Salen:", int(regulares.loc[regulares["origen"] == AEROPUERTO, "pasajeros"].sum()), "| llegan:", int(regulares.loc[regulares["destino"] == AEROPUERTO, "pasajeros"].sum()))

meses = pd.date_range(vuelos["fecha"].min(), vuelos["fecha"].max(), freq="MS")
venia = regulares.pivot_table(index="fecha", columns="ruta", values="pasajeros", aggfunc="sum").reindex(meses)
venia = venia[venia.sum().sort_values(ascending=False).index]
mensual = venia.fillna(0)                                            # decisión: lo que no viene en el dato se cuenta como cero
print(f"\nTabla mensual: {mensual.shape[0]} meses x {mensual.shape[1]} rutas | celdas que no venían en el dato: {int(venia.isna().sum().sum())} de {venia.size} ({venia.isna().mean().mean() * 100:.0f} %)")
print("Meses sin ninguna fila de vuelos regulares:", [str(m.date())[:7] for m in meses[venia.isna().all(axis=1)]])

principales = list(mensual.columns[:6])
nombres = regulares.assign(nombre_ruta=np.where(regulares["origen"] == AEROPUERTO, regulares["ciudad_destino"], regulares["ciudad_origen"])).groupby("ruta")["nombre_ruta"].first()
fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 3.8))
mensual[principales].plot(ax=izq, lw=1.3); izq.set(title=f"Pasajeros mensuales por ruta, {AEROPUERTO} (ida y vuelta)", xlabel=""); izq.legend(fontsize=8)
por_empresa = regulares.groupby(["anio", "nombre"])["pasajeros"].sum().unstack(fill_value=0)
por_empresa[por_empresa.sum().sort_values(ascending=False).index[:5]].plot.bar(stacked=True, ax=der, cmap="YlOrBr_r", rot=0)
der.set(title="Pasajeros por año y aerolínea (2026 va hasta el último mes del dato)", xlabel=""); der.legend(fontsize=7)
plt.tight_layout(); plt.show()
print("Rutas con más pasajeros en todo el periodo:", {f"{r} ({nombres[r]})": int(mensual[r].sum()) for r in principales})
''', r'''
# Limpieza (qué vuelos entran), tabla mensual completa por ruta, decisión sobre lo que falta y exploración.
'''),
("md", r'''
**Mi lectura** (¿qué se decidió con los meses y las celdas que no venían, y qué otra interpretación tienen?):

### 3.2 La pérdida de cada ruta, con su intervalo
'''),
("code", r'''
def combinar(cifras):
    cifras = np.asarray(cifras, dtype=float)
    m, media, B = len(cifras), cifras.mean(), cifras.var(ddof=1)
    T = (1 + 1 / m) * B
    return media, media - 1.96 * np.sqrt(T), media + 1.96 * np.sqrt(T)

azar = np.random.default_rng(0)
recientes = mensual.iloc[-12:]
filas = []
for ruta in mensual.columns:
    movil = mensual[ruta].rolling(12).sum()
    if movil.max() < 1000:                                           # rutas que nunca movieron mil pasajeros en un año: se dejan fuera
        continue
    fin = movil.idxmax()
    referencia = mensual.loc[fin - pd.DateOffset(months=11):fin, ruta].values
    reciente = recientes[ruta].values
    replicas = [12 * (azar.choice(referencia, 12).mean() - azar.choice(reciente, 12).mean()) for _ in range(REPLICAS)]
    media, inferior, superior = combinar(replicas)
    filas.append({"ruta": ruta, "ciudad": nombres[ruta], "mejor año (termina en)": str(fin.date())[:7], "pasajeros en el mejor año": int(referencia.sum()),
                  "últimos 12 meses": int(reciente.sum()), "pérdida anual": 12 * (referencia.mean() - reciente.mean()),
                  "inferior": inferior, "superior": superior, "pérdida (%)": (1 - reciente.sum() / referencia.sum()) * 100})
perdidas = pd.DataFrame(filas).set_index("ruta").sort_values("pérdida anual", ascending=False)
perdidas["ancho del intervalo (% de la pérdida)"] = (perdidas["superior"] - perdidas["inferior"]) / perdidas["pérdida anual"] * 100
display(perdidas.round(0))
print(f"Últimos 12 meses: de {str(recientes.index[0].date())[:7]} a {str(recientes.index[-1].date())[:7]} | {REPLICAS} réplicas por ruta")
''', r'''
# Para cada ruta: mejor ventana de 12 meses, últimos 12 meses, pérdida anual, intervalo con réplicas bootstrap
# combinadas y porcentaje de pérdida. Una tabla ordenada de mayor a menor pérdida.
'''),
("md", r'''
**Mi lectura** (¿qué ruta perdió más?, ¿su intervalo deja por fuera el cero?):

### 3.3 La trampa: comparar años calendario cuando el último está incompleto
'''),
("code", r'''
anual = mensual.groupby(mensual.index.year).sum()
meses_por_anio = mensual.groupby(mensual.index.year).size()
ultimo, ruta_mayor = anual.index[-1], mensual.columns[0]
mejor_anio = anual[ruta_mayor].idxmax()
ingenua = (1 - anual.loc[ultimo, ruta_mayor] / anual.loc[mejor_anio, ruta_mayor]) * 100
correcta = (1 - recientes[ruta_mayor].sum() / mensual[ruta_mayor].rolling(12).sum().max()) * 100
print("Meses con dato en cada año:", meses_por_anio.to_dict())
print(f"Ruta {ruta_mayor} ({nombres[ruta_mayor]}): {ultimo} contra {mejor_anio}, por año calendario: «perdió» el {ingenua:.0f} %")
print(f"La misma ruta, últimos 12 meses contra su mejor ventana de 12 meses: perdió el {correcta:.0f} %")
print(f"Diferencia: {ingenua - correcta:.0f} puntos que no son pasajeros perdidos, son meses que todavía no han pasado")
''', r'''
# La caída de la ruta principal calculada de dos maneras: último año calendario contra el mejor año calendario,
# y últimos 12 meses contra la mejor ventana de 12 meses. Mostrar las dos cifras y cuántos meses tiene cada año.
'''),
("md", r'''
**Mi explicación:**

### 3.4 ¿Qué tan bueno es «lo que movía antes» como pronóstico de «lo que movería hoy»?

La pérdida supone que, si nada hubiera cambiado, la ruta movería hoy lo mismo que en su mejor año. Ese supuesto se puede poner a prueba con la ruta principal: en cada mes del pasado, pronostiquen los doce meses siguientes con tres reglas y midan el error porcentual absoluto, $|\,\text{real}-\text{pronóstico}\,|/\text{real}$.
'''),
("code", r'''
serie = mensual[ruta_mayor]
filas = []
for corte in range(24, len(serie) - 12 + 1):
    real = serie.iloc[corte:corte + 12].sum()
    if real == 0:
        continue
    anterior, previo = serie.iloc[corte - 12:corte].sum(), serie.iloc[corte - 24:corte - 12].sum()
    reglas = {"los 12 meses anteriores (línea base)": anterior,
              "promedio de los 24 meses anteriores": (anterior + previo) / 2,
              "los 12 anteriores, con la tendencia del último año": anterior * (anterior / previo if previo > 0 else 1),
              "el mejor año visto hasta ese momento": serie.iloc[:corte].rolling(12).sum().max()}
    filas.append({"corte": serie.index[corte], **{n: abs(real - v) / real * 100 for n, v in reglas.items()}})
errores = pd.DataFrame(filas).set_index("corte")
corte_validacion = errores.index[len(errores) // 2]
resumen = pd.DataFrame({"validación (primera mitad)": errores[errores.index < corte_validacion].median(),
                        "prueba (segunda mitad)": errores[errores.index >= corte_validacion].median()})
display(resumen.round(1))
elegida = resumen["validación (primera mitad)"].idxmin()
print(f"Ruta {ruta_mayor}: {len(errores)} cortes | regla con menor error mediano en validación: {elegida} | su error mediano en prueba: {resumen.loc[elegida, 'prueba (segunda mitad)']:.0f} %")
print(f"Error mediano en prueba de «el mejor año visto»: {resumen.loc['el mejor año visto hasta ese momento', 'prueba (segunda mitad)']:.0f} %")
''', r'''
# Para la ruta principal, en cada mes del pasado: pronosticar el total de los 12 meses siguientes con al menos
# tres reglas (por ejemplo, los 12 meses anteriores; el promedio de 24; el mejor año visto) y medir el error
# porcentual. Separar los cortes en validación y prueba, en orden de tiempo. Comparar con el intervalo de 3.2.
'''),
("md", r'''
**Mi lectura** (el intervalo de 3.2 mide cuánto varían los meses; este error mide cuánto falla el supuesto de «si nada hubiera cambiado». ¿Cuál de las dos dudas es mayor, y cuál debe ir en la página?):

### 3.5 Explicar: ¿qué pasó con la ruta que más perdió?
'''),
("code", r'''
perdida_mayor = perdidas.index[0]
quien = regulares[regulares["ruta"] == perdida_mayor].groupby(["anio", "nombre"])["pasajeros"].sum().unstack(fill_value=0)
display(quien)
ultimo_mes = {empresa: str(regulares[(regulares["ruta"] == perdida_mayor) & (regulares["nombre"] == empresa) & (regulares["pasajeros"] > 0)]["fecha"].max().date())[:7] for empresa in quien.columns}
print(f"Ruta {perdida_mayor} ({nombres[perdida_mayor]}): último mes con pasajeros en vuelo regular, por aerolínea:", ultimo_mes)
print("El dato dice quién dejó de reportar pasajeros y cuándo. No dice por qué.")
''', r'''
# Para la ruta que más perdió: pasajeros por año y aerolínea, y último mes con pasajeros de cada aerolínea.
'''),
("md", r'''
**Mi lectura** (¿qué muestra el dato y qué habría que preguntarle a alguien para saber el porqué?):

### 3.6 El tablero
'''),
("code", r'''
import plotly.graph_objects as go
from plotly.subplots import make_subplots

coordenadas = aeropuertos.set_index("iata_code")[["latitude_deg", "longitude_deg", "municipality"]]
sin_coordenadas = [r for r in perdidas.index if r not in coordenadas.index]
en_mapa = [r for r in perdidas.index if r in coordenadas.index]
aqui = coordenadas.loc[AEROPUERTO]

tablero = make_subplots(rows=2, cols=2, specs=[[{"type": "geo"}, {"type": "xy"}], [{"type": "xy", "colspan": 2}, None]],
                        subplot_titles=("Rutas: grosor según pasajeros del mejor año, rojo si se perdió más de la mitad",
                                        "Pérdida anual de pasajeros, con intervalo de 95 %", "Pasajeros mensuales por ruta"),
                        row_heights=[0.55, 0.45], vertical_spacing=0.12)
maximo = perdidas["pasajeros en el mejor año"].max()
for ruta in en_mapa:
    fila, alla = perdidas.loc[ruta], coordenadas.loc[ruta]
    tablero.add_trace(go.Scattergeo(lon=[aqui["longitude_deg"], alla["longitude_deg"]], lat=[aqui["latitude_deg"], alla["latitude_deg"]], mode="lines+markers",
                                    line=dict(width=1 + 7 * fila["pasajeros en el mejor año"] / maximo, color="#A8432C" if fila["pérdida (%)"] > 50 else "#16302B"),
                                    name=ruta, showlegend=False, hoverinfo="text",
                                    text=f"{AEROPUERTO} – {ruta} ({fila['ciudad']})<br>mejor año: {fila['pasajeros en el mejor año']:,.0f}<br>últimos 12 meses: {fila['últimos 12 meses']:,.0f}"), row=1, col=1)
orden = perdidas.sort_values("pérdida anual")
tablero.add_trace(go.Bar(y=[f"{r} ({c})" for r, c in zip(orden.index, orden["ciudad"])], x=orden["pérdida anual"], orientation="h", marker_color="#A8432C", showlegend=False,
                         error_x=dict(type="data", symmetric=False, array=orden["superior"] - orden["pérdida anual"], arrayminus=orden["pérdida anual"] - orden["inferior"])), row=1, col=2)
for ruta in perdidas.index:
    tablero.add_trace(go.Scatter(x=mensual.index, y=mensual[ruta], mode="lines", name=f"{ruta} ({perdidas.loc[ruta, 'ciudad']})"), row=2, col=1)

mayor = perdidas.iloc[0]
tablero.update_geos(fitbounds="locations", showcountries=True, countrycolor="gray", showland=True, landcolor="#f4efe4")
tablero.update_layout(
    height=820, template="plotly_white", legend=dict(orientation="h", y=-0.08),
    title=dict(text=(f"<b>{AEROPUERTO}: la ruta a {mayor['ciudad'].title()} perdió cerca de {mayor['pérdida anual']:,.0f} pasajeros al año</b>"
                     f"<br><sup>Intervalo de 95 %: {mayor['inferior']:,.0f} a {mayor['superior']:,.0f}. Mejor año de la ruta contra los últimos 12 meses "
                     f"({str(recientes.index[0].date())[:7]} a {str(recientes.index[-1].date())[:7]}).</sup>").replace(",", ".")),
    margin=dict(t=110, b=150))
tablero.add_annotation(xref="paper", yref="paper", x=0, y=-0.22, showarrow=False, align="left", font=dict(size=11),
    text=("<b>Cómo leer esta página.</b> Vuelos regulares, ida y vuelta sumadas. La pérdida compara los 12 meses seguidos con más pasajeros de cada ruta con los últimos 12 meses.<br>"
          "El intervalo mide cuánto varían los meses; no mide si la ruta movería hoy lo mismo que en su mejor año. Un mes que no aparece en el dato se contó como cero.<br>"
          f"Fuente: Aeronáutica Civil, datos.gov.co, conjunto gb6w-ynu4, con datos hasta {str(mensual.index[-1].date())[:7]}. Coordenadas: OurAirports (no oficial). El dato no dice por qué se perdió una ruta."))
tablero.write_html("tablero_vuelos.html", include_plotlyjs="cdn", full_html=True)

archivo = pathlib.Path("tablero_vuelos.html")
contenido = archivo.read_text(encoding="utf-8")
print(f"Tablero escrito: {archivo.resolve().name}, {archivo.stat().st_size / 1e3:.0f} kB | trazas: {len(tablero.data)} | rutas sin coordenadas: {sin_coordenadas}")
print("Revisión del archivo:", {"mapa": "scattergeo" in contenido, "barras con intervalo": "error_x" in contenido, "series": contenido.count('"scatter"') > 0, "nota de lectura": "leer esta" in contenido, "fuente": "gb6w-ynu4" in contenido})
tablero.show()
''', r'''
# El tablero, con plotly, guardado como tablero_vuelos.html: mapa de rutas, pérdida por ruta con su intervalo,
# serie mensual por ruta, un título que diga el hallazgo con su intervalo y una nota de lectura con la fuente.
'''),
("md", r'''
Abran `tablero_vuelos.html` en otra pestaña, fuera del cuaderno, y léanlo como lo leería alguien que no estuvo en la clase. ¿Entiende la cifra? ¿Sabe qué no dice?

### La decisión en tres frases

Para quien se va a sentar con las aerolíneas.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 3.7 Bitácora, riesgos y variante

**Bitácora de decisiones.** Al menos cinco filas.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| | | |

**Riesgos y mantenimiento.** Respondan en dos o tres líneas cada una.

- Un mes que no aparece en el dato se contó como cero. ¿En qué caso eso es falso, y cómo lo comprobarían con la Aeronáutica Civil?
- La cifra es de pasajeros que volaron. ¿Qué diferencia hay entre «pasajeros perdidos» y «demanda perdida»? ¿A dónde pudieron irse esos pasajeros?
- El mejor año de una ruta puede ser un año raro. ¿Qué le pasa a la pérdida si la referencia fue un pico que no se iba a repetir?
- El conjunto se actualiza. ¿Qué parte del tablero queda vieja primero y cómo sabría el lector de qué fecha es?

**Cambien una línea.** Pongan `AEROPUERTO = "PEI"` (Matecaña, en Pereira), corran todo y llenen la tabla.

| | MZL | PEI |
|---|---|---|
| Rutas con más de mil pasajeros en su mejor año | | |
| Ruta que más perdió | | |
| Pérdida anual de esa ruta, con intervalo | | |
| Celdas mes por ruta que no venían en el dato (%) | | |
| Error mediano en prueba de «el mejor año visto» (%) | | |

¿Qué cuentan las dos tablas juntas que ninguna cuenta sola? **Mi respuesta:**

### Procedencia y entrega
'''),
("dado", r'''
{nombre: ficha for nombre, ficha in procedencia().items() if nombre in ("vuelos", "aeropuertos")}
'''),
("md", r'''
**Para entregar:** el cuaderno corrido y `tablero_vuelos.html` en su portafolio, carpeta `clase10/`. En Colab el tablero queda en el panel de archivos: descárguenlo y súbanlo junto al cuaderno.
'''),
]

VUELOS_SOLUCION = [("md", c[1]) if c[0] == "md" else ("code", c[1]) for c in VUELOS]
VUELOS_ESTUDIANTE = [("md", c[1]) if c[0] == "md" else ("code", c[1] if c[0] == "dado" else c[2]) for c in VUELOS]
