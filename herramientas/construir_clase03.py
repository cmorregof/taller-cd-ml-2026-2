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

# -------------------------------------------------------- LA FACTURA DE LONDRES
LONDRES = [
("md", r'''
# Clase 3 · La factura de Londres

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. Ustedes lo siguen, lo corren y me interrumpen. En la segunda mitad cada uno resuelve el gemelo, *El cielo de Caldas*, con la misma técnica y otros datos.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Trabajar con un dato que no cabe en memoria: leerlo por pedazos, resumir cada pedazo, juntar los resúmenes sin equivocarse y dejar cada versión del dato registrada con su huella. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 2 (seleccionar y preparar datos), 3 (flujo reproducible y trazable), 4 (implementar y evaluar un modelo) y 5 (comunicar a quien no es técnico). |
| **Uso de IA** | Pueden pedirle a Claude que explique una línea de código, que proponga tipos de dato más livianos o que escriba una gráfica. Deben poder explicar sin ayuda por qué el archivo no se lee de un golpe, qué resúmenes se pueden juntar entre pedazos y cuáles no, qué es la huella de una tabla y por qué un resultado sin versión del dato no se puede reproducir. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con las predicciones escritas, la decisión en tres frases y la tabla de la sección 4.5 llenas. |
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

Entre 2011 y 2014, la empresa UK Power Networks instaló medidores inteligentes en 5.567 hogares de Londres y guardó el consumo de cada uno cada media hora. El resultado es una tabla de 167 millones de filas que pesa unos 10 GB. Un computador de Colab tiene 12 GB de memoria.

Imagínense una comercializadora de energía que atiende esos hogares. Compra la energía un día antes y la factura a una tarifa plana de 14,228 peniques por kilovatio hora. Si compra de menos, paga caro lo que falta. Necesita tres respuestas:

1. **Predecir.** ¿Cuánto consumirá mañana el hogar promedio?
2. **Explicar.** ¿Qué mueve el consumo: la estación, el día de la semana, la tarifa?
3. **Decidir.** La víspera de un día de consumo alto la empresa quiere avisarles a los hogares. ¿Con qué umbral del pronóstico se manda el aviso, y cuántos avisos en falso cuesta?

**La métrica.** Para el pronóstico usamos el error porcentual absoluto medio, como en la clase 1. Llamo $C_t$ al consumo promedio por hogar en el día $t$, en kWh:

$$\text{MAPE}=\frac{100}{n}\sum_{t}\left|\frac{C_{t+1}-\hat C_{t+1}}{C_{t+1}}\right|.$$

Para el aviso contamos aciertos y falsas alarmas, y eso se define en la sección 4.

**Lo nuevo de hoy** está antes del modelo. La pregunta de la clase es cómo se llega de 167 millones de filas a una serie diaria sin quedarse sin memoria y sin perder el rastro de lo que se hizo.

### Predicción escrita

Antes de ver un solo dato, escriban aquí su apuesta.

- ¿Cuántos kWh consume al día un hogar de Londres? Un hogar colombiano de estrato medio consume cerca de 5. **Mi apuesta:**
- Si una tabla de un millón de filas ocupa 50 MB en disco, ¿cuánta memoria ocupa al abrirla con pandas sin decirle nada? **Mi apuesta:**
- ¿Qué tan lejos queda el consumo de mañana del de hoy, en porcentaje? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

### 2.1 El tamaño del problema

La fuente oficial es London Datastore. Publica un archivo ZIP con 168 archivos CSV adentro. La primera decisión es no bajarlo. Un ZIP guarda al final un índice con el nombre, el tamaño y la posición de cada archivo interno, y el servidor acepta que le pidan un rango de bytes. Así que se pide el índice, que pesa unos pocos kilobytes, y después solo los archivos que se vayan a usar.
'''),
("code", r'''
ARCHIVOS = 24                # cuántos de los 168 archivos oficiales se procesan. Al final lo cambiarán por 48
'''),
("code", r'''
from taller.fuentes_clase03 import abrir_londres, cuales, TARIFA_PLANA

paquete, internos, remoto = abrir_londres()
comprimido = sum(i.compress_size for i in internos) / 1e9
sin_comprimir = sum(i.file_size for i in internos) / 1e9
print(f"Archivos en el ZIP: {len(internos)}")
print(f"Tamaño comprimido: {comprimido:.2f} GB | sin comprimir: {sin_comprimir:.2f} GB")
print(f"Hasta ahora se han bajado {remoto.bytes_bajados / 1e3:.0f} kB en {remoto.peticiones} peticiones.")
print("Primeros nombres:", [i.filename.split('/')[-1] for i in internos[:3]])
'''),
("md", r'''
### 2.2 Un pedazo, mirado con calma

Antes de procesar nada conviene abrir un solo archivo y medirlo. Lo leo de la manera ingenua, con todo como texto, que es parecido a lo que hace pandas cuando no reconoce una columna.
'''),
("code", r'''
import time

inicio = time.time()
with paquete.open(internos[0]) as flujo:
    pedazo = pd.read_csv(flujo, dtype=str)
segundos = time.time() - inicio

memoria = pedazo.memory_usage(deep=True).sum() / 1e6
print(f"Filas: {len(pedazo)} | leído en {segundos:.1f} s")
print("Columnas:", list(pedazo.columns))
print(f"En disco: {internos[0].file_size / 1e6:.0f} MB | en memoria: {memoria:.0f} MB")
print(f"Los 168 archivos, leídos así, pedirían cerca de {memoria * 168 / 1e3:.0f} GB de memoria.")
pedazo.head()
'''),
("md", r'''
Comparen con su segunda apuesta. La tabla ocupa en memoria más de lo que pesa en disco. Cuánto más depende de la versión de pandas: las anteriores a la 3 guardan cada celda de texto como un objeto de Python con su propio envoltorio, y con ellas el mismo millón de filas se multiplica varias veces. La última línea de la celda hace la cuenta para los 168 archivos con la versión que ustedes tienen. Compárenla con los 12 GB de Colab.

Miren ahora las trampas de este pedazo. Hay tres.
'''),
("code", r'''
columna_kwh = pedazo.columns[3]
print("Nombre de la cuarta columna, con comillas:", repr(columna_kwh))
print("Lecturas que no son números:", pedazo[columna_kwh][pd.to_numeric(pedazo[columna_kwh], errors="coerce").isna()].value_counts().to_dict())
print("Filas repetidas:", int(pedazo.duplicated().sum()))
print("Hogares en este archivo:", pedazo["LCLid"].nunique(), "| tarifas:", pedazo["stdorToU"].value_counts().to_dict())
print("Lecturas por hogar:"); print(pedazo.groupby("LCLid").size().describe()[["min", "50%", "max"]].round(0).to_string())
'''),
("md", r'''
La primera es que el nombre de la columna de consumo termina en un espacio. La segunda es que algunas lecturas dicen `Null`, que es texto, y por eso pandas no puede tratar la columna como número. La tercera es que hay filas repetidas idénticas, y si se suman cuentan doble.

Hay una cuarta cosa que no es un error y que importa más que las otras tres. El archivo trae un millón de filas, pero solo unas pocas decenas de hogares. El archivo está ordenado por hogar. Volvemos a eso en la sección 2.5.

### 2.3 Tipos de dato: la misma tabla en menos memoria

El texto es la forma más cara de guardar un dato. El código del hogar se repite decenas de miles de veces, así que se guarda como **categoría**: una lista corta de nombres y un número pequeño por fila. El consumo cabe en un decimal de 32 bits. De la fecha solo necesito el día.
'''),
("code", r'''
liviano = pedazo.rename(columns=lambda nombre: nombre.strip()).rename(columns={"KWH/hh (per half hour)": "kwh"})
liviano["kwh"] = pd.to_numeric(liviano["kwh"], errors="coerce").astype("float32")
liviano["LCLid"] = liviano["LCLid"].astype("category")
liviano["stdorToU"] = liviano["stdorToU"].astype("category")
liviano["DateTime"] = pd.to_datetime(liviano["DateTime"].str[:19])
memoria_liviana = liviano.memory_usage(deep=True).sum() / 1e6
print(f"Memoria con tipos adecuados: {memoria_liviana:.0f} MB, {memoria / memoria_liviana:.0f} veces menos.")
print(f"Los 168 archivos pedirían {memoria_liviana * 168 / 1e3:.1f} GB.")
print(liviano.dtypes.to_string())
'''),
("md", r'''
Con tipos adecuados el dato completo cabría en un computador de Colab. Pero caber no es lo mismo que poder trabajar. Una sola operación que copie la tabla duplica la memoria, y para llegar a los tipos livianos primero hay que leer el texto. La salida de fondo es otra.

### 2.4 Resumir cada pedazo y juntar los resúmenes

La comercializadora no necesita las medias horas. Necesita el consumo por hogar y por día. Entonces cada archivo se lee, se limpia, se **resume** a una fila por hogar y día, y se suelta. En memoria solo hay un archivo a la vez. Al final se **juntan** los resúmenes.

El resumen de cada hogar $j$ en el día $d$ guarda dos números, la suma de las lecturas y cuántas fueron:

$$s_{jd}=\sum_{i\in(j,d)} x_i,\qquad n_{jd}=\#\{i\in(j,d)\}.$$

Guardo la cuenta porque un día completo tiene 48 medias horas, y un día con menos lecturas no se puede comparar con uno completo.
'''),
("code", r'''
def resumir(crudo):
    """De lecturas de media hora a una fila por hogar y por día."""
    crudo = crudo.rename(columns=lambda nombre: nombre.strip())                       # trampa 1: el espacio en el nombre
    crudo = crudo.rename(columns={"KWH/hh (per half hour)": "kwh"}).drop_duplicates()  # trampa 3: las filas repetidas
    crudo["kwh"] = pd.to_numeric(crudo["kwh"], errors="coerce")                        # trampa 2: Null pasa a faltante
    crudo["fecha"] = crudo["DateTime"].str[:10]
    crudo["media_hora"] = crudo["DateTime"].str[11:16]
    diario = (crudo.groupby(["LCLid", "stdorToU", "fecha"])["kwh"].agg(kwh="sum", lecturas="count").reset_index()
              .rename(columns={"LCLid": "hogar", "stdorToU": "tarifa"}))
    perfil = crudo.groupby(["stdorToU", "media_hora"])["kwh"].agg(suma="sum", cuenta="count")
    return diario, perfil

numeros = cuales(ARCHIVOS)
print(f"Se procesan {len(numeros)} de 168 archivos, repartidos por igual: {numeros[:4]} ... {numeros[-2:]}")
resumenes, perfiles, filas_leidas, inicio = [], [], 0, time.time()
try:
    for numero in numeros:
        with paquete.open(internos[numero]) as flujo:
            crudo = pd.read_csv(flujo, dtype=str)
        diario, perfil = resumir(crudo)
        resumenes.append(diario); perfiles.append(perfil); filas_leidas += len(crudo)
        print(f"  archivo {numero:3d}: {len(crudo)} filas -> {len(diario)} filas de resumen | {time.time() - inicio:4.0f} s")
    # Un hogar puede quedar partido entre dos archivos. Por eso, al juntar, se vuelve a sumar.
    consumo = (pd.concat(resumenes).groupby(["hogar", "tarifa", "fecha"])[["kwh", "lecturas"]].sum().reset_index()
               .sort_values(["hogar", "fecha"]).reset_index(drop=True))
    consumo["kwh"] = consumo["kwh"].round(4)
    perfil = pd.concat(perfiles).groupby(level=[0, 1]).sum()
    origen = "London Datastore, leído por rangos"
except Exception as error:                      # si London Datastore se cae en clase, se usa la copia ya resumida
    print("La fuente no respondió:", repr(error)[:120])
    consumo, perfil, origen = descargar("londres_diario", archivos=ARCHIVOS), None, "copia de respaldo"
segundos_proceso = time.time() - inicio

print(f"\nOrigen: {origen}")
print(f"De {filas_leidas} lecturas quedaron {len(consumo)} filas de hogar y día, de {consumo['hogar'].nunique()} hogares.")
print(f"Tiempo: {segundos_proceso:.0f} s | bajado: {remoto.bytes_bajados / 1e6:.0f} MB | memoria del resumen: {consumo.memory_usage(deep=True).sum() / 1e6:.0f} MB")
'''),
("md", r'''
Esto es todo el secreto de trabajar con datos grandes. La tabla nunca estuvo completa en memoria, y lo que quedó es decenas de veces más pequeño.

Procesamos una parte de los 168 archivos y no todos por tiempo de clase. El subconjunto es real y está definido: son los archivos cuyos números imprime la celda, tomados a intervalos iguales para recorrer todo el listado de hogares. El código no cambia si se procesan los 168. Solo tarda siete veces más.

**¿Qué resúmenes se pueden juntar?** La suma y la cuenta se juntan sin problema, porque la suma de las sumas es la suma total. Con ellas sale el promedio global:

$$\bar x=\frac{\sum_k s_k}{\sum_k n_k},\qquad\text{que en general no es}\qquad \frac{1}{K}\sum_k \bar x_k.$$

El promedio de los promedios solo coincide con el promedio global cuando todos los pedazos pesan lo mismo. La mediana es peor: no hay manera de obtener la mediana global a partir de las medianas de los pedazos. Por eso el resumen guarda sumas y cuentas, y nunca promedios.

### 2.5 La trampa de la clase: el primer pedazo no es una muestra

Cuando un archivo es demasiado grande, lo primero que uno hace es leer el principio y sacar conclusiones. Con este dato eso sale mal, y se puede medir.

**Predicción escrita:** el primer archivo tiene un millón de lecturas. ¿Su consumo promedio por hogar y día se parecerá al de todos los archivos procesados? **Mi apuesta:**
'''),
("code", r'''
completos = consumo[consumo["lecturas"] == 48]
primer = resumenes[0][resumenes[0]["lecturas"] == 48] if resumenes else completos[completos["hogar"].isin(completos["hogar"].unique()[:27])]
por_archivo = pd.Series([r.loc[r["lecturas"] == 48, "kwh"].mean() for r in resumenes]) if resumenes else pd.Series(dtype=float)

print(f"Primer archivo: {primer['hogar'].nunique()} hogares | {primer['kwh'].mean():.2f} kWh por hogar y día")
print(f"Todos los archivos procesados: {completos['hogar'].nunique()} hogares | {completos['kwh'].mean():.2f} kWh por hogar y día")
if len(por_archivo):
    print(f"Entre archivos, el promedio va de {por_archivo.min():.2f} a {por_archivo.max():.2f} kWh.")
    ax = por_archivo.plot.bar(color="#16302B", figsize=(9, 3)); ax.axhline(completos["kwh"].mean(), color="#A8432C")
    ax.set(title="Consumo promedio por hogar y día, según el archivo que uno hubiera mirado", xlabel="archivo procesado", ylabel="kWh"); plt.show()
'''),
("md", r'''
Un millón de filas suena a muestra de sobra. Pero el archivo está ordenado por hogar, así que un millón de filas son unas pocas decenas de hogares, y los hogares son muy distintos entre sí. El tamaño que importa para generalizar es el número de hogares, no el número de filas. Leer el principio de un archivo ordenado es tomar una muestra de conveniencia.

Por eso el subconjunto del curso recorre el listado a intervalos iguales en vez de tomar los primeros archivos. Aun así es una decisión, y va a la bitácora.

### 2.6 Versionar el dato

Acabamos de producir una tabla que costó tiempo y decisiones. Dentro de un mes alguien va a preguntar con qué dato salió un resultado. Para poder responder, cada tabla que se guarda lleva tres cosas: una **huella**, de dónde salió y qué regla la produjo.

La huella es una función $h$ que convierte el contenido de la tabla en una cadena corta. Si dos tablas son iguales, tienen la misma huella. Si cambia una sola celda, la huella cambia:

$$h(T)=h(T')\ \text{casi con certeza implica}\ T=T'.$$

Usamos SHA-256 sobre el contenido de la tabla. El código tiene Git para saber qué versión se corrió. Esto es lo mismo, para el dato.
'''),
("code", r'''
import hashlib, json
from datetime import datetime, timezone

carpeta = raiz / "datos" / "cache" / "versiones"
carpeta.mkdir(parents=True, exist_ok=True)

def huella(tabla):
    return hashlib.sha256(tabla.to_csv(index=False).encode("utf-8")).hexdigest()[:12]

def versionar(tabla, nombre, regla, padre=None):
    """Guarda la tabla en parquet con su huella en el nombre y la anota en el registro."""
    marca = huella(tabla)
    tabla.to_parquet(carpeta / f"{nombre}_{marca}.parquet", index=False)
    registro = carpeta / "registro.json"
    versiones = json.loads(registro.read_text(encoding="utf-8")) if registro.exists() else []
    versiones = [v for v in versiones if v["huella"] != marca]          # la misma tabla no se anota dos veces
    versiones.append({"nombre": nombre, "huella": marca, "filas": len(tabla), "regla": regla, "padre": padre,
                      "guardada": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")})
    registro.write_text(json.dumps(versiones, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Versión {marca}: {len(tabla)} filas | {regla}")
    return marca

v1 = versionar(consumo, "londres_diario", f"{origen}; {len(numeros)} de 168 archivos; suma y cuenta por hogar y día; Null faltante; sin filas repetidas")
'''),
("md", r'''
La tabla quedó en formato **parquet**, que guarda por columnas, comprime y conserva los tipos. Para tablas grandes es más pequeño y más rápido que un CSV.

Si ustedes y yo procesamos los mismos archivos con el mismo código, las dos huellas deben coincidir. Si no coinciden, alguno tiene otro dato, y es mejor saberlo ahora que después de entrenar.

### 2.7 Limpieza: días completos y un panel que cambia

Dos decisiones más, y cada una produce una versión nueva.
'''),
("code", r'''
consumo["fecha"] = pd.to_datetime(consumo["fecha"])
print("Lecturas por hogar y día:"); print(consumo["lecturas"].value_counts().sort_index(ascending=False).head(4).to_string())
print(f"Filas con menos de 48 lecturas: {(consumo['lecturas'] < 48).mean() * 100:.1f} %")

limpio = consumo[consumo["lecturas"] == 48].drop(columns="lecturas")
hogares_por_dia = limpio.groupby("fecha")["hogar"].nunique()
total_por_dia = limpio.groupby("fecha")["kwh"].sum()

fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 3.6))
hogares_por_dia.plot(ax=izq, color="#16302B"); izq.set(title="Hogares con medición completa, por día", xlabel="")
(total_por_dia / 1000).plot(ax=der, color="#A8432C"); der.set(title="Consumo total de la muestra (MWh por día)", xlabel="")
plt.tight_layout(); plt.show()
print(f"Hogares el 15 de enero de 2012: {hogares_por_dia.get(pd.Timestamp('2012-01-15'), 0)} | el 15 de enero de 2013: {hogares_por_dia.get(pd.Timestamp('2013-01-15'), 0)}")
'''),
("md", r'''
Un día con 47 lecturas no es un día de bajo consumo: es un día al que le falta media hora. **Decisión:** solo entran los días con las 48 lecturas.

La gráfica de la derecha parece una demanda que se multiplicó durante 2012. La de la izquierda explica por qué: los medidores se instalaron poco a poco, y lo que creció fue el número de hogares. Quien sume kWh por día sin mirar cuántos hogares hay, le reporta a la empresa un crecimiento que no existió. **Decisión:** la variable de trabajo es el consumo **promedio por hogar**, y la serie empieza el primer día con al menos la mitad del máximo de hogares.
'''),
("code", r'''
minimo = hogares_por_dia.max() / 2
primer_dia = hogares_por_dia[hogares_por_dia >= minimo].index.min()
ultimo_dia = hogares_por_dia[hogares_por_dia >= minimo].index.max()
limpio = limpio[(limpio["fecha"] >= primer_dia) & (limpio["fecha"] <= ultimo_dia)].reset_index(drop=True)
v2 = versionar(limpio.assign(fecha=limpio["fecha"].dt.strftime("%Y-%m-%d")), "londres_diario",
               f"solo días con 48 lecturas; del {primer_dia.date()} al {ultimo_dia.date()}", padre=v1)

serie = limpio.groupby("fecha")["kwh"].mean().rename("kwh")
serie = serie.asfreq("D").interpolate()                      # si falta algún día suelto, se interpola
print(f"Serie: {len(serie)} días, del {serie.index.min().date()} al {serie.index.max().date()}")
print(f"Consumo promedio: {serie.mean():.2f} kWh por hogar y día | factura diaria promedio: {serie.mean() * TARIFA_PLANA / 100:.2f} libras")
display(pd.DataFrame(json.loads((carpeta / "registro.json").read_text(encoding="utf-8"))).query("nombre == 'londres_diario'").tail(4))
'''),
("md", r'''
Comparen con su primera apuesta. El registro muestra las dos versiones con su huella, su regla y su versión madre. Todo lo que sigue se calcula con la segunda, y al final el resultado quedará amarrado a esa huella.

### 2.8 Conocer el dato antes de modelarlo

Cuatro miradas: cómo se reparte el consumo entre hogares, cómo cambia con la estación, con el día de la semana y con la hora.
'''),
("code", r'''
dias = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
fig, ejes = plt.subplots(2, 2, figsize=(13, 7.5))

por_hogar = limpio.groupby("hogar")["kwh"].mean()
ejes[0, 0].hist(por_hogar, bins=60, color="#16302B")
ejes[0, 0].axvline(por_hogar.mean(), color="#A8432C"); ejes[0, 0].axvline(por_hogar.median(), color="#E0A526")
ejes[0, 0].set(title="Consumo promedio de cada hogar (rojo: media; amarillo: mediana)", xlabel="kWh por día", ylabel="hogares")

serie.plot(ax=ejes[0, 1], color="#16302B", lw=1); ejes[0, 1].set(title="Consumo promedio por hogar", xlabel="", ylabel="kWh por día")

por_dia_semana = serie.groupby(serie.index.dayofweek).mean() / serie.mean() * 100 - 100
ejes[1, 0].bar(dias, por_dia_semana.values, color="#E0A526"); ejes[1, 0].set(title="Día de la semana: desvío frente al promedio", ylabel="%")

if perfil is not None:
    en_punto = perfil[perfil["cuenta"] > perfil["cuenta"].max() / 10]            # hay unas pocas lecturas a deshoras, como 15:15: se dejan fuera de la curva
    curva = (en_punto["suma"] / en_punto["cuenta"]).unstack(0)
    curva.plot(ax=ejes[1, 1], color=["#16302B", "#A8432C"]); ejes[1, 1].set(title="Consumo por media hora, según la tarifa", xlabel="hora", ylabel="kWh por media hora")
plt.tight_layout(); plt.show()
print(f"Media por hogar: {por_hogar.mean():.1f} kWh | mediana: {por_hogar.median():.1f} | el 10 % que más consume gasta {por_hogar[por_hogar >= por_hogar.quantile(0.9)].sum() / por_hogar.sum() * 100:.0f} % del total")
print(f"Mes de mayor consumo: {serie.groupby(serie.index.month).mean().idxmax()} | de menor: {serie.groupby(serie.index.month).mean().idxmin()} | razón entre los dos: {serie.groupby(serie.index.month).mean().max() / serie.groupby(serie.index.month).mean().min():.2f}")
'''),
("md", r'''
Tres lecturas. La distribución entre hogares es asimétrica: pocos hogares consumen mucho y la media queda por encima de la mediana. La estación domina la serie, con el invierno muy por encima del verano, que es lo esperable en una ciudad donde hace frío y oscurece temprano. El domingo se consume más que cualquier otro día.

La curva por media hora salió del mismo recorrido de los archivos, con otro resumen de suma y cuenta. Un solo paso por el dato puede alimentar varias tablas pequeñas.

### 2.9 PCA: ¿todos los hogares viven la misma estación?

Para cada hogar con el año 2013 completo calculo su consumo promedio en cada mes y lo divido por su promedio anual. Queda una matriz con una fila por hogar y doce columnas, que describe la **forma** del año de cada hogar sin importar su tamaño. Descompongo su matriz de covarianzas:

$$S=\frac{1}{n-1}Z^\top Z=V\Lambda V^\top,\qquad \text{varianza explicada por el componente } j=\frac{\lambda_j}{\sum_k\lambda_k}.$$

**Predicción escrita:** ¿qué forma tendrá el primer componente a lo largo de los doce meses? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA

anio_2013 = limpio[limpio["fecha"].dt.year == 2013]
mensual = anio_2013.pivot_table(index="hogar", columns=anio_2013["fecha"].dt.month, values="kwh", aggfunc="mean").dropna()
forma = mensual.div(mensual.mean(axis=1), axis=0)
forma = forma[(mensual.mean(axis=1) > 0.5)]                       # un hogar casi sin consumo no tiene forma que medir
pca = PCA(n_components=4).fit(forma - forma.mean())
explicada = pca.explained_variance_ratio_ * 100
puntajes_pca = pca.transform(forma - forma.mean())

meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
fig, (izq, centro, der) = plt.subplots(1, 3, figsize=(14, 3.8))
izq.bar(range(1, 5), explicada, color="#16302B"); izq.set(title="Varianza explicada", xlabel="componente", ylabel="%")
centro.plot(meses, pca.components_[0], color="#A8432C", marker="o", label="CP1"); centro.plot(meses, pca.components_[1], color="#3E7CB1", marker="o", label="CP2")
centro.axhline(0, color="gray", lw=0.5); centro.set(title="Cargas por mes"); centro.legend()
extremos = np.argsort(puntajes_pca[:, 0])
der.plot(meses, forma.iloc[extremos[:30]].mean(), color="#3E7CB1", label="30 hogares con CP1 más bajo")
der.plot(meses, forma.iloc[extremos[-30:]].mean(), color="#A8432C", label="30 hogares con CP1 más alto")
der.axhline(1, color="gray", lw=0.5); der.set(title="Forma del año en los dos extremos", ylabel="consumo del mes / promedio anual"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
print(f"Hogares con 2013 completo: {len(forma)} | El primer componente explica {explicada[0]:.0f} % de la varianza de la forma.")
'''),
("md", r'''
El primer componente opone los meses fríos a los cálidos. Un hogar con puntaje alto en ese componente tiene un invierno mucho más pesado que su verano, y uno con puntaje bajo tiene un año casi plano. Una lectura razonable es que los primeros calientan la casa con electricidad y los segundos con gas, pero el dato no trae esa variable y no puedo comprobarlo.

Para la comercializadora esto dice que el «hogar promedio» mezcla dos clases de cliente, y que el frío le pega a su demanda por una parte de los hogares.
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base

Con horizonte de un día hay dos pronósticos que no cuestan nada:

$$\hat C^{\text{ing}}_{t+1}=C_t\qquad\text{y}\qquad \hat C^{\text{sem}}_{t+1}=C_{t-6}.$$

El primero dice que mañana se consume lo mismo que hoy. El segundo, lo mismo que el mismo día de la semana pasada.
'''),
("code", r'''
H = 1
tabla = pd.DataFrame({"real": serie.shift(-H), "ingenuo": serie, "semanal": serie.shift(7 - H)}).dropna()

def mape(real, pronostico):
    return float((abs(real - pronostico) / real).mean() * 100)

for metodo in ["ingenuo", "semanal"]:
    print(f"MAPE {metodo}: {mape(tabla['real'], tabla[metodo]):.2f} %")
'''),
("md", r'''
Comparen con su tercera apuesta. La serie es mucho más dócil que el precio de la luz de la clase 1: promediar cientos de hogares cancela los caprichos de cada uno.

### 3.2 Cinco modelos y el protocolo de siempre

Los candidatos son los de la clase 1: regresión lineal, Ridge, $k$ vecinos, bosque aleatorio y *gradient boosting*. Les pido el cambio relativo y no el nivel, por la razón que vimos con el sancocho:

$$y_t=\frac{C_{t+1}}{C_t},\qquad \hat C_{t+1}=\hat y_t\,C_t.$$

Las entradas son el consumo de ayer, el de hace seis días y los promedios de 7 y de 28 días, todos divididos por el de hoy, más el día de la semana de mañana y la época del año escrita como $\sin(2\pi d/365{,}25)$ y $\cos(2\pi d/365{,}25)$, con $d$ el día del año. No hay temperatura, porque el conjunto no la trae. Es la variable que más falta hace y lo anoto en los riesgos.

El tiempo se parte en tres tramos:

$$\underbrace{\text{hasta jun } 2013}_{\text{entrenamiento}}\ \Big|\ \underbrace{\text{jul}\ \dots\ \text{oct } 2013}_{\text{validación: se escoge}}\ \Big|\ \underbrace{\text{nov } 2013\ \dots\ \text{feb } 2014}_{\text{prueba: se reporta}}$$
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
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1)),
        "k vecinos": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
        "bosque": RandomForestRegressor(n_estimators=300, random_state=0),
        "boosting": GradientBoostingRegressor(random_state=0),
    }

manana = serie.index + pd.Timedelta(days=H)
X = pd.DataFrame({
    "ayer": serie.shift(1) / serie,
    "hace_6": serie.shift(7 - H) / serie,
    "promedio_7": serie.rolling(7).mean() / serie,
    "promedio_28": serie.rolling(28).mean() / serie,
    "manana_es_finde": (manana.dayofweek >= 5).astype(int),
    "manana_es_lunes": (manana.dayofweek == 0).astype(int),
    "epoca_sen": np.sin(2 * np.pi * manana.dayofyear / 365.25),
    "epoca_cos": np.cos(2 * np.pi * manana.dayofyear / 365.25),
}, index=serie.index)
base = X.join(serie.rename("hoy")).join(serie.shift(-H).rename("real")).dropna()
variables = list(X.columns)
razon = base["real"] / base["hoy"]

objetivo = base.index + pd.Timedelta(days=H)
E = objetivo <= pd.Timestamp("2013-06-30")
V = (base.index >= "2013-07-01") & (objetivo <= pd.Timestamp("2013-10-31"))
P = base.index >= "2013-11-01"
print(f"Entrenamiento: {E.sum()} días | Validación: {V.sum()} | Prueba: {P.sum()}")

errores, pronosticos = {}, {}
for nombre, modelo in cinco_modelos().items():
    modelo.fit(base[variables][E], razon[E])
    en_validacion = mape(base["real"][V], modelo.predict(base[variables][V]) * base["hoy"][V])
    modelo.fit(base[variables][E | V], razon[E | V])
    pronosticos[nombre] = pd.Series(modelo.predict(base[variables][P]) * base["hoy"][P].values, index=base.index[P])
    errores[nombre] = {"validación": en_validacion, "prueba": mape(base["real"][P], pronosticos[nombre])}
for metodo in ["ingenuo", "semanal"]:
    errores[metodo] = {"validación": mape(base["real"][V], tabla.loc[base.index[V], metodo]), "prueba": mape(base["real"][P], tabla.loc[base.index[P], metodo])}
    pronosticos[metodo] = tabla.loc[base.index[P], metodo]
todos = pd.DataFrame(errores).T
pronosticos_prueba = pd.DataFrame(pronosticos)

todos.sort_values("validación").plot.barh(figsize=(9, 4), color=["#16302B", "#E0A526"])
plt.xlabel("MAPE (%)"); plt.title("Siete candidatos: validación para escoger, prueba para reportar"); plt.gca().invert_yaxis(); plt.show()
display(todos.sort_values("validación").round(2))
'''),
("code", r'''
mejor = todos["validación"].idxmin()
resumen = todos["prueba"].round(2)
elegido = mejor if resumen[mejor] < resumen["ingenuo"] else "ingenuo"
print(f"Ganador en validación: {mejor}")
print(f"Su MAPE en prueba: {resumen[mejor]} %   |   MAPE del ingenuo en prueba: {resumen['ingenuo']} %")

fechas = base.index[P] + pd.Timedelta(days=H)
plt.figure(figsize=(11, 3.8))
plt.plot(fechas, base["real"][P], color="black", lw=1.8, label="real")
plt.plot(fechas, pronosticos_prueba["ingenuo"], color="#E0A526", lw=1, label="ingenuo")
if mejor != "ingenuo":
    plt.plot(fechas, pronosticos_prueba[mejor], color="#A8432C", lw=1.2, ls="--", label=mejor)
plt.title("Pronóstico a un día contra el consumo real, en la prueba"); plt.ylabel("kWh por hogar"); plt.legend(); plt.show()

# El resultado queda amarrado a la versión del dato con que se calculó.
resultado = {"dato": v2, "archivos": len(numeros), "modelo": mejor, "mape_prueba": float(resumen[mejor]), "mape_ingenuo": float(resumen["ingenuo"])}
print(resultado)
'''),
("md", r'''
El último renglón es la mitad de lo que significa versionar. Un MAPE suelto no se puede reproducir. Un MAPE con la huella del dato al lado sí, porque dice sobre qué tabla se midió.

### 3.3 El mismo modelo sobre otra versión del dato

Para ver por qué importa, repito el cálculo de la línea base sobre la primera versión, la que incluía los días incompletos.
'''),
("code", r'''
primera = pd.read_parquet(carpeta / f"londres_diario_{v1}.parquet")
primera["fecha"] = pd.to_datetime(primera["fecha"])
primera = primera[(primera["fecha"] >= primer_dia) & (primera["fecha"] <= ultimo_dia)]
serie_v1 = primera.groupby("fecha")["kwh"].mean().asfreq("D").interpolate()
comparacion = pd.DataFrame({
    f"versión {v1} (con días incompletos)": {"consumo promedio (kWh)": serie_v1.mean(), "MAPE del ingenuo (%)": mape(serie_v1.shift(-1).dropna(), serie_v1[:-1])},
    f"versión {v2} (solo días completos)": {"consumo promedio (kWh)": serie.mean(), "MAPE del ingenuo (%)": mape(serie.shift(-1).dropna(), serie[:-1])},
}).T
display(comparacion.round(3))
print(f"Días en que las dos series difieren en más de 1 %: {(abs(serie_v1 / serie - 1) > 0.01).sum()}")
'''),
("md", r'''
Los dos renglones salen del mismo código y de la misma fuente. Lo único que cambió fue una regla de limpieza. La diferencia puede ser pequeña o grande, y ese no es el punto: sin la huella, dos personas con resultados distintos no tienen cómo saber si la diferencia viene del modelo o del dato.

### 3.4 Explicar: ¿qué mueve el consumo?

Dos miradas. La primera es la importancia que el bosque aleatorio le da a cada entrada. La segunda compara a los hogares con tarifa plana (`Std`) y con tarifa dinámica (`ToU`), que durante 2013 recibían un día antes el precio de cada franja.
'''),
("code", r'''
bosque = cinco_modelos()["bosque"].fit(base[variables], razon)
fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 3.8))
pd.Series(bosque.feature_importances_, index=variables).sort_values().plot.barh(ax=izq, color="#16302B")
izq.set(title="Importancia de cada entrada en el bosque aleatorio")

por_tarifa = limpio.pivot_table(index="fecha", columns="tarifa", values="kwh", aggfunc="mean").rolling(14).mean()
por_tarifa.plot(ax=der, color=["#16302B", "#A8432C"]); der.axvspan(pd.Timestamp("2013-01-01"), pd.Timestamp("2013-12-31"), color="#E0A526", alpha=0.15)
der.set(title="Consumo por hogar según la tarifa (promedio de 14 días)", xlabel="", ylabel="kWh por día"); plt.tight_layout(); plt.show()

antes = limpio[limpio["fecha"].dt.year == 2012].groupby("tarifa")["kwh"].mean()
durante = limpio[limpio["fecha"].dt.year == 2013].groupby("tarifa")["kwh"].mean()
print(pd.DataFrame({"hogares": limpio.groupby("tarifa")["hogar"].nunique(), "kWh por día en 2012": antes.round(2), "kWh por día en 2013": durante.round(2),
                    "cambio (%)": ((durante / antes - 1) * 100).round(1)}).to_string())
'''),
("md", r'''
La importancia dice qué usa el modelo para pronosticar el cambio de mañana, y eso no es lo mismo que explicar el consumo. La estación casi no aparece ahí porque ya está metida en el consumo de hoy.

La comparación de tarifas pide cuidado. La franja amarilla es el año en que rigió la tarifa dinámica. Los dos grupos ya eran distintos en 2012, antes de que empezara, así que la diferencia de nivel no se le puede atribuir a la tarifa. Lo que sí se puede mirar es si el cambio de 2012 a 2013 fue distinto en los dos grupos, y aun así 2012 no está completo para todos los hogares. Con este dato la respuesta honesta es una asociación, y de tamaño modesto.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿cuándo se manda el aviso?

Llamo **día alto** a un día cuyo consumo supera un nivel $U$. Fijo $U$ en el percentil 75 del consumo anterior a la prueba. La empresa manda el aviso la víspera si el pronóstico supera un umbral $u$. Con cada $u$ salen cuatro casillas, y de ellas dos medidas:

$$\text{aciertos}=\frac{\#\{\hat C_{t+1}>u\ \text{y}\ C_{t+1}>U\}}{\#\{C_{t+1}>U\}},\qquad \text{falsas alarmas}=\frac{\#\{\hat C_{t+1}>u\ \text{y}\ C_{t+1}\le U\}}{\#\{\hat C_{t+1}>u\}}.$$

Bajar $u$ atrapa más días altos y también manda más avisos en falso. No hay un umbral correcto. Hay un precio, y la tabla lo muestra.
'''),
("code", r'''
U = serie[:"2013-10-31"].quantile(0.75)
real, pronostico = base["real"][P], pronosticos_prueba[elegido]
es_alto = real > U
filas = []
for factor in [0.90, 0.94, 0.97, 1.00, 1.03, 1.06]:
    aviso = pronostico > U * factor
    filas.append({"umbral (kWh)": U * factor, "avisos": int(aviso.sum()), "días altos avisados": int((aviso & es_alto).sum()),
                  "días altos sin aviso": int((~aviso & es_alto).sum()), "avisos en falso": int((aviso & ~es_alto).sum()),
                  "aciertos": (aviso & es_alto).sum() / max(es_alto.sum(), 1), "falsas alarmas": (aviso & ~es_alto).sum() / max(aviso.sum(), 1)})
decision = pd.DataFrame(filas)
print(f"Nivel de día alto: {U:.2f} kWh por hogar | días altos en la prueba: {int(es_alto.sum())} de {len(real)} | método: {elegido}")
display(decision.round(2))

plt.figure(figsize=(10, 3.6))
plt.plot(fechas, real, color="black", label="real"); plt.plot(fechas, pronostico, color="#A8432C", ls="--", label=f"pronóstico ({elegido})")
plt.axhline(U, color="#E0A526", label="nivel de día alto"); plt.ylabel("kWh por hogar"); plt.legend(fontsize=8); plt.title("Días altos en la prueba"); plt.show()
'''),
("md", r'''
Lean la tabla de arriba hacia abajo. Cada renglón es una política. Entre los renglones cambian dos columnas en sentido contrario: los días altos que se quedan sin aviso y los avisos en falso. La empresa escoge según cuál error le cuesta más. Un aviso en falso gasta la paciencia del cliente. Un día alto sin aviso le cuesta energía comprada a última hora.

### La decisión en tres frases

Escríbanla para una gerente comercial que no sabe qué es un percentil. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Leer el ZIP por rangos y procesar archivo por archivo | Bajar los 796 MB y abrir todo | El dato completo pide más memoria que la que hay |
| Subconjunto de archivos a intervalos iguales | Los primeros archivos | El archivo está ordenado por hogar: el principio no es una muestra |
| Resumen con suma y cuenta | Guardar promedios por pedazo | Los promedios y las medianas no se pueden juntar entre pedazos |
| `Null` es faltante y las filas repetidas se cuentan una vez | Dejarlas | Una lectura repetida suma doble |
| Solo días con 48 lecturas | Todos los días | Un día incompleto no es un día de bajo consumo |
| Consumo promedio por hogar, desde que hay la mitad del panel | Consumo total | El número de hogares creció durante 2012 |
| Pronosticar el cambio relativo | El nivel | Árboles y vecinos no extrapolan |
| Día alto: percentil 75 anterior a la prueba | Un nivel fijado mirando la prueba | El nivel no puede conocer el futuro |
| Cada tabla guardada lleva huella, regla y versión madre | Sobrescribir el archivo | Un resultado sin versión del dato no se puede reproducir |

### 4.3 Riesgos y mantenimiento

Respondan en dos o tres líneas cada una.

- El modelo no conoce la temperatura. ¿En qué días va a fallar más, y qué dato le agregarían primero?
- La prueba cubre un solo invierno. ¿Qué tanto confían en la tabla de umbrales para el invierno siguiente?
- Estos hogares aceptaron participar en un estudio. ¿En qué podrían diferenciarse del resto de Londres? Recuerden el sesgo de selección de la clase 2.
- El consumo cada media hora de un hogar dice a qué hora se levanta la gente, cuándo sale y si la casa está vacía. ¿Qué cuidados pide un dato así, aunque venga sin nombres?
- Si London Datastore reemplazara el archivo por una versión corregida, ¿cómo se enterarían ustedes?

### 4.4 Procedencia

En esta clase la procedencia tiene dos partes: lo que anota `descargar` y el registro de versiones que construimos.
'''),
("code", r'''
print("Fuente:", "UK Power Networks, SmartMeter Energy Consumption Data in London Households, London Datastore, CC BY 4.0")
print("Origen en esta corrida:", origen, "| archivos:", numeros)
print(json.dumps([v for v in json.loads((carpeta / "registro.json").read_text(encoding="utf-8")) if v["huella"] in (v1, v2)], ensure_ascii=False, indent=2))
'''),
("md", r'''
### 4.5 Cambien una línea

Vuelvan a la celda donde dice `ARCHIVOS = 24`, pongan `ARCHIVOS = 48` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. El proceso tardará cerca del doble. Llenen la tabla con sus dos corridas.

| | 24 archivos | 48 archivos |
|---|---|---|
| Lecturas procesadas y tiempo | | |
| Hogares | | |
| Huella de la versión limpia | | |
| Consumo promedio por hogar (kWh por día) | | |
| MAPE del ingenuo en prueba | | |
| Ganador en validación y su MAPE en prueba | | |
| Días altos en la prueba | | |

Con la tabla llena, respondan en tres líneas. ¿Qué cambió al duplicar el dato y qué no? ¿Vale la pena procesar los 168 archivos para esta pregunta? ¿Para cuál pregunta sí valdría la pena? **Mi respuesta:**
'''),
]

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
medidos_por_mes.plot(color="#16302B", figsize=(10, 2.8)); plt.title("% de días medidos en cada mes"); plt.xlabel(""); plt.show()
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

ejes[0, 0].hist(lluvia.dropna(), bins=60, color="#16302B"); ejes[0, 0].set_yscale("log")
ejes[0, 0].set(title="Lluvia diaria (eje vertical logarítmico)", xlabel="mm", ylabel="días")

por_mes = lluvia.groupby(lluvia.index.month).mean() * 30
ejes[0, 1].bar([meses[m - 1] for m in por_mes.index], por_mes.values, color="#3E7CB1"); ejes[0, 1].set(title="Lluvia de un mes típico", ylabel="mm")

fechas_completas = set(lluvia_limpia["fecha"])
horas = por_hora[por_hora["fecha"].isin(fechas_completas)]
ejes[1, 0].bar(range(24), horas.groupby("hora")["mm"].mean().reindex(range(24)).values, color="#E0A526")
ejes[1, 0].set(title="Lluvia promedio por hora del día", xlabel="hora", ylabel="mm")

rezagos = range(1, 15)
ejes[1, 1].bar(rezagos, [lluvia.autocorr(k) for k in rezagos], color="#A8432C")
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
izq.bar(range(1, 5), explicada, color="#16302B"); izq.set(title="Varianza explicada", xlabel="componente", ylabel="%")
der.plot(range(24), pca.components_[0], color="#A8432C", marker="o", label="CP1"); der.plot(range(24), pca.components_[1], color="#3E7CB1", marker="o", label="CP2")
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

todos.sort_values("validación").plot.barh(figsize=(9, 4), color=["#16302B", "#E0A526"])
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
eje.plot(despues.index, despues["acumulado"], color="#3E7CB1", lw=1, label=f"lluvia acumulada de {K} día(s), hasta ayer")
eje.scatter(despues.index[despues["evento"] == 1], despues.loc[despues["evento"] == 1, "acumulado"], color="#A8432C", s=18, zorder=3, label="día con movimiento en masa en Caldas")
eje.axhline(decision.loc[decision["percentil"] == 80, "umbral (mm)"].iloc[0], color="#E0A526", label="umbral del percentil 80")
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
