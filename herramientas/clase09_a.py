"""Celdas de la demostración de la clase 9, «El noticiero de un minuto»."""
from molde import ARRANQUE

NOTICIERO = [
("md", r'''
# Clase 9 · El noticiero de un minuto

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo. Hoy el resultado es un servicio que otra persona puede usar desde su teléfono. Tomamos una noticia real, la resume un modelo de lenguaje, la lee una voz sintética, y todo queda detrás de una API con su página. El gemelo, *El parte de incendios*, lo construyen ustedes en trabajo autónomo.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Llevar un análisis a producción: medir un resumen contra una línea base, envolver el flujo en una API con FastAPI, probarla, desplegarla y decidir qué controles necesita antes de salir al aire. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular el problema y la métrica), 3 (flujo reproducible y trazable), 4 (implementar y evaluar), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Claude hace hoy dos oficios: escribe el resumen dentro del producto y les ayuda a escribir código. Deben poder explicar sin ayuda qué mide ROUGE y qué no, qué es una ruta de una API, qué significan los códigos 200, 404, 422, 502 y 503, y por qué una llave nunca se escribe en el código. |
| **Entregable** | Este cuaderno corrido en su portafolio, con la decisión de la sección 4 escrita por ustedes. Quien tenga llaves entrega además la dirección de su página desplegada. |

**Qué corre sin llaves y qué no.** Las secciones 1 a 3.5 y las pruebas de la API corren sin ninguna cuenta. Las dos celdas que llaman a Claude y a ElevenLabs necesitan llaves: si no las encuentran, lo dicen y siguen. Ninguna celda inventa la respuesta de un servicio.
'''),
("code", ARRANQUE),
("code", r'''
# El parámetro del cuaderno. Al final de la clase se cambia esta línea y se corre todo de nuevo.
TEMA = "economia actualidad"       # sección de El País con que se evalúan los resumidores

import os, re, time
from pathlib import Path
PROYECTO = Path(raiz).resolve() / "clases" / "09_api" / "noticiero"      # el servicio vive en esta carpeta del repositorio
sys.path.insert(0, str(PROYECTO / "api"))

import importlib.util
faltan = [paquete for paquete in ["fastapi", "httpx", "anthropic"] if importlib.util.find_spec(paquete) is None]
if faltan:                                                      # en Colab casi todo viene instalado; lo que falte se instala aquí
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *faltan], check=True)
print("TEMA:", TEMA, "| paquetes instalados ahora:", faltan or "ninguno")
'''),
("md", r'''
## 1. Formulación

Imagínense una emisora comunitaria que quiere abrir cada hora con una noticia contada en un minuto. No tiene redactor de turno a las tres de la mañana. Alguien propone que una máquina lea la noticia, escriba el resumen y lo grabe con una voz sintética.

La emisora necesita tres respuestas:

1. **Predecir.** Dada una noticia, ¿qué resumen escribiría el periódico?
2. **Explicar.** ¿De qué parte de la noticia sale un buen resumen?
3. **Decidir.** ¿Sale al aire un resumen que ninguna persona leyó? ¿Con qué controles?

**El dato.** MLSUM es un corpus de investigación con noticias de periódicos en cinco idiomas. La parte en español son noticias de El País, cada una con su **entradilla**, que es el resumen de una o dos líneas que el propio periódico publica debajo del título. Esa entradilla es nuestra respuesta de referencia: la escribió un periodista.

**La métrica.** ROUGE compara las palabras del resumen candidato $c$ con las de la referencia $r$. Si $|c\cap r|$ es el número de palabras que coinciden, contando repeticiones,

$$P_1=\frac{|c\cap r|}{|c|},\qquad R_1=\frac{|c\cap r|}{|r|},\qquad \text{ROUGE-1}=F_1=\frac{2\,P_1R_1}{P_1+R_1}.$$

ROUGE-L hace lo mismo con la subsecuencia común más larga, $\text{LCS}(c,r)$, que premia conservar el orden de las palabras:

$$P_L=\frac{\text{LCS}(c,r)}{|c|},\qquad R_L=\frac{\text{LCS}(c,r)}{|r|},\qquad \text{ROUGE-L}=\frac{2\,P_LR_L}{P_L+R_L}.$$

Un producto de radio tiene además una medida propia, la duración. Si el locutor dice $v$ palabras por minuto, un texto de $n$ palabras dura

$$\hat d=\frac{60\,n}{v}\ \text{segundos},$$

y con $v=150$ un minuto son 150 palabras. Ese 150 es un supuesto de locución que hoy contrastamos con el audio.

### Predicción escrita

- Si mi «resumen» es copiar la primera frase de la noticia, ¿qué ROUGE-1 obtengo contra la entradilla? **Mi apuesta:**
- ¿En qué parte de la noticia está la frase que más se parece a la entradilla: al comienzo, en la mitad o al final? **Mi apuesta:**
- Si a un resumen le cambio una cifra por otra, ¿cuánto baja su ROUGE? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

Una línea trae las 13.920 noticias de la partición de prueba de MLSUM en español. Pesa 44 MB y tarda menos de medio minuto. Los textos son de El País: los usamos en clase con cita y enlace, y no los republicamos.
'''),
("code", r'''
noticias = descargar("mlsum_es")
print("Noticias:", len(noticias), "| Columnas:", list(noticias.columns))
unica = noticias.iloc[0]
print("\nTítulo:    ", unica["title"]); print("Entradilla:", unica["summary"]); print("Fecha:     ", unica["date"], "| Tema:", unica["topic"])
print("Texto:     ", unica["text"][:330], "…")
'''),
("md", r'''
### 2.1 Las trampas de limpieza

Miren el comienzo del texto de esa noticia. Antes de la primera frase del periodista viene el pie de foto, con el crédito del fotógrafo. Quien tome «la primera frase» como resumen de esa noticia se lleva el pie de foto. Hay que medir qué tan frecuente es, y hay otras tres trampas.
'''),
("code", r'''
# Trampa 1: la fecha llega como texto, día/mes/año
print("Fecha tal como llega:", repr(noticias.loc[0, "date"]))
noticias["fecha"] = pd.to_datetime(noticias["date"], format="%d/%m/%Y")
print("Rango:", noticias["fecha"].min().date(), "a", noticias["fecha"].max().date())

# Trampa 2: no todo está en español
print("\nTemas con más noticias:"); print(noticias["topic"].value_counts().head(8))
en_ingles = noticias["text"].str.contains(" the ", regex=False)
print("\nNoticias con la palabra «the» (%): tema «elpais inenglish»", round(float(en_ingles[noticias["topic"] == "elpais inenglish"].mean() * 100), 1),
      "| los demás temas", round(float(en_ingles[noticias["topic"] != "elpais inenglish"].mean() * 100), 1))

# Trampa 3: pies de foto y créditos al comienzo del texto
comienzo = noticias["text"].str[:250]
print("\nNoticias con «Foto:», «Vídeo:» o una agencia en los primeros 250 caracteres (%):",
      round(float(comienzo.str.contains(r"Foto:|Vídeo:|\bEFE\b|REUTERS|\bAFP\b|GETTY|EUROPA PRESS", regex=True).mean() * 100), 1))

# Trampa 4: textos repetidos y longitudes extremas
noticias["palabras"] = noticias["text"].str.split().str.len()
noticias["palabras_entradilla"] = noticias["summary"].str.split().str.len()
print("Direcciones repetidas:", int(noticias["url"].duplicated().sum()), "| textos repetidos:", int(noticias["text"].duplicated().sum()))
print("Palabras por noticia: mínimo", noticias["palabras"].min(), "| mediana", noticias["palabras"].median(), "| máximo", noticias["palabras"].max())
'''),
("md", r'''
Decisiones de limpieza, para la bitácora. Dejo por fuera el tema `elpais inenglish` porque el noticiero es en español. Dejo por fuera las noticias de menos de 150 palabras, que no dan para un minuto, y las de más de 3.000, que son reportajes largos. Hay textos repetidos con direcciones distintas, y de cada grupo conservo uno. Los pies de foto aparecen en cerca de una de cada veinte noticias y no tienen un formato fijo, así que no intento borrarlos con una regla. Queda anotado como un defecto conocido del dato.
'''),
("code", r'''
limpias = noticias[(noticias["topic"] != "elpais inenglish") & noticias["palabras"].between(150, 3000)].copy()
limpias = limpias.drop_duplicates("text").reset_index(drop=True)
print("Noticias que quedan:", len(limpias), "de", len(noticias))
'''),
("md", r'''
### 2.2 Exploración

Cuatro miradas: cuánto mide una noticia y cuánto su entradilla, cuánto comprime el periodista, cómo cambia eso por sección y cuántas noticias hay por semana.
'''),
("code", r'''
limpias["compresion"] = limpias["palabras"] / limpias["palabras_entradilla"]
principales = limpias["topic"].value_counts().head(10).index

fig, ejes = plt.subplots(1, 4, figsize=(16, 3.6))
ejes[0].hist(limpias["palabras"], bins=50, color="#16302B"); ejes[0].set(title="Palabras por noticia")
ejes[1].hist(limpias["palabras_entradilla"], bins=range(5, 60), color="#E0A526"); ejes[1].set(title="Palabras por entradilla")
limpias[limpias["topic"].isin(principales)].groupby("topic")["palabras"].median().sort_values().plot.barh(ax=ejes[2], color="#16302B")
ejes[2].set(title="Mediana de palabras por sección", ylabel=""); ejes[2].tick_params(axis="y", labelsize=7)
limpias.set_index("fecha").resample("W").size().plot(ax=ejes[3], color="#A8432C"); ejes[3].set(title="Noticias por semana", xlabel="")
plt.tight_layout(); plt.show()

print("Entradilla: mediana", limpias["palabras_entradilla"].median(), "palabras | a 150 palabras por minuto dura",
      round(limpias["palabras_entradilla"].median() / 150 * 60, 1), "segundos")
print("Compresión mediana (palabras de la noticia por cada palabra de entradilla):", round(limpias["compresion"].median(), 1))
print("Noticias del tema elegido:", int((limpias["topic"] == TEMA).sum()))
'''),
("md", r'''
Una lectura que cambia el problema. La entradilla del periódico dura menos de diez segundos leída en voz alta. El producto de la emisora dura un minuto. La referencia que tenemos mide un resumen más corto que el que vamos a producir, y eso va a pesar cuando midamos con ROUGE.

### 2.3 Un análisis no supervisado: ¿las secciones se ven en las palabras?

Represento cada noticia por sus palabras con pesos TF-IDF. Si la palabra $t$ aparece $\text{tf}_{t,d}$ veces en la noticia $d$ y está en $\text{df}_t$ de las $N$ noticias,

$$w_{t,d}=\text{tf}_{t,d}\cdot\Bigl(\ln\frac{1+N}{1+\text{df}_t}+1\Bigr).$$

Después reduzco esa matriz enorme a dos dimensiones con una descomposición en valores singulares truncada, $X\approx U_k\Sigma_kV_k^\top$, que es el equivalente del PCA para matrices dispersas. En texto se le llama análisis semántico latente.
'''),
("code", r'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

cinco = ["deportes actualidad", "economia actualidad", "cultura actualidad", "internacional actualidad", "elpais ciencia"]
muestra = limpias[limpias["topic"].isin(cinco)]
tfidf = TfidfVectorizer(max_features=20000, min_df=5, max_df=0.5, sublinear_tf=True)
svd = TruncatedSVD(n_components=10, random_state=0)
Z = svd.fit_transform(tfidf.fit_transform(muestra["text"]))

plt.figure(figsize=(8, 5))
for tema, color in zip(cinco, ["#A8432C", "#16302B", "#E0A526", "#3B7EA1", "#7B5EA7"]):
    plt.scatter(Z[muestra["topic"] == tema, 1], Z[muestra["topic"] == tema, 2], s=5, alpha=0.5, color=color, label=tema)
plt.xlabel("componente 2"); plt.ylabel("componente 3"); plt.legend(fontsize=8, markerscale=3); plt.title("Noticias de cinco secciones en el espacio latente"); plt.show()
print("Varianza explicada por los diez componentes (%):", (svd.explained_variance_ratio_ * 100).round(1))
vocabulario = np.array(tfidf.get_feature_names_out())
for c in (1, 2):
    print(f"Componente {c + 1}: palabras con más peso positivo:", list(vocabulario[np.argsort(-svd.components_[c])[:8]]), "| negativo:", list(vocabulario[np.argsort(svd.components_[c])[:8]]))
'''),
("md", r'''
Dibujo los componentes 2 y 3 porque en TF-IDF el primero casi siempre mide lo que todas las noticias tienen en común. Las secciones forman regiones reconocibles sin que nadie le haya dicho al método cuál es cuál, y las palabras con más peso dicen qué separa cada eje.

Miren esas palabras con cuidado. Junto a «equipo» y «partido» aparecen «suscribirte», «newsletter», «facebook» y «twitter». Esas palabras no hablan de deportes: vienen de un texto fijo de invitación a seguir la sección, que acompaña a muchas noticias. El método encontró una costura del dato que no habíamos visto en la limpieza. Un análisis no supervisado sirve también para eso.

Este análisis es descriptivo. Lo que sí vamos a reutilizar es el TF-IDF, como herramienta para escoger frases.
'''),
("md", r'''
## 3. Modelado

### 3.1 La línea base y cinco resumidores que no necesitan llave

Antes de pagarle a un modelo de lenguaje hay que saber cuánto logra lo que no cuesta nada. Todos estos candidatos son **extractivos**: no escriben, escogen frases de la noticia.

| Candidato | Qué entrega |
|---|---|
| Línea base: el título | El título de la noticia |
| Primera frase | La frase 1 del texto |
| Dos primeras frases | Las frases 1 y 2 |
| Frase central | La frase cuyo vector TF-IDF se parece más al de la noticia completa |
| Frase más parecida al título | La de mayor similitud de coseno con el título |
| TextRank | La frase más conectada en el grafo de similitudes entre frases |

La similitud entre dos frases con vectores $a$ y $b$ es el coseno, $\cos(a,b)=a^\top b/(\lVert a\rVert\,\lVert b\rVert)$. TextRank arma una matriz $S$ con las similitudes entre todas las frases de la noticia, la normaliza por columnas en $\tilde S$ y busca el vector de puntajes que cumple

$$p=\frac{1-d}{n}\,\mathbf 1+d\,\tilde S\,p,\qquad d=0{,}85,$$

que es el mismo PageRank con que se ordenan páginas web. Una frase es importante si se parece a otras frases importantes.

**El protocolo.** Las noticias del tema elegido se ordenan por fecha y se parten en tres tramos. Con el primero se ajusta el vocabulario TF-IDF, que es lo único que aquí se «entrena». Con el segundo se escoge el candidato. Con el tercero se reporta.
'''),
("code", r'''
def frases(texto):
    """Parte un texto en frases: corta después de punto, interrogación o exclamación cuando sigue una mayúscula."""
    partes = re.split(r"(?<=[.!?…])\s+(?=[A-ZÁÉÍÓÚÑ¿¡“\"])", texto.strip())
    return [p for p in partes if len(p.split()) >= 4] or [texto]

def palabras(texto):
    return re.findall(r"\w+", texto.lower())

def rouge(candidato, referencia):
    """Devuelve ROUGE-1 (precisión, cobertura, F) y ROUGE-L (F)."""
    c, r = palabras(candidato), palabras(referencia)
    if not c or not r:
        return 0.0, 0.0, 0.0, 0.0
    cuenta_r = {}
    for p in r:
        cuenta_r[p] = cuenta_r.get(p, 0) + 1
    comunes = 0
    for p in c:
        if cuenta_r.get(p, 0) > 0:
            comunes += 1; cuenta_r[p] -= 1
    p1, r1 = comunes / len(c), comunes / len(r)
    anterior = [0] * (len(r) + 1)                              # subsecuencia común más larga, por programación dinámica
    for a in c:
        actual = [0]
        for j, b_ in enumerate(r):
            actual.append(anterior[j] + 1 if a == b_ else max(anterior[j + 1], actual[j]))
        anterior = actual
    lcs = anterior[-1]
    f = lambda p, q: 2 * p * q / (p + q) if p + q else 0.0
    return p1, r1, f(p1, r1), f(lcs / len(c), lcs / len(r))

print("Prueba de la métrica con dos frases que comparten 3 de 4 palabras:", [round(v, 2) for v in rouge("el gato come pescado", "el gato come carne")])

del_tema = limpias[limpias["topic"] == TEMA].sort_values("fecha").reset_index(drop=True)
cortes = del_tema["fecha"].quantile([0.5, 0.75]).values
E = (del_tema["fecha"] < cortes[0]).values
V = ((del_tema["fecha"] >= cortes[0]) & (del_tema["fecha"] < cortes[1])).values
P_ = (del_tema["fecha"] >= cortes[1]).values
print(f"Tema «{TEMA}»: {len(del_tema)} noticias | ajuste del vocabulario: {E.sum()} (hasta {pd.Timestamp(cortes[0]).date()}) | validación: {V.sum()} | prueba: {P_.sum()}")
'''),
("code", r'''
from sklearn.metrics.pairwise import cosine_similarity

vectorizador = TfidfVectorizer(min_df=2, sublinear_tf=True).fit(del_tema.loc[E, "text"])      # el vocabulario solo ve el primer tramo

def textrank(S, d=0.85, pasos=50):
    S = S.copy(); np.fill_diagonal(S, 0)
    columnas = S.sum(axis=0); columnas[columnas == 0] = 1
    S, p = S / columnas, np.full(len(S), 1 / len(S))
    for _ in range(pasos):
        p = (1 - d) / len(S) + d * S @ p
    return p

def candidatos(fila):
    fr = frases(fila["text"])[:40]                             # las primeras 40 frases bastan y acotan el tiempo
    M = vectorizador.transform(fr + [fila["text"], fila["title"]])
    F, todo, titulo = M[:-2], M[-2], M[-1]
    return {
        "título (línea base)": fila["title"],
        "primera frase": fr[0],
        "dos primeras frases": " ".join(fr[:2]),
        "frase central": fr[int(cosine_similarity(F, todo).ravel().argmax())],
        "parecida al título": fr[int(cosine_similarity(F, titulo).ravel().argmax())],
        "TextRank": fr[int(textrank(cosine_similarity(F)).argmax())],
    }

inicio = time.time()
filas = []
for i, fila in del_tema[V | P_].iterrows():
    for nombre, texto in candidatos(fila).items():
        p1, r1, f1, fl = rouge(texto, fila["summary"])
        filas.append({"noticia": i, "tramo": "validación" if V[i] else "prueba", "candidato": nombre,
                      "ROUGE-1": f1, "ROUGE-L": fl, "precisión": p1, "cobertura": r1, "palabras": len(texto.split())})
medidas = pd.DataFrame(filas)
print(f"{len(medidas)} resúmenes medidos en {time.time() - inicio:.0f} s")

tabla = medidas.pivot_table(index="candidato", columns="tramo", values=["ROUGE-1", "ROUGE-L"]).round(3)
tabla["palabras"] = medidas.groupby("candidato")["palabras"].median()
display(tabla.sort_values(("ROUGE-L", "validación"), ascending=False))
'''),
("code", r'''
orden = tabla.sort_values(("ROUGE-L", "validación")).index
fig, (izq, der) = plt.subplots(1, 2, figsize=(13, 3.8))
tabla["ROUGE-L"].loc[orden, ["validación", "prueba"]].plot.barh(ax=izq, color=["#16302B", "#E0A526"])
izq.set(title="ROUGE-L contra la entradilla", ylabel="")
tabla["ROUGE-1"].loc[orden, ["validación", "prueba"]].plot.barh(ax=der, color=["#16302B", "#E0A526"])
der.set(title="ROUGE-1 contra la entradilla", ylabel="")
plt.tight_layout(); plt.show()

sin_base = tabla.drop(index="título (línea base)")
mejor = sin_base[("ROUGE-L", "validación")].idxmax()
print(f"Ganador en validación (ROUGE-L): {mejor}")
print(f"En prueba: ROUGE-L {tabla.loc[mejor, ('ROUGE-L', 'prueba')]:.3f} | línea base (título): {tabla.loc['título (línea base)', ('ROUGE-L', 'prueba')]:.3f}")

# ¿La diferencia con la línea base es más que ruido? Bootstrap sobre las noticias de prueba.
en_prueba = medidas[medidas["tramo"] == "prueba"].pivot(index="noticia", columns="candidato", values="ROUGE-L")
diferencia = (en_prueba[mejor] - en_prueba["título (línea base)"]).values
azar = np.random.default_rng(0)
remuestreos = [diferencia[azar.integers(0, len(diferencia), len(diferencia))].mean() for _ in range(2000)]
print(f"Diferencia media con el título: {diferencia.mean():+.3f} (IC 95 %: {np.percentile(remuestreos, 2.5):+.3f} a {np.percentile(remuestreos, 97.5):+.3f}), con {len(diferencia)} noticias")
'''),
("md", r'''
Comparen con su primera apuesta. Lean dos cosas en la tabla. La primera es quién gana y por cuánto, con su intervalo: si el intervalo de la diferencia contiene el cero, el ganador no se distingue de la línea base. La segunda es que todos los valores son bajos. Dos personas que resumen la misma noticia usan palabras distintas, así que un ROUGE bajo no significa que el resumen sea malo.

### 3.2 Explicar: ¿de qué parte de la noticia sale el resumen?

Para cada noticia de validación busco la frase que más se parece a la entradilla, medida con ROUGE-L, y anoto en qué posición está.
'''),
("code", r'''
posiciones = []
for i, fila in del_tema[V].iterrows():
    fr = frases(fila["text"])[:40]
    posiciones.append(int(np.argmax([rouge(f, fila["summary"])[3] for f in fr])) + 1)
posiciones = np.array(posiciones)

plt.figure(figsize=(9, 3.4))
plt.hist(posiciones, bins=np.arange(0.5, 41.5), color="#16302B")
plt.xlabel("posición de la frase más parecida a la entradilla"); plt.ylabel("noticias"); plt.title("¿Dónde está el resumen dentro de la noticia?"); plt.show()
print(f"La mejor frase es la primera en el {np.mean(posiciones == 1) * 100:.0f} % de las noticias, está entre las tres primeras en el {np.mean(posiciones <= 3) * 100:.0f} % y después de la décima en el {np.mean(posiciones > 10) * 100:.0f} %")

oraculo = np.mean([max(rouge(f, fila["summary"])[3] for f in frases(fila["text"])[:40]) for _, fila in del_tema[V].iterrows()])
print(f"Techo extractivo: si alguien escogiera siempre la mejor frase, su ROUGE-L en validación sería {oraculo:.3f}")
'''),
("md", r'''
Comparen con su segunda apuesta. En periodismo se enseña la pirámide invertida: lo más importante va arriba. Si esa regla bastara, la frase más parecida a la entradilla sería casi siempre la primera. Lean en qué porcentaje lo es aquí y cuántas veces está después de la décima. En este corpus la entradilla no repite el comienzo de la noticia, y por eso los candidatos que copian el comienzo no ganan.

El techo extractivo da la otra mitad de la explicación. Es el ROUGE-L que tendría un resumidor que siempre escogiera la mejor frase posible. La distancia entre el ganador y ese techo es lo que se pierde por escoger mal la frase. La distancia entre el techo y 1 es lo que ninguna frase copiada puede dar, porque el periodista escribió la entradilla con palabras propias.

### 3.3 La trampa: una métrica que no ve los errores que importan

ROUGE cuenta palabras compartidas. No sabe si una cifra es correcta. Hagamos el experimento con resúmenes reales: tomo las entradillas de prueba que traen algún número y le agrego un cero al primero. «300 despidos» pasa a ser «3000 despidos». Mido el ROUGE de la entradilla dañada contra la original.

Después hago un segundo experimento. El producto de la emisora dura un minuto, así que mido como «resumen» las primeras 140 palabras de la noticia, contra la misma entradilla.
'''),
("code", r'''
from main import cifras_sin_respaldo, segundos_estimados, PALABRAS_OBJETIVO      # las piezas del servicio que se prueban sin llave

con_numero = del_tema[P_ & del_tema["summary"].str.contains(r"\d").values]
danadas = con_numero["summary"].str.replace(r"(\d+)", lambda m: m.group(1) + "0", n=1, regex=True)
r_danada = np.array([rouge(d, o)[2:] for d, o in zip(danadas, con_numero["summary"])])
detectadas = np.mean([len(cifras_sin_respaldo(d, t)) > 0 for d, t in zip(danadas, con_numero["text"])])
falsas = np.mean([len(cifras_sin_respaldo(o, t)) > 0 for o, t in zip(con_numero["summary"], con_numero["text"])])
print(f"Entradillas de prueba con algún número: {len(con_numero)}")
print("Ejemplo:", con_numero["summary"].iloc[0], "->", danadas.iloc[0])
print(f"ROUGE-1 de la entradilla con la cifra dañada, contra la original: {r_danada[:, 0].mean():.3f} | ROUGE-L: {r_danada[:, 1].mean():.3f}  (1 sería idéntica)")
print(f"El control de cifras marca el {detectadas * 100:.0f} % de las dañadas | y marca por error el {falsas * 100:.0f} % de las originales")

largo = np.array([rouge(" ".join(t.split()[:PALABRAS_OBJETIVO]), s)[:3] for t, s in zip(del_tema.loc[P_, "text"], del_tema.loc[P_, "summary"])])
print(f"\nPrimeras {PALABRAS_OBJETIVO} palabras como resumen: precisión {largo[:, 0].mean():.3f} | cobertura {largo[:, 1].mean():.3f} | ROUGE-1 {largo[:, 2].mean():.3f}")
print(f"Duración estimada de {PALABRAS_OBJETIVO} palabras: {segundos_estimados('palabra ' * PALABRAS_OBJETIVO)} segundos")
'''),
("md", r'''
Comparen con su tercera apuesta. Un resumen con una cifra multiplicada por diez conserva casi todo su ROUGE, y en una emisora ese es el error que termina en una rectificación. En el segundo experimento pasa lo contrario: un texto de un minuto queda castigado en precisión por el solo hecho de ser más largo que la entradilla, aunque todo lo que diga sea cierto.

La conclusión metodológica es que ROUGE contra la entradilla sirve para comparar resumidores cortos entre sí, y no sirve para aprobar el texto de un minuto que va a salir al aire. Para eso hacen falta controles de otra clase. Uno ya quedó escrito y probado: `cifras_sin_respaldo` revisa que todo número con dígitos del resumen esté en la noticia. Miren también su tasa de falsas alarmas sobre entradillas originales: el periodista a veces redondea o calcula, y el control lo marca. Un control barato deja pasar errores y además molesta de más. Hay que saber las dos tasas.

### 3.4 El resumen con Claude

Ahora sí, el modelo de lenguaje. La función `resumir_con_claude` vive en el servicio (`noticiero/api/main.py`) y hace una sola llamada a la API de mensajes, con las instrucciones del redactor de radio. La llave se lee de la variable de entorno `ANTHROPIC_API_KEY`. En Colab se guarda en el panel *Secretos* (el icono de la llave, a la izquierda) y esta celda la pasa al entorno.

**Si no hay llave, la celda lo dice y no produce nada.** No hay respuestas de ejemplo guardadas.
'''),
("code", r'''
for nombre in ["ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY"]:
    if not os.environ.get(nombre):
        try:                                                    # en Colab, las llaves viven en el panel Secretos
            from google.colab import userdata
            os.environ[nombre] = userdata.get(nombre)
        except Exception:
            pass
print({nombre: "encontrada" if os.environ.get(nombre) else "NO está" for nombre in ["ANTHROPIC_API_KEY", "ELEVENLABS_API_KEY"]})

import main
print("\nInstrucciones que recibe el modelo", f"({main.MODELO_CLAUDE}):\n")
print(main.INSTRUCCIONES)
'''),
("code", r'''
muestra_claude = del_tema[P_].head(5)                           # cinco noticias de prueba bastan para la clase
resumenes = {}
if os.environ.get("ANTHROPIC_API_KEY"):
    filas = []
    for i, fila in muestra_claude.iterrows():
        texto = main.resumir_con_claude({"titulo": fila["title"], "texto": fila["text"]})
        resumenes[i] = texto
        p1, r1, f1, fl = rouge(texto, fila["summary"])
        filas.append({"noticia": fila["title"][:50], "palabras": len(texto.split()), "segundos estimados": segundos_estimados(texto),
                      "cifras sin respaldo": cifras_sin_respaldo(texto, fila["text"]), "ROUGE-1": round(f1, 3), "cobertura": round(r1, 3)})
    display(pd.DataFrame(filas))
    print("\nPrimer resumen:\n", next(iter(resumenes.values())))
else:
    print("No hay ANTHROPIC_API_KEY en el entorno. Esta celda NO corrió y no hay resúmenes de Claude en este cuaderno.")
    print("Con la llave puesta, la celda resume cinco noticias de prueba y mide palabras, duración, cifras sin respaldo y ROUGE.")
'''),
("md", r'''
Cuando la celda corra, lean la tabla contra lo que aprendimos en 3.3. La columna que decide es `cifras sin respaldo`, junto con la duración. El ROUGE-1 saldrá bajo por la longitud, y la cobertura dirá cuánto de la entradilla quedó dentro del minuto. Después lean ustedes un resumen completo contra su noticia. Ese control humano no lo reemplaza ninguna de las columnas.

### 3.5 La voz

`voz_con_elevenlabs` manda el texto a la API de ElevenLabs y recibe un archivo MP3. Usa el modelo multilingüe y una de las voces de catálogo. La llave se lee de `ELEVENLABS_API_KEY`. Con el audio en la mano podemos contrastar el supuesto de 150 palabras por minuto: un MP3 de 128 kbit/s pesa 16.000 bytes por segundo, así que su duración es

$$d=\frac{\text{bytes}}{16\,000}\ \text{segundos}.$$
'''),
("code", r'''
if os.environ.get("ELEVENLABS_API_KEY") and resumenes:
    from IPython.display import Audio, display as mostrar
    filas = []
    for i, texto in resumenes.items():
        mp3 = main.voz_con_elevenlabs(texto)
        filas.append({"palabras": len(texto.split()), "segundos estimados": segundos_estimados(texto), "segundos reales": round(len(mp3) / 16000, 1)})
    duraciones = pd.DataFrame(filas)
    duraciones["palabras por minuto reales"] = (duraciones["palabras"] / duraciones["segundos reales"] * 60).round(0)
    display(duraciones)
    mostrar(Audio(mp3, autoplay=False))
else:
    print("Falta ELEVENLABS_API_KEY o no hay resúmenes de la celda anterior. Esta celda NO corrió y no hay audio en este cuaderno.")
    print("Con las dos llaves, la celda graba los cinco resúmenes, mide su duración real y deja oír el último.")
'''),
("md", r'''
### 3.6 De cuaderno a servicio: la API

Un cuaderno le sirve a quien lo corre. Para que le sirva a la emisora hay que dejar el flujo detrás de una dirección de internet que responda a pedidos. Eso es una API. La nuestra está en `noticiero/api/main.py`, escrita con FastAPI, y tiene cuatro rutas:

| Ruta | Qué hace | Necesita llave |
|---|---|---|
| `GET /` | Dice si el servicio está vivo y qué llaves tiene configuradas | No |
| `GET /noticia/{indice}` | Trae la noticia número `indice` desde el corpus en Hugging Face | No |
| `POST /resumen/{indice}` | Resume la noticia con Claude y le aplica los controles | Claude |
| `POST /voz` | Recibe un texto y devuelve el MP3 | ElevenLabs |

Cada respuesta lleva un código. 200 es «listo». 404 es «eso no existe». 422 es «el pedido está mal formado». 502 es «el servicio del que dependo falló». 503 es «no estoy en condiciones de atender esto», que es lo que la API responde cuando le falta una llave.

FastAPI trae un cliente de prueba que llama a la API sin prender un servidor. Probar una API es pedirle lo correcto y también lo incorrecto.
'''),
("code", r'''
print((PROYECTO / "api" / "main.py").read_text(encoding="utf-8").split("# ------------------------------------------------------------------ rutas")[1])
'''),
("code", r'''
from fastapi.testclient import TestClient
cliente = TestClient(main.app)

hay_claude, hay_voz = bool(os.environ.get("ANTHROPIC_API_KEY")), bool(os.environ.get("ELEVENLABS_API_KEY"))
pruebas = [
    ("El servicio responde", lambda: cliente.get("/"), 200),
    ("Una noticia que existe", lambda: cliente.get("/noticia/25"), 200),
    ("Una noticia que no existe", lambda: cliente.get("/noticia/99999"), 404),
    ("Un índice que no es un número", lambda: cliente.get("/noticia/abc"), 422),
    ("Voz con texto vacío", lambda: cliente.post("/voz", json={"texto": ""}), 422),
    ("Voz con un texto demasiado largo", lambda: cliente.post("/voz", json={"texto": "a" * 5000}), 422),
    ("Resumen" + ("" if hay_claude else " sin llave de Claude"), lambda: cliente.post("/resumen/25"), 200 if hay_claude else 503),
    ("Voz" + ("" if hay_voz else " sin llave de ElevenLabs"), lambda: cliente.post("/voz", json={"texto": "Buenos días."}), 200 if hay_voz else 503),
]
filas = []
for nombre, pedir, esperado in pruebas:
    respuesta = pedir()
    cuerpo = respuesta.json() if respuesta.headers["content-type"].startswith("application/json") else f"{len(respuesta.content)} bytes de audio"
    detalle = cuerpo.get("detail", cuerpo.get("titulo", cuerpo.get("estado", ""))) if isinstance(cuerpo, dict) else cuerpo
    filas.append({"prueba": nombre, "código": respuesta.status_code, "esperado": esperado, "pasa": respuesta.status_code == esperado, "detalle": str(detalle)[:70]})
resultado = pd.DataFrame(filas)
display(resultado)
print(f"Pasaron {int(resultado['pasa'].sum())} de {len(resultado)} pruebas.")
'''),
("md", r'''
Lean la columna `detalle` de las pruebas que fallan a propósito. Un buen mensaje de error le dice a quien llama qué hizo mal y qué puede hacer. La API no se cae cuando le falta una llave y tampoco inventa un resumen: responde 503 y explica.

### 3.7 ¿Y los agentes?

Lo que construimos es una **cadena**: noticia, resumen, control, voz, siempre en ese orden y decidido por nosotros. Un **agente** funciona distinto. Es un modelo de lenguaje al que se le dan herramientas y un objetivo, y que decide por sí mismo qué herramienta usar y cuándo parar. Para este producto, un agente podría escoger la noticia del día, pedir el resumen, leer el aviso del control de cifras y reescribir hasta que pase.

Tres preguntas para decidir entre cadena y agente. ¿Los pasos son siempre los mismos? Si lo son, una cadena es más barata, más rápida y más fácil de probar. ¿Qué puede romper el agente con las herramientas que le doy? Un agente que puede publicar al aire necesita un límite que no dependa de su buen juicio. ¿Cómo sé qué hizo? Sin bitácora de pasos no hay auditoría.

En el gemelo van a trabajar con un agente del otro lado: Claude les escribe la API y ustedes la aceptan o la rechazan con pruebas que ya están escritas.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Desplegar: de mi computador a una dirección pública

El servicio tiene dos piezas, y cada una va a un lugar distinto.

**La API, en Railway.** Railway toma una carpeta de un repositorio de GitHub, instala lo que diga `requirements.txt` y corre la orden del archivo `Procfile`.

1. Suban a su GitHub una copia de la carpeta `clases/09_api/noticiero/`.
2. En Railway, creen un proyecto nuevo desde ese repositorio y señalen como carpeta raíz `api`.
3. En la pestaña de variables escriban `ANTHROPIC_API_KEY` y `ELEVENLABS_API_KEY`. Ahí es donde viven las llaves: en el servicio, nunca en el código ni en el repositorio.
4. Generen un dominio público y ábranlo en el navegador. Debe responder con el estado del servicio, y `/docs` muestra la documentación que FastAPI escribe sola.

**La página, en Vercel.** La página es un solo archivo HTML que llama a la API desde el navegador de quien la visita.

1. En `web/config.js` cambien la dirección por la de su API en Railway.
2. En Vercel, importen el mismo repositorio y señalen como carpeta raíz `web`. No hay nada que compilar.
3. Abran la dirección que les da Vercel. Un enlace como `…/index.html?indice=25` hace el noticiero de la noticia 25.
4. Vuelvan a Railway y pongan en la variable `ORIGENES` la dirección de su página, para que la API solo atienda pedidos que vengan de ella.

Estos pasos los probé hasta donde se puede sin cuentas: la API y la página corren en el computador y se hablan entre sí. El despliegue en Railway y en Vercel lo hacemos juntos en clase.

### 4.2 Decidir: ¿sale al aire sin que nadie lo lea?

La API ya trae una regla escrita, en el campo `sale_al_aire` de `/resumen`: el resumen sale solo si no tiene cifras sin respaldo y si su duración estimada está entre 45 y 70 segundos. Lo que no cumple va a revisión humana.

Esa regla es una propuesta y tiene límites que ya medimos. El control de cifras deja pasar los números escritos con letras y los nombres cambiados, y marca por error una parte de los resúmenes correctos. Completen la decisión con lo que vieron hoy.

### La decisión en tres frases

Para la directora de la emisora, que no sabe qué es una API.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.3 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Entradilla de El País como referencia | Resúmenes escritos por nosotros | La escribió un periodista y viene con el dato |
| Dejar por fuera `elpais inenglish` y los textos de menos de 150 o más de 3.000 palabras | Usar todo | El noticiero es en español y dura un minuto |
| No borrar los pies de foto con una regla | Expresión regular de limpieza | No tienen formato fijo; se midió el daño en vez de esconderlo |
| Tres tramos por fecha: vocabulario, validación y prueba | Partir al azar | Las noticias de una misma semana comparten tema y palabras |
| ROUGE-L para escoger entre resumidores cortos | ROUGE para aprobar el texto de un minuto | Castiga la longitud y no ve cifras erradas |
| Control de cifras con dígitos | Confiar en el modelo | Es barato y atrapa el error que más cuesta; se conocen sus dos tasas de error |
| La API pide cada noticia a Hugging Face | Guardar las noticias en el repositorio | Los textos son de El País y no se republican |
| Llaves en variables de entorno | Llaves en el código | Un repositorio público con una llave adentro es una cuenta abierta a cualquiera |
| Tope de 1.500 caracteres en `/voz` y memoria de lo ya pedido | Sin límites | La voz se cobra por carácter y la dirección es pública |

### 4.4 Riesgos y mantenimiento

- **Verdad.** Un modelo de lenguaje puede escribir con soltura algo que la noticia no dice. ¿Qué control agregarían al de cifras? ¿Quién responde por lo que sale al aire?
- **Derechos.** Las noticias son de El País y el corpus se publicó para investigación. ¿Puede una emisora leer al aire resúmenes de noticias de otro medio? ¿Qué habría que acordar y cómo se da el crédito?
- **Voz.** La voz es sintética. ¿Debe saberlo el oyente? ¿Qué cambia si alguien propone clonar la voz de un locutor conocido?
- **Plata y abuso.** La dirección de la API es pública y cada pedido cuesta. Lean en `main.py` qué hace `CLAVE_DEL_SALON` y propongan un control más.
- **Dependencias.** El servicio depende de Hugging Face, de Anthropic y de ElevenLabs. ¿Qué ve el oyente cuando una de las tres falla, y qué debería ver?
- **Mantenimiento.** Los nombres de los modelos cambian. ¿Dónde está escrito el modelo de Claude en este servicio y cómo se cambia sin tocar el código?

### 4.5 Cambien una línea

Vuelvan a la celda de parámetros, cambien `TEMA = "economia actualidad"` por `TEMA = "deportes actualidad"` y corran todo de nuevo.

| | Economía | Deportes |
|---|---|---|
| Noticias del tema | | |
| Ganador en validación | | |
| ROUGE-L del ganador en prueba | | |
| ROUGE-L del título en prueba | | |
| Noticias cuya mejor frase es la primera (%) | | |
| Entradillas dañadas que marca el control de cifras (%) | | |

Con la tabla llena, respondan en tres líneas: ¿el mismo resumidor sirve para las dos secciones? ¿En cuál confiarían menos en un resumen automático, y por qué? **Mi respuesta:**

### 4.6 Procedencia
'''),
("code", r'''
{nombre: ficha for nombre, ficha in procedencia().items() if nombre == "mlsum_es"}
'''),
]
