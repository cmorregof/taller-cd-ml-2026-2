"""Genera los cuadernos de la clase 2: El ICFES (demostración) y Dengue (gemelo).

El cuaderno de los estudiantes sale del mismo texto que la solución: lo que va
entre « » en el código se reemplaza por ____.
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno

# -------------------------------------------------------------------- EL ICFES
ICFES = [
("md", r'''
# Clase 2 · El ICFES

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo durante la primera mitad de la clase. Ustedes lo siguen, lo corren y me interrumpen. En la segunda mitad cada uno resuelve el gemelo, *Dengue*, con la misma técnica y otros datos.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Limpiar un dato de personas dejando cada decisión a la vista, reconocer los sesgos que trae y los que agrega la limpieza, y escoger una métrica que responda a la decisión del cliente. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una línea de código, que proponga cómo codificar una categoría o que escriba una gráfica. Deben poder explicar sin ayuda por qué la exactitud engaña en este problema, qué es la captación, por qué borrar filas con faltantes sesga la muestra, qué columnas filtran la respuesta y por qué la partición se hace por colegio. |
| **Entregable** | Este cuaderno corrido de principio a fin en su portafolio, con las predicciones escritas, la decisión en tres frases y la tabla de la sección 4.5 llenas. |
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

Imagínense que la Secretaría de Educación de Caldas tiene un programa de refuerzo para estudiantes de grado once. El presupuesto alcanza para uno de cada cuatro. Los cupos se reparten meses antes del examen, cuando lo único que se sabe de cada estudiante es lo que respondió en el formulario de inscripción y en qué colegio estudia.

La Secretaría necesita tres respuestas:

1. **Predecir.** ¿Qué estudiantes van a quedar en el cuarto más bajo del país en Saber 11?
2. **Explicar.** ¿Qué condiciones del hogar y del colegio acompañan ese resultado?
3. **Decidir.** Con cupo para el 25 %, ¿a quiénes se les ofrece el refuerzo, y a quiénes deja por fuera esa regla?

**El dato.** El ICFES publica los resultados anonimizados de cada estudiante que presentó Saber 11. Usamos la aplicación 2020-2, que trae 504.872 estudiantes del país. Cada fila es una persona, con su puntaje, las respuestas del cuestionario socioeconómico y los datos del colegio. Nosotros bajamos un departamento.

**La respuesta que vamos a modelar.** El ICFES entrega con cada puntaje global su percentil nacional. Llamo $y_i=1$ al estudiante $i$ cuyo percentil es 25 o menos, y $y_i=0$ a los demás. El corte en 25 es una decisión mía, y por eso va a la bitácora.

**La métrica.** La Secretaría no quiere un porcentaje de aciertos. Quiere que sus cupos lleguen a quienes iban a quedar abajo. Si el modelo ordena a los $n$ estudiantes por riesgo y el programa atiende a los $k=\lfloor 0{,}25\,n\rceil$ primeros, la métrica es la **captación**:

$$\text{captación}=\frac{\#\{i:\ y_i=1\ \text{y}\ i\ \text{recibe cupo}\}}{\#\{i:\ y_i=1\}}.$$

Una captación de 0,60 significa que, de cada diez estudiantes que quedaron en el cuarto más bajo, seis tenían cupo.

### Predicción escrita

Antes de ver un solo dato, escriban aquí su apuesta. Después la comparamos con lo que salga.

- En Caldas, ¿qué fracción de los estudiantes queda en el cuarto más bajo del país? **Mi apuesta:**
- Si los cupos se repartieran por sorteo, ¿cuál sería la captación? **Mi apuesta:**
- Con lo que se sabe en la inscripción, ¿qué captación logrará el mejor modelo? **Mi apuesta:**
- ¿Qué variable pesará más: el estrato, la educación de la madre, los libros que hay en la casa o el tipo de colegio? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

La primera celda fija los dos parámetros del cuaderno. La segunda trae el dato desde datos.gov.co y tarda unos veinte segundos.
'''),
("code", r'''
DEPARTAMENTO = "CALDAS"      # al final de la clase lo cambiarán por otro departamento
CUPO = 0.25                  # fracción de estudiantes para la que alcanza el programa
'''),
("code", r'''
crudo = descargar("saber11", departamento=DEPARTAMENTO)
print("Filas:", len(crudo), "| Columnas:", crudo.shape[1], "| Colegios:", crudo["cole_cod_dane_establecimiento"].nunique())
print("¿Algún estudiante repetido?", crudo["estu_consecutivo"].duplicated().any())
from taller.presentacion import ficha_del_dato, RegistroDeLimpieza, resumen_pca, nube_3d, CIELO, ROJO
display(ficha_del_dato(crudo, f"Saber 11 2020-2, {DEPARTAMENTO.title()}"))
crudo.head(3).T
'''),
("md", r'''
### 2.1 Las trampas de la limpieza

Limpiar es decidir. Cada regla de limpieza cambia quién queda en la tabla, y con eso cambia la respuesta. Voy a mostrar cinco trampas de este dato y la decisión que tomo en cada una.

**Primera trampa: fechas de nacimiento imposibles.** La edad no viene en la tabla. Hay que calcularla con la fecha de nacimiento, y esa fecha la digitó una persona.
'''),
("code", r'''
anio_nacimiento = pd.to_numeric(crudo["estu_fechanacimiento"].str[:4], errors="coerce")
edad = 2020 - anio_nacimiento
print("Edades más frecuentes:"); print(edad.value_counts().head(5).sort_index().to_string())
print("\nEdades que no pueden ser de un estudiante de once:")
print(edad[(edad < 13) | (edad > 70)].value_counts().sort_index().to_string())
'''),
("md", r'''
Nadie presenta Saber 11 a los cero años. Esas fechas son errores de digitación. Si las dejo, la variable edad tendrá valores absurdos. Si borro las filas, pierdo estudiantes que sí presentaron el examen. **Decisión:** la edad por fuera del rango de 13 a 70 años se declara faltante y la fila se conserva.

**Segunda trampa: puntajes que no miden nada.**
'''),
("code", r'''
areas = ["punt_lectura_critica", "punt_matematicas", "punt_c_naturales", "punt_sociales_ciudadanas", "punt_ingles", "punt_global"]
print("Estudiantes con puntaje global igual a cero:", int((crudo["punt_global"] == 0).sum()))
display(crudo.loc[crudo["punt_global"] == 0, areas + ["percentil_global", "estu_estadoinvestigacion"]])
print(crudo["estu_estadoinvestigacion"].value_counts().to_string())
'''),
("md", r'''
Un cero en las cinco áreas no describe a un estudiante que respondió mal. Describe una prueba que no se respondió o que no se calificó. La tabla no dice cuál de las dos. **Decisión:** esas filas salen, porque no hay un resultado que predecir. Es la única regla de este cuaderno que borra filas.

**Tercera trampa: categorías que parecen respuestas.** Miren el estrato y la educación de la madre.
'''),
("code", r'''
for columna in ["fami_estratovivienda", "fami_educacionmadre"]:
    print(crudo[columna].value_counts(dropna=False).to_string(), "\n")
'''),
("md", r'''
Hay tres clases de valor que conviene distinguir. Están las respuestas con orden, como «Estrato 1» o «Primaria completa». Está la celda vacía, que es un estudiante que no respondió. Y están «Sin Estrato», «No sabe» y «No Aplica», que sí son respuestas pero no caben en el orden. Si convierto «Sin Estrato» en un cero, le digo al modelo que es un estrato más bajo que el uno, y eso no lo dice el dato.

**Decisión:** las categorías con orden se vuelven números. «No sabe» y «No Aplica» se tratan como faltantes. «Sin Estrato» queda con su propia columna, porque es información distinta.
'''),
("code", r'''
niveles_educacion = {
    "Ninguno": 0, "Primaria incompleta": 1, "Primaria completa": 2,
    "Secundaria (Bachillerato) incompleta": 3, "Secundaria (Bachillerato) completa": 4,
    "Técnica o tecnológica incompleta": 5, "Técnica o tecnológica completa": 6,
    "Educación profesional incompleta": 7, "Educación profesional completa": 8, "Postgrado": 9,
}
libros = {"0 A 10 LIBROS": 0, "11 A 25 LIBROS": 1, "26 A 100 LIBROS": 2, "MÁS DE 100 LIBROS": 3}
personas = {"1 a 2": 1.5, "3 a 4": 3.5, "5 a 6": 5.5, "7 a 8": 7.5, "9 o más": 9}
cuartos = {"Uno": 1, "Dos": 2, "Tres": 3, "Cuatro": 4, "Cinco": 5, "Seis o mas": 6}
lectura = {"No leo por entretenimiento": 0, "30 minutos o menos": 1, "Entre 30 y 60 minutos": 2, "Entre 1 y 2 horas": 3, "Más de 2 horas": 4}
trabajo = {"0": 0, "Menos de 10 horas": 1, "Entre 11 y 20 horas": 2, "Entre 21 y 30 horas": 3, "Más de 30 horas": 4}

X = pd.DataFrame(index=crudo.index)
X["mujer"] = (crudo["estu_genero"] == "F").astype(float).where(crudo["estu_genero"].notna())
X["edad"] = edad.where(edad.between(13, 70))
X["etnia"] = (crudo["estu_tieneetnia"] == "Si").astype(float).where(crudo["estu_tieneetnia"].notna())
X["estrato"] = pd.to_numeric(crudo["fami_estratovivienda"].str.extract(r"(\d)")[0], errors="coerce")
X["sin_estrato"] = (crudo["fami_estratovivienda"] == "Sin Estrato").astype(float)
X["educacion_madre"] = crudo["fami_educacionmadre"].map(niveles_educacion)
X["educacion_padre"] = crudo["fami_educacionpadre"].map(niveles_educacion)
for cosa in ["internet", "computador", "automovil"]:
    X[cosa] = (crudo[f"fami_tiene{cosa}"] == "Si").astype(float).where(crudo[f"fami_tiene{cosa}"].notna())
X["libros"] = crudo["fami_numlibros"].map(libros)
X["personas_hogar"] = crudo["fami_personashogar"].map(personas)
X["cuartos"] = crudo["fami_cuartoshogar"].map(cuartos)
X["lectura_diaria"] = crudo["estu_dedicacionlecturadiaria"].map(lectura)
X["horas_trabajo"] = crudo["estu_horassemanatrabaja"].astype(str).map(trabajo)
X["colegio_oficial"] = (crudo["cole_naturaleza"] == "OFICIAL").astype(float)
X["colegio_rural"] = (crudo["cole_area_ubicacion"] == "RURAL").astype(float)
X["nocturna_o_sabatina"] = crudo["cole_jornada"].isin(["NOCHE", "SABATINA"]).astype(float)
X["jornada_completa"] = crudo["cole_jornada"].isin(["COMPLETA", "UNICA"]).astype(float)

y = (crudo["percentil_global"] <= 25).astype(int).rename("bajo")
colegio = crudo["cole_cod_dane_establecimiento"]

validas = crudo["punt_global"] > 0                     # la única regla que borra filas
X, y, colegio, datos = X[validas], y[validas], colegio[validas], crudo[validas]
variables = list(X.columns)
print(f"Quedan {len(X)} estudiantes y {len(variables)} variables.")
print(f"Quedan en el cuarto más bajo del país: {y.mean() * 100:.1f} %")

registro = RegistroDeLimpieza(len(crudo))
registro.anotar("Puntaje global igual a cero", "la fila sale: la prueba no se calificó", int((~validas).sum()), int(validas.sum()))
registro.anotar("Fecha de nacimiento imposible o vacía", "la edad queda como faltante; la fila se queda", int(X["edad"].isna().sum()))
registro.anotar("«No sabe» y «No Aplica» en la educación de la madre", "se tratan como faltantes; la fila se queda", int(datos["fami_educacionmadre"].isin(["No sabe", "No Aplica"]).sum()))
registro.anotar("Filas con alguna celda vacía", "no se borran: se imputan con la mediana del entrenamiento", int(X.isna().any(axis=1).sum()))
display(registro.tabla())
'''),
("md", r'''
Comparen con su primera apuesta. Si el departamento fuera una copia del país, ese número sería 25 %.

**Cuarta trampa: los faltantes no caen al azar.** La salida fácil para las celdas vacías es `dropna()`, que borra toda fila con algún faltante. Antes de usarla hay que preguntarse quiénes son los que no respondieron. Llamo $m_i=1$ al estudiante con alguna variable faltante. Borrar filas es inofensivo solo si la respuesta no depende de eso:

$$P(y=1\mid m=1)=P(y=1\mid m=0).$$

Esa igualdad es una hipótesis nula, $H_0$, y se puede contrastar. Con la tabla de dos por dos que cruza «tiene faltantes» con «quedó abajo», la prueba ji-cuadrado de independencia compara lo observado, $O_{jk}$, con lo que se esperaría si $H_0$ fuera cierta, $E_{jk}$:

$$\chi^2=\sum_{j,k}\frac{(O_{jk}-E_{jk})^2}{E_{jk}}.$$

**Predicción escrita:** ¿los estudiantes con respuestas faltantes quedan abajo con la misma frecuencia que los demás? **Mi apuesta:**
'''),
("code", r'''
faltantes = (X.isna().mean() * 100).sort_values()
faltantes[faltantes > 0].plot.barh(color="#1D4E89", figsize=(8, 3.6))
plt.xlabel("% de estudiantes sin dato"); plt.title("Faltantes por variable"); plt.show()

incompleta = X.isna().any(axis=1)
comparacion = pd.DataFrame({
    "estudiantes": incompleta.value_counts(),
    "% en el cuarto más bajo": (y.groupby(incompleta).mean() * 100).round(1),
}).rename(index={False: "fila completa", True: "con algún faltante"})
display(comparacion)
print(f"dropna() borraría el {incompleta.mean() * 100:.1f} % de las filas.")

from scipy.stats import chi2_contingency
ji2, valor_p, _, _ = chi2_contingency(pd.crosstab(incompleta, y))
print(f"Prueba ji-cuadrado de independencia entre «tener faltantes» y «quedar abajo»: estadístico {ji2:.1f}, valor p {valor_p:.2g}")
'''),
("md", r'''
El valor $p$ de la última línea dice si la diferencia entre los dos porcentajes puede ser casualidad. Los dos porcentajes de la tabla no coinciden. Quien deja preguntas sin responder queda en el cuarto más bajo con más frecuencia que quien responde todo. `dropna()` sacaría de la tabla, con preferencia, a los estudiantes que la Secretaría está buscando, y lo haría sin avisar. Eso es un **sesgo de selección** creado por la limpieza.

**Decisión:** no se borra ninguna fila por faltantes. Cada faltante se reemplaza por la mediana de su variable, calculada solo con los estudiantes de entrenamiento. Además, la educación del padre trae muchos «No sabe» y «No Aplica», y eso también es información sobre el hogar.

**Quinta trampa: columnas que ya conocen la respuesta.** La tabla trae columnas que se calcularon después del examen. Con ellas cualquier modelo acierta, pero ninguna existe el día en que se reparten los cupos.
'''),
("code", r'''
filtran = ["punt_lectura_critica", "punt_matematicas", "punt_c_naturales", "punt_sociales_ciudadanas", "punt_ingles",
           "punt_global", "percentil_global", "desemp_matematicas", "estu_generacion_e"]
print(pd.crosstab(datos["estu_generacion_e"], y.map({0: "no quedó abajo", 1: "quedó abajo"})).to_string())
'''),
("md", r'''
Los puntajes por área y el nivel de desempeño son pedazos de la respuesta. `estu_generacion_e` es más sutil. Marca a quién se le reconoció después un beneficio del programa Generación E, y las dos categorías de «Excelencia» no tienen a nadie en el cuarto más bajo: el beneficio se otorgó mirando el puntaje. En la sección 3.5 medimos cuánto cambia el resultado con cada una de estas columnas.

Hay otras dos columnas que dejo por fuera por una razón distinta. `estu_inse_individual` y `estu_nse_individual` son el índice socioeconómico que el ICFES calcula con el cuestionario. No filtran la respuesta, pero el ICFES las publica con los resultados y no tengo cómo saber si la Secretaría las tendría a tiempo. Uso las respuestas originales del cuestionario, y en la sección 2.3 comparo con el índice.

### 2.2 Conocer el dato antes de modelarlo

Cuatro miradas: cómo se reparte el puntaje, cómo cambia la tasa de puntaje bajo entre grupos, quiénes presentan el examen y quiénes no están en la tabla. Para un grupo $g$, la tasa es

$$\hat p_g=\frac{1}{n_g}\sum_{i\in g}y_i,\qquad \text{con error estándar}\quad \sqrt{\frac{\hat p_g(1-\hat p_g)}{n_g}}.$$

El error estándar importa porque algunos grupos son pequeños. Una tasa calculada con cuarenta estudiantes se mueve mucho más que una calculada con cuatro mil.
'''),
("code", r'''
corte = datos.loc[y == 1, "punt_global"].max()
fig, (izq, der) = plt.subplots(1, 2, figsize=(12, 3.8))
izq.hist(datos["punt_global"], bins=60, color="#1D4E89")
izq.axvline(corte, color="#D7263D", lw=2); izq.text(corte - 8, izq.get_ylim()[1] * 0.85, f"percentil 25 nacional\n({corte} puntos)", color="#D7263D", fontsize=9, ha="right")
izq.set(title=f"Puntaje global, {DEPARTAMENTO.title()}", xlabel="puntos (0 a 500)", ylabel="estudiantes")

colegios = pd.DataFrame({"n": y.groupby(colegio).size(), "tasa": y.groupby(colegio).mean() * 100})
der.scatter(colegios["n"], colegios["tasa"], s=14, color="#5BC0EB", alpha=0.7)
der.axhline(y.mean() * 100, color="gray", ls=":")
der.set(title="Cada punto es un colegio", xlabel="estudiantes que presentaron", ylabel="% en el cuarto más bajo")
plt.tight_layout(); plt.show()
print(f"Colegios: {len(colegios)} | mediana de estudiantes por colegio: {colegios['n'].median():.0f}")
print(f"Colegios sin ningún estudiante abajo: {(colegios['tasa'] == 0).sum()} | con más de la mitad abajo: {(colegios['tasa'] > 50).sum()}")
'''),
("md", r'''
La gráfica de la derecha anticipa una decisión de la sección 3. Los estudiantes de un mismo colegio se parecen entre sí: hay colegios donde casi nadie queda abajo y colegios donde queda más de la mitad. Los colegios pequeños ocupan los dos extremos, que es lo que predice la fórmula del error estándar.
'''),
("code", r'''
def tasa_por(grupo, orden=None):
    tabla = pd.DataFrame({"n": y.groupby(grupo).size(), "tasa": y.groupby(grupo).mean() * 100})
    tabla["ee"] = np.sqrt(tabla["tasa"] * (100 - tabla["tasa"]) / tabla["n"])
    return tabla.reindex(orden) if orden else tabla

paneles = [
    ("Educación de la madre", tasa_por(datos["fami_educacionmadre"].fillna("(sin respuesta)"), list(niveles_educacion) + ["No sabe", "(sin respuesta)"])),
    ("Estrato de la vivienda", tasa_por(datos["fami_estratovivienda"].fillna("(sin respuesta)"), [f"Estrato {e}" for e in range(1, 7)] + ["Sin Estrato", "(sin respuesta)"])),
    ("Jornada del colegio", tasa_por(datos["cole_jornada"]).sort_values("tasa")),
    ("Naturaleza y zona", tasa_por(datos["cole_naturaleza"].str.capitalize() + ", " + datos["cole_area_ubicacion"].str.lower()).sort_values("tasa")),
]
fig, ejes = plt.subplots(2, 2, figsize=(13, 8))
for eje, (titulo, tabla) in zip(ejes.ravel(), paneles):
    tabla = tabla.dropna()
    eje.barh([f"{nombre[:34]} (n={int(n)})" for nombre, n in zip(tabla.index, tabla["n"])], tabla["tasa"], xerr=2 * tabla["ee"], color="#1B998B", ecolor="#1D4E89")
    eje.axvline(y.mean() * 100, color="gray", ls=":"); eje.invert_yaxis(); eje.tick_params(axis="y", labelsize=8)
    eje.set(title=titulo, xlabel="% en el cuarto más bajo")
plt.tight_layout(); plt.show()
'''),
("md", r'''
Tres lecturas. La primera es que la tasa baja de forma ordenada con la educación de la madre. La segunda es que el estrato se comporta peor de lo que uno espera en la parte alta: los estratos 5 y 6 tienen pocos estudiantes y barras de error anchas, así que su tasa dice poco. La tercera es que las jornadas nocturna y sabatina están muy por encima del resto. En esas jornadas estudian, sobre todo, adultos que retomaron el bachillerato.

En las dos gráficas de arriba, la barra de «sin respuesta» queda a la derecha de la línea punteada, que es la tasa del departamento. Es la cuarta trampa otra vez.

**¿Quiénes están en la tabla y quiénes no?** Este es el sesgo que ninguna limpieza arregla.
'''),
("code", r'''
print(f"Mujeres entre quienes presentaron: {X['mujer'].mean() * 100:.1f} %")
print(f"Mayores de 20 años: {(X['edad'] > 20).mean() * 100:.1f} % | en jornada nocturna o sabatina: {X['nocturna_o_sabatina'].mean() * 100:.1f} %")
print(f"Trabajan más de 20 horas a la semana: {(X['horas_trabajo'] >= 3).mean() * 100:.1f} %")
resumen_sexo = pd.DataFrame({"n": y.groupby(datos["estu_genero"]).size(),
                             "puntaje medio": datos.groupby("estu_genero")["punt_global"].mean().round(1),
                             "% en el cuarto más bajo": (y.groupby(datos["estu_genero"]).mean() * 100).round(1)})
display(resumen_sexo)
'''),
("md", r'''
La tabla solo contiene a quienes llegaron a presentar el examen. Los jóvenes que dejaron el colegio antes de once no aparecen, y son los que más habrían necesitado el refuerzo. Por eso las mujeres son más de la mitad de la tabla: ese número habla de quién llega a once, y este dato no alcanza para decir por qué.

De ahí sale una advertencia para leer la tabla de arriba. La diferencia entre hombres y mujeres se calcula sobre dos grupos que pasaron filtros distintos antes de llegar al examen. Sirve para describir a quienes presentaron. No sirve para comparar a todos los jóvenes de la misma edad.

### 2.3 PCA: ¿cuántas dimensiones tiene «el nivel socioeconómico»?

Nueve variables describen el hogar: estrato, educación de la madre y del padre, internet, computador, automóvil, libros, cuartos y personas. Si todas midieran lo mismo, bastaría un solo número. Tomo la matriz $Z$ de esas variables estandarizadas y descompongo su matriz de correlaciones:

$$R=\frac{1}{n-1}Z^\top Z=V\Lambda V^\top,\qquad \text{varianza explicada por el componente } j=\frac{\lambda_j}{\sum_k\lambda_k}.$$

El primer componente, $z_i^\top v_1$, es la combinación de las nueve variables que más varía entre estudiantes. El ICFES publica su propio índice socioeconómico, el INSE. Si el primer componente y el INSE cuentan la misma historia, su correlación será alta.

**Predicción escrita:** ¿cuánta varianza explicará el primer componente? Con nueve variables independientes sería 11 %. **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

hogar = ["estrato", "educacion_madre", "educacion_padre", "internet", "computador", "automovil", "libros", "cuartos", "personas_hogar"]
Z = StandardScaler().fit_transform(X[hogar].fillna(X[hogar].median()))     # mediana de todos: aquí solo se describe
pca = PCA().fit(Z)
explicada = pca.explained_variance_ratio_ * 100
signo = -1 if pca.components_[0].sum() < 0 else 1                           # el signo de un componente es arbitrario
cargas = pd.DataFrame(pca.components_.T[:, :3], index=hogar, columns=["CP1", "CP2", "CP3"])
puntajes = pd.DataFrame(pca.transform(Z)[:, :3], index=X.index, columns=["CP1", "CP2", "CP3"])
cargas["CP1"], puntajes["CP1"] = signo * cargas["CP1"], signo * puntajes["CP1"]
cp1 = puntajes["CP1"]

resumen_pca(explicada, cargas).show()
'''),
("code", r'''
muestra = puntajes.sample(min(4000, len(puntajes)), random_state=0)
nube_3d(muestra,
        grupo=y[muestra.index].map({0: "no quedó abajo", 1: "quedó en el cuarto más bajo"}),
        colores={"no quedó abajo": CIELO, "quedó en el cuarto más bajo": ROJO},
        texto="puntaje global: " + datos.loc[muestra.index, "punt_global"].astype(str),
        flechas=cargas, largo=5, tamano=2.5,
        titulo="Los estudiantes en las tres primeras dimensiones del hogar",
        subtitulo="Cada punto es un estudiante. Las flechas verdes son las cargas. Arrastren para girar; un clic en la leyenda apaga un grupo.",
        ejes=[f"CP{j + 1} ({explicada[j]:.0f} %)" for j in range(3)]).show()

quintil = pd.qcut(cp1, 5, labels=["1 (más bajo)", "2", "3", "4", "5 (más alto)"])
display((y.groupby(quintil, observed=True).mean() * 100).round(1).rename("% en el cuarto más bajo").rename_axis("quintil del primer componente").to_frame())
inse = datos["estu_inse_individual"]
print(f"El primer componente explica {explicada[0]:.0f} % de la varianza; el segundo, {explicada[1]:.0f} %.")
print(f"Correlación entre el primer componente y el INSE del ICFES: {cp1.corr(inse):.2f}")
'''),
("md", r'''
El primer componente carga con el mismo signo sobre casi todas las variables y explica varias veces lo que explicaría una variable suelta. Se puede leer como un índice de recursos del hogar, y su correlación con el INSE dice qué tan cerca está del índice oficial. La gráfica de la derecha muestra que ese solo número ya ordena a los estudiantes por riesgo.

El resto de la varianza no desaparece. Tener automóvil y tener libros no son la misma cosa, y por eso el modelo recibirá las variables originales y no el componente. El PCA aquí sirve para describir.
'''),
("md", r'''
## 3. Modelado

### 3.1 La métrica y la trampa de la exactitud

Un modelo de clasificación entrega para cada estudiante una probabilidad estimada $\hat p_i$ de quedar abajo. Para medirlo hay que cruzar lo que predijo con lo que pasó. Con un umbral de 0,5 salen cuatro casillas: verdaderos positivos (VP), falsos positivos (FP), verdaderos negativos (VN) y falsos negativos (FN). Las tres medidas más usadas son

$$\text{exactitud}=\frac{VP+VN}{n},\qquad \text{sensibilidad}=\frac{VP}{VP+FN},\qquad \text{precisión}=\frac{VP}{VP+FP}.$$

La exactitud es la que reporta todo el mundo. Veamos qué dice aquí, con el modelo más sencillo y con un «modelo» que no hace nada.

Primero la partición. Todos los modelos de este cuaderno se entrenan con unos colegios, se escogen con otros y se reportan con otros:

$$\underbrace{60\,\%\ \text{de los colegios}}_{\text{entrenamiento}}\ \Big|\ \underbrace{20\,\%}_{\text{validación: se escoge}}\ \Big|\ \underbrace{20\,\%}_{\text{prueba: se reporta}}$$

Se parte por colegio y no por estudiante por lo que vimos en la sección 2.2. Si los compañeros de un mismo salón quedan repartidos entre entrenamiento y prueba, el examen del modelo se parece al entrenamiento más de lo que se parecerá un colegio nuevo.
'''),
("code", r'''
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

resto, prueba = next(GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=0).split(X, y, colegio))
entrena, valida = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X.iloc[resto], y.iloc[resto], colegio.iloc[resto]))
E, V, P = (np.zeros(len(X), dtype=bool) for _ in range(3))
E[resto[entrena]] = True; V[resto[valida]] = True; P[prueba] = True

for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {tramo.sum():5d} estudiantes | {colegio[tramo].nunique():3d} colegios | {y[tramo].mean() * 100:.1f} % abajo")
'''),
("code", r'''
logistica = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[E], y[E])
prob = logistica.predict_proba(X[V])[:, 1]
dice_bajo = prob >= 0.5
real = y[V].values

def tres_medidas(real, dice):
    vp, fp, fn = ((dice == 1) & (real == 1)).sum(), ((dice == 1) & (real == 0)).sum(), ((dice == 0) & (real == 1)).sum()
    return {"exactitud": (dice == real).mean(), "sensibilidad": vp / (vp + fn), "precisión": vp / (vp + fp) if vp + fp else np.nan}

display(pd.DataFrame({"decir que nadie queda abajo": tres_medidas(real, np.zeros(len(real))),
                      "regresión logística, umbral 0,5": tres_medidas(real, dice_bajo)}).T.round(3))
print(f"El modelo señala a {dice_bajo.sum()} de {len(real)} estudiantes. Quedaron abajo {real.sum()}.")
'''),
("md", r'''
Decir que nadie queda abajo acierta con casi todos los estudiantes, porque la mayoría no queda abajo. El modelo apenas mejora esa exactitud, y su sensibilidad muestra que deja pasar a la mayoría de quienes la Secretaría busca. La exactitud premia acertar en el grupo grande y no se entera de lo que pasa en el pequeño.

El umbral de 0,5 tampoco tiene nada que ver con el problema. La Secretaría tiene un número de cupos, así que lo que necesita del modelo es un **orden**. Por eso medimos con la captación de la sección 1 y, como segunda medida, con el área bajo la curva ROC, que resume la calidad del orden sin fijar un cupo:

$$\text{AUC}=P\bigl(\hat p_i>\hat p_j\ \big|\ y_i=1,\ y_j=0\bigr).$$

Un AUC de 0,5 es ordenar al azar y un AUC de 1 es ordenar perfecto.

### 3.2 La línea base

Dos reglas que no necesitan modelo. La primera reparte los cupos por sorteo, y su captación esperada es el mismo cupo. La segunda ordena a los estudiantes por la educación de la madre, de menor a mayor, que es lo que haría alguien con sentido común y una sola columna.
'''),
("code", r'''
def captacion(real, puntaje, cupo=CUPO):
    """Fracción de los que quedaron abajo que cae entre los primeros `cupo` del orden. Los empates se rompen por sorteo."""
    real, puntaje = np.asarray(real), np.asarray(puntaje, dtype=float)
    k = int(round(cupo * len(real)))
    orden = np.lexsort((np.random.default_rng(0).random(len(real)), -puntaje))
    return float(real[orden[:k]].sum() / real.sum())

mediana_madre = X.loc[E, "educacion_madre"].median()
reglas = {
    "sorteo": pd.Series(np.random.default_rng(1).random(len(X)), index=X.index),
    "regla: educación de la madre": -X["educacion_madre"].fillna(mediana_madre),
}
errores = {nombre: {"captación en validación": captacion(y[V], regla[V]), "captación en prueba": captacion(y[P], regla[P]),
                    "AUC en prueba": roc_auc_score(y[P], regla[P])} for nombre, regla in reglas.items()}
display(pd.DataFrame(errores).T.round(3))
print(f"Techo de la captación en prueba: {min(1, CUPO / y[P].mean()):.2f}")
'''),
("md", r'''
La última línea trae un detalle que se olvida con facilidad. La captación tiene un techo, que depende de cuántos estudiantes quedan abajo:

$$\text{captación máxima}=\min\!\left(1,\ \frac{\text{cupo}}{\hat p}\right).$$

Si quedan abajo más estudiantes que cupos, ni un oráculo los cubre a todos. Guarden esa idea para la sección 4.5.

### 3.3 Cinco modelos y un protocolo para escoger

| Modelo | Idea en una línea |
|---|---|
| Regresión logística | $\log\dfrac{p_i}{1-p_i}=\beta_0+x_i^\top\beta$: cada variable suma o resta al logaritmo de la razón de probabilidades |
| $k$ vecinos | La fracción que quedó abajo entre los $k$ estudiantes de entrenamiento más parecidos |
| Árbol de decisión | Preguntas de sí o no, una tras otra, que van separando grupos con tasas distintas |
| Bosque aleatorio | El promedio de muchos árboles entrenados sobre remuestreos del dato |
| *Gradient boosting* | Una suma de árboles pequeños, cada uno corrigiendo el error de los anteriores |

La regla queda fijada antes de mirar: gana quien tenga mayor captación en **validación**, y su captación en **prueba** es la que se le reporta a la Secretaría. Los faltantes se llenan con la mediana de entrenamiento dentro de cada modelo, para que la validación y la prueba no participen en ese cálculo.
'''),
("code", r'''
def cinco_modelos():
    llenar = lambda: SimpleImputer(strategy="median")
    return {
        "logística": make_pipeline(llenar(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "k vecinos": make_pipeline(llenar(), StandardScaler(), KNeighborsClassifier(n_neighbors=50)),
        "árbol": make_pipeline(llenar(), DecisionTreeClassifier(max_depth=5, random_state=0)),
        "bosque": make_pipeline(llenar(), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)),
        "boosting": make_pipeline(llenar(), HistGradientBoostingClassifier(random_state=0)),
    }

puntajes = {nombre: regla[P] for nombre, regla in reglas.items()}        # el orden que cada candidato propone en la prueba
for nombre, modelo in cinco_modelos().items():
    modelo.fit(X[E], y[E])                                                # se entrena con los colegios de entrenamiento
    en_validacion = captacion(y[V], modelo.predict_proba(X[V])[:, 1])
    modelo.fit(X[E | V], y[E | V])                                        # se reentrena con todo lo que no es prueba
    puntajes[nombre] = pd.Series(modelo.predict_proba(X[P])[:, 1], index=X.index[P])
    errores[nombre] = {"captación en validación": en_validacion, "captación en prueba": captacion(y[P], puntajes[nombre]),
                       "AUC en prueba": roc_auc_score(y[P], puntajes[nombre])}
todos = pd.DataFrame(errores).T

todos[["captación en validación", "captación en prueba"]].sort_values("captación en validación").plot.barh(figsize=(9, 4), color=["#1D4E89", "#1B998B"])
plt.axvline(CUPO, color="gray", ls=":"); plt.xlabel("captación"); plt.title("Siete candidatos: validación para escoger, prueba para reportar"); plt.show()
display(todos.sort_values("captación en validación", ascending=False).round(3))
'''),
("code", r'''
fracciones = np.linspace(0.02, 1, 50)
plt.figure(figsize=(8, 4.2))
for nombre, puntaje in puntajes.items():
    estilo = {"color": "gray", "ls": ":"} if nombre == "sorteo" else {"color": "#1B998B", "lw": 2} if nombre.startswith("regla") else {"lw": 1.2}
    plt.plot(fracciones * 100, [captacion(y[P], puntaje, f) for f in fracciones], label=nombre, **estilo)
plt.axvline(CUPO * 100, color="#D7263D", lw=1); plt.text(CUPO * 100 + 1, 0.05, "cupo", color="#D7263D")
plt.xlabel("% de estudiantes con cupo"); plt.ylabel("captación"); plt.title("Captación según el tamaño del programa, en la prueba"); plt.legend(fontsize=8); plt.show()
'''),
("md", r'''
La segunda gráfica muestra la captación para cualquier tamaño de programa, y la línea vertical marca el cupo de la Secretaría. Mientras más arriba queda una curva sobre la diagonal del sorteo, mejor ordena ese candidato. Fíjense en qué tan juntas van las curvas de los cinco modelos y en qué tan lejos queda la regla de una sola columna.

### 3.4 El veredicto
'''),
("code", r'''
mejor = todos.drop(index="sorteo")["captación en validación"].idxmax()
resumen = todos["captación en prueba"].round(3)
regla = "regla: educación de la madre"
print(f"Ganador en validación: {mejor}")
print(f"Captación en prueba: {resumen[mejor]:.3f} | regla de una columna: {resumen[regla]:.3f} | sorteo: {resumen['sorteo']:.3f} | techo: {min(1, CUPO / y[P].mean()):.2f}")
'''),
("md", r'''
Lean los tres números juntos. El sorteo dice cuánto se logra sin saber nada. La regla dice cuánto se logra con una columna y sin modelo. La distancia entre la regla y el ganador es lo que de verdad aporta el aprendizaje automático en este problema, y la distancia entre el ganador y el techo es lo que el cuestionario no alcanza a ver.

### 3.5 La trampa de la clase: columnas que filtran la respuesta

En la sección 2.1 separé nueve columnas que se calcularon después del examen. Ahora medimos qué pasa si alguien las deja en la tabla por descuido. El modelo es el mismo y la partición es la misma. Solo cambian las columnas.

**Predicción escrita:** ¿cuánto subirá la captación si el modelo recibe el nivel de desempeño en matemáticas? **Mi apuesta:**
'''),
("code", r'''
extras = pd.DataFrame({
    "generacion_e": (datos["estu_generacion_e"] != "NO").astype(float),
    "generacion_e_excelencia": datos["estu_generacion_e"].str.contains("EXCELENCIA").astype(float),
    "desemp_matematicas": datos["desemp_matematicas"].astype(float),
    "punt_ingles": datos["punt_ingles"].astype(float),
    "punt_lectura_critica": datos["punt_lectura_critica"].astype(float),
})
escenarios = {
    "solo lo que se sabe en la inscripción": [],
    "+ Generación E": ["generacion_e", "generacion_e_excelencia"],
    "+ desempeño en matemáticas": ["desemp_matematicas"],
    "+ dos puntajes por área": ["punt_ingles", "punt_lectura_critica"],
}
fuga = {}
for nombre, columnas in escenarios.items():
    X_con = pd.concat([X, extras[columnas]], axis=1)
    modelo = cinco_modelos()["boosting"].fit(X_con[E | V], y[E | V])
    prob = modelo.predict_proba(X_con[P])[:, 1]
    fuga[nombre] = {"captación en prueba": captacion(y[P], prob), "AUC en prueba": roc_auc_score(y[P], prob)}
display(pd.DataFrame(fuga).T.round(3))
'''),
("md", r'''
La primera fila es el resultado honesto. Las demás son resultados que no se pueden cumplir, porque usan información que no existe cuando se reparten los cupos. A esto se le llama **fuga de información**. Lo peligroso es que nada falla. El código corre, la métrica sube y el informe se ve mejor.

La fila de Generación E enseña otra cosa. Esa columna casi no mueve la captación, porque delata a los estudiantes de arriba y la Secretaría busca a los de abajo. Una columna puede filtrar la respuesta y no notarse en la métrica que uno escogió. Por eso las columnas con fuga se sacan por lo que son, y no por lo que le hacen al número.

La fuga no siempre es tan visible como un puntaje. La pregunta que la detecta es siempre la misma: el día en que se toma la decisión, ¿esta columna ya existe?

### 3.6 Explicar: ¿qué acompaña al puntaje bajo?

La regresión logística permite leer cada variable. Como las variables entraron estandarizadas, $e^{\beta_j}$ es la **razón de probabilidades** asociada a subir una desviación estándar en la variable $j$, dejando quietas las demás:

$$\text{OR}_j=e^{\beta_j}=\frac{p/(1-p)\ \text{con}\ x_j+1\ \text{d.e.}}{p/(1-p)\ \text{con}\ x_j}.$$

Un valor de 1,3 dice que la razón entre quedar abajo y no quedar abajo sube 30 %. Un valor de 0,8 dice que baja 20 %.
'''),
("code", r'''
logistica = cinco_modelos()["logística"].fit(X[E | V], y[E | V])
razones = pd.Series(np.exp(logistica[-1].coef_[0]), index=variables).sort_values()
razones.plot.barh(figsize=(8, 5.5), color=["#5BC0EB" if r < 1 else "#D7263D" for r in razones])
plt.axvline(1, color="gray"); plt.xlabel("razón de probabilidades por desviación estándar"); plt.title("Qué acompaña a quedar en el cuarto más bajo"); plt.show()
print(razones.round(2).to_string())
'''),
("md", r'''
Comparen con su cuarta apuesta. Antes de sacar conclusiones, tres advertencias.

La primera es que estos números describen asociaciones entre estudiantes que presentaron el examen. Que los libros en la casa acompañen un mejor resultado no significa que regalar libros lo produzca.

La segunda es que las variables se pisan. Estrato, computador e internet miden cosas parecidas, como mostró el PCA, y la regresión reparte entre ellas un efecto que comparten. Un coeficiente pequeño en el estrato no dice que el estrato no importe.

La tercera es que hay variables, como la edad y la jornada, que cuentan una historia de trayectoria escolar. Un estudiante de 25 años en jornada sabatina no llegó ahí por su edad.

### 3.7 Sesgos: ¿a quién le acierta y a quién le falla el modelo?

Un modelo puede tener buena captación en promedio y repartir sus errores de forma desigual. Para cada grupo miro cuatro números en la prueba: qué fracción quedó abajo, qué fracción recibe cupo, la captación dentro del grupo y la fracción de **cupos de sobra**, que son los estudiantes que no quedaron abajo y recibieron cupo:

$$\text{cupos de sobra}_g=\frac{\#\{i\in g:\ y_i=0\ \text{y recibe cupo}\}}{\#\{i\in g:\ y_i=0\}}.$$
'''),
("code", r'''
elegido = mejor if resumen[mejor] > resumen[regla] else regla          # el más simple entre los que no pierden
puntaje = puntajes[elegido]
k = int(round(CUPO * P.sum()))
orden = np.lexsort((np.random.default_rng(0).random(P.sum()), -puntaje.values))
cupo = pd.Series(False, index=X.index[P]); cupo.iloc[orden[:k]] = True

def auditar(grupo):
    partes = {}
    for valor, indices in grupo[P].groupby(grupo[P]).groups.items():
        yy, cc = y[indices], cupo[indices]
        partes[valor] = {"n": len(yy), "% que quedó abajo": yy.mean() * 100, "% con cupo": cc.mean() * 100,
                         "captación": cc[yy == 1].mean(), "cupos de sobra": cc[yy == 0].mean()}
    return pd.DataFrame(partes).T

grupos = {
    "sexo": datos["estu_genero"].map({"F": "mujeres", "M": "hombres"}),
    "zona": datos["cole_area_ubicacion"].str.lower(),
    "naturaleza": datos["cole_naturaleza"].str.lower(),
    "jornada": X["nocturna_o_sabatina"].map({1.0: "nocturna o sabatina", 0.0: "diurna"}),
}
auditoria = pd.concat({nombre: auditar(grupo) for nombre, grupo in grupos.items()})
print("Candidato auditado:", elegido)
display(auditoria.round({"n": 0, "% que quedó abajo": 1, "% con cupo": 1, "captación": 2, "cupos de sobra": 2}))
'''),
("md", r'''
La tabla se lee por parejas. Cuando dos grupos tienen tasas reales parecidas y el modelo le da muchos más cupos a uno, hay algo que revisar. Cuando un grupo tiene una tasa real mucho más alta, es razonable que reciba más cupos, y aun así conviene mirar su columna de cupos de sobra: ahí se ve si el modelo aprendió a señalar al grupo entero en vez de distinguir a las personas dentro de él.

Miren primero el sexo y después la jornada. En el grupo donde la captación es más baja están los estudiantes que quedaron abajo sin que el modelo los viera. Esa es la gente que la decisión deja por fuera, y la sección 4 tiene que decirlo.

Esta auditoría tiene un límite. Solo puedo auditar por las variables que el dato trae.
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿a quiénes se les ofrece el cupo?

La captación de la prueba salió de unos pocos colegios. Con otros colegios habría salido otro número. Para ponerle un margen uso *bootstrap* por colegio: saco muchas veces, con reposición, una muestra de los colegios de prueba y recalculo la captación. El intervalo va del percentil 5 al 95 de esos resultados,

$$\bigl[\hat Q_{0{,}05}(\text{captación}^*),\ \hat Q_{0{,}95}(\text{captación}^*)\bigr].$$

Se remuestrean colegios y no estudiantes por la misma razón por la que se partió por colegio.
'''),
("code", r'''
colegios_prueba = colegio[P].unique()
filas_por_colegio = {c: np.flatnonzero(colegio[P].values == c) for c in colegios_prueba}
sorteo = np.random.default_rng(0)
replicas = {"elegido": [], "regla": []}
for _ in range(500):
    filas = np.concatenate([filas_por_colegio[c] for c in sorteo.choice(colegios_prueba, len(colegios_prueba))])
    replicas["elegido"].append(captacion(y[P].values[filas], puntajes[elegido].values[filas]))
    replicas["regla"].append(captacion(y[P].values[filas], puntajes[regla].values[filas]))
bajo_, alto_ = np.percentile(replicas["elegido"], [5, 95])
ventaja = np.array(replicas["elegido"]) - np.array(replicas["regla"])

precision = y[P][cupo].mean()
print(f"Candidato elegido: {elegido}")
print(f"Captación en prueba: {resumen[elegido]:.2f}, con intervalo de 90 % entre {bajo_:.2f} y {alto_:.2f}")
print(f"Ventaja sobre la regla de una columna: {np.median(ventaja):+.2f} (intervalo de 90 %: {np.percentile(ventaja, 5):+.2f} a {np.percentile(ventaja, 95):+.2f})")
print(f"De cada 100 cupos, {precision * 100:.0f} llegan a estudiantes que quedaron en el cuarto más bajo. Por sorteo llegarían {y[P].mean() * 100:.0f}.")
print(f"De cada 100 estudiantes que quedaron abajo, {(1 - resumen[elegido]) * 100:.0f} se quedan sin cupo.")
'''),
("md", r'''
Los últimos dos renglones son los que entiende quien no es técnico. El primero dice qué tan bien gastados quedan los cupos. El segundo dice cuánta gente se queda por fuera, y esa cifra hay que entregarla siempre junto con la primera.

### La decisión en tres frases

Escríbanla para una secretaria de educación que no sabe qué es una regresión logística. Una frase para lo que recomiendan, una para la evidencia y una para lo que puede salir mal.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| «Bajo» es percentil nacional de 25 o menos | Un puntaje fijo; el cuartil del departamento | El percentil lo publica el ICFES y permite comparar departamentos |
| Edad por fuera de 13 a 70 años se declara faltante | Borrar la fila; dejar el valor | La fecha es un error de digitación, el estudiante sí presentó |
| Puntaje global igual a cero sale de la tabla | Tratarlo como el puntaje más bajo | No hay una prueba respondida que predecir |
| «No sabe» y «No Aplica» se tratan como faltantes; «Sin Estrato» tiene su columna | Ponerles un número dentro del orden | No son un nivel más bajo ni más alto |
| Los faltantes se llenan con la mediana de entrenamiento | `dropna()` | Quien no responde queda abajo con más frecuencia: borrar sesga |
| Fuera las nueve columnas calculadas después del examen, y el INSE | Dejarlas | No existen el día en que se reparten los cupos |
| Partición y *bootstrap* por colegio | Por estudiante | Los compañeros de colegio se parecen |
| Captación con cupo de 25 % como métrica | Exactitud | La Secretaría reparte un número fijo de cupos |

### 4.3 Riesgos, ética y mantenimiento

Este dato es de personas, casi todas menores de edad cuando presentaron el examen. Respondan en dos o tres líneas cada una.

- El modelo se entrenó con la cohorte de 2020 y se usaría con una cohorte posterior. ¿Qué habría que comprobar antes de confiar en él?
- La lista de cupos sale de variables como la educación de los padres y el estrato. ¿Qué pasa si un estudiante o un colegio se entera de qué respuestas dan cupo?
- Ofrecer refuerzo no le hace daño a nadie. ¿Cambiaría su opinión sobre este mismo modelo si se usara para negar una beca?
- Según la auditoría de la sección 3.7, ¿qué grupo queda peor atendido? ¿Qué le propondrían a la Secretaría para ese grupo?
- Los jóvenes que dejaron el colegio antes de once no están en la tabla. ¿Qué decisión de política quedaría mal informada si alguien olvida eso?

### 4.4 Procedencia
'''),
("code", r'''
procedencia()["saber11"]
'''),
("md", r'''
### 4.5 Cambien una línea

Vuelvan a la celda donde dice `DEPARTAMENTO = "CALDAS"`, pongan `DEPARTAMENTO = "CHOCO"` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. El ICFES escribe los departamentos en mayúsculas y casi siempre sin tilde. Pueden escoger otro departamento si lo prefieren. Llenen la tabla con sus dos corridas.

| | Caldas | Su departamento |
|---|---|---|
| Estudiantes y colegios | | |
| % en el cuarto más bajo del país | | |
| Techo de la captación | | |
| Ganador en validación | | |
| Captación del ganador en prueba | | |
| Captación de la regla de una columna | | |
| De cada 100 cupos, cuántos llegan a quien quedó abajo | | |

Con la tabla llena, respondan en tres líneas. El corte del «cuarto más bajo» es nacional y el cupo es de 25 % en los dos departamentos. ¿Qué le pasa a la métrica cuando la tasa del departamento se aleja de 25 %? ¿Qué cambiarían en la formulación, el corte o el cupo, y por qué? **Mi respuesta:**
'''),
]

# ---------------------------------------------------------------------- DENGUE
DENGUE = [
("md", r'''
# Clase 2 · Dengue (gemelo)

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelve cada uno, de forma individual, en la segunda mitad de la clase. Es el gemelo de *El ICFES*: la misma técnica en otro mundo.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Repetir por cuenta propia el camino de la clase con un dato de vigilancia en salud: limpiar dejando cada decisión a la vista, encontrar los faltantes disfrazados, escoger la métrica según la decisión y detectar las columnas que filtran la respuesta. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular el problema, los objetivos y la métrica), 2 (seleccionar y preparar datos), 4 (implementar y evaluar un modelo), 5 (comunicar a quien no es técnico) y 6 (reflexión ética, técnica y operativa). |
| **Uso de IA** | Pueden pedirle a Claude que explique una celda, que les ayude a llenar un hueco o que interprete un error. Deben poder explicar sin ayuda qué significa un `-89`, por qué se descartaron unos años, cómo se dedujo el significado de los códigos, por qué `conducta` y `clasfinal` no pueden entrar al modelo y qué mide la captación. |
| **Entregable** | El cuaderno completo y corrido en su portafolio de GitHub, con las predicciones escritas, la decisión en tres frases, la bitácora y la tabla de la sección 4.5 llenas. |

**Cómo se trabaja.** El código está escrito casi todo. Donde vean `____` falta una pieza, y el comentario de al lado dice qué va. Lean la celda entera antes de llenar. Si una celda falla, lean el error antes de pedir ayuda.
'''),
("code", ARRANQUE),
("md", r'''
## 1. Formulación

Cada vez que un médico en Colombia atiende un caso de dengue llena una ficha de notificación para el sistema de vigilancia, SIVIGILA. La ficha registra quién es el paciente, qué síntomas tiene, si quedó hospitalizado y cómo se clasificó el caso. Las alcaldías de Bucaramanga y de Tuluá publican esas fichas, sin nombres, en datos.gov.co.

Imagínense el equipo de vigilancia de la Secretaría de Salud de Bucaramanga en plena epidemia. Llegan cientos de fichas por semana y el equipo alcanza a hacerle seguimiento telefónico a tres de cada diez casos. Necesita tres respuestas:

1. **Predecir.** Con lo que dice la ficha sobre el paciente y sus síntomas, ¿qué casos terminan hospitalizados?
2. **Explicar.** ¿Qué síntomas y qué condiciones del paciente acompañan la hospitalización?
3. **Decidir.** Si el equipo solo puede seguir al 30 % de los casos, ¿a cuáles llama, y a quiénes deja sin llamada esa regla?

**La respuesta que vamos a modelar.** $y_i=1$ si la ficha dice que el paciente $i$ fue hospitalizado, y $y_i=0$ si no.

**La métrica** es la misma de la clase. El modelo ordena los casos por riesgo, el equipo llama a los primeros $k$ y medimos la **captación**:

$$\text{captación}=\frac{\#\{i:\ y_i=1\ \text{y}\ i\ \text{recibe llamada}\}}{\#\{i:\ y_i=1\}}.$$

### Predicción escrita

Escriban su apuesta antes de correr nada.

- De cada cien casos de dengue notificados, ¿cuántos terminan hospitalizados? **Mi apuesta:**
- ¿Qué síntoma separa mejor a los hospitalizados de los demás? **Mi apuesta:**
- El protocolo médico ya dice que un paciente con signos de alarma se hospitaliza. ¿Un modelo le ganará a esa regla? **Mi apuesta:**
'''),
("md", r'''
## 2. Datos

La primera celda fija los parámetros del cuaderno. La segunda trae las fichas y tarda unos veinte segundos.
'''),
("code", r'''
CIUDAD = "bucaramanga"       # al final lo cambiarán por "tulua"
CUPO = 0.30                  # fracción de casos a la que el equipo alcanza a llamar
DESDE = 2021                 # primer año que entra al análisis. La sección 2.4 explica por qué
'''),
("code", r'''
crudo = descargar("dengue", ciudad=CIUDAD)
print("Filas:", len(crudo), "| Columnas:", crudo.shape[1])
from taller.presentacion import ficha_del_dato, RegistroDeLimpieza, resumen_pca, nube_3d, CIELO, ROJO
display(ficha_del_dato(crudo, f"Fichas de dengue, {CIUDAD.title()}", maximo=80))
crudo.head(3).T
'''),
("md", r'''
### 2.1 Primera trampa: números y letras en la misma columna

Si están con Bucaramanga, la descarga soltó un aviso en rojo, `DtypeWarning`, que dice que dos columnas mezclan tipos. Ese aviso es una pista. Miren qué hay en tres columnas.
'''),
("code", r'''
for columna in ["clasfinal", "conducta", "tip_ss_"]:
    print(crudo[columna].astype(str).value_counts().to_string(), "\n")
'''),
("md", r'''
En Bucaramanga conviven dos maneras de escribir lo mismo. Unas fichas dicen `1`, `2`, `3` y otras dicen el nombre completo. Además el nombre viene dañado: donde debería haber un espacio dice `-89`, y por eso se lee `DENGUE-89CON-89SIGNOS-89DE-89ALARMA`. Alguien reemplazó todos los espacios vacíos de la tabla por `-89` y se llevó por delante los espacios entre palabras. En Tuluá todo viene en códigos, pero el mismo código aparece como `2` y como `2.0`.

Las tres funciones de la celda siguiente son las herramientas de limpieza de todo el cuaderno. `texto` deja una columna en mayúsculas y sin el `-89`. `codigo` la convierte en número, y lo que no sea número queda como faltante. `fecha` se explica en la sección 2.3.
'''),
("code", r'''
def texto(columna):
    return crudo[columna].astype(str).str.replace("-89", " ").str.strip().str.upper()

def codigo(columna):
    return pd.to_numeric(crudo[columna], errors="coerce")

print(texto("clasfinal").value_counts().head(8).to_string())
'''),
("md", r'''
### 2.2 Segunda trampa: faltantes disfrazados

Un faltante honesto es una celda vacía. Un faltante disfrazado es un valor que parece un dato. Esta tabla tiene de los dos, y el disfraz más usado es el `-89`.
'''),
("code", r'''
disfrazados = crudo.apply(lambda columna: (pd.to_numeric(columna, errors="coerce") == -89).sum())
vacios = crudo.isna().sum()
resumen_faltantes = pd.DataFrame({"-89": disfrazados, "vacíos": vacios})
print(resumen_faltantes[(resumen_faltantes.sum(axis=1) > 0)].sort_values("-89", ascending=False).head(12).to_string())
print("\nEstrato, tal como viene:")
print(crudo["estrato_"].astype(str).value_counts().to_string())
'''),
("md", r'''
Si alguien promedia el estrato sin mirar, cada `-89` entra a la cuenta como un estrato negativo. El promedio sale absurdo y nadie avisa. **Decisión:** todo `-89` se convierte en faltante.
'''),
("code", r'''
estrato = codigo("estrato_").replace(«-89», np.nan)          # HUECO 1: el número que disfraza los faltantes
print("Estrato promedio sin limpiar:", round(codigo("estrato_").mean(), 2), "| limpio:", round(estrato.mean(), 2))
print("Fichas sin estrato:", int(estrato.isna().sum()))
'''),
("md", r'''
### 2.3 Tercera trampa: fechas en dos formatos y fechas de 1900

La ficha tiene tres fechas que nos interesan: inicio de síntomas (`ini_sin_`), consulta (`fec_con_`) y notificación (`fec_not`).
'''),
("code", r'''
for columna in ["fec_not", "ini_sin_", "fec_con_"]:
    valores = crudo[columna].astype(str)
    print(f"{columna:9s} con barras: {valores.str.contains('/').sum():4d} | de 1900: {valores.str.startswith('1900').sum():4d} | ejemplos: {valores.iloc[0]}  {valores[valores.str.contains('/')].head(1).tolist()}")
'''),
("md", r'''
Casi todas las fechas vienen como `2024-06-24T00:00:00.000`. En Bucaramanga, unas pocas vienen como `24/06/2024`. Si se le pide a pandas que adivine, puede leer `02/01/2024` como el 1 de febrero. Por eso la función `fecha` prueba los dos formatos de manera explícita. También convierte en faltante cualquier fecha de 1900, que es otro disfraz.
'''),
("code", r'''
def fecha(columna):
    valores = crudo[columna].astype(str)
    iso = pd.to_datetime(valores.str[:10], format="%Y-%m-%d", errors="coerce")
    con_barras = pd.to_datetime(valores, format=«"%d/%m/%Y"», errors="coerce")     # HUECO 2: el formato día/mes/año con barras, entre comillas
    resultado = iso.fillna(con_barras)
    return resultado.where(resultado.dt.year > 1900)

notificacion, sintomas, consulta = fecha("fec_not"), fecha("ini_sin_"), fecha("fec_con_")
print("Fechas de notificación sin leer:", int(notificacion.isna().sum()))
print("Notificaciones entre", notificacion.min().date(), "y", notificacion.max().date())
'''),
("md", r'''
**Compruébenlo ustedes mismos.** Deben quedar cero fechas de notificación sin leer. Si no, revisen el hueco 2.

### 2.4 Cuarta trampa: años en que nadie llenó los síntomas

El dengue se define como una enfermedad con fiebre. Una ficha de dengue sin fiebre es rara, y un año entero sin fiebre no puede ser. Miren, año por año, qué fracción de las fichas marca fiebre, cuántas no tienen estrato y cuántas dicen hospitalizado. En las columnas de síntomas, el `1` es «sí» y el `2` es «no».
'''),
("code", r'''
anio = notificacion.dt.year
por_anio = pd.DataFrame({
    "fichas": anio.value_counts().sort_index(),
    "% con fiebre": ((codigo("fiebre") == 1).groupby(anio).mean() * 100).round(1),
    "% sin estrato": (estrato.isna().groupby(anio).mean() * 100).round(1),
    "% hospitalizado": ((codigo("pac_hos_") == «1»).groupby(anio).mean() * 100).round(1),      # HUECO 3: el código que significa "sí"
})
por_anio.index = por_anio.index.astype(int)
display(por_anio)
'''),
("md", r'''
En Bucaramanga hay años en que casi ninguna ficha marca fiebre. Esos años las columnas de síntomas dicen «no» en todo, y ese «no» es otro faltante disfrazado: el síntoma no se capturó. Un modelo entrenado con esos años aprendería que hay hospitalizados sin un solo síntoma. Los primeros años tampoco traen estrato.

**Decisión:** el análisis usa las fichas desde `DESDE = 2021`, que es el primer año de Bucaramanga con síntomas y estrato. Para Tuluá, que empieza en 2023, la regla no quita nada. Anótenla en la bitácora.

### 2.5 Quinta trampa: códigos sin diccionario

Las alcaldías publican la tabla sin diccionario. La ficha de datos básicos del INS, que es pública, documenta algunos códigos: en `tip_cas_`, 2 es probable, 3 es confirmado por laboratorio y 5 es confirmado por nexo epidemiológico; en `ajuste_`, **6 es caso descartado**; en `con_fin_`, 1 es vivo y 2 es muerto.

Para `conducta` y `clasfinal` no conseguí el diccionario oficial al preparar la clase, porque el sitio del INS no respondió. Lo que sí hay es evidencia dentro del dato. Bucaramanga trae los dos sistemas, códigos y nombres, y se puede ver qué código se comporta como qué nombre.
'''),
("code", r'''
reciente = (anio >= DESDE).values
hospitalizado = codigo("pac_hos_") == 1
grave = codigo("cod_eve") == 220                       # 210 es dengue, 220 es dengue grave, 580 es mortalidad por dengue
for columna in ["conducta", "clasfinal"]:
    valor = texto(columna).str.replace(r"\.0$", "", regex=True)[reciente]
    evidencia = pd.DataFrame({"fichas": valor.value_counts(),
                              "% hospitalizado": (hospitalizado[reciente].groupby(valor).mean() * 100).round(1),
                              "% notificado como dengue grave": (grave[reciente].groupby(valor).mean() * 100).round(1)})
    print(columna); print(evidencia[evidencia["fichas"] >= 20].sort_index().to_string(), "\n")
'''),
("md", r'''
Con Bucaramanga la tabla permite emparejar. En `conducta`, el código 1 casi nunca coincide con hospitalización, igual que `AMBULATORIA`. El 2 coincide casi siempre, igual que `HOSPITALIZACION PISO`. El 3 coincide casi siempre y concentra los casos graves, igual que `UNIDAD DE CUIDADOS INTENSIVOS`. El 4 y el 5 se parecen a `OBSERVACION` y a `REMISION PARA HOSPITALIZACION`. En `clasfinal`, el 3 es el único código que coincide con la notificación de dengue grave, y el 1 y el 2 se comportan como «sin signos de alarma» y «con signos de alarma».

| Columna | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `conducta` | Ambulatoria | Hospitalización en piso | Cuidados intensivos | Observación | Remisión para hospitalización |
| `clasfinal` | Sin signos de alarma | Con signos de alarma | Dengue grave | | |

Este diccionario es una **deducción**, y la evidencia es de comportamiento: los códigos y los nombres están en fichas distintas, así que no hay una sola fila que traiga los dos. Para Tuluá supongo que rige la misma ficha nacional. Antes de usar esto fuera de clase habría que confirmarlo con el INS.

**Sexta trampa: fichas que no son casos.** Una ficha con `ajuste_` igual a 6 es un caso que se descartó: se sospechó dengue y resultó otra cosa. Las fichas con `cod_eve` 580 son registros de mortalidad.
'''),
("code", r'''
print("Valores de ajuste_:"); print(codigo("ajuste_")[reciente].value_counts().sort_index().to_string())
print("\nFichas de mortalidad (cod_eve 580):", int((codigo("cod_eve")[reciente] == 580).sum()))
'''),
("md", r'''
**Decisión:** las fichas descartadas salen, porque esos pacientes no tuvieron dengue. Las de mortalidad también salen, porque registran otro evento. Comparen cuántas fichas descartadas hay en cada ciudad cuando lleguen a la sección 4.5.

### 2.6 La tabla limpia

Esta celda reúne todas las decisiones. La edad viene en grupos con nombres distintos en cada ciudad, y por eso hay un pequeño diccionario que lleva las dos a la misma escala de curso de vida.
'''),
("code", r'''
COLUMNA_EDAD = {"bucaramanga": "def_clas_edad", "tulua": "clas_edad"}
CURSO_DE_VIDA = {"PRIMERA INFANCIA": 0, "INFANCIA": 1, "ADOLESCENCIA": 2, "ADOLECENCIA": 2, "JOVENES": 3, "JUVENTUD": 3,
                 "ADULTEZ": 4, "ADULTES": 4, "VEJEZ": 5, "PERSONA MAYOR": 5}
SINTOMAS = ["fiebre", "cefalea", "dolrretroo", "malgias", "artralgia", "erupcionr"]
ALARMA = ["dolor_abdo", "vomito", "diarrea", "somnolenci", "hipotensio", "hepatomeg", "hem_mucosa", "hipotermia", "aum_hemato", "caida_plaq", "acum_liqui"]
CONDUCTA = {"AMBULATORIA": 1, "HOSPITALIZACION PISO": 2, "UNIDAD DE CUIDADOS INTENSIVOS": 3, "OBSERVACION": 4, "REMISION PARA HOSPITALIZACION": 5}
CLASIFICACION = {"DENGUE SIN SIGNOS DE ALARMA": 1, "DENGUE CON SIGNOS DE ALARMA": 2, "DENGUE GRAVE": 3}

limpio = pd.DataFrame(index=crudo.index)
limpio["fecha"] = notificacion
limpio["hospitalizado"] = (codigo("pac_hos_") == 1).astype(int)
limpio["mujer"] = (texto("sexo_") == "F").astype(float)
limpio["curso_de_vida"] = texto(COLUMNA_EDAD[CIUDAD]).map(CURSO_DE_VIDA)
limpio["estrato"] = estrato.where(estrato.between(1, 6))
limpio["subsidiado"] = (texto("tip_ss_").str[0] == "S").astype(float)       # la primera letra sirve para los dos sistemas: S y SUBSIDIADO
limpio["dias_hasta_consulta"] = (consulta - «sintomas»).dt.days            # HUECO 4: la fecha que se resta para saber cuánto esperó el paciente
for columna in SINTOMAS + ALARMA:
    limpio[columna] = (codigo(columna) == 1).astype(float)
limpio["signos_de_alarma"] = limpio[ALARMA].sum(axis=1)                     # cuántos signos de alarma marca cada ficha

# Dos columnas que NO entran al modelo. Se guardan para la sección 3.5.
limpio["conducta"] = codigo("conducta").fillna(texto("conducta").map(CONDUCTA))
limpio["clasfinal"] = codigo("clasfinal").fillna(texto("clasfinal").map(CLASIFICACION))

antes = len(limpio)
es_caso = (codigo("ajuste_") != «6») & (codigo("cod_eve") != 580)           # HUECO 6: el código de ajuste que marca un caso descartado
limpio = limpio[es_caso & (limpio["fecha"].dt.year >= «DESDE»)]            # HUECO 7: el parámetro que fija el primer año
limpio = limpio[limpio["dias_hasta_consulta"].between(0, 30) | limpio["dias_hasta_consulta"].isna()]
limpio = limpio.sort_values("fecha").reset_index(drop=True)

y = limpio["hospitalizado"]
print(f"De {antes} fichas quedan {len(limpio)}, entre {limpio['fecha'].min().date()} y {limpio['fecha'].max().date()}.")
print(f"Hospitalizados: {y.mean() * 100:.1f} %")
print("Faltantes que quedan (%):"); print((limpio.isna().mean() * 100).round(1)[lambda s: s > 0].to_string())

registro = RegistroDeLimpieza(antes)
n_caso = int(es_caso.sum()); n_desde = int((es_caso & (notificacion.dt.year >= DESDE)).sum())
registro.anotar("Casos descartados y fichas de mortalidad", "salen de la tabla", antes - n_caso, n_caso)
registro.anotar(f"Fichas anteriores a {DESDE}", "salen: antes casi ninguna trae síntomas", n_caso - n_desde, n_desde)
registro.anotar("Consulta antes de los síntomas o más de un mes después", "salen: son errores de fecha", n_desde - len(limpio), len(limpio))
registro.anotar("Estrato faltante o fuera de 1 a 6", "queda como faltante; la ficha se queda", int(limpio["estrato"].isna().sum()))
display(registro.tabla())
'''),
("md", r'''
Comparen con su primera apuesta. Dos notas sobre la celda.

La lista `ALARMA` reúne los once signos que el protocolo de vigilancia de dengue del Instituto Nacional de Salud (códigos 210, 220 y 580, versión de marzo de 2022) pone en la definición de «dengue con signos de alarma»: dolor abdominal, vómito persistente, diarrea, acumulación de líquidos, sangrado de mucosas, letargo, hipotensión postural, hepatomegalia, caída de la temperatura y caída de plaquetas con aumento del hematocrito. Las columnas de la ficha siguen esa lista. El mismo protocolo define los códigos de `ajuste_` que usamos: 3 es confirmado por laboratorio, 5 es confirmado por nexo epidemiológico y 6 es descartado. Lo que sigue siendo una deducción mía es el significado de los códigos de `conducta`, porque el protocolo no los describe.

La penúltima regla quita las fichas en que la consulta aparece antes de los síntomas o más de un mes después. Son errores de fecha, y son pocas.

### 2.7 Conocer el dato antes de modelarlo

Cuatro miradas: cómo llegan los casos en el tiempo, cómo cambia la hospitalización con la edad, cuánto esperan los pacientes antes de consultar y qué síntomas distinguen a los hospitalizados. La tasa de un grupo $g$ y su error estándar son los de la clase:

$$\hat p_g=\frac{1}{n_g}\sum_{i\in g}y_i,\qquad \text{ee}(\hat p_g)=\sqrt{\frac{\hat p_g(1-\hat p_g)}{n_g}}.$$
'''),
("code", r'''
nombres_edad = ["primera infancia", "infancia", "adolescencia", "juventud", "adultez", "persona mayor"]
semana = limpio["fecha"].dt.to_period("W").dt.start_time
fig, ejes = plt.subplots(2, 2, figsize=(13, 7.5))

por_semana = y.groupby(semana).agg(["size", "mean"])
ejes[0, 0].bar(por_semana.index, por_semana["size"], width=6, color="#1D4E89")
ejes[0, 0].set(title=f"Casos notificados por semana, {CIUDAD.title()}", ylabel="casos")
otro = ejes[0, 0].twinx(); otro.plot(por_semana.index, por_semana["mean"].rolling(8, min_periods=4).mean() * 100, color="#D7263D")
otro.set_ylabel("% hospitalizado (promedio de 8 semanas)", color="#D7263D"); otro.spines["right"].set_visible(True)

por_edad = y.groupby(limpio["curso_de_vida"]).agg(["size", "mean"])
error = np.sqrt(por_edad["mean"] * (1 - por_edad["mean"]) / por_edad["size"])
ejes[0, 1].bar([nombres_edad[int(i)] for i in por_edad.index], por_edad["mean"] * 100, yerr=2 * error * 100, color="#1B998B", ecolor="#1D4E89")
ejes[0, 1].set(title="Hospitalización por curso de vida", ylabel="%"); ejes[0, 1].tick_params(axis="x", rotation=20)

dias = limpio["dias_hasta_consulta"].dropna()
ejes[1, 0].hist([dias[y == 0], dias[y == 1]], bins=range(0, 16), density=True, color=["#5BC0EB", "#D7263D"], label=["no hospitalizado", "hospitalizado"])
ejes[1, 0].set(title="Días entre el inicio de síntomas y la consulta", xlabel="días"); ejes[1, 0].legend()

frecuencia = limpio.groupby("hospitalizado")[SINTOMAS + ALARMA].«mean»().T * 100          # HUECO 8: la operación que convierte ceros y unos en una fracción
frecuencia.columns = ["no hospitalizado", "hospitalizado"]
frecuencia.sort_values("hospitalizado").plot.barh(ax=ejes[1, 1], color=["#5BC0EB", "#D7263D"])
ejes[1, 1].set(title="% de fichas con cada síntoma", xlabel="%"); ejes[1, 1].tick_params(axis="y", labelsize=8)
plt.tight_layout(); plt.show()
'''),
("md", r'''
**Una prueba de hipótesis: ¿los hospitalizados consultaron más tarde?** El histograma de abajo a la izquierda compara dos grupos. La prueba de Mann-Whitney contrasta

$$H_0:\ P(D_1>D_0)=\tfrac12\qquad\text{contra}\qquad H_1:\ P(D_1>D_0)\neq\tfrac12,$$

donde $D_1$ son los días hasta la consulta de un hospitalizado tomado al azar y $D_0$ los de un no hospitalizado. Usa rangos, así que no supone normalidad, que aquí no hay.
'''),
("code", r'''
from scipy.stats import mannwhitneyu

dias_h, dias_n = dias[y.loc[dias.index] == 1], dias[y.loc[dias.index] == 0]
estadistico, valor_p = mannwhitneyu(dias_h, dias_n)
print(f"Mediana de días hasta la consulta | hospitalizados: {dias_h.median():.0f} | no hospitalizados: {dias_n.median():.0f}")
print(f"Media | hospitalizados: {dias_h.mean():.2f} | no hospitalizados: {dias_n.mean():.2f}")
print(f"P(D1 > D0) estimada: {estadistico / (len(dias_h) * len(dias_n)):.3f} | valor p: {valor_p:.2g}")
'''),
("md", r'''
- Con miles de fichas, casi cualquier diferencia sale con un valor $p$ diminuto. Miren la probabilidad estimada: ¿qué tan lejos está de un medio? ¿La diferencia importa en la práctica? **Mi lectura:**

Respondan en una línea cada una.

- ¿El porcentaje de hospitalizados se mantiene estable en el tiempo, o cambia cuando llega la epidemia? **Mi lectura:**
- ¿Qué síntomas aparecen casi igual en los dos grupos y cuáles separan? Comparen con su segunda apuesta. **Mi lectura:**
- ¿Los hospitalizados consultaron antes o después que los demás? **Mi lectura:**

**¿Quiénes están en la tabla y quiénes no?** Una ficha existe solo si la persona consultó y el médico notificó. Quien pasó el dengue en la casa con acetaminofén no aparece. Respondan: ¿el porcentaje de hospitalizados de esta tabla sirve como estimación del riesgo de hospitalización de una persona con dengue? ¿Queda por encima o por debajo? **Mi respuesta:**

### 2.8 PCA: ¿los síntomas vienen en paquetes?

En clase usamos PCA para ver si las variables del hogar medían lo mismo. Aquí la pregunta es si los diecisiete síntomas se agrupan. Se estandariza la matriz de síntomas y se descompone su matriz de correlaciones:

$$R=V\Lambda V^\top,\qquad \text{varianza explicada por el componente } j=\frac{\lambda_j}{\sum_k\lambda_k}.$$

**Predicción escrita:** con diecisiete síntomas independientes, cada componente explicaría cerca de 6 %. ¿Cuánto explicará el primero? **Mi apuesta:**
'''),
("code", r'''
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

Z = StandardScaler().fit_transform(limpio[SINTOMAS + ALARMA])
pca = PCA().fit(Z)
explicada = pca.explained_variance_ratio_ * 100
cargas = pd.DataFrame(pca.components_.T[:, :3], index=SINTOMAS + ALARMA, columns=["CP1", "CP2", "CP3"])

resumen_pca(explicada, cargas, "Cargas: qué mezcla de síntomas es cada componente").show()
tipo = pd.Series(["síntomas generales"] * len(SINTOMAS) + ["signos de alarma"] * len(ALARMA), index=cargas.index)
nube_3d(cargas, grupo=tipo, colores={"síntomas generales": CIELO, "signos de alarma": ROJO}, etiquetas=True, tamano=7,
        titulo="Los diecisiete síntomas en tres componentes",
        subtitulo="Cada punto es un síntoma. Dos síntomas quedan cerca si aparecen en las mismas fichas. Arrastren para girar.",
        ejes=[f"CP{j + 1} ({explicada[j]:.0f} %)" for j in range(3)]).show()
print(f"Los tres primeros componentes explican {explicada[:3].sum():.0f} % de la varianza.")
'''),
("md", r'''
- ¿Los síntomas generales y los signos de alarma quedan en zonas distintas de la gráfica? **Mi lectura:**
- Si una ciudad marca fiebre en todas las fichas, esa columna no varía y su punto queda en el origen. ¿Qué le aporta al modelo una columna así? **Mi respuesta:**
'''),
("md", r'''
## 3. Modelado

### 3.1 La partición y la trampa de la exactitud

En el ICFES partimos por colegio. Aquí lo que manda es el tiempo, como en la clase 1: el equipo siempre va a usar el modelo con casos posteriores a los que sirvieron para entrenarlo. Las fichas ya están ordenadas por fecha de notificación, y se cortan así:

$$\underbrace{\text{primer } 60\,\%}_{\text{entrenamiento}}\ \Big|\ \underbrace{20\,\%\ \text{siguiente}}_{\text{validación: se escoge}}\ \Big|\ \underbrace{\text{último } 20\,\%}_{\text{prueba: se reporta}}$$

Las tres medidas de la clase, con verdaderos y falsos positivos y negativos:

$$\text{exactitud}=\frac{VP+VN}{n},\qquad \text{sensibilidad}=\frac{VP}{VP+FN},\qquad \text{precisión}=\frac{VP}{VP+FP}.$$
'''),
("code", r'''
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

variables = ["mujer", "curso_de_vida", "estrato", "subsidiado", "dias_hasta_consulta"] + SINTOMAS + ALARMA
X = limpio[variables]

n = len(limpio)
posicion = np.arange(n)
corte_1, corte_2 = int(«0.60» * n), int(0.80 * n)              # HUECO 10: la fracción de fichas que va a entrenamiento
E, V, P = posicion < corte_1, (posicion >= corte_1) & (posicion < corte_2), posicion >= corte_2
for nombre, tramo in [("Entrenamiento", E), ("Validación", V), ("Prueba", P)]:
    print(f"{nombre:14s} {tramo.sum():5d} fichas | {limpio['fecha'][tramo].min().date()} a {limpio['fecha'][tramo].max().date()} | {y[tramo].mean() * 100:.1f} % hospitalizado")
'''),
("code", r'''
logistica = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(X[E], y[E])
dice_hospital = logistica.predict_proba(X[V])[:, 1] >= 0.5
real = y[V].values

def tres_medidas(real, dice):
    vp, fp, fn = ((dice == 1) & (real == 1)).sum(), ((dice == 1) & (real == 0)).sum(), ((dice == 0) & (real == 1)).sum()
    return {"exactitud": (dice == real).mean(), "sensibilidad": vp / (vp + «fn»), "precisión": vp / (vp + fp) if vp + fp else np.nan}   # HUECO 11: los hospitalizados que el modelo dejó pasar

display(pd.DataFrame({"decir que nadie se hospitaliza": tres_medidas(real, np.zeros(len(real))),
                      "regresión logística, umbral 0,5": tres_medidas(real, dice_hospital)}).T.round(3))
'''),
("md", r'''
En el ICFES, decir que nadie quedaba abajo tenía casi la misma exactitud que el modelo. Miren si aquí pasa lo mismo y respondan: ¿por qué la exactitud engaña menos, o más, en este dato? Piensen en qué fracción de los casos es positiva y en qué tan fuerte es la señal de los síntomas. **Mi respuesta:**

Aun así, el equipo no decide con un umbral de 0,5. Decide con un número de llamadas. Por eso la métrica sigue siendo la captación, acompañada del área bajo la curva ROC, $\text{AUC}=P(\hat p_i>\hat p_j\mid y_i=1,\ y_j=0)$.

### 3.2 La línea base

Dos reglas sin modelo. La primera reparte las llamadas por sorteo. La segunda es el protocolo: llamar primero a quien tenga más signos de alarma. Esta segunda línea base es exigente, porque resume lo que la medicina ya sabe.
'''),
("code", r'''
def captacion(real, puntaje, cupo=CUPO):
    """Fracción de los hospitalizados que cae entre los primeros `cupo` del orden. Los empates se rompen por sorteo."""
    real, puntaje = np.asarray(real), np.asarray(puntaje, dtype=float)
    k = int(round(cupo * len(real)))
    orden = np.lexsort((np.random.default_rng(0).random(len(real)), -puntaje))
    return float(real[orden[:k]].sum() / «real.sum()»)          # HUECO 12: el denominador de la captación: todos los hospitalizados

reglas = {
    "sorteo": pd.Series(np.random.default_rng(1).random(n)),
    "regla: signos de alarma": limpio["signos_de_alarma"],
}
errores = {nombre: {"captación en validación": captacion(y[V], regla[V]), "captación en prueba": captacion(y[P], regla[P]),
                    "AUC en prueba": roc_auc_score(y[P], regla[P])} for nombre, regla in reglas.items()}
display(pd.DataFrame(errores).T.round(3))
print(f"Techo de la captación en prueba: {min(1, CUPO / y[P].mean()):.2f}")
'''),
("md", r'''
Recuerden el techo de la clase: $\text{captación máxima}=\min(1,\ \text{cupo}/\hat p)$. Si se hospitalizan más casos de los que el equipo alcanza a llamar, ni un oráculo los cubre a todos.

### 3.3 Cinco modelos y el mismo protocolo de la clase

Los cinco candidatos son los de la clase: regresión logística, $\log\dfrac{p_i}{1-p_i}=\beta_0+x_i^\top\beta$; $k$ vecinos; árbol de decisión; bosque aleatorio y *gradient boosting*. Gana quien tenga mayor captación en **validación** y se reporta su captación en **prueba**.

**Predicción escrita:** ¿algún modelo le ganará a la regla de los signos de alarma en la prueba? ¿Por cuánto? **Mi apuesta:**
'''),
("code", r'''
def cinco_modelos():
    llenar = lambda: SimpleImputer(strategy="median")
    return {
        "logística": make_pipeline(llenar(), StandardScaler(), LogisticRegression(max_iter=1000)),
        "k vecinos": make_pipeline(llenar(), StandardScaler(), KNeighborsClassifier(n_neighbors=50)),
        "árbol": make_pipeline(llenar(), DecisionTreeClassifier(max_depth=5, random_state=0)),
        "bosque": make_pipeline(llenar(), RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1)),
        "boosting": make_pipeline(llenar(), HistGradientBoostingClassifier(random_state=0)),
    }

puntajes = {nombre: regla[P] for nombre, regla in reglas.items()}
for nombre, modelo in cinco_modelos().items():
    modelo.fit(X[E], y[E])                                                # se entrena solo con el pasado
    en_validacion = captacion(y[V], modelo.predict_proba(X[V])[:, 1])
    modelo.fit(X[E | V], y[E | V])                                        # se reentrena con todo lo anterior a la prueba
    puntajes[nombre] = pd.Series(modelo.predict_proba(X[P])[:, 1], index=X.index[P])
    errores[nombre] = {"captación en validación": en_validacion, "captación en prueba": captacion(y[P], puntajes[nombre]),
                       "AUC en prueba": roc_auc_score(y[P], puntajes[nombre])}
todos = pd.DataFrame(errores).T

fig, (izq, der) = plt.subplots(1, 2, figsize=(14, 4.2))
todos[["captación en validación", "captación en prueba"]].sort_values("captación en validación").plot.barh(ax=izq, color=["#1D4E89", "#1B998B"])
izq.axvline(CUPO, color="gray", ls=":"); izq.set(xlabel="captación", title="Siete candidatos: validación para escoger, prueba para reportar")
fracciones = np.linspace(0.02, 1, 50)
for nombre, puntaje in puntajes.items():
    estilo = {"color": "gray", "ls": ":"} if nombre == "sorteo" else {"color": "#1B998B", "lw": 2} if nombre.startswith("regla") else {"lw": 1.2}
    der.plot(fracciones * 100, [captacion(y[P], puntaje, f) for f in fracciones], label=nombre, **estilo)
der.axvline(CUPO * 100, color="#D7263D", lw=1)
der.set(xlabel="% de casos con llamada", ylabel="captación", title="Captación según la capacidad del equipo, en la prueba"); der.legend(fontsize=8)
plt.tight_layout(); plt.show()
display(todos.sort_values("captación en validación", ascending=False).round(3))
'''),
("md", r'''
**¿Por qué el ganador se escoge en validación y no en prueba?** Respondan en dos líneas, con sus palabras: **Mi respuesta:**
'''),
("code", r'''
mejor = todos.drop(index="sorteo")[«"captación en validación"»].idxmax()        # HUECO 13: la columna con la que se escoge al ganador
resumen = todos["captación en prueba"].round(3)
regla = "regla: signos de alarma"
print(f"Ganador en validación: {mejor}")
print(f"Captación en prueba: {resumen[mejor]:.3f} | regla de los signos de alarma: {resumen[regla]:.3f} | sorteo: {resumen['sorteo']:.3f} | techo: {min(1, CUPO / y[P].mean()):.2f}")
'''),
("md", r'''
Comparen con su tercera apuesta. La distancia entre el ganador y la regla es lo que el modelo agrega a lo que el protocolo ya sabía. Escriban en una línea si esa distancia justifica cambiar la regla por un modelo: **Mi lectura:**

**Por qué quedan tan cerca.** La respuesta que modelamos es una decisión del médico, y la guía de atención le indica hospitalizar a quien tiene signos de alarma. El modelo aprende, en buena parte, la regla que produjo el dato. Dos preguntas para cerrar:

- Si el modelo imita al protocolo, ¿puede este cuaderno decir si el protocolo se equivoca? **Mi respuesta:**
- ¿Qué habría que medir para saberlo, y está ese dato en la ficha? **Mi respuesta:**

### 3.4 La trampa de la clase: columnas que filtran la respuesta

En la sección 2.6 guardé dos columnas sin dárselas al modelo. `conducta` dice qué se hizo con el paciente, y uno de sus valores es «hospitalización en piso». `clasfinal` es la clasificación con que se cerró el caso. Las dos se llenan cuando ya se sabe en qué terminó el paciente.

**Predicción escrita:** ¿a cuánto llegará el AUC si el modelo recibe `conducta`? **Mi apuesta:**
'''),
("code", r'''
escenarios = {
    "solo paciente y síntomas": [],
    "+ clasfinal": ["clasfinal"],
    "+ conducta": [«"conducta"»],                         # HUECO 14: la columna que dice qué se hizo con el paciente, entre comillas
    "+ las dos": ["clasfinal", "conducta"],
}
fuga = {}
for nombre, columnas in escenarios.items():
    X_con = limpio[variables + columnas]
    modelo = cinco_modelos()["boosting"].fit(X_con[E | V], y[E | V])
    prob = modelo.predict_proba(X_con[P])[:, 1]
    fuga[nombre] = {"captación en prueba": captacion(y[P], prob), "AUC en prueba": roc_auc_score(y[P], prob),
                    "exactitud con umbral 0,5": ((prob >= 0.5) == y[P]).mean()}
display(pd.DataFrame(fuga).T.round(3))
'''),
("md", r'''
- ¿Cuál de las cuatro filas es el resultado que se le puede prometer al equipo? **Mi respuesta:**
- La pregunta que detecta la fuga es: el día en que se toma la decisión, ¿esta columna ya existe? Aplíquenla a `caida_plaq` (caída de plaquetas) y a `aum_hemato` (aumento del hematocrito), que salen de un examen de laboratorio. ¿El equipo las tiene cuando decide a quién llamar? **Mi respuesta:**

### 3.5 Explicar: ¿qué acompaña a la hospitalización?

Como en clase, $e^{\beta_j}$ es la razón de probabilidades asociada a subir una desviación estándar en la variable $j$, con las demás quietas:

$$\text{OR}_j=e^{\beta_j}.$$
'''),
("code", r'''
logistica = cinco_modelos()["logística"].fit(X[E | V], y[E | V])
razones = pd.Series(np.exp(logistica[-1].coef_[0]), index=variables).sort_values()
razones.plot.barh(figsize=(8, 6), color=["#5BC0EB" if r < 1 else "#D7263D" for r in razones])
plt.axvline(1, color="gray"); plt.xlabel("razón de probabilidades por desviación estándar")
plt.title("Qué acompaña a la hospitalización"); plt.show()
print(razones.round(2).to_string())
'''),
("md", r'''
- ¿Qué tres variables pesan más? ¿Son signos de alarma? **Mi lectura:**
- Estos números describen asociaciones. Un signo de alarma puede aparecer en la ficha porque el paciente ya estaba hospitalizado y le hicieron exámenes. ¿Cómo cambia eso la lectura de la gráfica? **Mi respuesta:**

### 3.6 Sesgos: ¿a quién le acierta y a quién le falla el modelo?

La misma auditoría de la clase. Para cada grupo: qué fracción se hospitalizó, qué fracción recibe llamada, la captación dentro del grupo y las **llamadas de sobra**, que son los casos que no se hospitalizaron y recibieron llamada.
'''),
("code", r'''
elegido = mejor if resumen[mejor] > resumen[regla] else regla            # el más simple entre los que no pierden
puntaje = puntajes[elegido]
k = int(round(CUPO * P.sum()))
orden = np.lexsort((np.random.default_rng(0).random(P.sum()), -puntaje.values))
llamada = pd.Series(False, index=X.index[P]); llamada.iloc[orden[:k]] = True

def auditar(grupo):
    partes = {}
    for valor, indices in grupo[P].groupby(grupo[P]).groups.items():
        yy, cc = y[indices], llamada[indices]
        partes[valor] = {"n": len(yy), "% hospitalizado": yy.mean() * 100, "% con llamada": cc.mean() * 100,
                         "captación": cc[yy == 1].mean(), "llamadas de sobra": cc[yy == 0].mean()}
    return pd.DataFrame(partes).T

grupos = {
    "sexo": limpio["mujer"].map({1.0: "mujeres", 0.0: "hombres"}),
    "régimen": limpio["subsidiado"].map({1.0: "subsidiado", 0.0: "otros regímenes"}),
    "edad": (limpio["curso_de_vida"] <= 2).map({True: "menores de 18", False: "adultos"}),
}
auditoria = pd.concat({nombre: auditar(grupo) for nombre, grupo in grupos.items()})
print("Candidato auditado:", elegido)
display(auditoria.round({"n": 0, "% hospitalizado": 1, "% con llamada": 1, "captación": 2, "llamadas de sobra": 2}))
'''),
("md", r'''
- ¿En qué grupo es más baja la captación? Esos son los hospitalizados que la regla deja sin llamada. **Mi lectura:**
- El régimen subsidiado cubre a la población con menos recursos. ¿El modelo los atiende igual que a los demás? **Mi lectura:**
'''),
("md", r'''
## 4. Comunicación

### 4.1 Decidir: ¿a quiénes llama el equipo?

Para ponerle margen a la captación usamos *bootstrap* por semana: se sacan muchas veces, con reposición, las semanas de la prueba y se recalcula la captación. El intervalo de 90 % va del percentil 5 al 95,

$$\bigl[\hat Q_{0{,}05}(\text{captación}^*),\ \hat Q_{0{,}95}(\text{captación}^*)\bigr].$$

Se remuestrean semanas completas porque los casos de una misma semana comparten la misma etapa de la epidemia.
'''),
("code", r'''
semana_prueba = limpio["fecha"][P].dt.to_period("W").astype(str).values
semanas = np.unique(semana_prueba)
filas_por_semana = {s: np.flatnonzero(semana_prueba == s) for s in semanas}
sorteo = np.random.default_rng(0)
replicas = {"elegido": [], "regla": []}
for _ in range(500):
    filas = np.concatenate([filas_por_semana[s] for s in sorteo.choice(semanas, len(semanas))])
    replicas["elegido"].append(captacion(y[P].values[filas], puntajes[elegido].values[filas]))
    replicas["regla"].append(captacion(y[P].values[filas], puntajes[regla].values[filas]))
bajo_, alto_ = np.percentile(replicas["elegido"], [«5», 95])              # HUECO 15: el percentil inferior de un intervalo de 90 %
ventaja = np.array(replicas["elegido"]) - np.array(replicas["regla"])

precision = y[P][llamada].mean()
print(f"Ciudad: {CIUDAD} | Candidato elegido: {elegido}")
print(f"Captación en prueba: {resumen[elegido]:.2f}, con intervalo de 90 % entre {bajo_:.2f} y {alto_:.2f}")
print(f"Ventaja sobre la regla de los signos de alarma: {np.median(ventaja):+.2f} (intervalo de 90 %: {np.percentile(ventaja, 5):+.2f} a {np.percentile(ventaja, 95):+.2f})")
print(f"De cada 100 llamadas, {precision * 100:.0f} van a pacientes que terminaron hospitalizados. Por sorteo irían {y[P].mean() * 100:.0f}.")
print(f"De cada 100 hospitalizados, {(1 - resumen[elegido]) * 100:.0f} se quedan sin llamada.")
'''),
("md", r'''
### La decisión en tres frases

Para la secretaria de salud, que es médica y no sabe qué es un *bootstrap*.

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

### 4.2 Bitácora de decisiones

Llenen las filas vacías. La primera va de ejemplo.

| Decisión | Alternativa descartada | Por qué |
|---|---|---|
| Todo `-89` es un faltante | Dejarlo como número | No es un valor de la variable: es la marca de una celda vacía |
| Fechas con dos formatos explícitos | | |
| Solo fichas desde 2021 | | |
| Diccionario de `conducta` y `clasfinal` deducido del dato | | |
| Fuera las fichas con `ajuste_` 6 y con `cod_eve` 580 | | |
| Faltantes llenos con la mediana de entrenamiento | | |
| Partición por tiempo | | |
| `conducta` y `clasfinal` fuera del modelo | | |
| Captación con cupo de 30 % como métrica | | |

### 4.3 Riesgos, ética y mantenimiento

Este dato es de pacientes. Viene sin nombres, pero trae edad, sexo, régimen, estrato, grupo poblacional y fechas.

- La ficha no dice si los síntomas se anotaron antes o después de hospitalizar al paciente. ¿Qué le pasa al modelo si se anotaron después?
- El diccionario de códigos es una deducción. ¿Qué parte de sus resultados cambiaría si el código 4 de `conducta` no fuera «observación»?
- En un municipio pequeño, ¿se podría reconocer a una persona cruzando edad, sexo, fecha y grupo poblacional? ¿Qué columnas quitarían antes de publicar una tabla así?
- El modelo se entrenó en una epidemia. ¿Cada cuánto habría que revisarlo, y qué señal les diría que dejó de servir?
- Una llamada de seguimiento no le hace daño a nadie. ¿Qué cambiaría si el mismo modelo se usara para decidir a quién no se hospitaliza?

### 4.4 Procedencia
'''),
("code", r'''
procedencia()["dengue"]
'''),
("md", r'''
### 4.5 Cambien una línea

Vuelvan a la celda donde dice `CIUDAD = "bucaramanga"`, pongan `CIUDAD = "tulua"` y corran todo de nuevo con *Entorno de ejecución*, *Ejecutar todo*. Ninguna otra línea cambia: las funciones de limpieza sirven para las dos ciudades. Llenen la tabla con sus dos corridas.

| | Bucaramanga | Tuluá |
|---|---|---|
| Fichas descargadas y fichas que quedan | | |
| Fichas descartadas (`ajuste_` 6) | | |
| % hospitalizado | | |
| Techo de la captación | | |
| Ganador en validación | | |
| Captación del ganador en prueba | | |
| Captación de la regla de los signos de alarma | | |
| AUC honesto y AUC con `conducta` | | |

Con la tabla llena, respondan en tres líneas. ¿Qué trampa de limpieza pesó más en cada ciudad? ¿El modelo de una ciudad serviría en la otra, y qué habría que comprobar antes de intentarlo? **Mi respuesta:**

Dejen el cuaderno entregado con `CIUDAD = "tulua"` corrido y la tabla completa.

**Para entregar:** en Colab, menú *Archivo*, *Guardar una copia en GitHub*, y escojan su repositorio de portafolio, carpeta `clase02/`. Un cuaderno que no corre de principio a fin desde cero no se califica: antes de guardar usen *Entorno de ejecución*, *Reiniciar sesión y ejecutar todo*.
'''),
]

if __name__ == "__main__":
    destino = RAIZ / "clases" / "02_limpieza"
    destino.mkdir(parents=True, exist_ok=True)
    nbf.write(cuaderno(ICFES, False), destino / "02a_el_icfes.ipynb")
    if DENGUE:
        nbf.write(cuaderno(DENGUE, True), destino / "02b_dengue.ipynb")
        (RAIZ / "_profesor").mkdir(exist_ok=True)
        nbf.write(cuaderno(DENGUE, False), RAIZ / "_profesor" / "02b_dengue_SOLUCION.ipynb")
    print("Cuadernos de la clase 2 escritos.")
