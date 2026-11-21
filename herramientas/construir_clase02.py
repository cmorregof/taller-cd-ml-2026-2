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
# La brecha del ICFES en Caldas

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Este cuaderno lo resuelvo yo en vivo. Ustedes lo siguen, lo corren y me interrumpen. Termina, como *El IPC del sancocho*, en un carrusel para LinkedIn.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Limpiar un dato de personas dejando cada decisión a la vista, escoger una métrica que responda a la decisión del cliente, revisar a quién le falla el modelo y contar el resultado. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 1 (formular), 2 (seleccionar y preparar datos), 4 (implementar y evaluar), 5 (comunicar) y 6 (reflexión ética). |
| **Uso de IA** | Pueden pedirle a Claude que explique una línea o que escriba una gráfica. Deben poder explicar sin ayuda por qué la exactitud engaña aquí, qué es la captación y por qué la partición se hace por colegio. |
| **Entregable** | Este cuaderno corrido en su portafolio, con las predicciones y la decisión en tres frases. |
'''),
("code", ARRANQUE),
("md", r'''
## 1. La pregunta

La Secretaría de Educación de Caldas tiene un programa de refuerzo para grado once. El presupuesto alcanza para uno de cada cuatro estudiantes, y los cupos se reparten antes del examen, cuando solo se sabe lo que cada uno respondió en la inscripción.

1. **Predecir.** ¿Qué estudiantes van a quedar en el cuarto más bajo del país?
2. **Explicar.** ¿Qué condiciones del hogar y del colegio acompañan ese resultado?
3. **Decidir.** Con cupo para el 25 %, ¿a quiénes se les ofrece el refuerzo, y a quiénes deja por fuera esa regla?

**El dato.** Saber 11, aplicación 2020-2, publicado por el ICFES en datos.gov.co. Cada fila es un estudiante anonimizado, con su puntaje, sus respuestas sobre el hogar y los datos de su colegio.

**La respuesta.** $y_i=1$ si el percentil nacional del estudiante $i$ es 25 o menos.

**La métrica.** Si el modelo ordena a los estudiantes por riesgo y el programa atiende al primer 25 %, la **captación** es la fracción de los que quedaron abajo que recibió cupo:

$$\text{captación}=\frac{\#\{i:\ y_i=1\ \text{y}\ i\ \text{recibe cupo}\}}{\#\{i:\ y_i=1\}}.$$

### Predicción escrita

- En Caldas, ¿qué fracción de los estudiantes queda en el cuarto más bajo del país? **Mi apuesta:**
- ¿Qué pesa más: el colegio, la zona o la familia? **Mi apuesta:**
- Si los cupos se reparten por sorteo, ¿cuál es la captación? **Mi apuesta:**
'''),
("code", r'''
from taller.icfes import preparar, tasa_por, tramos, competir, asignar_cupos, auditar

DEPARTAMENTO = "CALDAS"      # al final pueden cambiarlo por otro departamento
CUPO = 0.25
crudo = descargar("saber11", departamento=DEPARTAMENTO)
print(f"{len(crudo):,} estudiantes de {crudo['cole_cod_dane_establecimiento'].nunique()} colegios".replace(",", "."))
'''),
("md", r'''
## 2. Tres trampas antes de modelar

**Edades imposibles.** La fecha de nacimiento trae errores de digitación.
'''),
("code", r'''
edad = 2020 - pd.to_numeric(crudo["estu_fechanacimiento"].astype(str).str[:4], errors="coerce")
print("Edades fuera de 13 a 70 años:", int((~edad.between(13, 70)).sum()))
'''),
("md", r'''
**Puntajes en cero.** Un puntaje global de cero es sospechoso en un examen de cinco pruebas. Miremos cuántos hay y en qué estado los publica el ICFES.
'''),
("code", r'''
puntaje = pd.to_numeric(crudo["punt_global"], errors="coerce")
print("Puntaje global igual a cero:", int((puntaje == 0).sum()))
print(crudo.loc[puntaje == 0, "estu_estadoinvestigacion"].value_counts().to_string())
'''),
("md", r'''
**Faltantes que no son al azar.** Quienes dejan preguntas en blanco no se parecen a los demás. Si borráramos esas filas, la muestra quedaría más privilegiada de lo que es. Por eso no se borran: los modelos llenan los huecos con la mediana del entrenamiento, y lo decimos.

`preparar` convierte el formulario en números, deja las edades imposibles como faltantes y quita solo los puntajes en cero, porque no describen un desempeño. Es una decisión, y va a la bitácora.
'''),
("code", r'''
X, y, colegio, datos = preparar(crudo)
incompleta = X.isna().any(axis=1)
print(f"Quedan {len(X):,} estudiantes. En el cuarto más bajo del país: {y.mean() * 100:.1f} %".replace(",", "."))
print(f"Con algún faltante: {incompleta.mean() * 100:.1f} % | de ellos, en el cuarto más bajo: {y[incompleta].mean() * 100:.1f} %")
'''),
("md", r'''
## 3. La brecha

Antes de cualquier modelo, la pregunta descriptiva: ¿en qué grupos se concentran los puntajes bajos? Las barras llevan dos errores estándar, para no leer diferencias donde hay ruido.
'''),
("code", r'''
madre = {"Ninguno": "ninguna", "Primaria incompleta": "primaria", "Primaria completa": "primaria",
         "Secundaria (Bachillerato) incompleta": "bachillerato", "Secundaria (Bachillerato) completa": "bachillerato",
         "Técnica o tecnológica incompleta": "técnica", "Técnica o tecnológica completa": "técnica",
         "Educación profesional incompleta": "profesional", "Educación profesional completa": "profesional",
         "Postgrado": "posgrado"}
paneles = {
    "Colegio y zona": tasa_por(y, datos["cole_naturaleza"].str.capitalize() + ", " + datos["cole_area_ubicacion"].str.lower()),
    "Educación de la madre": tasa_por(y, datos["fami_educacionmadre"].map(madre),
                                      ["ninguna", "primaria", "bachillerato", "técnica", "profesional", "posgrado"]),
    "Internet en la casa": tasa_por(y, datos["fami_tieneinternet"].map({"Si": "sí", "No": "no"})),
    "Jornada": tasa_por(y, datos["cole_jornada"].str.lower()),
}
fig, ejes = plt.subplots(1, 4, figsize=(17, 3.8))
for eje, (titulo, tabla) in zip(ejes, paneles.items()):
    eje.barh(tabla.index, tabla["tasa"], xerr=2 * tabla["ee"], color=PALETA[0], ecolor="#9AA5B1")
    eje.axvline(y.mean() * 100, color="#D7263D", ls=":"); eje.set(title=titulo, xlabel="% en el cuarto más bajo")
plt.tight_layout(); plt.show()
'''),
("code", r'''
municipios = tasa_por(y, datos["cole_mcpio_ubicacion"].str.title())
municipios = municipios[municipios["n"] >= 80]
print("Municipios con 80 estudiantes o más, del más alto al más bajo en el cuarto inferior del país:")
print(municipios.sort_values("tasa", ascending=False)[["n", "tasa"]].round(1).to_string())
'''),
("md", r'''
**Lectura.** Las brechas más grandes aparecen en lo que hay en la casa y en dónde queda el colegio. La diferencia entre colegio oficial y privado es menor. La jornada sabatina y la nocturna agrupan a muchos adultos que validan el bachillerato, y por eso conviene leerlas aparte. Ninguna de estas barras es una causa: son grupos que se superponen.

## 4. Predecir: ¿a quién le llega el cupo?

**Los tramos se parten por colegio.** Dos estudiantes del mismo colegio se parecen. Si uno queda en entrenamiento y el otro en prueba, el modelo parece mejor de lo que es. Por eso cada colegio va entero a un solo tramo: 60 % para entrenar, 20 % para escoger y 20 % para reportar.

Compiten cinco candidatos: el sorteo, una regla de una sola columna (la educación de la madre) y tres modelos (regresión logística, un árbol y un bosque aleatorio).
'''),
("code", r'''
E, V, P = tramos(X, y, colegio)
todos, puntajes = competir(X, y, E, V, P, CUPO)
display(todos.sort_values("captación en validación", ascending=False).round(3))
print(f"Techo de la captación con cupo del {CUPO:.0%}: {min(1, CUPO / y[P].mean()):.2f}")
'''),
("code", r'''
mejor = todos.drop(index="sorteo")["captación en validación"].idxmax()
regla = "regla: educación de la madre"
elegido = mejor if todos.loc[mejor, "captación en prueba"] > todos.loc[regla, "captación en prueba"] else regla
print(f"Ganador en validación: {mejor} | se usa: {elegido}")
print(f"Captación en prueba: {todos.loc[elegido, 'captación en prueba']:.2f} "
      f"| regla de una columna: {todos.loc[regla, 'captación en prueba']:.2f} | sorteo: {todos.loc['sorteo', 'captación en prueba']:.2f}")
'''),
("md", r'''
**Por qué no la exactitud.** Un modelo que dijera «nadie queda abajo» acertaría en tres de cada cuatro estudiantes y no le serviría de nada a la Secretaría. La captación mide lo que importa: cuántos de los que necesitaban el cupo lo recibieron.

## 5. ¿A quién le falla?

Repartimos los cupos con el candidato elegido y miramos, por grupo, qué fracción de los que quedaron abajo recibió cupo.
'''),
("code", r'''
cupos = asignar_cupos(puntajes[elegido], CUPO)
grupos = {"zona": datos["cole_area_ubicacion"].str.lower(),
          "sexo": datos["estu_genero"].map({"F": "mujeres", "M": "hombres"}),
          "jornada": X["nocturna_o_sabatina"].map({1.0: "nocturna o sabatina", 0.0: "diurna"})}
auditoria = pd.concat({nombre: auditar(y, cupos, grupo) for nombre, grupo in grupos.items()})
auditoria.round({"n": 0, "% que quedó abajo": 1, "% con cupo": 1, "captación": 2})
'''),
("md", r'''
Una captación más baja en un grupo significa que el programa deja por fuera, en proporción, a más estudiantes de ese grupo que lo necesitaban. Esa es una decisión política que el modelo no puede tomar solo: la Secretaría puede, por ejemplo, reservar cupos por zona.

### La decisión en tres frases

1. **Recomendación:**
2. **Evidencia:**
3. **Riesgo:**

## 6. El post

Tres láminas con lo que dicen los datos de Caldas. Las cifras salen de las celdas de arriba.
'''),
("code", r'''
from taller import carrusel as c
c.preparar_fuentes()
PIE = "Fuente: ICFES, Saber 11 2020-2, en datos.gov.co. Cálculos propios.\nCarlos Manuel Orrego Franco"
zona = paneles["Colegio y zona"].query("n >= 50")["tasa"].round(1)      # grupos con muy pocos estudiantes no van al post
c.lamina_barras(zona.to_dict(), {k: (c.ROJO if "rural" in k else c.AZUL) for k in zona.index}, "SABER 11 EN CALDAS",
                f"En los colegios oficiales rurales,\n{zona.max():.0f} % queda en el cuarto\nmás bajo del país",
                "Porcentaje de estudiantes con percentil nacional 25 o menos, 2020-2.", PIE, "2/4", "icfes_2.png")
plt.show()
'''),
("code", r'''
madre_tasa = paneles["Educación de la madre"]["tasa"].round(1)
c.lamina_barras(madre_tasa.to_dict(), {k: c.MORADO for k in madre_tasa.index}, "LA FAMILIA PESA",
                f"Si la madre no estudió, {madre_tasa.iloc[0]:.0f} %\nqueda abajo; si tiene posgrado, {madre_tasa.iloc[-1]:.0f} %",
                "Porcentaje en el cuarto más bajo del país\nsegún la educación de la madre.", PIE, "3/4", "icfes_3.png")
plt.show()
'''),
("code", r'''
extremos = pd.concat([municipios.nlargest(4, "tasa"), municipios.nsmallest(4, "tasa")]).sort_values("tasa")
c.lamina_ranking(extremos["tasa"], f"{datos['cole_mcpio_ubicacion'].nunique()} MUNICIPIOS",
                 f"En {extremos['tasa'].idxmax()}, {extremos['tasa'].max():.0f} %.\nEn {extremos['tasa'].idxmin()}, {extremos['tasa'].min():.0f} %",
                 "Porcentaje en el cuarto más bajo del país. Los cuatro\nmunicipios más altos y los cuatro más bajos con 80 o más estudiantes.",
                 PIE, "4/4", "icfes_4.png", resaltar=["Manizales"])
plt.show()
'''),
("md", r'''
## Bitácora

| Decisión | Alternativa | Por qué |
|---|---|---|
| Cuarto más bajo = percentil nacional 25 o menos | Otro corte | El cupo alcanza para uno de cada cuatro |
| Quitar solo los puntajes en cero | Borrar filas con faltantes | Los que dejan preguntas en blanco no se parecen a los demás |
| Llenar faltantes con la mediana del entrenamiento | La mediana de todos | Usar la prueba para llenar sería mirar el examen |
| Tramos por colegio | Por estudiante | Los compañeros de colegio se parecen |
| Captación como métrica | Exactitud | La exactitud premia decir que nadie queda abajo |

## Cambien una línea

Pongan otro departamento en `DEPARTAMENTO` (por ejemplo `"RISARALDA"` o `"CHOCO"`, como lo escribe el ICFES) y corran todo. ¿Las brechas son las mismas?
'''),
("code", r'''
procedencia()
'''),
]


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
