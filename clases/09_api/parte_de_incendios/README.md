# El parte de incendios

Gemelo de la clase 9 · Taller en Ciencia de Datos y Aprendizaje Automático · UNAL Manizales · 2026-II

## El encargo

Imagínense la sala de crisis de una entidad de gestión del riesgo. Cada lunes alguien pregunta cuántos focos de calor hubo en un departamento y a dónde conviene mandar las brigadas esta semana. Hoy esa respuesta la arma una persona a mano con una hoja de cálculo.

Ustedes van a construir **una API que entrega el parte de focos de calor por departamento**. Es trabajo autónomo. El código se lo piden a Claude y ustedes lo aceptan o lo rechazan con las pruebas que ya están escritas en este repositorio.

Las tres preguntas del proyecto:

1. **Predecir.** ¿Cuántos focos de calor habrá la próxima semana en cada departamento?
2. **Explicar.** ¿Qué pesa más en esa cifra, la época del año o lo que pasó la semana anterior?
3. **Decidir.** ¿A qué departamentos mandas las brigadas la próxima semana?

## El dato

NASA FIRMS publica los focos de calor que detecta el sensor MODIS de los satélites Terra y Aqua. Cada fila es un píxel de cerca de un kilómetro donde el satélite vio una anomalía térmica. `datos.py` ya trae los focos de Colombia de 2016 a 2025 y les agrega el departamento.

Tres cosas que deben saber antes de contar:

- Un foco no es un incendio. Un incendio grande produce muchos focos y una quema pequeña bajo nubes no produce ninguno.
- La columna `type` distingue el presunto incendio de vegetación (0) de volcanes, fuentes fijas y mar adentro. **El parte cuenta solo los de tipo 0 que cayeron en un departamento.**
- Los límites departamentales vienen de OpenStreetMap, vía geoBoundaries. No son la cartografía oficial.

## Lo que hay en el repositorio

| Archivo | Estado |
|---|---|
| `datos.py` | Escrito. No se cambia. |
| `pruebas_aceptacion.py` | Escrito. No se cambia. Es el contrato. |
| `main.py` | Casi vacío. Aquí va su API. |
| `requirements.txt`, `Procfile` | Listos para Railway. |

## Criterios de aceptación

La entrega se acepta cuando `python pruebas_aceptacion.py` termina con «10 de 10 criterios cumplidos».

| # | Ruta | Lo que debe hacer |
|---|---|---|
| 1 | `GET /salud` | Responde `{"estado": "ok", "focos": N, "desde": "AAAA-MM-DD", "hasta": "AAAA-MM-DD"}`, donde `N` cuenta los focos de tipo 0 con departamento. |
| 2 | `GET /departamentos` | La lista de los 33 nombres, en orden alfabético (la misma de `datos.departamentos()`). |
| 3 | `GET /parte/{departamento}?desde=&hasta=` | `departamento`, `desde`, `hasta`, `focos`, `frp_total`, `confianza_media` y `por_dia`, una lista de `{"fecha", "focos"}` solo con los días que tuvieron focos. Las dos fechas se incluyen. |
| 4 | `…&confianza_minima=80` | Parámetro opcional, de 0 a 100, 0 por defecto. Cuenta solo focos con esa confianza o más. |
| 5 | `GET /parte/bolivar` | El nombre se entiende sin tildes y sin importar mayúsculas. La respuesta trae el nombre oficial. |
| 6 | `GET /parte/Narnia` | 404, con un mensaje en `detail` que diga dónde consultar los nombres válidos. |
| 7 | Fechas | Una fecha mal escrita, o `desde` posterior a `hasta`, responde 422. |
| 8 | `GET /ranking?desde=&hasta=&n=5` | Los `n` departamentos con más focos, de mayor a menor, como `{"departamento", "focos"}`. `n` va de 1 a 33; fuera de ese rango, 422. |
| 9 | `GET /brigadas?fecha=&n=5` | Para la semana de siete días que empieza en `fecha`: `metodo` (una frase que explique cómo se calcula), `semana_desde`, `semana_hasta` y `departamentos`, una lista de `{"departamento", "esperado"}` de mayor a menor. |
| 10 | `/brigadas` y el tiempo | El cálculo usa **solo focos anteriores a `fecha`**. Si no hay al menos un año de datos antes de `fecha`, responde 422. |

Además, y esto no lo revisa el programa:

- En el repositorio no hay llaves ni contraseñas.
- `/docs` describe cada ruta con una frase en español.
- El cuaderno `09b_el_parte_de_incendios.ipynb` justifica con números el método de `/brigadas`.

## Cómo se trabaja

1. Copien esta carpeta a un repositorio suyo de GitHub.
2. `pip install -r requirements.txt` y `python pruebas_aceptacion.py`. Al comienzo fallan las diez.
3. Pídanle a Claude una ruta a la vez. Peguen el criterio de la tabla tal como está. Corran las pruebas después de cada ruta.
4. Lean lo que Claude escribió antes de aceptarlo. Las pruebas no lo ven todo.
5. Cuando pasen las diez, despliéguenla en Railway como hicimos en clase con el noticiero.

## Entrega

La dirección de su repositorio, la dirección pública de su API y el cuaderno corrido.
