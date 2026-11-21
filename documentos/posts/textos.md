# Posts de LinkedIn del taller

Un post diario, en ciclos de dos en español y uno en inglés. Cada post sale de un cuaderno del curso; las láminas las genera el guion de `_profesor/posts/` (fuera de git) o la última sección del cuaderno, y el PDF del carrusel queda en esta carpeta.

| Día | Post | Idioma | Carrusel | Estado |
|---|---|---|---|---|
| Sáb 21 nov, 8:00 | El IPC del sancocho | español | `../post_sancocho/carrusel_indice_del_sancocho.pdf` | programado |
| Dom 22 nov | ¿Quién se queda atrás en el ICFES? | español | `carrusel_la_brecha_del_icfes.pdf` | listo para programar |
| Lun 23 nov | Can you predict tomorrow's price of electricity? | inglés | `carousel_colombia_power_price.pdf` | listo para programar |

---

## Dom 22 nov · ¿Quién se queda atrás en el ICFES?

Título del documento en LinkedIn: **La brecha del ICFES en Caldas**

¿Quién se queda atrás en el ICFES?

Analicé los resultados de Saber 11 de 2020 que publica el ICFES para los 9.941 estudiantes de grado once de Caldas. Miré quiénes quedaron en el cuarto más bajo del país y qué haría un modelo de aprendizaje automático si tuviera que repartir cupos de refuerzo.

Encontré tres cosas:

1. En los colegios oficiales rurales, el 37 % de los estudiantes quedó en el cuarto más bajo del país. En los oficiales urbanos fue el 21 %, y en los privados urbanos, el 15 %.

2. La educación de la madre marca la diferencia más grande. Si la madre no estudió, el 47 % queda abajo. Si tiene posgrado, el 7 %.

3. El municipio también pesa: 49 % en Belalcázar y 13 % en Villamaría.

Después puse a un modelo a repartir cupos de refuerzo para uno de cada cuatro estudiantes, solo con lo que se sabe en la inscripción. Encontró al 48 % de los que iban a quedar abajo, frente al 25 % de un sorteo. Al auditarlo apareció un sesgo. Le dio cupo a 8 de cada 10 estudiantes de jornada nocturna o sabatina, y solo alcanzó al 31 % de los de jornada diurna que lo necesitaban. Una buena métrica global puede esconder un mal reparto, y por eso el modelo se revisa grupo por grupo.

Lo trabajamos en clase en la Especialización en Estadística de la Universidad Nacional, sede Manizales. El cuaderno, con el código y los datos: https://colab.research.google.com/github/cmorregof/taller-cd-ml-2026-2/blob/main/clases/02_limpieza/02a_el_icfes.ipynb

¿Qué creen que pesa más en su departamento?

#CienciaDeDatos #Educación #ICFES #Caldas #Colombia

---

## Lun 23 nov · Can you predict tomorrow's price of electricity?

Document title on LinkedIn: **Colombia's power price, 2023 to 2026**

Can you predict tomorrow's price of electricity in Colombia?

I took four years of hourly spot prices from Colombia's wholesale power market, published by the market operator XM, together with the daily level of the country's hydro reservoirs. Then I asked three questions.

1. How volatile is it? The monthly average went from 1,529 pesos per kWh in October 2024 to 113 in June 2025. That is a thirteen-fold swing in eight months.

2. Do reservoirs explain it? Partly. When reservoirs sit below 40 %, the median daily price is 559 pesos; between 60 and 70 %, it is 381. Day to day, though, the correlation is weak (Spearman −0.17).

3. Can machine learning beat a naive forecast? Not reliably. One day ahead, "same as today" misses by 12.5 % on average, and no model beat it on the validation period. A week ahead, its error more than doubles to 27.3 %, and a random forest misses by 61.8 %.

The data needed one decision before any model. Each hour is settled several times, and two settlements of the same day can differ by a factor of almost three. I kept the most recent numbered settlement.

My graduate students in Statistics at Universidad Nacional de Colombia, Manizales, solve this same problem in class. Code and data: https://github.com/cmorregof/taller-cd-ml-2026-2

Would you sign a fixed-price contract, or stay on the spot market?

#DataScience #Forecasting #Energy #Colombia #MachineLearning
