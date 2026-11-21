# Posts de LinkedIn del taller

Un post diario, en ciclos de dos en español y uno en inglés. Cada post sale de un cuaderno del curso. Las láminas las generan los guiones de `herramientas/posts/` y la portada sale de una foto de `fotos/` con `herramientas/posts/portada.py`. El PDF del carrusel queda en esta carpeta.

| Día | Post | Idioma | Carrusel | Estado |
|---|---|---|---|---|
| Sáb 21 nov, 8:00 | El IPC del sancocho | español | `../post_sancocho/carrusel_indice_del_sancocho.pdf` | programado con el texto anterior |
| Dom 22 nov | ¿Quién se queda atrás en el ICFES? | español | `carrusel_la_brecha_del_icfes.pdf` (7 láminas) | listo para programar |
| Lun 23 nov | Can you predict tomorrow's price of electricity? | inglés | `carousel_colombia_power_price.pdf` (6 láminas) | listo para programar |

|---

## Sáb 21 nov · El IPC del sancocho

Título del documento en LinkedIn: **El índice del sancocho**

El sancocho de Manizales subió menos que la comida durante doce años. La excepción fue 2022.

Con los precios mayoristas que el DANE publica en el SIPSA armé un índice con la lógica del IPC, pero con una sola canasta. Es un almuerzo de sancocho para seis, con arroz, banano y una ensalada de zanahoria, tomate y lechuga. Cada mes tomé la variedad más barata de cada ingrediente, porque así se compra en la galería. El espinazo es el único que no tiene reemplazo.

En enero de 2013, los ingredientes de ese almuerzo costaban 18.181 pesos en la central mayorista de Manizales. En diciembre de 2024 costaban 37.112, es decir, un 104 % más. En el mismo periodo, el IPC de alimentos de Manizales subió 137 % y el IPC total del país, 85 %.

El espinazo explica la mitad del aumento, porque es lo que más pesa en la olla. Aun así, en porcentaje subió menos que el resto de la receta, que aumentó 119,5 %.

En 2022 todo se aceleró. El sancocho subió 33 %, más que los alimentos de la ciudad (27 %) y que el IPC total (13 %). El espinazo subió 22 % ese año. En cambio, el kilo de plátano verde pasó de 1.183 a 2.804 pesos y el arroz subió 73 %.

Al armar el índice aparecieron meses sin precio, y la manera de rellenarlos cambia la conclusión. Para medir cuánto, borré a propósito el espinazo de 2022 y lo rellené de cuatro formas. Con el promedio histórico, la inflación del sancocho en 2022 baja de 33 % a 9 %, porque el relleno usa precios de toda la década. Con interpolación queda en 33 %.

Hoy abrimos con este ejercicio el Taller de Ciencia de Datos de la Especialización en Estadística de la Universidad Nacional, sede Manizales. El cuaderno, con el código y los datos: https://colab.research.google.com/github/cmorregof/taller-cd-ml-2026-2/blob/main/clases/01_formular/01c_el_ipc_del_sancocho.ipynb

¿Qué plato usarían para medir la inflación de su ciudad?

#CienciaDeDatos #Estadística #Inflación #Manizales #Colombia

Nota: el IPC de alimentos de Manizales tiene huecos en 2016, rellenos con la variación nacional (ver el cuaderno).

---

## Dom 22 nov · ¿Quién se queda atrás en el ICFES?

Título del documento en LinkedIn: **La brecha del ICFES en Caldas**

En Caldas, un estudiante de colegio rural cuya madre es profesional se queda atrás en el ICFES tan a menudo como un estudiante urbano cuya madre solo hizo primaria. En los dos casos le pasa a uno de cada tres.

Analicé los resultados de Saber 11 de 2020 que publica el ICFES para los 9.941 estudiantes de grado once de Caldas. Comparé cada puntaje con los de todo el país. Digo que un estudiante "se queda atrás" cuando su puntaje está entre el 25 % más bajo de Colombia. Cuento como profesional a la madre que llegó a la universidad, terminara o no, o que hizo un posgrado.

En los colegios urbanos, la educación de la madre marca una distancia grande. Si la madre es profesional, se queda atrás el 8,4 %. Si solo hizo primaria o no estudió, la cifra sube a 33,3 %. En los colegios rurales esa ventaja casi desaparece. Se queda atrás el 33,3 % de los hijos de madres profesionales y el 39,6 % de los hijos de madres con primaria o menos. El dato rural de madres profesionales sale de solo 72 estudiantes, así que hay que leerlo con cuidado.

Lo rural junta varias desventajas en el mismo estudiante. Solo el 32,1 % de los estudiantes rurales tiene internet en casa, frente al 78,6 % de los urbanos. Además, el 55,3 % tiene una madre que estudió como mucho primaria, mientras que en la ciudad es el 24,8 %.

Después le pedí a un modelo de aprendizaje automático que repartiera cupos de refuerzo para uno de cada cuatro estudiantes, solo con lo que se sabe al inscribirse. El modelo encontró al 48 % de los que iban a quedarse atrás. Un sorteo encuentra al 25 %, y repartir según la educación de la madre, al 39 %.

Al auditarlo por grupos vimos dónde los encontraba. Le dio cupo al 81 % de los estudiantes de jornada nocturna o sabatina, y allí alcanzó al 88 % de los que lo necesitaban. En la jornada diurna solo alcanzó al 31 %. Los estudiantes nocturnos y sabatinos son mayores: su mediana de edad es 20 años y el 43 % pasa de 20, mientras que en la diurna la mediana es 17. El modelo aprendió a reconocer la jornada, y la mayoría de los estudiantes diurnos que iban a quedarse atrás se quedó sin cupo.

Lo trabajamos en clase en la Especialización en Estadística de la Universidad Nacional, sede Manizales. El cuaderno, con el código y los datos: https://colab.research.google.com/github/cmorregof/taller-cd-ml-2026-2/blob/main/clases/02_limpieza/02a_el_icfes.ipynb

¿Cómo repartirían ustedes esos cupos?

#CienciaDeDatos #Educación #ICFES #Caldas #Colombia

---

## Lun 23 nov · Can you predict tomorrow's price of electricity?

Document title on LinkedIn: **Colombia's power price, 2023 to 2026**

Colombia's most expensive month for electricity came six months after its driest.

I took four years of hourly spot prices from Colombia's wholesale power market, published by the market operator XM, together with the daily level of the country's hydro reservoirs. The usual explanation is simple. When reservoirs drop, thermal plants switch on and the price goes up.

The data only partly agrees. In April 2024, reservoirs reached 31 %, their lowest monthly level in the period, and power averaged 749 pesos per kWh. In May, with reservoirs still at 38 %, the average fell to 292. The most expensive month came in October 2024, at 1,529 pesos, when reservoirs had recovered to 53 %. In September 2023, with reservoirs at 77 %, the price was already above 1,000.

Allowing for a lag of up to three months does not rescue the story. The monthly Spearman correlation between reservoir level and price stays between −0.15 and −0.21.

The price also has a daily rhythm. The median at 7 pm is 621 pesos, about twice the 304 at 3 am.

Before any model, the data needed one decision. XM settles each hour several times, and two settlements of the same day can differ by a factor of almost three. I kept the most recent numbered settlement.

Then I asked whether machine learning can forecast tomorrow's price. One day ahead, "same as today" misses by 12.5 % on average, and no model beat it on the validation period. A week ahead, the naive error rises to 27.3 %, and a random forest misses by 61.8 %.

My graduate students in Statistics at Universidad Nacional de Colombia, Manizales, work through the same problem in class. Code and data: https://github.com/cmorregof/taller-cd-ml-2026-2

If you work in energy, what moved the price in October 2024?

#DataScience #Forecasting #Energy #Colombia #MachineLearning
