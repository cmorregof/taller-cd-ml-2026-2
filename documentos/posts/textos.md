# Posts de LinkedIn del taller

Un post diario, en ciclos de dos en español y uno en inglés. Cada post sale de un cuaderno del curso. Las láminas las generan los guiones de `herramientas/posts/` y la portada sale de una foto de `fotos/` con `herramientas/posts/portada.py`. El PDF del carrusel queda en esta carpeta.

| Día | Post | Idioma | Carrusel | Estado |
|---|---|---|---|---|
| Sáb 21 nov, 8:00 | El IPC del sancocho | español | `../post_sancocho/carrusel_indice_del_sancocho.pdf` | publicado, con el texto anterior |
| Sáb 21 nov, 11:07 | Can you predict tomorrow's price of electricity? | inglés | `carousel_colombia_power_price.pdf` (6 láminas) | publicado |
| Dom 22 nov, 10:00 | ¿Quién se queda atrás en el ICFES? | español | `carrusel_la_brecha_del_icfes.pdf` (8 láminas, datos de 2022) | programado |

---

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

En los colegios rurales de Caldas, uno de cada tres estudiantes se queda atrás en el ICFES. En los urbanos, menos de uno de cada cinco.

Analicé los resultados de Saber 11 de 2022 que publica el ICFES en datos.gov.co para los 9.932 estudiantes de grado once de Caldas. Digo que un estudiante "se queda atrás" cuando su puntaje global es de 209 o menos, que es el 25 % más bajo del país. Calculé ese umbral con los resultados de todo Colombia, porque la tabla no trae el percentil. Antes hubo que limpiar una trampa: cada estudiante aparecía dos veces.

En la ciudad, la educación de la madre marca una distancia grande. Se queda atrás el 31,0 % de los hijos de madres que solo hicieron primaria o no estudiaron, y el 7,7 % de los hijos de madres profesionales. En el campo las dos cifras suben, al 36,0 % y al 16,5 %. Cuento como profesional a la madre que llegó a la universidad, terminara o no, o que hizo un posgrado.

Lo rural junta varias desventajas en el mismo estudiante. El 40,8 % de los estudiantes rurales tiene internet en casa, frente al 79,0 % de los urbanos. Además, el 49,0 % tiene una madre que estudió como mucho primaria, mientras que en la ciudad es el 21,1 %.

Primero hice el análisis con 2020 y apareció un dato llamativo. Ese año, en los colegios rurales, se quedaba atrás el 33,3 % de los hijos de madres profesionales, casi tanto como el resto. En 2022 esa cifra bajó a 16,5 %. No sé cuánto de la diferencia viene de la pandemia y cuánto del azar, porque son grupos de 72 y 97 estudiantes. Por eso no construí el post sobre ese dato.

Después le pedí a un modelo de aprendizaje automático que repartiera cupos de refuerzo para uno de cada cuatro estudiantes, solo con lo que se sabe al inscribirse. En colegios que no vio al entrenar, encontró al 52 % de los que iban a quedarse atrás. Un sorteo encuentra al 23 %, y repartir según la educación de la madre, al 36 %.

Al auditarlo por jornada vimos cómo los encontraba. En esos colegios de prueba, le dio cupo a todos los estudiantes de jornada nocturna o sabatina. En esas jornadas se queda atrás el 52,2 %, y la mediana de edad es 20 años, frente a 17 en la diurna. En la jornada diurna, en cambio, solo alcanzó al 35 % de quienes lo necesitaban. El modelo aprendió a reconocer la jornada, y la mayoría de los estudiantes diurnos que iban a quedarse atrás se quedó sin cupo.

Lo trabajamos en clase en la Especialización en Estadística de la Universidad Nacional, sede Manizales. El cuaderno, que reproduce cada cifra en Colab: https://colab.research.google.com/github/cmorregof/taller-cd-ml-2026-2/blob/main/posts/la_brecha_del_icfes_2022.ipynb

¿Cómo repartirían ustedes esos cupos?

#CienciaDeDatos #Educación #ICFES #Caldas #Colombia

Notas: los datos son el periodo 20224 de datos.gov.co `kgxf-xxbe` (cuaderno `posts/la_brecha_del_icfes_2022.ipynb`; láminas con `herramientas/posts/laminas_icfes_2022.py`). Las cifras de 2020 salen del cuaderno de clase (`rnvb-vnyh`, percentil del ICFES). El cuaderno de clase sigue con 2020. Se pidió acceso a DataIcfes el 21 de noviembre para comparar con 2024.

---

## Sáb 21 nov, 11:07 · Can you predict tomorrow's price of electricity?

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
