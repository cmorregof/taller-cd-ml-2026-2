# IPC del DANE para el índice del sancocho

Dos tablas con estadísticas públicas del DANE, citadas fila por fila en la columna `url_fuente`.

- `ipc_total_indice.csv`: índice del IPC total nacional, serie de empalme, base diciembre de 2018 = 100, enero de 2013 a diciembre de 2024. Sale del anexo «Índices, series de empalme» de octubre de 2026.
- `ipc_alimentos_variacion_mensual.csv`: variación mensual (%) del IPC de alimentos por ciudad y nacional, enero de 2013 a diciembre de 2024, leída de los anexos y boletines mensuales del DANE. Hasta 2018 el grupo se llama «Alimentos» (base 2008); desde 2019, «Alimentos y bebidas no alcohólicas» (base 2018).

Advertencias:

- De junio a diciembre de 2016 el DANE no publicó la tabla por ciudades en el boletín. Esos meses no están para las ciudades; el cuaderno los llena con la variación nacional y lo dice.
- Quibdó, San Andrés y «Otras áreas urbanas» solo aparecen desde 2019.
- Encadenadas, las variaciones nacionales reproducen la inflación anual de alimentos publicada por el DANE (por ejemplo, 4,69 % en 2014, 17,22 % en 2021 y 27,82 % en 2022).
