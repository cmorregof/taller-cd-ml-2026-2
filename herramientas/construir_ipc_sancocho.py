"""Genera el cuaderno «El IPC del sancocho» (clase 2, primera hora).

Es un cuaderno para mostrar y correr en vivo: no tiene huecos. La plomería vive en
taller/canasta.py y taller/carrusel.py; aquí quedan las decisiones a la vista.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno

CELDAS = [
("md", r'''
# El IPC del sancocho

**Taller en Ciencia de Datos y Aprendizaje Automático** · Especialización en Estadística · UNAL Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Ayer pronosticamos el costo de una olla de sancocho. Hoy usamos los mismos datos para algo más grande: medir cuánto subió la comida en Manizales, con el mismo método que usa el DANE para el IPC, y comparar nuestro resultado con el oficial. En el camino aparecen datos que faltan, y vamos a decidir qué hacer con ellos. Al final, el cuaderno produce un carrusel listo para LinkedIn.

| Ficha | |
|---|---|
| **Objetivo pedagógico** | Construir un índice de precios con datos reales, decidir con evidencia cómo tratar los datos faltantes y convertir el resultado en una pieza de comunicación. |
| **Enfoque** | Ciencia de Datos. |
| **Resultados de aprendizaje** | 2 (seleccionar y preparar datos), 3 (flujo reproducible) y 5 (comunicar a quien no es técnico). |
| **Uso de IA** | Pueden pedirle a Claude que explique cualquier línea. Deben poder explicar sin ayuda por qué falta un dato antes de imputarlo, qué método de imputación escogimos y qué riesgo tiene. |
| **Entregable** | Este cuaderno corrido en su portafolio y un post propio con una de las gráficas. |
'''),
("code", ARRANQUE),
("md", r'''
## 1. Cómo se mide la inflación

El DANE escoge una canasta de bienes, fija las cantidades y cada mes vuelve a preguntar los precios. El índice compara lo que cuesta hoy la misma canasta con lo que costaba en un mes de referencia. Esa fórmula se llama índice de Laspeyres:

$$
I_t = 100 \cdot \frac{\sum_i q_i \, p_{i,t}}{\sum_i q_i \, p_{i,0}}
$$

Aquí $q_i$ es la cantidad fija del producto $i$, $p_{i,t}$ su precio en el mes $t$ y $p_{i,0}$ su precio en el mes base. Usamos la base del DANE, diciembre de 2018, así que $I_t = 150$ significa que la canasta cuesta 50 % más que en diciembre de 2018.

**Nuestra canasta:** un almuerzo para seis en Manizales.

| Plato | Ingredientes (kg) |
|---|---|
| Sancocho | espinazo de cerdo 1,5 · yuca 1 · papa 1 · mazorca 0,9 · plátano verde 1,4 |
| Acompañantes | arroz 0,75 · banano 1 |
| Ensalada | zanahoria 0,5 · tomate 0,5 · lechuga 0,5 |

**El criterio colombiano.** Si en la plaza hay tres variedades de papa, uno compra la más barata. Por eso, cada mes tomamos la variedad más barata de cada ingrediente. El espinazo es la excepción, porque no tiene sustituto.

### Predicción escrita

- Entre 2013 y 2024, ¿cuánto subió este almuerzo? **Mi apuesta:**
- ¿Subió más o menos que la comida en general (IPC de alimentos)? ¿Y que el costo de vida (IPC total)? **Mi apuesta:**
'''),
("code", r'''
from taller.canasta import RECETA, precios_ciudad, indice, ipc_total, ipc_alimentos, indice_colombia

sipsa = descargar("sipsa")
precios, variedad = precios_ciudad(sipsa, "Manizales")
precios.tail()
'''),
("md", r'''
La tabla `variedad` guarda cuál variedad fue la más barata cada mes. Veamos tres ingredientes:
'''),
("code", r'''
for ingrediente in ["arroz", "tomate", "zanahoria"]:
    print(ingrediente, variedad[ingrediente].value_counts().to_dict())
'''),
("md", r'''
## 2. Los datos que faltan

Antes de llenar un hueco hay que preguntar por qué está ahí. Un dato puede faltar por tres razones, y cada una pide un tratamiento distinto:

1. **Falta al azar.** Se perdió una planilla. La ausencia no dice nada sobre el precio.
2. **Falta por algo que sí vemos.** Por ejemplo, el mercado no reportó porque era festivo. Otras columnas ayudan a llenarlo.
3. **Falta por su propio valor.** El producto no llegó a la plaza porque escaseaba, y justo entonces estaba caro. Esta es la peligrosa, porque cualquier relleno subestima el precio.

En estadística se llaman MCAR, MAR y MNAR. Contemos los huecos de Manizales:
'''),
("code", r'''
print(precios.isna().sum())
pd.DataFrame({"precio": precios["zanahoria"], "variedad": variedad["zanahoria"]}).loc["2021-09-01":"2022-03-01"]
'''),
("md", r'''
La zanahoria parece desaparecer en noviembre de 2021, pero no faltaba: el DANE le cambió el nombre a «Zanahoria bogotana». El criterio de la variedad más barata lo resolvió solo, porque toma la que haya. Lección: muchos «faltantes» son cambios de nombre, de código o de formato, y se arreglan sin imputar nada.

Queda un hueco real, enero de 2022. Antes de llenarlo, escojamos el método con evidencia.
'''),
("md", r'''
## 3. El experimento: esconder lo que sí sabemos

Para comparar métodos de imputación necesitamos saber la respuesta correcta. El truco: tomamos el espinazo, que tiene los 144 meses, escondemos todo 2022 y lo rellenamos de cuatro maneras.

- **Promedio histórico:** cada hueco recibe el promedio de toda la serie.
- **Último valor:** se repite el último precio conocido.
- **Interpolación:** se traza una línea recta entre el último precio antes del hueco y el primero después.
- **Ciudad vecina:** se usa el precio de Medellín, ajustado por la diferencia típica entre las dos ciudades en 2021.

Medimos dos cosas: qué tan cerca queda cada relleno del precio real (MAPE) y cuánto cambia la conclusión, que aquí es la inflación del almuerzo en 2022.
'''),
("code", r'''
real = precios["espinazo"]
hueco = (real.index >= "2022-01-01") & (real.index <= "2022-12-01")
con_hueco = real.mask(hueco)

vecina, _ = precios_ciudad(sipsa, "Medellín")
razon = (con_hueco / vecina["espinazo"]).loc["2021-01-01":"2021-12-01"].median()

metodos = {
    "promedio histórico": con_hueco.fillna(con_hueco.mean()),
    "último valor": con_hueco.ffill(),
    "interpolación": con_hueco.interpolate(),
    "ciudad vecina": con_hueco.fillna(vecina["espinazo"] * razon),
}
'''),
("code", r'''
def mape(real, estimado):
    return float((abs(real - estimado) / real).mean() * 100)

def inflacion_2022(serie_espinazo):
    canasta = precios.assign(espinazo=serie_espinazo, zanahoria=precios["zanahoria"].interpolate())
    i = indice(canasta)
    return 100 * (i["2022-12-01"] / i["2021-12-01"] - 1)

resultado = pd.DataFrame({
    nombre: {"MAPE en los meses escondidos (%)": mape(real[hueco], serie[hueco]),
             "inflación del almuerzo en 2022 (%)": inflacion_2022(serie)}
    for nombre, serie in metodos.items()}).T
resultado.loc["dato real"] = [0, inflacion_2022(real)]
resultado.round(1)
'''),
("code", r'''
fig, ax = plt.subplots(figsize=(10, 4))
ventana = slice("2021-06-01", "2023-06-01")
ax.plot(real[ventana], color="black", lw=2.5, label="real")
for (nombre, serie), color in zip(metodos.items(), PALETA[1:]):
    ax.plot(serie.where(hueco)[ventana], color=color, lw=2, label=nombre)
ax.axvspan("2022-01-01", "2022-12-01", color="#EEF1F4")
ax.set(title="Espinazo en Manizales: el año escondido y cuatro maneras de llenarlo", ylabel="pesos por kilo")
ax.legend(frameon=False, ncol=3); plt.show()
'''),
("md", r'''
**Lo que dice el experimento.** El promedio histórico parece inofensivo y es el peor: rellena 2022 con precios de 2013 a 2024 y se come casi toda la inflación de ese año. El último valor congela el precio y también la subestima. La interpolación y la ciudad vecina quedan cerca del dato real.

**Los riesgos de imputar**, aunque el método sea bueno:

- **Aplana.** La interpolación traza una recta y borra los saltos de cada mes. La serie se ve más tranquila de lo que fue.
- **Usa el futuro.** Para interpolar enero hace falta febrero. Sirve para describir el pasado, pero no se puede usar para pronosticar, porque en el momento del pronóstico febrero no existe.
- **Contagia.** La ciudad vecina copia los choques de Medellín, aunque Manizales haya vivido otros.
- **No arregla lo que falta por su valor.** Si el dato faltaba porque había escasez, todos los métodos subestiman.

**Decisión.** Para el único hueco real, la zanahoria de enero de 2022, usamos interpolación. Es un solo mes, entre dos meses conocidos, y la zanahoria pesa poco en la canasta. Lo anotamos en la bitácora.
'''),
("code", r'''
precios["zanahoria"] = precios["zanahoria"].interpolate()
print("Huecos que quedan:", int(precios.isna().sum().sum()))
'''),
("md", r'''
## 4. El sancocho contra el IPC

Ahora sí, el índice. Lo comparamos con dos series oficiales del DANE, guardadas en `datos/ipc/` con la ruta de cada anexo:

- **IPC de alimentos de Manizales:** la canasta completa de comida que mide el DANE en la ciudad.
- **IPC total de Colombia:** el costo de vida.

El DANE publica la variación mensual de alimentos por ciudad. Al encadenarla se obtiene un índice. De junio a diciembre de 2016 no publicó la tabla por ciudades, así que esos siete meses toman la variación nacional. Otra imputación, y también la decimos.
'''),
("code", r'''
sancocho = indice(precios)
alimentos, _ = ipc_alimentos()
total = ipc_total()

def cambio(serie, desde, hasta):
    return 100 * (serie[hasta] / serie[desde] - 1)

comparacion = pd.DataFrame({
    "2013 a 2024 (%)": [cambio(s, "2013-01-01", "2024-12-01") for s in (sancocho, alimentos["Manizales"], total)],
    "solo 2022 (%)": [cambio(s, "2021-12-01", "2022-12-01") for s in (sancocho, alimentos["Manizales"], total)],
}, index=["sancocho", "IPC de alimentos de Manizales", "IPC total de Colombia"])
comparacion.round(1)
'''),
("code", r'''
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(sancocho, lw=2.5, label="sancocho")
ax.plot(alimentos["Manizales"], lw=2, label="IPC de alimentos, Manizales")
ax.plot(total, lw=2, ls="--", label="IPC total, Colombia")
ax.axhline(100, color="black", lw=0.6)
ax.set(title="Índice, diciembre de 2018 = 100", ylabel="índice"); ax.legend(frameon=False); plt.show()
print(f"Correlación de las variaciones anuales, sancocho y alimentos: "
      f"{sancocho.pct_change(12).corr(alimentos['Manizales'].pct_change(12)):.2f}")
'''),
("md", r'''
## 5. ¿Qué ingrediente empujó?

Como el costo es una suma, el alza total se reparte exacta entre ingredientes:

$$
C_T - C_0 = \sum_i q_i \,(p_{i,T} - p_{i,0}).
$$
'''),
("code", r'''
cantidades = pd.Series(RECETA)
aporte = (precios.iloc[-1] - precios.iloc[0]) * cantidades
aporte = (aporte / aporte.sum() * 100).sort_values()
peso_2024 = (precios.iloc[-1] * cantidades / (precios.iloc[-1] * cantidades).sum() * 100)

aporte.plot.barh(figsize=(8, 4), color=PALETA[0])
plt.title("De cada 100 pesos que subió el almuerzo, ¿cuántos puso cada ingrediente?"); plt.xlabel("pesos"); plt.show()
print(f"El espinazo pesa {peso_2024['espinazo']:.0f} % del costo en diciembre de 2024.")
'''),
("md", r'''
## 6. Colombia: ¿hasta dónde se puede imputar?

Repetimos el índice en las 71 plazas del SIPSA. No todas reportan los diez ingredientes. Por eso armamos tres versiones según cuántos datos reales tiene cada plaza:

- plazas con los diez ingredientes en al menos el 90 % de los meses;
- plazas con al menos el 75 %;
- todas, imputando lo que falte.

La función `indice_colombia` interpola los huecos internos, completa los bordes con el cambio mediano del país y usa los mismos pesos en todas las plazas. Así comparamos cambios, no niveles.
'''),
("code", r'''
indices, imputado, cobertura = indice_colombia(sipsa)
grupos = {
    "≥ 90 % de los meses": cobertura.index[(cobertura >= 0.9).all(axis=1)].tolist(),
    "≥ 75 % de los meses": cobertura.index[(cobertura >= 0.75).all(axis=1)].tolist(),
    "todas las plazas": cobertura.index.tolist(),
}
fig, ejes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
for ax, (nombre, plazas) in zip(ejes, grupos.items()):
    ax.plot(indices[plazas], color="#9AA5B1", lw=0.6, alpha=0.6)
    ax.plot(indices[plazas].median(axis=1), color=PALETA[0], lw=2.5, label="sancocho, mediana")
    ax.plot(alimentos["Nacional"], color=PALETA[3], lw=2, label="IPC de alimentos")
    ax.set_title(f"{nombre}: {len(plazas)} plazas, {imputado[plazas].mean() * 100:.0f} % imputado")
ejes[0].legend(frameon=False); plt.tight_layout(); plt.show()
'''),
("md", r'''
Miren la tercera gráfica. Con la mitad de los datos imputados, las plazas se parecen entre sí porque sus huecos se llenaron con el cambio del país. La imputación borró la diversidad que queríamos medir. Por eso el post usa las plazas con datos de verdad.
'''),
("md", r'''
## 7. El post

Las láminas salen del mismo cuaderno, con las cifras calculadas arriba. Si mañana cambia un dato, el post cambia solo.
'''),
("code", r'''
from taller import carrusel as c
c.preparar_fuentes()
PIE = "Fuentes: DANE, SIPSA (precios mayoristas) e IPC. Cálculos propios.\nCarlos Manuel Orrego Franco"

c.lamina_lineas({"Sancocho": sancocho, "Alimentos": alimentos["Manizales"], "Costo de vida\n(Colombia)": total},
                {"Sancocho": c.AZUL, "Alimentos": c.MORADO, "Costo de vida\n(Colombia)": c.GRIS},
                "MANIZALES, 2013 A 2024", "El sancocho le sigue el paso\nal IPC de alimentos",
                "Índice, diciembre de 2018 = 100. Diez precios de plaza\nsiguen a la canasta completa del DANE.",
                PIE, "2/5", "lamina_2.png")
plt.show()
'''),
("code", r'''
anio_2022 = comparacion["solo 2022 (%)"].set_axis(["Sancocho", "Alimentos\nManizales", "Costo de vida\nColombia"])
c.lamina_barras(anio_2022.to_dict(), dict(zip(anio_2022.index, [c.AZUL, c.MORADO, c.GRIS])),
                "2022, EL AÑO DEL GOLPE",
                f"En 2022 el almuerzo subió\n{anio_2022.iloc[0]:.0f} %, más del doble\nque el costo de vida",
                "Variación entre diciembre de 2021 y diciembre de 2022.", PIE, "3/5", "lamina_3.png")
plt.show()
'''),
("code", r'''
ciudades = grupos["≥ 75 % de los meses"]
tabla = pd.DataFrame({"Sancocho": [cambio(indices[x], "2013-01-01", "2024-12-01") for x in ciudades],
                      "Alimentos": [cambio(alimentos[x], "2013-01-01", "2024-12-01") for x in ciudades]},
                     index=ciudades).sort_values("Sancocho")
costo_vida = cambio(total, "2013-01-01", "2024-12-01")
c.lamina_pares(tabla, ["Sancocho", "Alimentos"], [c.AZUL, c.MORADO], f"{len(ciudades)} CIUDADES",
               f"El mismo almuerzo subió\n{tabla.Sancocho.min():.0f} % en {tabla.Sancocho.idxmin()} y "
               f"{tabla.Sancocho.max():.0f} % en {tabla.Sancocho.idxmax()}",
               "Cambio entre enero de 2013 y diciembre de 2024 (%).\nAzul: sancocho, con los mismos pesos en todas.\n"
               "Morado: IPC de alimentos de cada ciudad.", PIE, "4/5",
               referencia=(f"Costo de vida, Colombia: {costo_vida:.0f} %", costo_vida), ruta="lamina_4.png")
plt.show()
'''),
("md", r'''
## Para ir más allá: las plazas en tres dimensiones

Cada plaza es un punto. Dos plazas quedan cerca si su sancocho subió y bajó en los mismos meses. El color dice cuánto de su serie fue imputado. Giren la nube: las plazas casi todas imputadas se amontonan juntas, porque comparten el mismo relleno.
'''),
("code", r'''
from sklearn.decomposition import PCA
from taller.presentacion import nube_3d

cambios = np.log(indices).diff().dropna().T                     # una fila por plaza, una columna por mes
pca = PCA(n_components=3).fit(cambios)
puntos = pd.DataFrame(pca.transform(cambios), index=cambios.index, columns=["CP1", "CP2", "CP3"])
nivel = pd.cut(imputado, [-0.01, 0.1, 0.5, 1], labels=["casi nada imputado", "algo imputado", "mayoría imputada"])
nube_3d(puntos, grupo=nivel.astype(str), titulo="Las plazas del SIPSA según cómo se movió su sancocho",
        subtitulo="Arrastren para girar. El color es la fracción de datos imputados.",
        ejes=[f"CP{j + 1} ({v:.0f} %)" for j, v in enumerate(pca.explained_variance_ratio_ * 100)]).show()
'''),
("md", r'''
## Bitácora

| Decisión | Alternativa | Por qué |
|---|---|---|
| Variedad más barata de cada mes | Una variedad fija | Así compra la gente, y resuelve los cambios de nombre |
| Espinazo sin sustituto | Cualquier corte de cerdo | El lomo no reemplaza al espinazo en un sancocho |
| Interpolar la zanahoria de enero de 2022 | Promedio, último valor, ciudad vecina | Ganó en el experimento; es un mes entre dos conocidos |
| IPC de alimentos de junio a diciembre de 2016 con la variación nacional | Dejar el hueco | El DANE no publicó la tabla por ciudades esos meses |
| Post con las plazas de datos reales | Todas las plazas | Imputar la mitad borra las diferencias entre ciudades |
'''),
("code", r'''
procedencia()
'''),
]

destino = RAIZ / "clases" / "01_formular"
nbf.write(cuaderno(CELDAS, False), destino / "01c_el_ipc_del_sancocho.ipynb")
print("Cuaderno escrito.")
