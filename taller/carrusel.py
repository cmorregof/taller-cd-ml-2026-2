"""Láminas de carrusel para LinkedIn (1080 × 1350) con estilo editorial.

Titular en Source Serif 4 y rótulos en Fira Sans Condensed, las dos con licencia libre (OFL).
Si no se pueden descargar, se usa la letra por defecto de matplotlib y todo sigue funcionando.

    from taller.carrusel import lamina_lineas, lamina_barras, lamina_pares, lamina_texto
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

RAIZ = Path(__file__).resolve().parent.parent
FUENTES = RAIZ / "datos" / "cache" / "fuentes"
URLS = {
    "SourceSerif4.ttf": "https://github.com/google/fonts/raw/main/ofl/sourceserif4/SourceSerif4%5Bopsz,wght%5D.ttf",
    "FiraSansCondensed-Regular.ttf": "https://github.com/google/fonts/raw/main/ofl/firasanscondensed/FiraSansCondensed-Regular.ttf",
    "FiraSansCondensed-Medium.ttf": "https://github.com/google/fonts/raw/main/ofl/firasanscondensed/FiraSansCondensed-Medium.ttf",
    "FiraSansCondensed-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/firasanscondensed/FiraSansCondensed-SemiBold.ttf",
}
ROJO, AZUL, MORADO, GRIS, TINTA, FONDO = "#D7263D", "#1D4E89", "#6C4AB6", "#9AA5B1", "#121212", "#FBFAF7"
SERIF, COND = "DejaVu Serif", "DejaVu Sans"


def preparar_fuentes() -> None:
    """Descarga y registra las fuentes. Se llama una vez; si falla, quedan las de siempre."""
    global SERIF, COND
    try:
        FUENTES.mkdir(parents=True, exist_ok=True)
        for nombre, url in URLS.items():
            destino = FUENTES / nombre
            if not destino.exists():
                urllib.request.urlretrieve(url, destino)
            if nombre.startswith("Fira"):
                font_manager.fontManager.addfont(str(destino))
        # La serif viene en un solo archivo variable; matplotlib necesita un archivo por peso.
        from fontTools.ttLib import TTFont
        from fontTools.varLib import instancer
        for peso, valor in [("Regular", 400), ("Bold", 700)]:
            fija = FUENTES / f"SourceSerif4-{peso}.ttf"
            if not fija.exists():
                variable = TTFont(FUENTES / "SourceSerif4.ttf")
                instancer.instantiateVariableFont(variable, {"wght": valor, "opsz": 36}, updateFontNames=True).save(fija)
            font_manager.fontManager.addfont(str(fija))
        SERIF, COND = "Source Serif 4", "Fira Sans Condensed"
    except Exception as error:
        print(f"No pude bajar las fuentes ({type(error).__name__}); uso las de matplotlib.")


def _lienzo(etiqueta: str, titular: str, bajada: str, pie: str, pagina: str, izquierda: float = 0.1,
            derecha: float = 0.86):
    """Marco común: filete, etiqueta, titular, bajada, pie y número. Se acomoda al largo del titular."""
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=FONDO)
    fig.patches.append(plt.Rectangle((0, 0.9926), 1, 0.0074, transform=fig.transFigure, color=ROJO))
    fig.text(0.067, 0.935, f" {etiqueta} ", fontsize=19, family=COND, weight="semibold", color="white",
             va="center", bbox=dict(boxstyle="square,pad=0.35", fc=ROJO, ec="none"))
    lineas_t, lineas_b = titular.count("\n") + 1, bajada.count("\n") + 1
    fig.text(0.067, 0.885, titular, fontsize=40, family=SERIF, weight="bold", color=TINTA, va="top", linespacing=1.05)
    y_bajada = 0.885 - 0.047 * lineas_t - 0.036
    fig.text(0.067, y_bajada, bajada, fontsize=21, family=SERIF, color="#464646", va="top", linespacing=1.25)
    techo = y_bajada - 0.024 * lineas_b - 0.05
    fig.text(0.067, 0.035, pie, fontsize=15, family=COND, color="#5F6B7A", va="bottom", linespacing=1.35)
    fig.text(0.933, 0.035, pagina, fontsize=16, family=COND, weight="semibold", color="#5F6B7A", ha="right", va="bottom")
    ax = fig.add_axes([izquierda, 0.15, derecha - izquierda, techo - 0.15], facecolor=FONDO)
    for lado in ["top", "right", "left"]:
        ax.spines[lado].set_visible(False)
    ax.tick_params(labelsize=17, colors="#3A3A3A", length=0)
    ax.grid(axis="y", color="#DADDE1", lw=1)
    ax.set_axisbelow(True)
    return fig, ax


def _numero(valor: float, decimales: int = 0, idioma: str = "es") -> str:
    """1.529,5 en español y 1,529.5 en inglés."""
    texto = f"{valor:,.{decimales}f}"
    return texto if idioma == "en" else texto.replace(",", "X").replace(".", ",").replace("X", ".")


def _fuente_ejes(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_family(COND)


def lamina_lineas(series: dict, colores: dict, etiqueta: str, titular: str, bajada: str, pie: str, pagina: str,
                  ruta: str | None = None, base: float | None = 100):
    """Varias series de tiempo con el nombre escrito al final de cada línea, sin leyenda.
    `base` dibuja una línea de referencia (100 en un índice); None la quita."""
    fig, ax = _lienzo(etiqueta, titular, bajada, pie, pagina, derecha=0.8)
    for nombre, serie in series.items():
        grueso = 4.2 if colores[nombre] not in (GRIS,) else 2.6
        ax.plot(serie.index, serie.values, color=colores[nombre], lw=grueso, solid_capstyle="round")
        ax.text(serie.index[-1], serie.values[-1], f"  {nombre}", color=colores[nombre], fontsize=18,
                family=COND, weight="semibold", va="center")
    if base is not None:
        ax.axhline(base, color="#3A3A3A", lw=1)
    ax.margins(x=0.02)
    _fuente_ejes(ax)
    if ruta:
        fig.savefig(ruta, dpi=100, facecolor=FONDO)
    return fig


def lamina_barras(valores: dict, colores: dict, etiqueta: str, titular: str, bajada: str, pie: str, pagina: str,
                  ruta: str | None = None, sufijo: str = " %", decimales: int = 1, idioma: str = "es"):
    """Barras verticales con el valor escrito encima de cada una."""
    fig, ax = _lienzo(etiqueta, titular, bajada, pie, pagina)
    nombres = list(valores)
    barras = ax.bar(nombres, [valores[n] for n in nombres], color=[colores[n] for n in nombres], width=0.62)
    for b, n in zip(barras, nombres):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), _numero(valores[n], decimales, idioma) + sufijo,
                ha="center", va="bottom", fontsize=26, family=SERIF, weight="bold", color=colores[n])
    ax.set_yticks([])
    ax.spines["bottom"].set_color("#3A3A3A")
    ax.grid(False)
    _fuente_ejes(ax)
    if ruta:
        fig.savefig(ruta, dpi=100, facecolor=FONDO)
    return fig


def lamina_pares(tabla, columnas: list, colores: list, etiqueta: str, titular: str, bajada: str, pie: str,
                 pagina: str, referencia: tuple | None = None, ruta: str | None = None):
    """Barras horizontales en pares (una fila por ciudad) con una línea de referencia opcional."""
    fig, ax = _lienzo(etiqueta, titular, bajada, pie, pagina, izquierda=0.17)
    filas = list(tabla.index)
    for j, (col, color) in enumerate(zip(columnas, colores)):
        pos = [i + (0.2 if j == 0 else -0.2) for i in range(len(filas))]
        ax.barh(pos, tabla[col], height=0.38, color=color)
        for p, v in zip(pos, tabla[col]):
            ax.text(v + 2, p, f"{v:.0f}", va="center", fontsize=15, family=COND, color=color, weight="semibold")
    if referencia:
        ax.axvline(referencia[1], color=ROJO, lw=2, ls=(0, (4, 3)))
        ax.text(referencia[1], -0.75, f" {referencia[0]}", color=ROJO, fontsize=16, family=COND,
                weight="semibold", va="center", ha="left",
                bbox=dict(boxstyle="square,pad=0.2", fc=FONDO, ec="none"))
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels(filas, fontsize=19)
    ax.set_ylim(-1.15, len(filas) - 0.45)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#DADDE1", lw=1)
    ax.spines["bottom"].set_visible(False)
    _fuente_ejes(ax)
    if ruta:
        fig.savefig(ruta, dpi=100, facecolor=FONDO)
    return fig


def lamina_ranking(valores, etiqueta: str, titular: str, bajada: str, pie: str, pagina: str,
                   ruta: str | None = None, resaltar: list | None = None, sufijo: str = " %"):
    """Barras horizontales de una sola serie, ordenadas, con el valor al final. `resaltar` pinta algunas en rojo."""
    fig, ax = _lienzo(etiqueta, titular, bajada, pie, pagina, izquierda=0.24)
    resaltar = resaltar or []
    colores = [ROJO if n in resaltar else AZUL for n in valores.index]
    ax.barh(range(len(valores)), valores.values, color=colores, height=0.62)
    for i, v in enumerate(valores.values):
        ax.text(v + valores.max() * 0.015, i, f"{v:.0f}{sufijo}", va="center", fontsize=18, family=COND,
                weight="semibold", color="#3A3A3A")
    ax.set_yticks(range(len(valores)))
    ax.set_yticklabels(valores.index, fontsize=20)
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    _fuente_ejes(ax)
    if ruta:
        fig.savefig(ruta, dpi=100, facecolor=FONDO)
    return fig


def lamina_texto(etiqueta: str, titular: str, puntos: list, pie: str, pagina: str, ruta: str | None = None):
    """Lámina de cierre: un titular y tres o cuatro líneas de método."""
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=FONDO)
    fig.patches.append(plt.Rectangle((0, 0.9926), 1, 0.0074, transform=fig.transFigure, color=ROJO))
    fig.text(0.067, 0.935, f" {etiqueta} ", fontsize=19, family=COND, weight="semibold", color="white",
             va="center", bbox=dict(boxstyle="square,pad=0.35", fc=ROJO, ec="none"))
    fig.text(0.067, 0.885, titular, fontsize=44, family=SERIF, weight="bold", color=TINTA, va="top", linespacing=1.05)
    y = 0.70
    for punto in puntos:
        fig.patches.append(plt.Rectangle((0.067, y - 0.016), 0.012, 0.0096, transform=fig.transFigure, color=ROJO))
        fig.text(0.1, y, punto, fontsize=24, family=SERIF, color="#2B2B2B", va="top", linespacing=1.3)
        y -= 0.05 + 0.036 * punto.count("\n") + 0.05
    fig.text(0.067, 0.035, pie, fontsize=15, family=COND, color="#5F6B7A", va="bottom", linespacing=1.35)
    fig.text(0.933, 0.035, pagina, fontsize=16, family=COND, weight="semibold", color="#5F6B7A", ha="right", va="bottom")
    if ruta:
        fig.savefig(ruta, dpi=100, facecolor=FONDO)
    return fig
