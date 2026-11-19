"""Piezas de presentación que comparten los cuadernos del curso.

Aquí no hay análisis: hay tablas con formato y figuras interactivas con la
paleta del curso, para que cada cuaderno muestre su dato de la misma manera.

    from taller.presentacion import ficha_del_dato, RegistroDeLimpieza, resumen_pca, nube_3d
"""
from __future__ import annotations

import pandas as pd

PALETA = ["#1D4E89", "#D7263D", "#1B998B", "#6C4AB6", "#5BC0EB", "#F49097", "#9AA5B1"]
AZUL, ROJO, VERDE, MORADO, CIELO, ROSA, GRIS = PALETA
FUENTE = dict(family="Inter, Helvetica, Arial, sans-serif", size=13, color="#1F2937")


def ficha_del_dato(tabla: pd.DataFrame, titulo: str = "", maximo: int = 60):
    """Una fila por columna de la tabla: tipo, porcentaje sin dato, valores distintos y un ejemplo."""
    resumen = pd.DataFrame({
        "tipo": tabla.dtypes.astype(str),
        "sin dato (%)": tabla.isna().mean() * 100,
        "valores distintos": tabla.nunique(),
        "ejemplo": [str(tabla[c].dropna().iloc[0])[:40] if tabla[c].notna().any() else "" for c in tabla.columns],
    })
    resumen.index.name = "columna"
    pie = f"{titulo} · {len(tabla):,} filas y {tabla.shape[1]} columnas".replace(",", ".").strip(" ·")
    if len(resumen) > maximo:
        pie += f" (se muestran las primeras {maximo})"
    return (resumen.head(maximo).style.format({"sin dato (%)": "{:.1f}"})
            .bar(subset=["sin dato (%)"], color=ROSA, vmin=0, vmax=100)
            .set_caption(pie))


def _miles(valor) -> str:
    return f"{valor:,.0f}".replace(",", ".")


class RegistroDeLimpieza:
    """Lleva la cuenta de qué hizo cada regla de limpieza: cuántas filas tocó y cuántas quedan."""

    def __init__(self, filas_iniciales: int):
        self.inicio = int(filas_iniciales)
        self.pasos = []

    def anotar(self, regla: str, decision: str, afectadas: int, quedan: int | None = None):
        anterior = self.pasos[-1]["filas que quedan"] if self.pasos else self.inicio
        self.pasos.append({"regla": regla, "decisión": decision, "filas afectadas": int(afectadas),
                           "filas que quedan": int(anterior if quedan is None else quedan)})
        return self

    def tabla(self, titulo: str = "Registro de limpieza"):
        cuadro = pd.DataFrame(self.pasos)
        cuadro.index = range(1, len(cuadro) + 1)
        cuadro["% del dato original"] = cuadro["filas que quedan"] / self.inicio * 100
        return (cuadro.style.format({"filas afectadas": _miles, "filas que quedan": _miles, "% del dato original": "{:.1f}"})
                .bar(subset=["% del dato original"], color=CIELO, vmin=0, vmax=100)
                .set_caption(f"{titulo} · filas iniciales: {self.inicio:,}".replace(",", ".")))


def resumen_pca(explicada, cargas: pd.DataFrame, titulo_cargas: str = "Cargas: qué mezcla de variables es cada componente"):
    """Barras de varianza explicada y mapa de calor de las cargas, lado a lado."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    k = min(len(explicada), 10)
    figura = make_subplots(rows=1, cols=2, column_widths=[0.38, 0.62], horizontal_spacing=0.24,
                           subplot_titles=("Varianza explicada por componente", titulo_cargas))
    figura.add_trace(go.Bar(x=[f"CP{j + 1}" for j in range(k)], y=list(explicada[:k]), marker_color=AZUL,
                            text=[f"{v:.0f} %" for v in explicada[:k]], textposition="outside",
                            hovertemplate="%{x}: %{y:.1f} %<extra></extra>"), row=1, col=1)
    figura.add_trace(go.Heatmap(z=cargas.values, x=list(cargas.columns), y=[str(i) for i in cargas.index], zmid=0,
                                colorscale=[[0, ROJO], [0.5, "#FFFFFF"], [1, AZUL]], text=cargas.round(2).values,
                                texttemplate="%{text}", colorbar=dict(thickness=12, len=0.8),
                                hovertemplate="%{y}<br>%{x}: %{z:.2f}<extra></extra>"), row=1, col=2)
    figura.update_yaxes(title_text="%", range=[0, max(explicada[:k]) * 1.25], row=1, col=1)
    figura.update_layout(template="plotly_white", height=max(340, 26 * len(cargas) + 150), font=FUENTE, showlegend=False,
                         margin=dict(l=10, r=10, t=50, b=10))
    return figura


def nube_3d(puntos: pd.DataFrame, grupo=None, valor=None, texto=None, titulo: str = "", subtitulo: str = "",
            ejes=None, colores: dict | None = None, tamano: float = 4, etiquetas: bool = False,
            flechas: pd.DataFrame | None = None, largo: float = 3.0, nombre_valor: str = "", alto: int = 640):
    """Nube de puntos en tres dimensiones que se gira con el ratón.

    `puntos` trae tres columnas. `grupo` colorea por categoría y `valor` por una
    cantidad continua. `texto` es lo que aparece al pasar el ratón. `flechas`
    dibuja las cargas de un PCA sobre la misma figura.
    """
    import plotly.graph_objects as go

    x, y, z = puntos.columns[:3]
    texto = pd.Series(puntos.index.astype(str), index=puntos.index) if texto is None else pd.Series(texto, index=puntos.index)
    modo = "markers+text" if etiquetas else "markers"
    figura = go.Figure()
    if grupo is not None:
        grupo = pd.Series(grupo, index=puntos.index)
        nombres = list(colores) if colores else list(pd.unique(grupo.dropna()))
        for i, nombre in enumerate(nombres):
            parte = puntos[grupo == nombre]
            if len(parte) == 0:
                continue
            color = colores[nombre] if colores else PALETA[i % len(PALETA)]
            figura.add_trace(go.Scatter3d(
                x=parte[x], y=parte[y], z=parte[z], mode=modo, name=f"{nombre} ({len(parte)})",
                marker=dict(size=tamano, color=color, opacity=0.9, line=dict(color="white", width=0.5)),
                text=texto[parte.index], textposition="top center", textfont=dict(size=10, color=color),
                hovertemplate="<b>%{text}</b><extra>" + str(nombre) + "</extra>"))
    else:
        marcador = dict(size=tamano, opacity=0.9, line=dict(color="white", width=0.5), color=AZUL)
        if valor is not None:
            marcador.update(color=pd.Series(valor, index=puntos.index), colorscale=[[0, CIELO], [0.45, AZUL], [0.75, MORADO], [1, ROJO]],
                            colorbar=dict(title=nombre_valor, thickness=14, len=0.6))
        figura.add_trace(go.Scatter3d(x=puntos[x], y=puntos[y], z=puntos[z], mode=modo, marker=marcador, text=texto,
                                      textposition="top center", hovertemplate="<b>%{text}</b><extra></extra>", showlegend=False))
    if flechas is not None:
        for nombre, fila in flechas.iloc[:, :3].iterrows():
            v = fila.values * largo
            figura.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]], mode="lines+text", line=dict(color=VERDE, width=6),
                                          text=["", str(nombre)], textposition="top center", textfont=dict(size=11, color=VERDE),
                                          hoverinfo="skip", showlegend=False))
    ejes = ejes or [x, y, z]
    encabezado = f"<b>{titulo}</b>" + (f"<br><sup>{subtitulo}</sup>" if subtitulo else "")
    figura.update_layout(
        template="plotly_white", height=alto, font=FUENTE, margin=dict(l=0, r=0, t=70, b=0), title=dict(text=encabezado),
        legend=dict(itemsizing="constant", bgcolor="rgba(255,255,255,0.85)", x=0.01, y=0.98),
        scene=dict(xaxis_title=ejes[0], yaxis_title=ejes[1], zaxis_title=ejes[2], camera=dict(eye=dict(x=1.55, y=-1.55, z=0.85))))
    return figura
