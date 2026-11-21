"""Arma la hoja de bitácora imprimible de un cuaderno: un .tex y su PDF en documentos/bitacoras/.

    python herramientas/hoja.py            -> todas las hojas registradas en HOJAS
    python herramientas/hoja.py 03a        -> solo las que empiezan por 03a

Cada hoja sale de un diccionario que vive junto a las celdas de su cuaderno, para que
las preguntas del papel y las del cuaderno no se separen. Necesita pdflatex y tcolorbox.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from molde import RAIZ
from clase03_londres import HOJA_LONDRES

HOJAS = [HOJA_LONDRES]
DESTINO = RAIZ / "documentos" / "bitacoras"

PREAMBULO = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[margin=1.5cm,top=1.3cm,bottom=1.4cm]{geometry}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{tabularx,array}
\usepackage[table]{xcolor}
\usepackage[most]{tcolorbox}
\definecolor{rojo}{HTML}{D7263D}
\definecolor{azul}{HTML}{1D4E89}
\definecolor{gris}{HTML}{9AA5B1}
\definecolor{fondo}{HTML}{FBFAF7}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\raggedright\hyphenpenalty=10000
\renewcommand{\arraystretch}{1.25}
\arrayrulecolor{gris}
\newcommand{\renglon}{\vspace{6mm}\hrule height 0.3pt\relax}
\newtcolorbox{caja}[1]{enhanced,breakable,colback=white,colframe=azul,boxrule=0.6pt,arc=1.5mm,
  left=2mm,right=2mm,top=1.5mm,bottom=2mm,fonttitle=\bfseries\small,title={#1},
  colbacktitle=azul,coltitle=white,before skip=3mm,after skip=0mm}
\newtcolorbox{cajaroja}[1]{enhanced,breakable,colback=white,colframe=rojo,boxrule=0.6pt,arc=1.5mm,
  left=2mm,right=2mm,top=1.5mm,bottom=2mm,fonttitle=\bfseries\small,title={#1},
  colbacktitle=rojo,coltitle=white,before skip=3mm,after skip=0mm}
\begin{document}
"""


def tex(texto: str) -> str:
    """Escapa lo que LaTeX no deja pasar y convierte las comillas españolas."""
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("«", r"\guillemotleft{}"), ("»", r"\guillemotright{}"), ("·", r"\textperiodcentered{}")]:
        texto = texto.replace(a, b)
    return texto


def tabla(columnas, filas, anchos=None, alto="7mm"):
    """Una tabla con encabezado y filas vacías para llenar a mano. `filas` trae el texto de la primera columna."""
    n = len(columnas)
    spec = anchos or ("|l|" + "X|" * (n - 1))
    cuerpo = [r"\begin{tabularx}{\linewidth}{" + spec + r"}\hline",
              " & ".join(r"\textbf{\small " + tex(c) + "}" for c in columnas) + r"\\\hline"]
    for primera in filas:
        cuerpo.append(r"\rule{0pt}{" + alto + "}" + tex(primera) + " & " * (n - 1) + r"\\\hline")
    cuerpo.append(r"\end{tabularx}")
    return "\n".join(cuerpo)


def armar(hoja: dict) -> str:
    p = [PREAMBULO]
    p.append(r"{\color{rojo}\rule{\linewidth}{2pt}}\\[1.5mm]")
    p.append(r"{\LARGE\bfseries " + tex(hoja["titulo"]) + r"}\hfill{\small\color{azul}\bfseries HOJA DE BITÁCORA}\\[0.5mm]")
    p.append(r"{\small " + tex(hoja["clase"]) + r" \textperiodcentered{} Taller en Ciencia de Datos y Aprendizaje Automático \textperiodcentered{} UNAL Manizales}\\[3mm]")
    p.append(r"\textbf{Nombre:} \rule{0.62\linewidth}{0.3pt}\hfill \textbf{Fecha:} \rule{0.2\linewidth}{0.3pt}\\[1mm]")
    p.append(r"{\footnotesize\color{azul}Al terminar, tómenle una foto a las dos caras y adjúntenla a la tarea de Classroom.}\\[-2mm]")

    p.append(r"\begin{cajaroja}{Apuestas. Se escriben con esfero, antes de correr la celda, y no se corrigen}")
    filas = [r"\begin{tabularx}{\linewidth}{|c|X|p{3.1cm}|p{3.1cm}|}\hline",
             r" & \textbf{\small Pregunta} & \textbf{\small Mi apuesta} & \textbf{\small Lo que salió}\\\hline"]
    for codigo, pregunta in hoja["apuestas"]:
        filas.append(r"\textbf{" + codigo + r"} & \rule{0pt}{8mm}" + tex(pregunta) + r" & & \\\hline")
    filas.append(r"\end{tabularx}")
    p.append("\n".join(filas))
    p.append(r"\end{cajaroja}")

    for titulo, columnas, primeras in hoja["tablas"]:
        p.append(r"\begin{caja}{" + tex(titulo) + "}")
        p.append(tabla(columnas, primeras))
        p.append(r"\end{caja}")

    p.append(r"\begin{caja}{Mis respuestas, con mis palabras}")
    for pregunta in hoja["preguntas"]:
        p.append(r"{\small\bfseries " + tex(pregunta) + r"}\renglon\renglon\vspace{2mm}" + "\n")
    p.append(r"\end{caja}")

    p.append(r"\begin{cajaroja}{" + tex(hoja["decision"]) + "}")
    for frase in ["1. Recomendación:", "2. Evidencia:", "3. Riesgo:"]:
        p.append(r"{\small\bfseries " + frase + r"}\renglon\renglon\vspace{2mm}" + "\n")
    p.append(r"\end{cajaroja}")

    titulo, columnas, primeras = hoja["cambio"]
    p.append(r"\begin{caja}{" + tex(titulo) + "}")
    p.append(tabla(columnas, primeras, anchos="|p{7.2cm}|X|X|"))
    p.append(r"\par\vspace{3mm}{\small\bfseries " + tex(hoja["cierre"]) + r"}\renglon\renglon")
    p.append(r"\end{caja}")

    p.append(r"\end{document}")
    return "\n".join(p)


if __name__ == "__main__":
    DESTINO.mkdir(parents=True, exist_ok=True)
    filtro = sys.argv[1] if len(sys.argv) > 1 else ""
    for hoja in HOJAS:
        if not hoja["archivo"].startswith(filtro):
            continue
        fuente = DESTINO / f"{hoja['archivo']}_bitacora.tex"
        fuente.write_text(armar(hoja), encoding="utf-8")
        for _ in range(2):
            salida = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", fuente.name],
                                    cwd=DESTINO, capture_output=True, text=True)
        if salida.returncode != 0:
            print(salida.stdout[-1500:]); raise SystemExit(f"No compiló {fuente.name}")
        for sobra in (".aux", ".log"):
            (DESTINO / f"{hoja['archivo']}_bitacora{sobra}").unlink(missing_ok=True)
        print("Hoja lista:", DESTINO / f"{hoja['archivo']}_bitacora.pdf")
