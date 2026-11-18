"""Piezas comunes a todos los cuadernos del curso.

Cada clase tiene su generador `construir_claseNN.py`, que importa de aquí la celda
de arranque y la función `cuaderno`. En el código de las celdas, lo que va entre
« » es un hueco: en el cuaderno del estudiante sale como ____ y en la solución
sale lleno. Los comentarios `HUECO n:` se renumeran solos en orden de aparición.
"""
import re
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parent.parent

REPO = "https://github.com/cmorregof/taller-cd-ml-2026-2.git"
RAIZ = Path(__file__).resolve().parent.parent

ARRANQUE = f'''# Celda de arranque. No hay que cambiarle nada: trae las herramientas del curso.
import pathlib, subprocess, sys
raiz = next((p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "taller" / "datos.py").exists()), None)
if raiz is None:
    subprocess.run(["git", "clone", "-q", "{REPO}"], check=True)
    raiz = pathlib.Path("taller-cd-ml-2026-2")
sys.path.insert(0, str(raiz))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from taller import descargar, procedencia

plt.rcParams.update({{"figure.figsize": (9, 4), "axes.spines.top": False, "axes.spines.right": False}})
print("Herramientas listas.")'''

def cuaderno(celdas, estudiante):
    nb = nbf.v4.new_notebook()
    contador = [0]
    for tipo, texto in celdas:
        texto = texto.strip("\n")
        if tipo == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(texto))
        else:
            def numerar(m):
                contador[0] += 1
                return f"HUECO {contador[0]}:"
            texto = re.sub(r"HUECO \d+:", numerar, texto)
            texto = re.sub(r"«(.*?)»", "____" if estudiante else r"\1", texto)
            nb.cells.append(nbf.v4.new_code_cell(texto))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb

