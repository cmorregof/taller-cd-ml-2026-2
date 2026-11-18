"""Genera los cuadernos de la clase 5 (imágenes, transferencia y fuga de datos).

La demostración (Lunares) sale completa. El gemelo (La roya) tiene el andamiaje de
las clases 5 a 8: cada celda de código del estudiante trae solo un comentario que
dice qué debe lograr, y la solución agrega el código debajo del mismo comentario.
Por eso las celdas del gemelo se escriben como ("tarea", comentario, solución).
"""
import sys
from pathlib import Path
import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molde import ARRANQUE, RAIZ, cuaderno
from clase05_lunares import LUNARES
from clase05_roya import ROYA


def gemelo(celdas, estudiante):
    """Convierte las celdas ("tarea", comentario, solución) al formato de `cuaderno`."""
    salida = []
    for celda in celdas:
        if celda[0] == "tarea":
            comentario, solucion = celda[1].strip("\n"), celda[2].strip("\n")
            salida.append(("code", comentario if estudiante else comentario + "\n" + solucion))
        else:
            salida.append(celda)
    return cuaderno(salida, False)


if __name__ == "__main__":
    destino = RAIZ / "clases" / "05_imagenes"
    destino.mkdir(parents=True, exist_ok=True)
    (RAIZ / "_profesor").mkdir(exist_ok=True)
    nbf.write(cuaderno(LUNARES, False), destino / "05a_lunares.ipynb")
    nbf.write(gemelo(ROYA, True), destino / "05b_la_roya.ipynb")
    nbf.write(gemelo(ROYA, False), RAIZ / "_profesor" / "05b_la_roya_SOLUCION.ipynb")
    print("Cuadernos de la clase 5 escritos.")
