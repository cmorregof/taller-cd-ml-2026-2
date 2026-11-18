"""Corre cuadernos de principio a fin y deja al lado una copia *_corrido.ipynb.

    python herramientas/ejecutar.py ruta/al/cuaderno.ipynb [otro.ipynb ...]
"""
import sys, nbformat
from nbclient import NotebookClient
for ruta in sys.argv[1:]:
    nb=nbformat.read(ruta,as_version=4)
    NotebookClient(nb,timeout=900,kernel_name="python3",resources={"metadata":{"path":ruta.rsplit("/",1)[0]}}).execute()
    nbformat.write(nb,ruta.replace(".ipynb","_corrido.ipynb"))
    print("OK",ruta)
    for c in nb.cells:
        if c.cell_type=="code":
            for o in c.get("outputs",[]):
                if o.output_type=="stream": print(o.text.rstrip()[:900]); print("--")
