# Taller en Ciencia de Datos y Aprendizaje Automático

Especialización en Estadística · Universidad Nacional de Colombia, Sede Manizales · 2026-II
Profesor: Carlos Manuel Orrego Franco

Cada clase tiene dos cuadernos. El primero lo resuelve el profesor en vivo. El segundo, su gemelo, lo resuelve cada estudiante con la misma técnica y otros datos. Todos los datos son reales y se descargan de su fuente oficial.

## Clases

| # | Fecha | En vivo | Gemelo |
|---|---|---|---|
| 1 | Vie 20 nov | [El sancocho](clases/01_formular/01a_el_sancocho.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmorregof/taller-cd-ml-2026-2/blob/main/clases/01_formular/01a_el_sancocho.ipynb) | [El precio de la luz](clases/01_formular/01b_el_precio_de_la_luz.ipynb) [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cmorregof/taller-cd-ml-2026-2/blob/main/clases/01_formular/01b_el_precio_de_la_luz.ipynb) |

## Cómo se usan los datos

En cualquier cuaderno:

```python
from taller import descargar
tabla = descargar("sipsa")
```

La función baja el dato de la fuente oficial, lo deja en `datos/cache/` y anota de dónde salió, cuándo y con qué huella MD5. Los datos no se guardan en este repositorio: algunas fuentes no permiten republicarlos y otras pesan demasiado.

| Nombre | Qué es | Fuente | Tiempo aproximado |
|---|---|---|---|
| `sipsa` | Precios mayoristas mensuales de alimentos, 2013 a 2024 | DANE, Microdatos, catálogo 776 | 30 segundos |
| `xm_precio` | Precio de bolsa horario, 2023 a octubre de 2026 | XM, SiMEM, conjunto EC6945 | 3 a 4 minutos |
| `xm_embalses` | Energía útil por embalse y día, 2023 a octubre de 2026 | XM, SiMEM, conjunto c51127 | 2 a 3 minutos |

## Para correrlo en un computador propio

```bash
git clone https://github.com/cmorregof/taller-cd-ml-2026-2.git
cd taller-cd-ml-2026-2
pip install -r requirements.txt
python pruebas/probar_fuentes.py
```

## Estructura

- `clases/`: los cuadernos, una carpeta por clase.
- `taller/`: el código que descarga y normaliza los datos.
- `pruebas/`: revisa que las fuentes oficiales sigan respondiendo.
- `herramientas/`: los programas que generan los cuadernos.
