"""Portada de carrusel con foto: velo claro arriba, título serif, etiqueta roja y pie oscuro.

Se corre desde la raíz del repositorio. Las fuentes las baja taller.carrusel.preparar_fuentes().

    python herramientas/posts/portada.py icfes   -> documentos/posts/icfes_01_portada.png
    python herramientas/posts/portada.py luz
"""
import math, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, ".")
from taller import carrusel
W, H = 1080, 1350
FU = str(carrusel.FUENTES) + "/"
def serif(tam, peso):
    f = ImageFont.truetype(FU + 'SourceSerif4.ttf', tam); f.set_variation_by_axes([peso, min(60, tam)]); return f
def cond(tam, cual='SemiBold'): return ImageFont.truetype(FU + f'FiraSansCondensed-{cual}.ttf', tam)

def portada(foto, salida, etiqueta, contexto, lineas, bajada, firma, fuente, desliza, velo=700, desplazar=0):
    im = Image.open(foto).convert('RGB')
    im = im.resize((W, round(im.height * W / im.width)), Image.LANCZOS)
    y0 = min(max(0, (im.height - H) // 2 + desplazar), im.height - H)
    im = im.crop((0, y0, W, y0 + H)).convert('RGBA')
    capa = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(capa)
    corte = velo - 230
    for y in range(velo):
        a = 238 if y < corte else int(238 * 0.5 * (1 + math.cos(math.pi * (y - corte) / 230)))
        d.line([(0, y), (W, y)], fill=(251, 250, 247, a))
    for y in range(H - 220, H):
        a = int(205 * ((y - (H - 220)) / 220) ** 1.6); d.line([(0, y), (W, y)], fill=(12, 18, 32, a))
    im = Image.alpha_composite(im, capa); d = ImageDraw.Draw(im)
    M = 72; ROJO = (215, 38, 61); TINTA = (18, 18, 18); GRIS = (70, 70, 70)
    d.rectangle([0, 0, W, 10], fill=ROJO)
    k = cond(27, 'SemiBold'); w = d.textlength(etiqueta, font=k)
    d.rectangle([M, 78, M + w + 28, 78 + 44], fill=ROJO); d.text((M + 14, 84), etiqueta, font=k, fill=(255, 255, 255))
    d.text((M + w + 48, 84), contexto, font=cond(27, 'Medium'), fill=GRIS)
    t = serif(88, 700); y = 150
    for linea in lineas:
        d.text((M, y), linea, font=t, fill=TINTA); y += 98
    for i, b in enumerate(bajada):
        d.text((M, y + 20 + 48 * i), b, font=serif(38, 400), fill=GRIS)
    d.text((M, H - 94), firma, font=cond(28, 'SemiBold'), fill=(255, 255, 255))
    d.text((M, H - 58), fuente, font=cond(23, 'Regular'), fill=(226, 232, 240))
    f = cond(28, 'SemiBold'); d.text((W - M - d.textlength(desliza, font=f), H - 94), desliza, font=f, fill=(255, 255, 255))
    im.convert('RGB').save(salida, quality=95)

PORTADAS = {
    "icfes": dict(foto="documentos/posts/fotos/icfes_aula.png", salida="documentos/posts/icfes_01_portada.png",
                  etiqueta="SABER 11 EN CALDAS", contexto="9.941 estudiantes, 2020",
                  lineas=["¿Quién se queda", "atrás en el ICFES?"],
                  bajada=["La familia, la zona del colegio y un", "modelo que reparte cupos de refuerzo"],
                  firma="Carlos Manuel Orrego Franco", fuente="Datos: ICFES, Saber 11 2020-2", desliza="Desliza  →", velo=640),
    "luz": dict(foto="documentos/posts/fotos/luz_embalse.png", salida="documentos/posts/luz_01_cover.png",
                etiqueta="COLOMBIA'S POWER MARKET", contexto="2023 to 2026",
                lineas=["Can you predict", "tomorrow's price", "of electricity?"],
                bajada=["Four years of prices against the reservoirs"],
                firma="Carlos Manuel Orrego Franco", fuente="Data: XM, SiMEM", desliza="Swipe  →", velo=730),
}

if __name__ == '__main__':
    carrusel.preparar_fuentes()
    portada(**PORTADAS[sys.argv[1]])
