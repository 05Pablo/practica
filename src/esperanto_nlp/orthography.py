"""Normalización ortográfica del Esperanto.

Las letras con diacrítico (ĉ, ĝ, ĥ, ĵ, ŝ, ŭ) a menudo se escriben con el
"sistema x" (cx, gx, hx, jx, sx, ux) cuando el teclado no las tiene. Como la
letra x no existe en el alfabeto del Esperanto, la conversión no es ambigua
para palabras esperantistas (sí podría afectar a nombres extranjeros: "Linux").
"""

import re

_X_SYSTEM = {
    "cx": "ĉ", "gx": "ĝ", "hx": "ĥ", "jx": "ĵ", "sx": "ŝ", "ux": "ŭ",
    "Cx": "Ĉ", "Gx": "Ĝ", "Hx": "Ĥ", "Jx": "Ĵ", "Sx": "Ŝ", "Ux": "Ŭ",
    "CX": "Ĉ", "GX": "Ĝ", "HX": "Ĥ", "JX": "Ĵ", "SX": "Ŝ", "UX": "Ŭ",
}
_X_RE = re.compile("|".join(_X_SYSTEM))


def from_x_system(text):
    """Convierte texto en sistema x a letras con diacrítico: 'cxiu' -> 'ĉiu'."""
    return _X_RE.sub(lambda m: _X_SYSTEM[m.group()], text)
