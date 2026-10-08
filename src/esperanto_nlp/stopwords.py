"""Gestión de stop-words (palabras vacías) del Esperanto.

La lista base procede de `data/vortoj.json` (repositorio nlp-esperantilo,
Apache-2.0, que incluye a su vez stopwords-iso, MIT). Se completa por reglas
con la tabla de correlativos y las formas flexionadas de los pronombres.
"""

import json
from importlib import resources

PRONOUNS = ["mi", "ci", "vi", "li", "ŝi", "ĝi", "ni", "ili", "oni", "si"]

# Tabla de correlativos: 5 prefijos x 9 terminaciones (kiu, tiu, iu, ĉiu, neniu...).
CORRELATIVE_PREFIXES = ["ki", "ti", "i", "ĉi", "neni"]
CORRELATIVE_SUFFIXES = ["u", "o", "a", "e", "am", "al", "el", "om", "es"]


def _load_vortoj():
    path = resources.files("esperanto_nlp") / "data" / "vortoj.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)["vortoj"]


# Palabra funcional -> categoría del recurso (prepozicio, konjunkcio, pronomo...).
FUNCTION_WORDS = {entry["vorto"]: entry["kategorio"] for entry in _load_vortoj()}


def _inflections(word):
    """Formas en plural (-j) y acusativo (-n) que admite una palabra funcional."""
    if word.endswith(("u", "a")):
        return [word, word + "j", word + "n", word + "jn"]
    if word.endswith(("o", "e")):
        return [word, word + "n"]
    return [word]


def _build_stop_words():
    words = {entry["vorto"] for entry in _load_vortoj() if entry.get("ignorinda", True)}

    for prefix in CORRELATIVE_PREFIXES:
        for suffix in CORRELATIVE_SUFFIXES:
            words.update(_inflections(prefix + suffix))

    for pronoun in PRONOUNS:
        words.add(pronoun + "n")                      # acusativo: min, lin...
        words.update(_inflections(pronoun + "a"))     # posesivos: mia, miajn...

    return frozenset(words)


STOP_WORDS = _build_stop_words()
"""Conjunto de stop-words por defecto (en minúsculas)."""


def is_stop_word(word, stop_words=STOP_WORDS):
    """Indica si una palabra (en cualquier capitalización) es stop-word."""
    return word.lower() in stop_words
