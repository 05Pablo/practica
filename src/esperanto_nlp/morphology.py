"""Análisis morfológico del Esperanto basado en reglas.

El Esperanto marca la categoría gramatical con terminaciones regulares:

    -o  sustantivo    -a  adjetivo    -e  adverbio
    -i  infinitivo    -as/-is/-os presente/pasado/futuro
    -us condicional   -u  volitivo/imperativo
    -j  plural        -n  acusativo

`analyze` aplica estas reglas a una palabra aislada y devuelve su lema, su
categoría (etiquetas Universal Dependencies) y sus rasgos morfológicos. Las
palabras funcionales (artículo, preposiciones, pronombres, correlativos...) no
siguen las terminaciones y se resuelven con un léxico cerrado.

El lema solo elimina la flexión (número, caso, tiempo verbal), no la
derivación: "konataj" -> "konata" (y no "koni").
"""

import re
import unicodedata
from functools import lru_cache
from typing import NamedTuple

from .stopwords import CORRELATIVE_PREFIXES, FUNCTION_WORDS, PRONOUNS


class Analysis(NamedTuple):
    lemma: str
    pos: str
    morph: dict


# --- Léxico cerrado -----------------------------------------------------------

# Categoría de vortoj.json -> etiqueta UD. Las categorías que no aparecen
# (verboformo, ĝentilaĵo, titolo) se analizan por su terminación: "esti", "bona".
CATEGORY_POS = {
    "artikolo": "DET",
    "prepozicio": "ADP",
    "konjunkcio": "CCONJ",
    "pronomo": "PRON",
    "adverbo": "ADV",
    "partikulo": "PART",
    "nombro": "NUM",
    "interjekcio": "INTJ",
}
SUBORDINATING = {"ke", "se", "ĉar", "kvankam", "ol", "kvazaŭ"}

# Terminación de correlativo -> categoría. Los de -u (kiu, tiu...) pueden ser
# pronombre o determinante; lo decide el contexto en el Tagger.
CORRELATIVE_POS = {
    "u": "PRON", "o": "PRON", "a": "DET", "es": "DET",
    "e": "ADV", "am": "ADV", "al": "ADV", "el": "ADV", "om": "ADV",
}

# Abreviaturas: forma en minúsculas -> (lema, categoría).
ABBREVIATIONS = {
    "s-ro": ("sinjoro", "NOUN"), "s-ino": ("sinjorino", "NOUN"),
    "d-ro": ("doktoro", "NOUN"), "n-ro": ("numero", "NOUN"),
    "jc.": ("jarcento", "NOUN"), "ĉ.": ("ĉirkaŭ", "ADV"),
    "k.t.p.": ("kaj tiel plu", "ADV"), "ktp.": ("kaj tiel plu", "ADV"),
    "k.s.": ("kaj simile", "ADV"), "t.e.": ("tio estas", "ADV"),
    "ekz.": ("ekzemple", "ADV"),
    "p.k.": ("post Kristo", "ADV"), "a.k.": ("antaŭ Kristo", "ADV"),
}

# Unicode los considera puntuación, pero en UD son símbolos.
SYMBOLS = {"%", "‰", "&", "@", "#", "*", "/", "§", "†", "‡"}

PLURAL_ACC = {
    "": {"Number": "Sing", "Case": "Nom"},
    "j": {"Number": "Plur", "Case": "Nom"},
    "n": {"Number": "Sing", "Case": "Acc"},
    "jn": {"Number": "Plur", "Case": "Acc"},
}


def _build_lexicon():
    lexicon = {}

    for word, category in FUNCTION_WORDS.items():
        if category in CATEGORY_POS:
            pos = "SCONJ" if word in SUBORDINATING else CATEGORY_POS[category]
            lexicon[word] = Analysis(word, pos, {})

    for prefix in CORRELATIVE_PREFIXES:
        for suffix, pos in CORRELATIVE_POS.items():
            word = prefix + suffix
            lexicon[word] = Analysis(word, pos, {})
            if suffix in ("u", "a"):
                for ending in ("j", "n", "jn"):
                    lexicon[word + ending] = Analysis(word, pos, dict(PLURAL_ACC[ending]))
            elif suffix in ("o", "e"):
                lexicon[word + "n"] = Analysis(word, pos, {"Case": "Acc"})

    for pronoun in PRONOUNS:
        lexicon[pronoun] = Analysis(pronoun, "PRON", {"Case": "Nom"})
        lexicon[pronoun + "n"] = Analysis(pronoun, "PRON", {"Case": "Acc"})
        possessive = pronoun + "a"
        for ending, features in PLURAL_ACC.items():
            lexicon[possessive + ending] = Analysis(possessive, "DET", {"Poss": "Yes", **features})

    lexicon["l'"] = Analysis("la", "DET", {})
    lexicon["adiaŭ"] = Analysis("adiaŭ", "INTJ", {})
    lexicon["ambaŭ"] = Analysis("ambaŭ", "DET", {})
    lexicon["almenaŭ"] = Analysis("almenaŭ", "ADV", {})
    return lexicon


LEXICON = _build_lexicon()


# --- Terminaciones ------------------------------------------------------------

# (terminación, categoría, terminación del lema, rasgos), de la más larga a la
# más corta para que "-ojn" se pruebe antes que "-o".
ENDINGS = [
    ("ojn", "NOUN", "o", {"Number": "Plur", "Case": "Acc"}),
    ("ajn", "ADJ", "a", {"Number": "Plur", "Case": "Acc"}),
    ("oj", "NOUN", "o", {"Number": "Plur", "Case": "Nom"}),
    ("aj", "ADJ", "a", {"Number": "Plur", "Case": "Nom"}),
    ("on", "NOUN", "o", {"Number": "Sing", "Case": "Acc"}),
    ("an", "ADJ", "a", {"Number": "Sing", "Case": "Acc"}),
    ("en", "ADV", "e", {"Case": "Acc"}),  # dirección: hejmen "hacia casa"
    ("as", "VERB", "i", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Pres"}),
    ("is", "VERB", "i", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"}),
    ("os", "VERB", "i", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Fut"}),
    ("us", "VERB", "i", {"VerbForm": "Fin", "Mood": "Cnd"}),
    ("o", "NOUN", "o", {"Number": "Sing", "Case": "Nom"}),
    ("a", "ADJ", "a", {"Number": "Sing", "Case": "Nom"}),
    ("e", "ADV", "e", {}),
    ("i", "VERB", "i", {"VerbForm": "Inf"}),
    ("u", "VERB", "i", {"VerbForm": "Fin", "Mood": "Jus"}),
]
MIN_ROOT = 2  # evita analizar como raíz fragmentos de una sola letra

# Sufijos de participio: tiempo y voz.
PARTICIPLES = {
    "ant": ("Pres", "Act"), "int": ("Past", "Act"), "ont": ("Fut", "Act"),
    "at": ("Pres", "Pass"), "it": ("Past", "Pass"), "ot": ("Fut", "Pass"),
}
PARTICIPLE_RE = re.compile(r"^(.{2,}?)(ant|int|ont|at|it|ot)$")

# Raíces que terminan como un participio sin serlo (ŝtata, milita, spirita...).
NOT_PARTICIPLES = {
    "ŝtat", "milit", "rilat", "spirit", "privat", "konstant", "kvadrat",
    "sacerdot", "jezuit", "kvalit", "majoritat", "eksplicit", "vulgat",
    "golgot", "front", "spit", "debat", "subit", "akurat", "ermit", "edit",
    "kalifat", "sultanat", "patriarkat", "episkopat", "demokrat", "soldat",
    "kandidat", "advokat", "senat", "format", "klimat", "aŭtomat", "strat",
    "prelat", "apostat", "unit", "infinit", "kvant", "eksterordinat",
}


def _participle_features(root):
    match = PARTICIPLE_RE.match(root)
    if not match or root in NOT_PARTICIPLES:
        return {}
    tense, voice = PARTICIPLES[match.group(2)]
    return {"VerbForm": "Part", "Tense": tense, "Voice": voice}


def analyze_by_ending(word):
    """Analiza una palabra en minúsculas por su terminación, o devuelve None."""
    for ending, pos, lemma_ending, features in ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_ROOT:
            root = word[:-len(ending)]
            morph = dict(features)
            if pos in ("ADJ", "ADV"):
                morph.update(_participle_features(root))
            if pos == "VERB" and root == "est":
                pos = "AUX"
            return Analysis(root + lemma_ending, pos, morph)
    return None


def _proper_noun(text):
    """Nombre propio: se quita el acusativo/plural de los nombres esperantizados."""
    lemma = re.sub(r"-j?n$", "", text)                   # Noa-n -> Noa
    lemma = re.sub(r"(?<=[oa])(jn|j|n)$", "", lemma)     # Jesuon -> Jesuo
    return Analysis(lemma, "PROPN", {})


@lru_cache(maxsize=100_000)
def analyze(text, is_sent_start=False):
    """Devuelve el `Analysis` (lema, categoría UD, rasgos) de un token aislado.

    Args:
        text: el token tal como aparece en el texto.
        is_sent_start: si el token empieza frase (la mayúscula no indica
            entonces nombre propio).
    """
    if not any(c.isalnum() for c in text):
        is_punct = text not in SYMBOLS and all(
            unicodedata.category(c).startswith("P") for c in text
        )
        return Analysis(text, "PUNCT" if is_punct else "SYM", {})

    if text[0].isdigit():
        if "-" in text:                                   # ordinal: 20-a
            return Analysis(text.lower(), "ADJ", {"NumType": "Ord"})
        return Analysis(text, "NUM", {"NumType": "Card"})

    lower = text.lower()
    if lower in ABBREVIATIONS:
        lemma, pos = ABBREVIATIONS[lower]
        return Analysis(lemma, pos, {"Abbr": "Yes"})
    if lower in LEXICON:
        return LEXICON[lower]

    if text[0].isupper() and not is_sent_start:
        return _proper_noun(text)

    if lower.endswith(("'", "’")):                        # elisión poética: dom' -> domo
        return Analysis(lower[:-1] + "o", "NOUN", {})

    analysis = analyze_by_ending(lower)
    if analysis is not None:
        return analysis
    if text[0].isupper():
        return _proper_noun(text)
    return Analysis(lower, "X", {})


def lemmatize(text, is_sent_start=False):
    """Lema de un token aislado: lemmatize("estis") -> "esti"."""
    return analyze(text, is_sent_start).lemma


def pos_tag(text, is_sent_start=False):
    """Categoría UD de un token aislado: pos_tag("hundojn") -> "NOUN"."""
    return analyze(text, is_sent_start).pos
