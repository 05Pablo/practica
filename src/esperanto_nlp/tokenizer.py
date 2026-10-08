"""Tokenizador basado en expresiones regulares."""

import re

# Abreviaturas que contienen puntos y no deben partirse ni cerrar una frase.
ABBREVIATIONS = [
    "k.t.p.", "ktp.", "k.s.", "t.e.", "ekz.",  # etcétera, y similares, es decir...
    "p.K.", "a.K.", "a.n.e.", "n.e.",          # después/antes de Cristo, (antes de la) era común
    "jc.", "ĉ.", "vol.", "pp.",                # siglo, aproximadamente, volumen, páginas
]

_ABBR = "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True))
_LETTER = r"[^\W\d_]"  # cualquier letra Unicode (incluye ĉ, ĝ, ĥ, ĵ, ŝ, ŭ)

TOKEN_RE = re.compile(
    rf"""
      (?<!{_LETTER})(?i:{_ABBR})(?!{_LETTER})        # abreviaturas: p.K., k.t.p.
    | {_LETTER}+(?:-{_LETTER}+)*(?:['’](?!{_LETTER}))? # palabras: domo, s-ro, Noa-n, l'
    | \d+(?:[.,]\d+)*(?:-{_LETTER}+)?                 # números: 1200, 3,14, 20-a
    | \.\.\.                                          # puntos suspensivos
    | \S                                              # cualquier otro símbolo
    """,
    re.VERBOSE,
)

SENTENCE_END = {".", "!", "?", "...", "…"}
CLOSING_PUNCT = {"»", "”", '"', "'", ")", "]", "›"}


class Tokenizer:
    """Divide un texto en tokens y marca los inicios de frase."""

    def __call__(self, text):
        """Devuelve una lista de tuplas (texto, posición, espacio_posterior)."""
        matches = list(TOKEN_RE.finditer(text))
        tokens = []
        for n, match in enumerate(matches):
            end = matches[n + 1].start() if n + 1 < len(matches) else len(text)
            tokens.append((match.group(), match.start(), text[match.end():end]))
        return tokens

    @staticmethod
    def sentence_starts(words):
        """Indica, para cada token, si empieza una frase nueva.

        Una frase empieza tras un signo de final de frase (. ! ? ...). Los
        cierres de comillas o paréntesis pertenecen aún a la frase anterior, y
        tras ellos solo empieza frase si la palabra siguiente va en mayúscula:
        «Ne!» Bone (dos frases) frente a «Venu!» li diris (una frase).
        """
        starts = []
        for i, word in enumerate(words):
            if i == 0:
                starts.append(True)
                continue
            if word in CLOSING_PUNCT:
                starts.append(False)
                continue
            j = i - 1
            while j > 0 and words[j] in CLOSING_PUNCT:
                j -= 1
            after_closing = j < i - 1
            starts.append(
                words[j] in SENTENCE_END
                and (not after_closing or not word[:1].islower())
            )
        return starts
