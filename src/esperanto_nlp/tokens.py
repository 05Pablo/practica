"""Objetos Doc, Token y Span que estructuran el resultado del análisis."""

import unicodedata

NUMBER_WORDS = {
    "nul", "unu", "du", "tri", "kvar", "kvin", "ses", "sep", "ok", "naŭ",
    "dek", "cent", "mil", "miliono", "miliardo",
}


class Token:
    """Una palabra, número o signo de puntuación dentro de un Doc."""

    def __init__(self, doc, i, text, idx, whitespace="", is_sent_start=False):
        self.doc = doc
        self.i = i                      # posición del token en el Doc
        self.text = text
        self.idx = idx                  # posición del primer carácter en el texto
        self.whitespace_ = whitespace   # espacio que sigue al token
        self.is_sent_start = is_sent_start
        # Atributos que rellenan los componentes del pipeline.
        self.lemma_ = ""
        self.pos_ = ""
        self.morph = {}
        self.is_stop = False

    @property
    def text_with_ws(self):
        return self.text + self.whitespace_

    @property
    def lower_(self):
        return self.text.lower()

    @property
    def is_alpha(self):
        """Verdadero si es una palabra (letras, con guiones o apóstrofo internos)."""
        stripped = self.text.replace("-", "").replace("'", "").replace("’", "")
        return stripped.isalpha()

    @property
    def is_digit(self):
        return self.text.isdigit()

    @property
    def like_num(self):
        """Verdadero si representa un número: '1200', '3,14', '20-a', 'dek'."""
        return self.text[:1].isdigit() or self.lower_ in NUMBER_WORDS

    @property
    def is_punct(self):
        return all(unicodedata.category(c).startswith("P") for c in self.text)

    def __len__(self):
        return len(self.text)

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Span:
    """Secuencia contigua de tokens de un Doc (p. ej. una frase)."""

    def __init__(self, doc, start, end):
        self.doc = doc
        self.start = start
        self.end = end

    @property
    def text(self):
        if self.start >= self.end:
            return ""
        first, last = self.doc[self.start], self.doc[self.end - 1]
        return self.doc.text[first.idx:last.idx + len(last)]

    def __iter__(self):
        return iter(self.doc.tokens[self.start:self.end])

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, i):
        return list(self)[i]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text


class Doc:
    """Resultado de analizar un texto: secuencia de Tokens."""

    def __init__(self, text, words, starts, whitespaces, sent_starts):
        self.text = text
        self.tokens = [
            Token(self, i, word, idx, ws, sent_start)
            for i, (word, idx, ws, sent_start)
            in enumerate(zip(words, starts, whitespaces, sent_starts))
        ]

    @property
    def sents(self):
        """Genera las frases del documento como objetos Span."""
        start = 0
        for token in self.tokens[1:]:
            if token.is_sent_start:
                yield Span(self, start, token.i)
                start = token.i
        if self.tokens:
            yield Span(self, start, len(self.tokens))

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, end, step = key.indices(len(self.tokens))
            if step != 1:
                raise ValueError("Los Span solo admiten pasos de 1")
            return Span(self, start, end)
        return self.tokens[key]

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.text
