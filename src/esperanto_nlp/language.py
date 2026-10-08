"""Objeto Language: aplica el tokenizador y el pipeline de componentes."""

from .morphology import analyze
from .orthography import from_x_system
from .stopwords import CORRELATIVE_PREFIXES, STOP_WORDS
from .tokenizer import Tokenizer
from .tokens import Doc, MorphAnalysis

# Correlativos en -u (kiu, tiu, iu, ĉiu, neniu): pronombre o determinante.
U_CORRELATIVES = {prefix + "u" for prefix in CORRELATIVE_PREFIXES}
NOMINAL_POS = {"NOUN", "PROPN", "ADJ", "NUM", "DET"}


class Tagger:
    """Componente que asigna `token.pos_` y `token.morph` por reglas.

    Además del análisis por terminación, aplica dos reglas de contexto:
    - Una palabra en mayúscula al inicio de frase es nombre propio si aparece
      como nombre propio en otro punto del documento ("Jesuon ... la Jesuo").
    - Un correlativo en -u seguido de un sustantivo o adjetivo es determinante
      ("tiu libro"); si no, es pronombre ("tiu venis").
    """

    def __call__(self, doc):
        analyses = [analyze(token.text, token.is_sent_start) for token in doc]

        proper_nouns = {a.lemma for a in analyses if a.pos == "PROPN"}
        for i, token in enumerate(doc):
            if token.is_sent_start and analyses[i].pos != "PROPN":
                candidate = analyze(token.text, is_sent_start=False)
                if candidate.pos == "PROPN" and candidate.lemma in proper_nouns:
                    analyses[i] = candidate

        for token, analysis in zip(doc, analyses):
            token.pos_ = analysis.pos
            token.morph = MorphAnalysis(analysis.morph)

        for i, analysis in enumerate(analyses):
            if analysis.lemma in U_CORRELATIVES:
                j = i + 1
                if j < len(doc) and doc[j].lower_ == "ĉi":   # tiu ĉi libro
                    j += 1
                if j < len(doc) and doc[j].pos_ in NOMINAL_POS:
                    doc[i].pos_ = "DET"
        return doc


class Lemmatizer:
    """Componente que asigna `token.lemma_` por reglas (quita la flexión).

    Si el Tagger ya ha marcado el token como nombre propio, el lema conserva
    la mayúscula ("Jesuon" -> "Jesuo"), igual que en mitad de frase.
    """

    def __call__(self, doc):
        for token in doc:
            is_sent_start = token.is_sent_start and token.pos_ != "PROPN"
            token.lemma_ = analyze(token.text, is_sent_start).lemma
        return doc


class StopWordsMarker:
    """Componente que marca `token.is_stop` según una lista de stop-words."""

    def __init__(self, stop_words):
        self.stop_words = stop_words

    def __call__(self, doc):
        for token in doc:
            token.is_stop = (
                token.lower_ in self.stop_words
                or token.lemma_.lower() in self.stop_words
            )
        return doc


class Language:
    """Procesador de texto en Esperanto, análogo a `spacy.Language`.

    Ejemplo:
        >>> nlp = Language()
        >>> doc = nlp("La hundo kuras.")
        >>> [token.text for token in doc]
        ['La', 'hundo', 'kuras', '.']
    """

    lang = "eo"

    def __init__(self, stop_words=None, normalize_x=False):
        """
        Args:
            stop_words: conjunto de stop-words; por defecto `STOP_WORDS`.
            normalize_x: convertir el sistema x ('cx' -> 'ĉ') antes de analizar.
        """
        self.tokenizer = Tokenizer()
        self.stop_words = set(STOP_WORDS if stop_words is None else stop_words)
        self.normalize_x = normalize_x
        self.pipeline = [
            ("tagger", Tagger()),
            ("lemmatizer", Lemmatizer()),
            ("stopwords", StopWordsMarker(self.stop_words)),
        ]

    @property
    def pipe_names(self):
        return [name for name, _ in self.pipeline]

    def add_pipe(self, component, name, before=None):
        """Añade un componente (función Doc -> Doc) al pipeline."""
        position = self.pipe_names.index(before) if before else len(self.pipeline)
        self.pipeline.insert(position, (name, component))

    def make_doc(self, text):
        """Tokeniza el texto sin aplicar el resto del pipeline."""
        if self.normalize_x:
            text = from_x_system(text)
        tokens = self.tokenizer(text)
        words = [word for word, _, _ in tokens]
        return Doc(
            text,
            words,
            [idx for _, idx, _ in tokens],
            [ws for _, _, ws in tokens],
            self.tokenizer.sentence_starts(words),
        )

    def __call__(self, text):
        doc = self.make_doc(text)
        for _, component in self.pipeline:
            doc = component(doc)
        return doc

    def pipe(self, texts):
        """Procesa una secuencia de textos de forma perezosa."""
        for text in texts:
            yield self(text)


def load(**kwargs):
    """Crea un procesador de Esperanto: `nlp = esperanto_nlp.load()`."""
    return Language(**kwargs)
