"""Objeto Language: aplica el tokenizador y el pipeline de componentes."""

from .orthography import from_x_system
from .stopwords import STOP_WORDS
from .tokenizer import Tokenizer
from .tokens import Doc


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
        self.pipeline = [("stopwords", StopWordsMarker(self.stop_words))]

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
