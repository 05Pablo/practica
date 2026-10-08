"""Librería de NLP basada en reglas para Esperanto.

Uso:
    >>> import esperanto_nlp
    >>> nlp = esperanto_nlp.load()
    >>> doc = nlp("Mi legas la Biblion.")
    >>> [(token.text, token.lemma_, token.pos_, token.is_stop) for token in doc]
    [('Mi', 'mi', 'PRON', True), ('legas', 'legi', 'VERB', False), ('la', 'la', 'DET', True), ('Biblion', 'Biblio', 'PROPN', False), ('.', '.', 'PUNCT', False)]
"""

from .language import Language, Lemmatizer, Tagger, load
from .morphology import analyze, lemmatize, pos_tag
from .orthography import from_x_system
from .stopwords import STOP_WORDS, is_stop_word
from .tokenizer import Tokenizer
from .tokens import Doc, MorphAnalysis, Span, Token

__version__ = "0.2.0"

__all__ = [
    "Doc", "Language", "Lemmatizer", "MorphAnalysis", "Span", "STOP_WORDS",
    "Tagger", "Token", "Tokenizer", "analyze", "from_x_system", "is_stop_word",
    "lemmatize", "load", "pos_tag",
]
