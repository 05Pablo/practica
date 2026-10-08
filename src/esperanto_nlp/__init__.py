"""Librería de NLP basada en reglas para Esperanto.

Uso:
    >>> import esperanto_nlp
    >>> nlp = esperanto_nlp.load()
    >>> doc = nlp("Mi legas la Biblion.")
    >>> [(token.text, token.is_stop) for token in doc]
    [('Mi', True), ('legas', False), ('la', True), ('Biblion', False), ('.', False)]
"""

from .language import Language, load
from .orthography import from_x_system
from .stopwords import STOP_WORDS, is_stop_word
from .tokenizer import Tokenizer
from .tokens import Doc, Span, Token

__version__ = "0.1.0"

__all__ = [
    "Doc", "Language", "Span", "STOP_WORDS", "Token", "Tokenizer",
    "from_x_system", "is_stop_word", "load",
]
