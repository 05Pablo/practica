import pytest

import esperanto_nlp
from esperanto_nlp import STOP_WORDS, is_stop_word


@pytest.mark.parametrize("word", ["la", "kaj", "de", "en", "ke", "ne", "La", "KAJ"])
def test_resource_words_are_stop_words(word):
    assert is_stop_word(word)


@pytest.mark.parametrize("word", ["kiu", "kiuj", "kiujn", "tio", "tion", "ĉiam", "nenies", "iel"])
def test_correlatives_are_stop_words(word):
    assert is_stop_word(word)


@pytest.mark.parametrize("word", ["min", "lin", "mia", "miajn", "ilian", "sian"])
def test_pronoun_forms_are_stop_words(word):
    assert is_stop_word(word)


@pytest.mark.parametrize("word", ["Dio", "preĝo", "Korano", "sinagogo", "kristano"])
def test_content_words_are_not_stop_words(word):
    assert not is_stop_word(word)


def test_non_ignorable_resource_words_are_excluded():
    assert "bona" not in STOP_WORDS
    assert "miliono" not in STOP_WORDS


def test_doc_marks_stop_words():
    doc = esperanto_nlp.load()("Mi legas la Biblion.")
    assert [token.is_stop for token in doc] == [True, False, True, False, False]


def test_custom_stop_words():
    nlp = esperanto_nlp.load(stop_words={"biblion"})
    assert [token.is_stop for token in nlp("Mi legas la Biblion")] == [False, False, False, True]
