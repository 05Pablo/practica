import pytest

from esperanto_nlp import Tokenizer, from_x_system


def words(text):
    return [word for word, _, _ in Tokenizer()(text)]


@pytest.mark.parametrize("text, expected", [
    ("La hundo kuras.", ["La", "hundo", "kuras", "."]),
    ("Ĉu vi ŝatas ĝin?", ["Ĉu", "vi", "ŝatas", "ĝin", "?"]),
    ("Mi, vi kaj li!", ["Mi", ",", "vi", "kaj", "li", "!"]),
    ("«Venu», li diris.", ["«", "Venu", "»", ",", "li", "diris", "."]),
])
def test_basic_tokenization(text, expected):
    assert words(text) == expected


def test_hyphenated_words_are_one_token():
    assert words("s-ro Zamenhof kaj Noa-n") == ["s-ro", "Zamenhof", "kaj", "Noa-n"]


def test_elision_keeps_apostrophe():
    assert words("de l' mondo") == ["de", "l'", "mondo"]


def test_numbers():
    assert words("En 1887 kaj 3,14 en la 20-a jarcento") == [
        "En", "1887", "kaj", "3,14", "en", "la", "20-a", "jarcento",
    ]


def test_abbreviations_are_one_token():
    assert words("En 33 p.K. li mortis, k.t.p.") == [
        "En", "33", "p.K.", "li", "mortis", ",", "k.t.p.",
    ]


def test_ellipsis():
    assert words("Nu...") == ["Nu", "..."]


def test_positions_and_whitespace_rebuild_text():
    text = "La  hundo,\tkuras."
    tokens = Tokenizer()(text)
    assert "".join(word + ws for word, _, ws in tokens) == text
    for word, idx, _ in tokens:
        assert text[idx:idx + len(word)] == word


def test_empty_text():
    assert Tokenizer()("") == []


def test_sentence_starts():
    assert Tokenizer.sentence_starts(["Jes", ".", "Ne", "!", "»", "Bone"]) == [
        True, False, True, False, False, True,
    ]
    # Tras «...!» en minúscula la frase continúa.
    assert Tokenizer.sentence_starts(["«", "Venu", "!", "»", "li", "diris", "."]) == [
        True, False, False, False, False, False, False,
    ]


def test_x_system():
    assert from_x_system("cxiu gxardeno, Sxi auxdas") == "ĉiu ĝardeno, Ŝi aŭdas"
