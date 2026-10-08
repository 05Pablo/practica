import pytest

import esperanto_nlp
from esperanto_nlp import Doc, Language, Span, Token


@pytest.fixture
def nlp():
    return esperanto_nlp.load()


def test_load_returns_language(nlp):
    assert isinstance(nlp, Language)
    assert nlp.lang == "eo"


def test_doc_structure(nlp):
    doc = nlp("La hundo kuras.")
    assert isinstance(doc, Doc)
    assert len(doc) == 4
    assert doc.text == "La hundo kuras."
    assert all(isinstance(token, Token) for token in doc)
    assert [token.i for token in doc] == [0, 1, 2, 3]
    assert doc[1].text == "hundo"
    assert doc[-1].text == "."


def test_token_attributes(nlp):
    doc = nlp("Ĉe 20 hundoj, Dek!")
    ce, num, hundoj, comma, dek, excl = doc
    assert ce.lower_ == "ĉe"
    assert ce.is_alpha and not ce.is_punct
    assert num.is_digit and num.like_num and not num.is_alpha
    assert dek.like_num
    assert comma.is_punct and excl.is_punct
    assert hundoj.idx == 6
    assert hundoj.whitespace_ == "" and hundoj.text_with_ws == "hundoj"
    assert ce.text_with_ws == "Ĉe "


def test_text_with_ws_rebuilds_text(nlp):
    text = "Mi legas  la Biblion. Ĉu vi?"
    assert "".join(token.text_with_ws for token in nlp(text)) == text


def test_slicing_returns_span(nlp):
    doc = nlp("La granda hundo kuras.")
    span = doc[1:3]
    assert isinstance(span, Span)
    assert span.text == "granda hundo"
    assert [token.text for token in span] == ["granda", "hundo"]
    assert len(span) == 2


def test_sentences(nlp):
    doc = nlp("Jesuo naskiĝis ĉ. 4 a.K. en Betlehemo. Li mortis en Jerusalemo! Ĉu vere?")
    assert [sent.text for sent in doc.sents] == [
        "Jesuo naskiĝis ĉ. 4 a.K. en Betlehemo.",
        "Li mortis en Jerusalemo!",
        "Ĉu vere?",
    ]


def test_empty_doc(nlp):
    doc = nlp("")
    assert len(doc) == 0
    assert list(doc.sents) == []


def test_pipe(nlp):
    docs = list(nlp.pipe(["Unu.", "Du tri."]))
    assert [len(doc) for doc in docs] == [2, 3]


def test_normalize_x():
    nlp = esperanto_nlp.load(normalize_x=True)
    assert [token.text for token in nlp("cxu vi?")] == ["ĉu", "vi", "?"]


def test_add_pipe(nlp):
    def mark_upper(doc):
        for token in doc:
            token.pos_ = "X"
        return doc

    nlp.add_pipe(mark_upper, "marker", before="stopwords")
    assert nlp.pipe_names == ["tagger", "lemmatizer", "marker", "stopwords"]
    assert nlp("hundo")[0].pos_ == "X"
