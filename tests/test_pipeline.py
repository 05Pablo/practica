import pytest

import esperanto_nlp


@pytest.fixture
def nlp():
    return esperanto_nlp.load()


def test_pipe_names(nlp):
    assert nlp.pipe_names == ["tagger", "lemmatizer", "stopwords"]


def test_full_analysis(nlp):
    doc = nlp("La judoj preĝas en la sinagogo.")
    assert [t.lemma_ for t in doc] == ["la", "judo", "preĝi", "en", "la", "sinagogo", "."]
    assert [t.pos_ for t in doc] == ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"]


def test_morph_string(nlp):
    doc = nlp("Mi vidis hundojn.")
    assert str(doc[2].morph) == "Case=Acc|Number=Plur"
    assert doc[2].morph.to_dict() == {"Number": "Plur", "Case": "Acc"}


def test_correlative_determiner_or_pronoun(nlp):
    assert nlp("Tiu libro estas bona.")[0].pos_ == "DET"
    assert nlp("Tiu ĉi libro estas bona.")[0].pos_ == "DET"
    assert nlp("Tiu venis.")[0].pos_ == "PRON"
    assert nlp("La homo, kiu venis.")[3].pos_ == "PRON"


def test_proper_noun_at_sentence_start(nlp):
    doc = nlp("Jesuon oni krucumis. Laŭ la Biblio, Jesuo resurektis.")
    assert (doc[0].lemma_, doc[0].pos_) == ("Jesuo", "PROPN")
    # Sin otra aparición en el documento, decide la terminación.
    doc = nlp("Hundoj bojas.")
    assert (doc[0].lemma_, doc[0].pos_) == ("hundo", "NOUN")


def test_inflected_esti_is_stop_word(nlp):
    doc = nlp("Ĝi estis fondita en 1950.")
    assert doc[1].lemma_ == "esti"
    assert doc[1].is_stop
    assert not doc[2].is_stop
