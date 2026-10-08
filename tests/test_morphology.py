import pytest

from esperanto_nlp import analyze, lemmatize, pos_tag


@pytest.mark.parametrize("word, lemma, pos, morph", [
    ("hundo", "hundo", "NOUN", {"Number": "Sing", "Case": "Nom"}),
    ("hundojn", "hundo", "NOUN", {"Number": "Plur", "Case": "Acc"}),
    ("bonaj", "bona", "ADJ", {"Number": "Plur", "Case": "Nom"}),
    ("bonan", "bona", "ADJ", {"Number": "Sing", "Case": "Acc"}),
    ("rapide", "rapide", "ADV", {}),
    ("hejmen", "hejme", "ADV", {"Case": "Acc"}),
    ("legi", "legi", "VERB", {"VerbForm": "Inf"}),
    ("legas", "legi", "VERB", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Pres"}),
    ("legis", "legi", "VERB", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Past"}),
    ("legos", "legi", "VERB", {"VerbForm": "Fin", "Mood": "Ind", "Tense": "Fut"}),
    ("legus", "legi", "VERB", {"VerbForm": "Fin", "Mood": "Cnd"}),
    ("legu", "legi", "VERB", {"VerbForm": "Fin", "Mood": "Jus"}),
])
def test_regular_endings(word, lemma, pos, morph):
    assert analyze(word) == (lemma, pos, morph)


@pytest.mark.parametrize("word, lemma", [
    ("estas", "esti"), ("estis", "esti"), ("estos", "esti"), ("estu", "esti"),
])
def test_esti_is_auxiliary(word, lemma):
    assert analyze(word).lemma == lemma
    assert analyze(word).pos == "AUX"


def test_derived_verb_from_esti_is_not_auxiliary():
    assert analyze("estiĝis") == ("estiĝi", "VERB", analyze("estiĝis").morph)


@pytest.mark.parametrize("word, lemma, pos", [
    ("la", "la", "DET"), ("l'", "la", "DET"),
    ("en", "en", "ADP"), ("kun", "kun", "ADP"),
    ("kaj", "kaj", "CCONJ"), ("ke", "ke", "SCONJ"), ("ĉar", "ĉar", "SCONJ"),
    ("ne", "ne", "PART"), ("tre", "tre", "ADV"), ("tri", "tri", "NUM"),
])
def test_function_words(word, lemma, pos):
    assert analyze(word)[:2] == (lemma, pos)


@pytest.mark.parametrize("word, lemma, pos", [
    ("mi", "mi", "PRON"), ("lin", "li", "PRON"), ("ilin", "ili", "PRON"),
    ("mia", "mia", "DET"), ("miajn", "mia", "DET"), ("sian", "sia", "DET"),
])
def test_pronouns(word, lemma, pos):
    assert analyze(word)[:2] == (lemma, pos)


@pytest.mark.parametrize("word, lemma, pos", [
    ("kiu", "kiu", "PRON"), ("kiujn", "kiu", "PRON"), ("tion", "tio", "PRON"),
    ("kia", "kia", "DET"), ("ĉies", "ĉies", "DET"),
    ("kie", "kie", "ADV"), ("tien", "tie", "ADV"), ("neniam", "neniam", "ADV"),
])
def test_correlatives(word, lemma, pos):
    assert analyze(word)[:2] == (lemma, pos)


@pytest.mark.parametrize("word, tense, voice", [
    ("legita", "Past", "Pass"), ("legata", "Pres", "Pass"), ("legota", "Fut", "Pass"),
    ("leganta", "Pres", "Act"), ("leginta", "Past", "Act"), ("legonta", "Fut", "Act"),
])
def test_participles(word, tense, voice):
    lemma, pos, morph = analyze(word)
    assert (lemma, pos) == (word, "ADJ")
    assert morph["VerbForm"] == "Part"
    assert (morph["Tense"], morph["Voice"]) == (tense, voice)


@pytest.mark.parametrize("word", ["ŝtata", "milita", "spirita", "privata"])
def test_false_participles(word):
    assert "VerbForm" not in analyze(word).morph


def test_proper_nouns():
    assert analyze("Mohamedo") == ("Mohamedo", "PROPN", {})
    assert analyze("Jesuon").lemma == "Jesuo"
    assert analyze("Noa-n").lemma == "Noa"
    assert analyze("Zamenhof").pos == "PROPN"


def test_capital_at_sentence_start_is_not_proper_noun():
    assert analyze("Hundoj", is_sent_start=True)[:2] == ("hundo", "NOUN")
    assert analyze("Zamenhof", is_sent_start=True).pos == "PROPN"


@pytest.mark.parametrize("token, lemma, pos", [
    (".", ".", "PUNCT"), ("«", "«", "PUNCT"), ("%", "%", "SYM"),
    ("1887", "1887", "NUM"), ("20-a", "20-a", "ADJ"),
    ("p.K.", "post Kristo", "ADV"), ("s-ro", "sinjoro", "NOUN"),
])
def test_non_words(token, lemma, pos):
    assert analyze(token)[:2] == (lemma, pos)


def test_unknown_word_is_x():
    assert pos_tag("ibn") == "X"


def test_helpers():
    assert lemmatize("kristanojn") == "kristano"
    assert pos_tag("preĝis") == "VERB"
