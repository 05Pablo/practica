"""Código Python que la web ejecuta en el navegador con Pyodide.

- `analyze(text)`: pasa el texto por la librería `esperanto_nlp` y devuelve los
  tokens con su lema, POS, morfología y stop-word.
- `Classifier`: reproduce en Python puro (sin sklearn ni numpy) la predicción
  del Pipeline TF-IDF + regresión logística exportado por `build_web.py`.

`scripts/build_web.py` comprueba que `Classifier` da las mismas probabilidades
que sklearn antes de publicar la web.
"""

import json
import math
from collections import Counter

_nlp = {}


def get_nlp(normalize_x=False):
    if normalize_x not in _nlp:
        import esperanto_nlp
        _nlp[normalize_x] = esperanto_nlp.load(normalize_x=normalize_x)
    return _nlp[normalize_x]


def preprocess(text, normalize_x=False):
    """Mismo preprocesado que el cuaderno: lemas sin stop-words."""
    doc = get_nlp(normalize_x)(text)
    return [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]


def analyze(text, normalize_x=False):
    """Análisis token a token, en JSON para la web."""
    doc = get_nlp(normalize_x)(text)
    tokens = [{
        "text": t.text,
        "lemma": t.lemma_,
        "pos": t.pos_,
        "morph": str(t.morph),
        "stop": t.is_stop,
        "alpha": t.is_alpha,
        "sent_start": t.is_sent_start,
    } for t in doc]
    return json.dumps({"tokens": tokens, "sents": [s.text for s in doc.sents]},
                      ensure_ascii=False)


class Classifier:
    """TF-IDF (norma L2) + regresión logística multinomial (softmax)."""

    def __init__(self, model):
        if isinstance(model, str):
            model = json.loads(model)
        self.classes = model["classes"]
        self.terms = model["terms"]
        self.index = {term: i for i, term in enumerate(self.terms)}
        self.idf = model["idf"]
        self.coef = model["coef"]
        self.intercept = model["intercept"]

    def vectorize(self, tokens):
        """Vector TF-IDF disperso {índice: peso}, normalizado (L2) como sklearn."""
        counts = Counter(self.index[t] for t in tokens if t in self.index)
        weights = {i: n * self.idf[i] for i, n in counts.items()}
        norm = math.sqrt(sum(w * w for w in weights.values()))
        return {i: w / norm for i, w in weights.items()} if norm else {}

    def predict_tokens(self, tokens, n_terms=8):
        vec = self.vectorize(tokens)
        scores = [b + sum(w * row[i] for i, w in vec.items())
                  for row, b in zip(self.coef, self.intercept)]
        top = max(scores)
        exps = [math.exp(s - top) for s in scores]
        total = sum(exps)
        proba = [e / total for e in exps]
        best = max(range(len(proba)), key=proba.__getitem__)

        # Términos que más empujan hacia la clase predicha (peso TF-IDF x coef).
        contrib = sorted(((w * self.coef[best][i], self.terms[i]) for i, w in vec.items()),
                         reverse=True)
        return {
            "proba": proba,
            "pred": self.classes[best],
            "terms": [[t, round(c, 4)] for c, t in contrib[:n_terms] if c > 0],
            "n_tokens": len(tokens),
            "n_known": sum(1 for t in tokens if t in self.index),
        }

    def predict(self, text, normalize_x=False):
        return self.predict_tokens(preprocess(text, normalize_x))

    def predict_json(self, text, normalize_x=False):
        return json.dumps(self.predict(text, normalize_x), ensure_ascii=False)

    def predict_many_json(self, paragraphs_json, normalize_x=False):
        """Clasifica varios párrafos y promedia las probabilidades (como el cuaderno)."""
        results = [self.predict(p, normalize_x) for p in json.loads(paragraphs_json)]
        k = len(self.classes)
        mean = [sum(r["proba"][j] for r in results) / len(results) for j in range(k)] if results else []
        return json.dumps({"paragraphs": results, "mean": mean}, ensure_ascii=False)
