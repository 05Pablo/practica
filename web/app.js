/* Web del proyecto Esperanto-NLP.
 *
 * - Pinta el dataset y los resultados a partir de data/results.json (lo genera
 *   scripts/build_web.py).
 * - Carga la librería esperanto_nlp en el navegador con Pyodide para la demo de
 *   análisis y para el clasificador (data/model.json).
 *
 * Nada de la temática está escrito aquí: clases, textos y ejemplos salen de
 * results.json y site_config.json.
 */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const state = { config: null, results: null, classes: [], py: null, demo: null, clf: null };

/* ---------- Utilidades ---------- */
const SERIES = ["--c1", "--c2", "--c3", "--pos-adv", "--pos-adj", "--pos-propn", "--pos-noun", "--pos-verb"];
const DARK_TEXT_SLOTS = new Set([3]); // amarillo: texto oscuro encima

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
const pct = (x, d = 1) => `${(x * 100).toFixed(d).replace(".", ",")} %`;
const num = n => n.toLocaleString("es-ES");
const dec = (x, d = 3) => x.toFixed(d).replace(".", ",");

function classIndex(c) { return Math.max(0, state.classes.indexOf(c)); }
function classColor(c) { return `var(${SERIES[classIndex(c) % SERIES.length]})`; }
function classLabel(c) { return state.config?.classes?.[c]?.label || c; }
function classInk(c) { return DARK_TEXT_SLOTS.has(classIndex(c) % SERIES.length) ? "#0b0b0b" : "#ffffff"; }
function swatch(c) { return `<span class="swatch" style="background:${classColor(c)}"></span>`; }
function classTag(c) { return `<span class="class-dot">${swatch(c)}<span>${esc(c)}</span></span>`; }

function hbar({ label, value, max, text, color, tip, top = false }) {
  const w = max > 0 ? (value / max) * 100 : 0;
  return `<div class="hbar${top ? " is-top" : ""}" ${tip ? `data-tip="${esc(tip)}"` : ""}>
    <div class="hbar-label">${label}</div>
    <div class="hbar-track"><div class="hbar-fill" style="width:${w}%;background:${color}"></div></div>
    <div class="hbar-value">${text}</div></div>`;
}

function stack(proba, classes) {
  return `<div class="stack">${proba.map((p, i) =>
    `<span style="width:${p * 100}%;background:${classColor(classes[i])}" data-tip="${esc(classes[i])}: ${pct(p)}"></span>`).join("")}</div>`;
}

function legend(classes) {
  return classes.map(c => `<span class="legend-item">${swatch(c)}${esc(c)}</span>`).join("");
}

function segmented(container, items, onSelect, selected = 0) {
  container.innerHTML = items.map((it, i) =>
    `<button type="button" role="tab" aria-selected="${i === selected}" data-i="${i}">${it.html ?? esc(it.label)}</button>`).join("");
  container.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("button", container).forEach(x => x.setAttribute("aria-selected", x === b));
    onSelect(items[+b.dataset.i], +b.dataset.i);
  });
  onSelect(items[selected], selected);
}

/* ---------- Tema ---------- */
function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("theme"); } catch (_) { /* sin almacenamiento */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-toggle").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch (_) { /* nada */ }
  });
}

/* ---------- Tooltip ---------- */
function initTooltip() {
  const tip = $("#tooltip");
  document.addEventListener("mouseover", e => {
    const el = e.target.closest("[data-tip]");
    if (!el) { tip.hidden = true; return; }
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
  });
  document.addEventListener("mousemove", e => {
    if (tip.hidden) return;
    const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8);
    const y = e.clientY + 18 + tip.offsetHeight > innerHeight ? e.clientY - tip.offsetHeight - 12 : e.clientY + 18;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  });
  document.addEventListener("scroll", () => { tip.hidden = true; }, { passive: true });
}

/* ---------- Portada y textos ---------- */
function renderStatic() {
  const { config: cfg, results: r } = state;
  $$("[data-bind]").forEach(el => { el.textContent = cfg[el.dataset.bind] ?? ""; });
  $$("[data-href]").forEach(el => { el.href = cfg[el.dataset.href]; });
  $("#team").textContent = `Equipo: ${cfg.team.join(", ")}`;
  $("#lib-version").textContent = `esperanto_nlp ${r.library_version}`;

  const best = r.comparison.find(m => m.model === r.best_model);
  const articles = r.articles ? Object.values(r.articles).reduce((a, b) => a + b, 0) : null;
  const stats = [
    [num(r.n_paragraphs), "párrafos etiquetados"],
    [r.classes.length, `clases · ${cfg.topic}`],
    articles ? [num(articles), "artículos de Vikipedio"] : [num(r.split.test), "párrafos de test"],
    [dec(best.f1_macro), `F1 macro (${best.model})`],
  ];
  $("#hero-stats").innerHTML = stats.map(([v, l]) =>
    `<div class="stat"><div class="stat-value">${v}</div><div class="stat-label">${esc(l)}</div></div>`).join("");
}

/* ---------- Dataset ---------- */
function renderDataset() {
  const r = state.results;
  const counts = r.classes.map(c => r.class_counts[c]);
  const max = Math.max(...counts);
  $("#class-bars").innerHTML = r.classes.map(c => {
    const n = r.class_counts[c];
    const art = r.articles?.[c];
    const tip = `${classLabel(c)}: ${num(n)} párrafos (${pct(n / r.n_paragraphs)})` +
      (art ? ` de ${num(art)} artículos` : "") + ` · mediana ${r.median_words[c]} palabras`;
    return hbar({ label: `${swatch(c)}<span>${esc(c)}</span>`, value: n, max, text: num(n), color: classColor(c), tip });
  }).join("");
  const ratio = max / Math.min(...counts);
  $("#imbalance-note").textContent =
    `La clase mayor tiene ${dec(ratio, 1)} veces más párrafos que la menor. Por eso usamos F1 macro y class_weight="balanced".`;

  // Histograma de longitudes
  const bins = r.length_bins;
  const labels = bins.map((b, i) => (i === 0 ? `<${b}` : `${bins[i - 1]}–${b - 1}`)).concat(`≥${bins.at(-1)}`);
  const hmax = Math.max(...r.classes.flatMap(c => r.length_hist[c]));
  $("#length-legend").innerHTML = legend(r.classes);
  $("#length-hist").innerHTML = labels.map((lab, k) =>
    `<div class="vgroup">${r.classes.map(c => {
      const v = r.length_hist[c][k];
      return `<div class="vbar" style="height:${(v / hmax) * 100}%;background:${classColor(c)}" data-tip="${esc(c)} · ${lab} palabras: ${num(v)} párrafos"></div>`;
    }).join("")}<span class="vgroup-label">${lab}</span></div>`).join("");

  // Explorador de párrafos
  let current = r.classes[0], pos = 0;
  const show = () => {
    const list = r.samples[current] || [];
    $("#sample-text").textContent = list.length ? list[pos % list.length] : "—";
    $("#sample-text").style.borderLeftColor = classColor(current);
  };
  segmented($("#sample-tabs"), r.classes.map(c => ({ html: `${swatch(c)} ${esc(c)}`, c })), it => {
    current = it.c; pos = 0; show();
  });
  $("#sample-next").addEventListener("click", () => { pos++; show(); });
}

/* ---------- Resultados ---------- */
function renderResults() {
  const r = state.results;
  $("#split-note").textContent =
    `Partición estratificada: ${num(r.split.train)} párrafos de entrenamiento y ${num(r.split.test)} de test ` +
    `(semilla ${r.split.seed}). Mismo preprocesado y modelos que el cuaderno.`;

  // Comparación de modelos
  const rows = r.comparison.map(m => {
    const best = m.model === r.best_model;
    const metric = (v, name) => `<div class="metric" data-tip="${esc(m.model)} · ${name}: ${dec(v)}">
      <div class="hbar-track"><div class="hbar-fill metric-fill" style="width:${v * 100}%"></div></div>
      <span class="metric-value">${dec(v)}</span></div>`;
    return `<div class="compare-row${best ? " is-best" : ""}">
      <div class="compare-name">${esc(m.model)}${best ? '<span class="tag">mejor</span>' : ""}</div>
      ${metric(m.accuracy, "Accuracy")}${metric(m.f1_macro, "F1 macro")}</div>`;
  }).join("");
  $("#model-compare").innerHTML =
    `<div class="compare-row compare-head"><div class="compare-name">Modelo</div><div>Accuracy</div><div>F1 macro</div></div>${rows}`;
  $("#metric-legend").innerHTML =
    `<span class="muted">Escala 0–1 · en verde el mejor modelo según F1 macro (la métrica de referencia, porque las clases están desequilibradas).</span>`;

  // Matriz de confusión
  const models = Object.keys(r.confusion);
  segmented($("#cm-tabs"), models.map(m => ({ label: m, m })), it => renderConfusion(it.m),
    models.indexOf(r.best_model));

  // Métricas por clase
  $("#report-model").textContent = `(${r.best_model})`;
  $("#report-table").innerHTML =
    `<thead><tr><th>Clase</th><th class="num">Precisión</th><th class="num">Recall</th><th class="num">F1</th><th class="num">Soporte</th></tr></thead><tbody>` +
    r.classes.map(c => {
      const x = r.report[c];
      return `<tr><td>${classTag(c)}</td><td class="num">${dec(x.precision)}</td><td class="num">${dec(x.recall)}</td><td class="num">${dec(x["f1-score"])}</td><td class="num">${x.support}</td></tr>`;
    }).join("") + "</tbody>";

  // Errores más frecuentes
  const emax = Math.max(1, ...r.error_pairs.map(e => e[2]));
  const nerr = r.error_pairs.reduce((a, e) => a + e[2], 0);
  $("#error-pairs").innerHTML = r.error_pairs.map(([real, pred, n], i) =>
    hbar({ label: `<span>${esc(real)} → ${esc(pred)}</span>`, value: n, max: emax, text: num(n),
      color: "var(--ink-muted)", top: i === 0, tip: `${n} párrafos de ${real} clasificados como ${pred} (${pct(n / nerr)} de los errores)` })).join("") +
    `<p class="note">${num(nerr)} errores de ${num(r.split.test)} párrafos de test.</p>`;

  // Términos característicos
  const tt = r.top_terms;
  $("#top-terms").innerHTML = Object.keys(tt).length ? r.classes.map(c => {
    const terms = tt[c] || [];
    const tmax = Math.max(...terms.map(t => t[1]));
    return `<div><h4>${swatch(c)}${esc(c)}</h4><ol>${terms.slice(0, 10).map(([t, w]) =>
      `<li data-tip="${esc(t)}: coeficiente ${dec(w, 2)}"><span lang="eo">${esc(t)}</span><div class="hbar-track"><div class="hbar-fill" style="width:${(w / tmax) * 100}%;background:${classColor(c)}"></div></div></li>`).join("")}</ol></div>`;
  }).join("") : `<p class="muted">El mejor modelo no es lineal.</p>`;

  // Ejemplos
  const renderEx = kind => {
    $("#examples").innerHTML = r.examples[kind].map(ex => `<div class="example">
      <p lang="eo">${esc(ex.text)}${ex.text.length >= 400 ? "…" : ""}</p>
      <div class="example-meta"><span>Real: ${classTag(ex.real)}</span><span>Predicha: ${classTag(ex.pred)}</span>
      ${stack(ex.proba, r.classes)}<span>${pct(Math.max(...ex.proba))} de confianza</span></div></div>`).join("");
  };
  $("#ex-tabs").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("#ex-tabs button").forEach(x => x.setAttribute("aria-selected", x === b));
    renderEx(b.dataset.ex);
  });
  renderEx("correct");
}

function renderConfusion(model) {
  const r = state.results, cls = r.classes, m = r.confusion[model];
  const steps = 8;
  let html = `<div class="cm-head cm-axis">real \\ pred.</div>` +
    cls.map(c => `<div class="cm-head col">${swatch(c)}${esc(c)}</div>`).join("");
  m.forEach((row, i) => {
    const total = row.reduce((a, b) => a + b, 0) || 1;
    html += `<div class="cm-head">${swatch(cls[i])}${esc(cls[i])}</div>`;
    row.forEach((v, j) => {
      const share = v / total;
      const k = Math.min(steps - 1, Math.round(share * (steps - 1)));
      // Paso 0 = fondo neutro (sigue al tema); pasos 1-3 son azules claros en
      // ambos temas, así que llevan texto oscuro; del 4 en adelante, blanco.
      const ink = k === 0 ? "var(--ink)" : k < 4 ? "#0b0b0b" : "#ffffff";
      html += `<div class="cm-cell" style="background:var(--seq-${k});color:${ink}"
        data-tip="Real ${cls[i]} → predicha ${cls[j]}: ${v} párrafos (${pct(share)} de la fila)">
        <span><strong>${v}</strong><small>${pct(share, 0)}</small></span></div>`;
    });
  });
  const cm = $("#cm");
  cm.style.gridTemplateColumns = `minmax(90px, 140px) repeat(${cls.length}, minmax(64px, 1fr))`;
  cm.innerHTML = html;
}

/* ---------- Python en el navegador ---------- */
function setStatus(st, text) {
  const el = $("#py-status");
  el.dataset.state = st;
  $(".py-status-text", el).textContent = text;
}

async function initPython() {
  try {
    setStatus("loading", "Cargando Python…");
    if (typeof loadPyodide !== "function") throw new Error("no se pudo descargar Pyodide");
    const [py, zip, demoSrc, model] = await Promise.all([
      loadPyodide(),
      fetch("esperanto_nlp.zip").then(okOr("esperanto_nlp.zip")).then(r => r.arrayBuffer()),
      fetch("py/web_demo.py").then(okOr("web_demo.py")).then(r => r.text()),
      fetch("data/model.json").then(okOr("model.json")).then(r => r.text()),
    ]);
    setStatus("loading", "Cargando librería…");
    py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide" });
    py.FS.writeFile("/home/pyodide/web_demo.py", demoSrc);
    py.globals.set("MODEL_JSON", model);
    await py.runPythonAsync(`
import sys
if "/home/pyodide" not in sys.path:
    sys.path.insert(0, "/home/pyodide")
import web_demo
clf = web_demo.Classifier(MODEL_JSON)
del MODEL_JSON
web_demo.get_nlp()
`);
    state.py = py;
    state.demo = py.globals.get("web_demo");
    state.clf = py.globals.get("clf");
    state.modelClasses = state.clf.classes.toJs();
    setStatus("ready", "Python listo");
    $$("#lib-run, #clf-run, #wiki-run").forEach(b => { b.disabled = false; });
    runLibrary();
  } catch (err) {
    console.error(err);
    setStatus("error", "Python no disponible");
    const msg = `No se pudo cargar Python en el navegador (${err.message}). Las demos interactivas no están disponibles; el resto de la página sí.`;
    $("#clf-error").textContent = msg;
    $("#clf-error").hidden = false;
  }
}

function okOr(name) {
  return r => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r; };
}

/* ---------- Demo de la librería ---------- */
const POS_NAMES = {
  NOUN: "sustantivo", VERB: "verbo", AUX: "auxiliar", ADJ: "adjetivo", ADV: "adverbio", PROPN: "nombre propio",
  PRON: "pronombre", DET: "determinante", ADP: "preposición", CCONJ: "conj. coordinante", SCONJ: "conj. subordinante",
  NUM: "numeral", PART: "partícula", INTJ: "interjección", PUNCT: "puntuación", SYM: "símbolo", X: "otro",
};

function initLibrary() {
  const cfg = state.config;
  $("#lib-input").value = cfg.library_examples[0] || "";
  chips($("#lib-examples"), cfg.library_examples, t => { $("#lib-input").value = t; runLibrary(); });
  $("#lib-run").addEventListener("click", runLibrary);
  $("#lib-x").addEventListener("change", runLibrary);
  $("#lib-input").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runLibrary(); });
  $("#pos-legend").innerHTML = [["NOUN", "--pos-noun"], ["VERB", "--pos-verb"], ["ADJ", "--pos-adj"], ["ADV", "--pos-adv"], ["PROPN", "--pos-propn"]]
    .map(([p, v]) => `<span class="legend-item"><span class="swatch" style="background:var(${v})"></span>${p} · ${POS_NAMES[p]}</span>`).join("") +
    `<span class="legend-item"><span class="swatch" style="background:var(--axis)"></span>resto</span>
     <span class="legend-item" style="opacity:.55">atenuado = stop-word</span>`;
}

function chips(container, items, onPick) {
  container.innerHTML = items.map((t, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(t)}" lang="eo">${esc(t)}</button>`).join("");
  container.addEventListener("click", e => {
    const b = e.target.closest(".chip");
    if (b) onPick(items[+b.dataset.i]);
  });
}

function runLibrary() {
  if (!state.demo) return;
  const text = $("#lib-input").value.trim();
  if (!text) return;
  const out = JSON.parse(state.demo.analyze(text, $("#lib-x").checked));
  $("#lib-output").hidden = false;
  $("#lib-count").textContent = `(${out.tokens.length})`;
  $("#lib-inline").innerHTML = out.tokens.map(t => {
    const punct = t.pos === "PUNCT";
    const tip = `${t.text} → lema «${t.lemma}» · ${t.pos} (${POS_NAMES[t.pos] || t.pos})${t.morph ? " · " + t.morph : ""}${t.stop ? " · stop-word" : ""}`;
    return `<span class="itok${t.stop ? " is-stop" : ""}${punct ? " is-punct" : ""}" data-pos="${esc(t.pos)}" data-tip="${esc(tip)}">
      <span class="itok-text">${esc(t.text)}</span>${punct ? "" : `<span class="itok-lemma">${esc(t.lemma)}</span><span class="itok-pos">${esc(t.pos)}</span>`}</span>`;
  }).join("");
  $("#lib-tbody").innerHTML = out.tokens.map((t, i) => `<tr class="${t.stop ? "is-stop" : ""}">
    <td class="num muted">${i}</td><td lang="eo">${esc(t.text)}</td><td lang="eo">${esc(t.lemma)}</td>
    <td><span class="pos" data-pos="${esc(t.pos)}" data-tip="${esc(POS_NAMES[t.pos] || t.pos)}">${esc(t.pos)}</span></td>
    <td>${esc(t.morph) || '<span class="muted">—</span>'}</td><td>${t.stop ? '<span class="stop-yes">sí</span>' : ""}</td></tr>`).join("");
  $("#lib-sents").innerHTML = out.sents.map(s => `<li>${esc(s)}</li>`).join("");
}

/* ---------- Clasificador ---------- */
function initClassifier() {
  const cfg = state.config;
  $("#clf-input").value = cfg.classifier_examples[0] || "";
  chips($("#clf-examples"), cfg.classifier_examples, t => { $("#clf-input").value = t; classifyText(); });
  chips($("#wiki-examples"), cfg.wikipedia_examples, t => { $("#wiki-input").value = t; classifyWiki(); });
  $("#clf-run").addEventListener("click", classifyText);
  $("#wiki-run").addEventListener("click", classifyWiki);
  $("#wiki-input").addEventListener("keydown", e => { if (e.key === "Enter") classifyWiki(); });
  $("#clf-input").addEventListener("keydown", e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) classifyText(); });
  $("#clf-tabs").addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    $$("#clf-tabs button").forEach(x => x.setAttribute("aria-selected", x === b));
    $$("[data-panel]").forEach(p => { p.hidden = p.dataset.panel !== b.dataset.tab; });
  });
}

function showError(msg) {
  $("#clf-error").textContent = msg;
  $("#clf-error").hidden = !msg;
}

function renderPrediction({ title, proba, terms, meta, paragraphs }) {
  const classes = state.modelClasses;
  const best = proba.indexOf(Math.max(...proba));
  $("#clf-output").hidden = false;
  $("#clf-title").textContent = title;
  const badge = $("#clf-badge");
  badge.textContent = `${classes[best]} · ${pct(proba[best])}`;
  badge.style.background = classColor(classes[best]);
  badge.style.color = classInk(classes[best]);
  $("#clf-bars").innerHTML = classes.map((c, i) => hbar({
    label: `${swatch(c)}<span>${esc(c)}</span>`, value: proba[i], max: 1, text: pct(proba[i]),
    color: classColor(c), top: i === best, tip: `${classLabel(c)}: ${pct(proba[i], 2)}`,
  })).join("");
  $("#clf-meta").textContent = meta;
  $("#clf-terms-card").hidden = !terms.length;
  $("#clf-terms").innerHTML = terms.map(([t, w]) =>
    `<span class="term" lang="eo" data-tip="Aporta ${dec(w, 3)} a la puntuación de ${classes[best]}">${esc(t)} <small>+${dec(w, 2)}</small></span>`).join("");

  $("#clf-paras-card").hidden = !paragraphs;
  if (paragraphs) {
    $("#clf-paras").innerHTML = paragraphs.map(p => {
      const b = p.proba.indexOf(Math.max(...p.proba));
      return `<div class="para"><p lang="eo">${esc(p.text)}</p><div>${stack(p.proba, classes)}
        <div class="stack-caption">${classTag(classes[b])} ${pct(p.proba[b], 0)}</div></div></div>`;
    }).join("");
  }
}

function classifyText() {
  if (!state.clf) return;
  showError("");
  const text = $("#clf-input").value.trim();
  if (!text) return;
  const r = JSON.parse(state.clf.predict_json(text, false));
  if (!r.n_known) {
    showError("Ninguna palabra del texto está en el vocabulario del modelo: la predicción sería solo el sesgo de las clases. Prueba con un texto más largo en Esperanto.");
    $("#clf-output").hidden = true;
    return;
  }
  renderPrediction({
    title: "Resultado del texto",
    proba: r.proba,
    terms: r.terms,
    meta: `${r.n_tokens} lemas útiles tras el preprocesado, ${r.n_known} conocidos por el modelo.`,
  });
}

async function classifyWiki() {
  if (!state.clf) return;
  showError("");
  const title = $("#wiki-input").value.trim();
  if (!title) return;
  const btn = $("#wiki-run");
  btn.disabled = true;
  btn.textContent = "Descargando…";
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 10000);
  try {
    const params = new URLSearchParams({
      action: "query", prop: "extracts", explaintext: "1", redirects: "1",
      format: "json", formatversion: "2", origin: "*", titles: title,
    });
    const resp = await fetch(`${state.config.wikipedia_api}?${params}`, { signal: ctrl.signal });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const page = data?.query?.pages?.[0];
    if (!page || page.missing || page.invalid) throw new Error(`no existe el artículo «${title}» en Vikipedio`);
    const paragraphs = (page.extract || "").split(/\n+/)
      .map(p => p.trim())
      .filter(p => p && !p.startsWith("=") && p.split(/\s+/).length >= 15);
    if (!paragraphs.length) throw new Error("el artículo no tiene párrafos de al menos 15 palabras");
    const r = JSON.parse(state.clf.predict_many_json(JSON.stringify(paragraphs), false));
    // Términos de los párrafos que votan por la clase ganadora, sumados.
    const bestClass = state.modelClasses[r.mean.indexOf(Math.max(...r.mean))];
    const terms = r.paragraphs.filter(p => p.pred === bestClass).flatMap(p => p.terms)
      .reduce((acc, [t, w]) => acc.set(t, (acc.get(t) || 0) + w), new Map());
    renderPrediction({
      title: page.title,
      proba: r.mean,
      terms: [...terms].sort((a, b) => b[1] - a[1]).slice(0, 10),
      meta: `${paragraphs.length} párrafos de eo.wikipedia.org/wiki/${page.title.replace(/ /g, "_")}. Probabilidad media de los párrafos, como en el cuaderno.`,
      paragraphs: r.paragraphs.map((p, i) => ({ ...p, text: paragraphs[i] })),
    });
  } catch (err) {
    const reason = err.name === "AbortError" ? "la petición tardó más de 10 s" : err.message;
    showError(`No se pudo clasificar el artículo: ${reason}.`);
    $("#clf-output").hidden = true;
  } finally {
    clearTimeout(timer);
    btn.disabled = false;
    btn.textContent = "Descargar y clasificar";
  }
}

/* ---------- Arranque ---------- */
async function main() {
  initTheme();
  initTooltip();
  try {
    const [config, results] = await Promise.all([
      fetch("site_config.json").then(okOr("site_config.json")).then(r => r.json()),
      fetch("data/results.json").then(okOr("results.json")).then(r => r.json()),
    ]);
    state.config = config;
    state.results = results;
    state.classes = results.classes;
  } catch (err) {
    document.querySelector("main").insertAdjacentHTML("afterbegin",
      `<div class="wrap"><p class="error">No se pudieron cargar los datos de la web (${esc(err.message)}). Genera la web con <code>python scripts/build_web.py</code>.</p></div>`);
    return;
  }
  renderStatic();
  renderDataset();
  renderResults();
  initLibrary();
  initClassifier();
  initPython();
}

main();
