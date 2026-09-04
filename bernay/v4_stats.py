"""
v4_stats — consumer for the Western population-statistics KB
(v4_stats.json, built by v4_stats_seed.py).

  region_income(region_text)   -> sourced income record for an accent/
                                   region string ('US South' -> $93,650
                                   median family income [Census/Fed]), or
                                   None. This is the accent->income line
                                   the user asked for.
  occasion_audience(text)       -> sourced gift-occasion record if the ad
                                   names one ('Father's Day' -> $199.38/
                                   person [NRF]), or None.
  stats_sentences()             -> clean factual sentences distilled from
                                   the KB for the prime corpus (no slop).

Safe no-op (returns None / "") if the KB isn't built yet, so importing it
before v4_stats_seed.py has run never crashes a caller.
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "v4_stats.json")

_KB = None


def load_kb():
    global _KB
    if _KB is None:
        if os.path.exists(KB_PATH):
            with open(KB_PATH, encoding="utf-8") as f:
                _KB = json.load(f)
        else:
            _KB = {"regional_income": [], "occasions": [],
                   "spending_by_age": [], "population": [],
                   "state_income": [], "national_income": {},
                   "life_expectancy": {}, "health_burden": [],
                   "income_by_race": [], "income_by_age": []}
    return _KB


_GENERIC_US = re.compile(
    r"\b(u\.?s\.?a?|us general|general american|united states|america|"
    r"american)\b", re.I)


def region_income(region_text):
    """Match a free-text region/accent string to a regional income record.

    A SPECIFIC sub-region (matched by its full name or a distinctive alias,
    word-boundary) wins. A GENERIC country signal ('US', 'General
    American', 'USA') falls back to the NATIONAL figure — never a
    sub-region (the old substring match leaked Northeast for a bare 'US',
    because every region name starts with 'US '). Returns a record with
    'display'/'measure'/'source_name'/'source_url', or None to abstain."""
    if not region_text:
        return None
    low = region_text.lower()
    kb = load_kb()
    # MOST specific first: a named US state ("California" -> $95,521 household,
    # not the broad West region's family figure).
    for s in kb.get("state_income", []):
        if re.search(r"\b" + re.escape(s["name"].lower()) + r"\b", low):
            return {"region": s["name"],
                    "measure": "median household income",
                    "display": s["display"], "value": s["value"],
                    "year": s["year"], "source_name": s["source_name"],
                    "source_url": s["source_url"]}
    recs = kb.get("regional_income", [])
    # then a census region / European nation by full name OR distinctive alias,
    # but NOT the bare country token (so 'US' alone never selects a sub-region)
    for r in recs:
        names = [r["region"].lower()] + [a.lower() for a in r.get("aliases", [])]
        for name in names:
            if name in ("us", "usa"):
                continue
            if re.search(r"\b" + re.escape(name) + r"\b", low):
                return r
    # generic US / General American -> national median (honest: no region)
    if _GENERIC_US.search(low):
        nat = load_kb().get("national_income")
        if nat:
            return {"region": "US (national)", "measure": nat.get("measure"),
                    "display": nat.get("display"), "value": nat.get("value"),
                    "year": nat.get("year"),
                    "source_name": nat.get("source_name"),
                    "source_url": nat.get("url", nat.get("source_url"))}
    return None


def occasion_audience(text):
    """If the ad text/occasion names a known gift occasion, return its
    sourced spending record, else None."""
    if not text:
        return None
    low = text.lower()
    for o in load_kb().get("occasions", []):
        if o["name"].lower() in low or any(
                a in low for a in o.get("aliases", [])):
            return o
    return None


# ---- ethnicity: EXPLICIT-signal-only (US) ----------------------------------
# The race/ethnicity income layer is REFERENCE data, surfaced ONLY when the
# copy/creative EXPLICITLY signals an ethnic market — a Spanish-language ad or
# an explicit community reference. It is NEVER inferred from accent, name, or
# appearance (accent != race), and is US-only (European ads key on nationality
# via region_income). This keeps a sensitive axis grounded and abstaining.
_ES_MARK = re.compile(r"[ñ¿¡áéíóú]")
_ES_STOP = re.compile(
    r"\b(el|la|los|las|un|una|unos|unas|de|del|que|porque|por|para|con|sin|"
    r"su|sus|tu|tus|más|pero|cuando|este|esta|estos|estas|nuestro|nuestra|"
    r"también|años|salud|dolor|tu salud|sobre|muy|ya|cómo|qué)\b", re.I)
_ETHNIC_TERMS = [
    (re.compile(r"\b(hispanic|latino|latina|latinx|latin[ -]community|"
                r"comunidad latina|para latinos)\b", re.I),
     "Hispanic (any race)", "explicit Hispanic/Latino reference"),
    (re.compile(r"\b(african[ -]american|black[ -]community|black[ -]owned|"
                r"for the black community)\b", re.I),
     "Black", "explicit Black-community reference"),
    (re.compile(r"\b(asian[ -]american|aapi|asian[ -]community)\b", re.I),
     "Asian", "explicit Asian-American reference"),
]


def explicit_ethnic_signal(text):
    """-> (group, evidence) ONLY when the copy EXPLICITLY signals an ethnic
    market (a Spanish-language ad, or an explicit community reference);
    otherwise None. Deterministic and copy-only — never accent/appearance."""
    if not text:
        return None
    for rx, group, ev in _ETHNIC_TERMS:
        if rx.search(text):
            return group, ev
    # Spanish-language copy: Spanish diacritics/inverted punctuation
    # (ñ ¿ ¡ á é í ó ú) essentially never occur in English, so a couple of
    # them ALONGSIDE Spanish function words is decisive; a stray loanword
    # accent ("café") alone is not (it lacks the function words). The pure-
    # stopword path covers accent-stripped / all-caps Spanish.
    low = text.lower()
    marks = len(_ES_MARK.findall(low))
    stops = len(_ES_STOP.findall(low))
    words = max(len(low.split()), 1)
    if (marks >= 2 and stops >= 2) or (stops >= 6 and stops / words > 0.08):
        return "Hispanic (any race)", "Spanish-language copy"
    return None


def income_by_race(group):
    """Sourced US median household income for an explicitly-signaled group,
    or None. `group` is one of the strings explicit_ethnic_signal returns."""
    if not group:
        return None
    recs = load_kb().get("income_by_race", [])
    for r in recs:                                  # exact group match first
        if r["group"].lower() == group.lower():
            return r
    g = group.lower().split(" (")[0].split(",")[0].strip()   # leading token
    for r in recs:                                  # 'Hispanic' != 'non-Hispanic'
        if r["group"].lower().split(" (")[0].split(",")[0].strip() == g:
            return r
    return None


# Non-English-market detection: a lander WRITTEN in a European language is
# targeted for that country, so its income baseline must be that country's
# figure — NOT the US ACS age table. Distinctive stopwords + diacritics per
# language; returns a region NAME that region_income() resolves, or None
# (treat as US/English). Spanish is deliberately omitted — it's ambiguous
# between Spain and US-Hispanic, which explicit_ethnic_signal() already
# handles on its own axis.
# Unicode SCRIPT -> market. A script is definitive (unlike stopword density),
# so detect_market checks these first. Devanagari is listed before other Indic
# scripts because Hindi is the common DR case. Han/Kana/Hangul are separated so
# each routes to its own NLLB source language.
_SCRIPT_MARKETS = [
    ("[ऀ-ॿ]", "India"),        # Devanagari (Hindi/Marathi)
    ("[ঀ-৿]", "Bangladesh"),   # Bengali
    ("[؀-ۿ]", "Arabia"),       # Arabic/Persian/Urdu
    ("[฀-๿]", "Thailand"),     # Thai
    ("[Ѐ-ӿ]", "Russia"),       # Cyrillic
    ("[Ͱ-Ͽ]", "Greece"),       # Greek
    ("[֐-׿]", "Israel"),       # Hebrew
    ("[぀-ヿ]", "Japan"),        # Hiragana/Katakana
    ("[가-힯]", "Korea"),        # Hangul
    ("[一-鿿]", "China"),        # Han (after kana: JP mixes both)
    ("[஀-௿]", "Tamil"),        # Tamil
    ("[ఀ-౿]", "Telugu"),       # Telugu
]

_MARKET_LANGS = [
    ("Sweden", ["och", "är", "för", "att", "det", "som", "inte", "med",
                "den", "här", "köp", "läkare", "hälsa", "svenska", "viktet",
                "kroppen", "kvinnor"], "åäö"),
    ("Germany", ["und", "der", "die", "das", "ist", "nicht", "für", "mit",
                 "sie", "auch", "sich", "gesundheit", "abnehmen", "arzt",
                 "köper"], "äöüß"),
    ("France", ["le", "la", "les", "et", "des", "pour", "vous", "est", "une",
                "avec", "sur", "santé", "médecin", "ne", "pas"], "éèêàçù"),
    # Italian: the original 13 words missed the MOST common ones, so a real
    # Italian ad ("La trasformazione in 6 settimane con NAC di cui chi beve...
    # Il tuo fegato lavora di notte mentre dormi") tripped only 3 hits, stayed
    # under min_hits=5, and was never detected -> never translated -> the whole
    # decomposition ran on raw Italian and read it as "Learning a Language".
    ("Italy", ["il", "che", "di", "per", "non", "sono", "con", "una", "più",
               "della", "salute", "medico", "questo", "la", "le", "lo", "un",
               "e", "in", "si", "del", "al", "dei", "ma", "come", "anche",
               "tuo", "tua", "mio", "sua", "ti", "ci", "se", "da", "dopo",
               "quando", "chi", "cui", "ora", "senza", "molto", "già",
               "essere", "fare", "corpo", "pelle", "anni", "giorni"], "àèìòù"),
    ("Netherlands", ["het", "een", "van", "voor", "niet", "zijn", "ook",
                     "gezondheid", "afvallen", "arts", "dit", "deze"], ""),
    # Indonesian/Malay, Spanish, Portuguese: LATIN-script languages the table
    # never covered, so their ads fell through undetected (3 Indonesian ads in
    # the 966 corpus — kidney stones + scabies — decomposed as noise).
    ("Indonesia", ["yang", "dan", "untuk", "tidak", "ini", "itu", "dengan",
                   "dari", "kalau", "bisa", "saja", "sudah", "akan", "juga",
                   "ada", "kita", "anda", "kulit", "sehat", "obat", "tubuh",
                   "awal", "tanda", "oleh", "sangat", "tau", "banyak"], ""),
    ("Spain", ["que", "de", "la", "el", "los", "las", "para", "con", "una",
               "por", "más", "como", "pero", "este", "salud", "piel",
               "cuerpo", "años", "sin", "muy", "tu", "su"], "áéíóúñ¿¡"),
    ("Portugal", ["que", "de", "para", "com", "uma", "não", "mais", "como",
                  "por", "você", "seu", "sua", "saúde", "pele", "corpo",
                  "anos", "sem", "muito", "isso", "está"], "ãõáéíóúç"),
    ("Denmark", ["og", "det", "som", "ikke", "med", "for", "sundhed", "læge",
                 "dette", "kroppen"], "æø"),
    ("Norway", ["og", "det", "som", "ikke", "med", "for", "helse", "lege",
                "dette", "kroppen"], "æø"),
]


_EN_STOP = {"the", "and", "to", "of", "in", "is", "you", "your", "for",
            "with", "this", "that", "are", "it", "on", "we", "our", "have",
            "not", "can", "will", "how", "why", "what", "now", "more", "best",
            "all", "they", "from", "has", "was", "but", "out", "about", "if"}


def is_mostly_latin(text, thresh=0.85, min_letters=12):
    """True if the alphabetic content is predominantly Latin script — i.e. the
    OCR produced READABLE Latin text, not a mangled non-Latin script. Lets terse
    English ad copy (few function words) still count as readable/Western rather
    than being mislabeled 'a non-Latin script the OCR couldn't read'."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < min_letters:
        return False
    latin = sum(1 for c in letters
                if ("a" <= c.lower() <= "z") or ("À" <= c <= "ɏ"))
    return latin / len(letters) >= thresh


def looks_english(text, min_hits=5):
    """True if `text` is confidently English/Western. Primary signal: common
    English function-word density (>= min_hits distinct ones). FALLBACK for terse
    / graphical DR creatives (big bold phrases like 'VAGINAL pH RISES' carry
    almost no function words but are plainly readable English): accept when the
    copy is predominantly LATIN script, has >= 10 distinct words, and at least
    one English function word. This stops readable English ads from being
    mislabeled 'a non-Latin script the OCR couldn't read'. Genuinely non-Latin /
    mangled OCR (Devanagari, Cyrillic, CJK) fails is_mostly_latin and is still
    gated out; a real foreign LANGUAGE is caught earlier by detect_market."""
    if not text:
        return False
    words = set(re.findall(r"[a-z]+", text.lower()))
    if len(words & _EN_STOP) >= min_hits:
        return True
    return (is_mostly_latin(text) and len(words) >= 10
            and len(words & _EN_STOP) >= 1)


def detect_market(text, min_hits=5):
    """-> a region NAME (for region_income / translation) if `text` is clearly
    in a non-English language, else None (US/English). Used to pick the right
    income market AND the translation source language for a foreign lander.
    Non-Latin scripts are detected first (present once multilingual OCR has
    read them); then European languages by stopword density."""
    if not text:
        return None
    # NON-LATIN SCRIPTS -> their market (source lang mapped in
    # v4_translate.REGION_LANG, then to NLLB-200's FLORES code). A script is a
    # DEFINITIVE language signal — no stopword threshold needed — so this runs
    # first and catches languages the stopword table will never cover. Only
    # Devanagari was here before, so Bengali/Arabic/Thai/CJK/Cyrillic/Greek/
    # Hebrew ads were never detected -> never translated -> decomposed as noise
    # (measured: a Bengali pet-ear-cleaner ad in the 966 corpus).
    for _pat, _region in _SCRIPT_MARKETS:
        if re.search(_pat, text):
            return _region
    low = text.lower()
    words = set(re.findall(r"[a-zà-ÿäöüßæø]+", low))
    best, best_n = None, 0
    for region, stops, dia in _MARKET_LANGS:
        n = sum(1 for s in stops if s in words)
        if dia and any(c in low for c in dia):
            n += 2                       # diacritics are a strong tell
        if n > best_n:
            best, best_n = region, n
    # The winning language must also BEAT ENGLISH. An absolute floor alone is a
    # length trap: min_hits counts DISTINCT stopwords, so the longer a document
    # is the likelier it is to contain 5 of any language's function words by
    # accident. A 9,952-char ENGLISH supplement lander (RYZE mushroom coffee)
    # scored Italy on 'per' (per serving), 'a', 'in', 'no', 'e' (vitamin E),
    # 'da', 'come' — while neither of its two halves scored anything, because
    # only the concatenation crossed 5. It was then NLLB-translated
    # Italian->English TWICE at 251s each: 8.4 minutes of the 9-minute analyze,
    # and the copy every downstream reader sees was rewritten by a translator
    # fed the wrong source language.
    #
    # Comparing the two counts is the honest test — a real foreign lander has
    # far more of its own function words than English ones, and a mixed page
    # resolves to whichever language actually dominates. The floor stays, so
    # terse copy still abstains rather than guessing.
    en_hits = len(words & _EN_STOP)
    if best_n >= min_hits and best_n > en_hits:
        return best
    return None


def income_by_age(age_bucket):
    """Sourced median household income for a model age bucket (one of
    '18-24','25-34','35-44','45-54','55+'), or None to abstain. Returns a
    record with 'display'/'acs'/'source_name'/'source_url' — a POPULATION
    baseline for that age cohort (ACS B19049), used to ground an income read
    in the age we measured, NOT a claim about the individual buyer."""
    if not age_bucket:
        return None
    for r in load_kb().get("income_by_age", []):
        if r["bucket"] == age_bucket:
            return r
    return None


def stats_sentences():
    """Distil the KB into clean factual sentences for the prime corpus —
    the 'train it in' half. Quality register, every sentence a real
    sourced fact, deterministic order."""
    kb = load_kb()
    out = []
    nat = kb.get("national_income") or {}
    if nat:
        out.append(f"In {nat.get('year','recent years')}, the United States "
                   f"{nat.get('measure','median household income')} was "
                   f"{nat.get('display','')}.")
    for r in kb.get("regional_income", []):
        out.append(f"The {r['region']} {r['measure']} was about "
                   f"{r['display']} in {r['year']}.")
    for s in kb.get("state_income", []):
        out.append(f"In {s['name']}, the median household income was "
                   f"{s['display']} in {s['year']}.")
    for s in kb.get("spending_by_age", []):
        out.append(f"Households headed by someone aged {s['age']} spent on "
                   f"average {s['display']} per year in {s['year']} "
                   f"({s['note']}).")
    for o in kb.get("occasions", []):
        out.append(f"For {o['name']} in {o['year']}, shoppers spent about "
                   f"${o['avg_per_person']} per person, roughly {o['total']} "
                   f"in total, with about {int(o['participation']*100)} "
                   f"percent of people taking part.")
    for p in kb.get("population", []):
        out.append(f"The {p['fact']} was {p['value']} in {p['year']}.")
    le = kb.get("life_expectancy") or {}
    if le:
        out.append(f"US life expectancy at birth in {le.get('year')} was "
                   f"{le.get('overall')} overall — {le.get('men')} for men "
                   f"and {le.get('women')} for women.")
    for h in kb.get("health_burden", []):
        out.append(f"The {h['fact']} was {h['value']} in {h['year']}.")
    for r in kb.get("income_by_race", []):
        out.append(f"In {r['year']}, the US median household income for "
                   f"{r['group']} households was {r['display']}.")
    for a in kb.get("income_by_age", []):
        out.append(f"In {a['year']}, households headed by someone aged "
                   f"{a['acs']} had a median household income of "
                   f"{a['display']}.")
    return "\n".join(out)


if __name__ == "__main__":
    kb = load_kb()
    if not kb.get("regional_income"):
        print("(KB empty — run v4_stats_seed.py first)")
        raise SystemExit(0)
    print("=== region_income probes ===")
    for q in ["mild Southern US", "US South", "New England", "California",
              "Midwest", "UK", "Australia"]:
        r = region_income(q)
        if r:
            print(f"  {q:18} -> {r['display']} ({r['measure']}, "
                  f"{r['year']}) [{r['source_name']}]")
        else:
            print(f"  {q:18} -> (no match — abstain)")
    print("\n=== occasion probes ===")
    for q in ["the perfect gift for Father's Day", "back to school sale"]:
        o = occasion_audience(q)
        print(f"  {q:34} -> "
              + (f"${o['avg_per_person']}/person [{o['source_name']}]"
                 if o else "(no match)"))
    print("\n=== distilled corpus sentences ===")
    print(stats_sentences())
