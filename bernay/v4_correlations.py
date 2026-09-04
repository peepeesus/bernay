"""
Correlations knowledge base — grounds avatar/demographic inference in real
Western epidemiological/demographic studies instead of thin keyword
markers. Built by the v4-correlations-build workflow into
v4_correlations.json:

    {"painpoints": [{id,name,domain,description,cues[],product_categories[]}],
     "correlations": {painpoint_id: [{factor,value,magnitude,finding,
                       source_name,source_url,year,region,confidence}]}}

match_painpoints(text)  -> painpoints whose ad-language cues appear.
demographic_prior(text) -> weighted age/gender/life_stage priors implied
                            by those painpoints' studies, with citations.

No heavy deps (regex + json). Safe no-op if the KB isn't built yet, so it
can be imported before the workflow finishes. Figures are public-study
AGGREGATES used as directional priors — every contribution carries its
source.
"""

import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
KB_PATH = os.path.join(HERE, "v4_correlations.json")

AGE_BUCKETS = ["18-24", "25-34", "35-44", "45-54", "55+"]
_AGE_RANGES = {"18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
               "45-54": (45, 54), "55+": (55, 120)}
CONF_W = {"high": 1.0, "medium": 0.6, "low": 0.3}

_KB = None


def load_kb():
    global _KB
    if _KB is None:
        if os.path.exists(KB_PATH):
            with open(KB_PATH, encoding="utf-8") as f:
                _KB = json.load(f)
        else:
            _KB = {"painpoints": [], "correlations": {}}
    return _KB


_STOP = {"and", "the", "of", "a", "your", "&", "symptoms", "health",
         "issues", "problems", "slowdown"}

# generic head nouns that must NOT match a painpoint on their own — they
# turn up in unrelated copy ("high-PRESSURE washer" != "blood PRESSURE",
# "cleaning PERFORMANCE" != "sexual PERFORMANCE", "lose WEIGHT" is its own
# niche). A concept with a generic head only matches on its full two-word
# phrase; a DISTINCTIVE single word ("arthritis", "menopause") may match
# alone, since it almost never appears outside its condition.
_GENERIC_HEAD = {"pain", "loss", "fat", "weight", "health", "stress",
                 "decline", "performance", "pressure", "savings", "relief",
                 "support", "fear", "squeeze", "sugar", "disease", "care",
                 "aging", "dysfunction", "fatigue", "sleep", "energy",
                 "cancer", "enlargement", "mortgage", "security",  # phrase-only
                 # ('breast cancer','prostate enlargement','reverse mortgage',
                 # 'social security') — never bare cancer/enlargement/mortgage/
                 # security (a home-security ad isn't Social Security)
                 "shipping",  # 'Packaging & Shipping' (B2B fulfillment) must not
                 # fire on the 'Free shipping' every DR health ad offers — needs
                 # 'poly mailers'/'fulfillment'/'your orders', not bare shipping
                 "living"}   # 'Cost of Living' must not fire on "worms living
                             # inside you"; "cost of living" 2-word still matches


def _name_terms(name):
    """Distinctive match phrases for a painpoint name. The name is split on
    &/and/comma/slash into CONCEPTS — 'Joint Pain & Arthritis' -> 'joint
    pain' + 'arthritis', 'Low Libido & Sexual Performance' -> 'low libido'
    + 'sexual performance' — and each concept yields its core two-word
    phrase plus, when the head noun is distinctive, that head alone. This
    catches plain mentions ('arthritis', 'blood pressure') the cue list
    didn't enumerate while keeping the aquoxis guard: a bare generic head
    ('pressure', 'performance') is never returned, so a 'high-pressure
    water device' is not read as 'high blood pressure'."""
    out = []
    for concept in re.split(r"\s*(?:&|/|,|\band\b)\s*", name.lower()):
        words = [w for w in re.findall(r"[a-z]+", concept)
                 if w not in _STOP]
        if not words:
            continue
        if len(words) >= 2:
            out.append(" ".join(words[-2:]))       # 'blood pressure'
        head = words[-1]
        if head not in _GENERIC_HEAD and len(head) >= 5:
            out.append(head)                        # 'arthritis','menopause'
    seen, res = set(), []
    for t in out:                                   # dedup, keep order
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res


# Health-relevance gate for the semantic gap-filler: a broad body-part / symptom /
# health-and-pet lexicon. The 987 painpoint anchors are so broad that off-topic ads
# score ~0.5 against SOME painpoint, so the semantic fallback runs ONLY when the copy
# trips this gate — keeping hotels / e-commerce / SaaS / education as no-match.
_HEALTH_GATE = re.compile(
    r"health|wellness|supplement|vitamin|nutrient|remedy|symptom|condition|"
    r"pain|ache|sore|stiff|swell|inflam|cramp|numb|tingl|"
    r"blood|sugar|glucose|cholesterol|pressure|circulation|artery|heart|vein|"
    r"gut|digest|bloat|bowel|constipat|probiotic|"
    r"joint|arthrit|cartilage|bone|muscle|nerve|back|knee|posture|"
    r"skin|wrinkle|collagen|acne|eczema|hair|scalp|nail|"
    r"weight|belly|fat|metabol|slim|"
    r"sleep|insomnia|anxiety|stress|mood|depress|focus|memory|brain|fog|cortisol|"
    r"hormone|menopause|testosterone|thyroid|libido|erectile|prostate|menstru|"
    r"vision|eyesight|macular|lasik|hearing|tinnitus|"
    r"teeth|tooth|\bgum|dental|denture|"
    r"kidney|liver|bladder|urinary|"
    r"immune|detox|energy|fatigue|vitality|adaptogen|"
    r"diabet|cancer|tumou?r|neuropath|migraine|"
    r"doctor|clinic|medical|prescription|\bmeds\b|"
    r"\bdog|\bcat\b|\bpet|paws|fleas|kibble", re.I)

# Union TOPIC gate for the semantic gap-filler now that the KB spans the whole
# consumer-marketing universe (health + wealth + business + relationships +
# beauty + self-growth + survival + lifestyle). Fires the semantic rescue when
# >=2 DISTINCT topical tokens appear — an ad in ANY of these verticals clears
# it; only genuinely contentless/gibberish copy fails. The cosine>=0.58 in
# match() still resolves the SPECIFIC painpoint, so a broad gate + tight cosine
# keeps precision. Health terms are inherited from _HEALTH_GATE via the union.
_NONHEALTH_GATE = re.compile(
    # wealth / money
    r"money|income|cash|salary|paycheck|dollar|\$\d|debt|credit|loan|mortgage|"
    r"invest|stock|crypto|bitcoin|trading|retire|savings|wealth|millionaire|"
    r"budget|afford|financial|profit|earn|hustle|passive income|"
    # business / opportunity
    r"business|entrepreneur|startup|client|customer|\bsales\b|revenue|agency|"
    r"freelance|ecommerce|dropship|funnel|\bleads?\b|\bboss\b|scale your|saas|"
    # relationships / love
    r"dating|single|\bex\b|marriage|spouse|\bwife\b|husband|girlfriend|"
    r"boyfriend|relationship|divorce|breakup|lonely|attract|romance|soulmate|"
    r"partner|flirt|\bcrush\b|"
    # beauty / appearance + aesthetic-community tribes
    r"makeup|mascara|lipstick|foundation|wardrobe|outfit|wrinkle|\bglow\b|"
    r"complexion|lashes|\bstyle\b|fashion|jewelry|serum|"
    r"aesthetic|cottagecore|coquette|\by2k\b|retro|vintage|\bgoth|grunge|"
    r"gorpcore|streetwear|sneaker|hypebeast|tattoo|\bvinyl|cosplay|fandom|"
    r"\banime|couture|"
    # self-improvement / mind
    r"confidence|productiv|procrastinat|discipline|\bhabit|mindset|manifest|"
    r"\blearn\b|course|fluent|language|\bskill|charisma|"
    # survival / security
    r"survival|prepper|prepping|disaster|emergency|security|intruder|burglar|"
    r"self-defense|\bgold\b|silver|collapse|privacy|identity theft|tactical|"
    # lifestyle / consumer
    r"travel|flight|vacation|\bhotel|cruise|recipe|cooking|garden|hobby|gadget|"
    r"puppy|leash|renovat|declutter|"
    # automotive
    r"\bcar\b|\bcars\b|engine|horsepower|detailing|automotive|dealership|"
    r"\bauto\b|vehicle|mechanic|"
    # gifts & events
    r"\bgift|wedding|bridal|registry|birthday|\bparty\b|celebrate|anniversary|"
    # toys & games
    r"\btoy|gaming|gamer|console|board game|collectible|trading card|puzzle|"
    # sports & outdoors
    r"\bsport|workout|camping|hiking|fishing|hunting|outdoor|adventure|\btrail|"
    # electronics & tech
    r"headphone|earbuds|speaker|\bsmart home|audiophile|"
    # food & drink
    r"\bcoffee|matcha|\bsnack|gourmet|meal kit|barista|"
    # home & living
    r"\bdecor|furniture|cleaning|organize|declutter|interior|"
    # arts & hobbies
    r"art supplies|\bcraft|painting|instrument|\bguitar|"
    # books & education
    r"\bbook\b|\bbooks\b|reading|\bnovel\b|"
    # baby & parenting
    r"\bbaby|newborn|stroller|nursery|diaper|toddler", re.I)


def _topic_tokens(low):
    """Distinct topical tokens across health + all non-health verticals."""
    return set(_HEALTH_GATE.findall(low)) | set(_NONHEALTH_GATE.findall(low))


# the non-health domains added by the v4_nonhealth_painpoints taxonomy, plus the
# two pre-existing non-medical ones. Used to confine the semantic search to the
# ad's vertical when the gate tokens are clearly one-sided.
_NONHEALTH_DOMAINS = frozenset({
    "wealth_income", "business_growth", "relationships", "beauty_appearance",
    "self_growth", "survival_security", "lifestyle_consumer", "fashion",
    "automotive", "gifts_events", "toys_games", "sports_outdoors",
    "electronics_tech", "food_drink", "home_living", "arts_hobbies",
    "books_education", "baby_parenting", "pet_supplies", "travel_luggage",
    "financial", "relationship_loneliness"})

# strong pet-subject markers. 'vet' is EXCLUDED (collides with veteran). Used only
# by the species guard, which ALSO requires a veterinary painpoint to have matched,
# so a stray 'cat' can't hijack a human ad on its own.
# STRONG markers: unambiguous pet subjects. One is enough — no human ad says
# "kibble" or "your puppy" by accident.
# The possessives MUST end on a word boundary. Without it "your cat" matched
# inside "your cataractS" and one substring in a human ophthalmology
# advertorial ("if you are serious about clearing your cataracts without
# surgery") flipped the entire read into pet mode: the painpoint card filled
# with "Diabetes in Dogs"/"Ectropion in Dogs", the angles with "Tapeworms in
# Cats"/"Dog Staph Infection", and critic_species rejected the reader's
# CORRECT human answer ("foggy vision" / "protein clumps") three rounds
# running — leaving the fallback phrase-hunt to publish "Was Pain". Same trap
# waits in "your catalog", "my catheter", "your category", "my dogma".
_PET_STRONG = re.compile(
    r"\bpupp(?:y|ies)\b|\bkittens?\b|\bkibble\b|\bfleas?\b|\bcanine\b"
    r"|\bfeline\b|\bveterinar(?:y|ian)\b|\blitter box\b|\bgroomer\b"
    r"|\bkennel\b|\bfurry friend|\b(?:your|my) dogs?\b|\b(?:your|my) cats?\b"
    r"|\bdoggy\b|\bdoggie\b|\bkitty\b",
    re.I)
# WEAK markers: real pet vocabulary that also arrives as NOISE. A bare "cat"
# is one whisper error away from "cap" — which is exactly what happened on
# ?id=1819239872392567, a Kiierr laser-CAP hair-loss ad whose transcript read
# "...than admit that it's a cat". Two such tokens in 11,715 characters
# flipped the whole read into pet mode and the painpoint card filled with
# "Pica in Cats", "Cat Asthma", "Bird Flu in Cats".
_PET_WEAK = re.compile(
    r"\bdogs?\b|\bcats?\b|\bpets?\b|\bpaws?\b|\bleash\b|\bwhiskers\b", re.I)


def _is_pet_copy(low):
    """Is this ad ABOUT an animal?

    The old guard fired on any single marker, on the theory that it was
    doubly gated because a veterinary painpoint also had to match. With 399
    staged petMD rows in the KB that second gate is free — a veterinary match
    is always available — so in practice one stray token decided it.

    A real pet ad RETURNS to its subject; a transcription artefact does not.
    So weak markers must scale with the length of the copy, while one
    unambiguous marker still settles it outright.
    """
    if _PET_STRONG.search(low):
        return True
    weak = len(_PET_WEAK.findall(low))
    return bool(weak) and weak >= 1 + len(low) // 4000


# Kept as the union so existing callers that just want "any pet vocabulary"
# (e.g. the AVMA/APPA owner overlay) behave as before. The species GUARD must
# use _is_pet_copy, never this.
_PET_MARKER = re.compile(
    _PET_STRONG.pattern + "|" + _PET_WEAK.pattern, re.I)

# FOIL NEGATION: a pattern-interrupt hook names a condition only to DENY it —
# "IT'S NOT ALLERGIES, it's your gut", "not just a headache", "your problem
# isn't hormones". The denied condition is a false lead, not what the ad sells,
# so a term whose ONLY occurrences sit right after such a denial must not count.
# Matched against the ~40 chars ending just before the term: a negation of
# BEING (not / isn't / aren't / ...) followed by up to 3 filler words.
# Deliberately does NOT include "no", "no more", "stop", "end", "without",
# "get rid of" — those are RELIEF phrasings where the condition is still the
# subject ("no more back pain", "stop your knee pain", "no energy" = fatigue),
# so they keep matching.
_FOIL_NEG = re.compile(
    r"\b(?:not|isn'?t|aren'?t|wasn'?t|weren'?t|ain'?t|"
    r"nothing to do with|not about)\b(?:\s+\w+){0,3}\s*$")

# SEMANTIC ADJUDICATION knobs (see match_painpoints). The embedder is a JUDGE
# over the regex candidates, applied CONSERVATIVELY so it never demotes a
# specific lexical match on a noisy story-lede:
#   _SEM_FLOOR    a WEAK (single-cue) candidate is vetoed if the copy scores
#                 below this against it. Multi-cue (hits>=2) matches are never
#                 vetoed — they are lexically specific, not bare-word leaks.
#   _SEM_PROMOTE  a candidate may be lifted to #1 over the hit-count order only
#                 if its cosine is at least this (absolute confidence) ...
#   _SEM_RERANK   ... AND at least this much above the current top (decisive).
# So near-ties and low-confidence reads keep the regex order ("prediabetic,
# A1C 6.4" stays Blood Sugar), while a clear semantic winner (rosacea ad ->
# Rosacea over Wrinkles) is promoted. Env-tunable; V4_SEM_ADJUDICATE=0 disables
# (pure hit-count fallback, offline / embedder unavailable).
_SEM_FLOOR = float(os.environ.get("V4_SEM_FLOOR", "0.30"))
_SEM_PROMOTE = float(os.environ.get("V4_SEM_PROMOTE", "0.50"))
_SEM_RERANK = float(os.environ.get("V4_SEM_RERANK", "0.08"))

# analyze() calls match_painpoints several times on the SAME copy (desire,
# demographics, awareness), and each semantic embed is a subprocess model load.
# Memoize the copy vector so a copy is embedded at most once per unique text.
_QVEC_CACHE = {}


def _embed_copy(text):
    key = (text or "")[:1800]
    v = _QVEC_CACHE.get(key)
    if v is None:
        import v4_embed as _emb
        v = _emb.encode([key])[0]
        if len(_QVEC_CACHE) > 256:          # bounded; copies aren't reused often
            _QVEC_CACHE.clear()
        _QVEC_CACHE[key] = v
    return v


def _cos(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


# SEGMENT-MAX adjudication (story-lede fix). A long narrative lede dominates a
# single whole-copy embedding, so a condition revealed briefly and LATE
# ("...my hand went numb" after 500 chars of a bike-ride story) gets drowned:
# the whole-text vector sits far from the true painpoint. Fix: score each
# candidate by the BEST-matching chunk of the copy, not the whole. The whole
# (capped) text is always chunk 0, so segment-MAX can only RAISE a candidate's
# score above the single-vector path, never lower it — it strictly adds recall
# of the reveal without weakening the veto on genuine noise.
_SEG_CACHE = {}
_ANCHOR_VECS = None


def _segments(text):
    t = (text or "")[:6000]
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", t) if s.strip()]
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= 320:
            cur = (cur + " " + s).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s[:320]
        if len(chunks) >= 15:
            break
    if cur and len(chunks) < 16:
        chunks.append(cur)
    whole = t[:1800]
    return [whole] + [c for c in chunks if c and c != whole][:15]


def _segment_vecs(text):
    key = (text or "")[:1800]
    v = _SEG_CACHE.get(key)
    if v is None:
        import v4_embed as _emb
        v = _emb.encode(_segments(text))
        if len(_SEG_CACHE) > 256:
            _SEG_CACHE.clear()
        _SEG_CACHE[key] = v
    return v


def _anchor_vecs():
    global _ANCHOR_VECS
    if _ANCHOR_VECS is None:
        import v4_semantic_painpoint as _sem
        _ANCHOR_VECS = {a["id"]: a["vec"] for a in _sem._load_anchors()}
    return _ANCHOR_VECS


# Pet rows live in TWO domain families, and the guard only knew one. The 399
# `veterinary_*` conditions were filtered while the 6 `pet_supplies` rows
# walked straight past it — which is how "Pet Grooming" appeared as a
# painpoint on a women's hair-loss ad (2026-08-27).
_PET_DOMAINS = ("veterinary", "pet_supplies")


def _is_vet(p):
    return str(p.get("domain", "")).startswith(_PET_DOMAINS)


def rows_for_names(names):
    """KB rows for resolved painpoint NAMES, in order, as match tuples.

    match_painpoints() is regex-driven and misses ordinary phrasings — on the
    ad that prompted this, `Hair Loss & Thinning` fired on ZERO of its cues
    and ZERO of its name terms ("hair growth", "defend your hair from DHT",
    "less shedding" are none of them), so the only regex match was
    "Pet Grooming". The painpoint was still resolved correctly downstream, by
    the classifier/open-set route. Anything that needs the ad's REAL
    painpoints must therefore work from the resolved names, not the raw
    regex hits.
    """
    idx = {p["name"].lower(): p for p in load_kb()["painpoints"]}
    out = []
    for n in names or []:
        p = idx.get(str(n).lower())
        if p is not None:
            out.append((p, 1))
    return out


# A trailing painpoint is kept only if it holds this share of the LEADER's
# evidence. Tunable, but 0.25 is what the failure below actually needed.
_PAIN_DOMINANCE = float(os.environ.get("V4_PAIN_DOMINANCE", "0.25"))


def dominant_painpoints(names, matched, ratio=None):
    """Drop trailing painpoints whose evidence is a rounding error next to the
    leader's.

    match_painpoints' density guard uses an ABSOLUTE floor (2 cue hits), which
    means the same two hits count as "a second complaint the ad makes" whether
    the copy is a 200-char hook or a 7,000-char advertorial. On a real
    hyperpigmentation VSL the top row fired 17 times across 14 distinct cues,
    and `Wrinkles & Skin Aging` (2 hits: "skin barrier", "brighter skin") and
    `Swimwear` (2 hits, off one "bikini line" in a patient anecdote) were
    listed beside it as equals — the deck then showed a dark-spot ad
    complaining about wrinkles and swimwear.

    Evidence is only meaningful RELATIVE to how much of it the ad offered, so
    the floor scales with the leader. Names with no hit count (a brief-supplied
    painpoint, an open-set extraction, a classifier override) are never cut —
    unknown evidence is not zero evidence — and the leader always survives.

    The floor is purely relative, with no absolute minimum. An earlier version
    used max(2, ...) and re-broke two locked cases: when the leader itself has
    only 2 hits, a 1-hit row is HALF the available evidence, not noise. That
    absolute 2 was the exact bug this function exists to remove.
    """
    if not names or len(names) < 2:
        return list(names or [])
    hits = {}
    for p, h in matched or []:
        try:
            hits[str(p["name"]).lower()] = max(hits.get(str(p["name"]).lower(),
                                                        0), int(h))
        except (KeyError, TypeError, ValueError):
            continue
    if not hits:
        return list(names)
    top = max(hits.values())
    floor = max(1, int(math.ceil((ratio if ratio is not None
                                  else _PAIN_DOMINANCE) * top)))
    kept = [names[0]]
    for n in names[1:]:
        h = hits.get(str(n).lower())
        if h is None or h >= floor:
            kept.append(n)
    return kept


def _species_filter(out, low):
    """THE HUMAN/ANIMAL AXIS, applied wherever candidates are produced.

    The embedder cannot separate it ("dog gut health" sits next to human
    "Gut Health & Digestion"), so semantics will happily promote a pet
    condition onto a human ad and vice versa. Two-sided:
      pet subject in the copy  -> keep ONLY veterinary rows
      no pet subject           -> drop veterinary rows entirely

    It used to be one-sided (pet ads kept clean, human ads unguarded), which
    is how a human beauty ad reported "Glaucoma in Dogs" and "Pyometra in
    Cats" as the buyer's painpoints. Never empties a pet ad's list; on a
    human ad an emptied list is correct — the fallbacks resolve a human
    painpoint instead.
    """
    if not out:
        return out
    if _is_pet_copy(low):
        vet = [(p, h) for p, h in out if _is_vet(p)]
        return vet or out
    return [(p, h) for p, h in out if not _is_vet(p)]


def match_painpoints(text, extra_painpoints=None):
    """-> [(painpoint, hits)] for painpoints whose cues OR distinctive name
    terms appear. Cues catch ad-language ('menobelly'); name terms catch
    plain mentions ('arthritis', 'blood pressure') the cue list didn't list.

    `extra_painpoints` is a list of clean painpoint LABELS from the Gemini
    vision brief ('Joint pain', 'Low mobility'). They are matched the same
    way, so a condition Gemini SAW in the creative still grounds the
    age/gender prior even when the raw copy never used a KB cue phrase —
    the turmeric VSL says 'joint pain', not the cue 'knee pain'."""
    low = text.lower()
    # despaced copy: OCR of stylized ad text drops spaces ('FEELPUFFYFOR',
    # 'POOPONCEAWEEK', 'guthealth'), so word-boundary matching alone misses
    # cues. We also substring-match distinctive (>=5 char) terms here.
    low_ns = re.sub(r"[^a-z0-9]", "", low)
    extra = " \n ".join(extra_painpoints or []).lower()

    def _hit(t):
        if len(t) < 3:
            return False
        rx = r"\b" + re.escape(t) + r"\b"
        if re.search(rx, low):
            # NEGATION GUARD: count the term only if at least one occurrence is
            # NOT a foil ("it's not allergies"). If every occurrence sits right
            # after a denial-of-being, the ad names the condition to reject it.
            for mm in re.finditer(rx, low):
                if not _FOIL_NEG.search(low[max(0, mm.start() - 40):mm.start()]):
                    return True
            return False
        tns = re.sub(r"[^a-z0-9]", "", t)
        return len(tns) >= 5 and tns in low_ns          # OCR space-loss

    out = []
    for p in load_kb()["painpoints"]:
        # dedup: a word appearing in BOTH the cue list and the name-terms
        # split (e.g. 'digestion' is both a literal cue and the head of
        # 'Gut Health & Digestion') was silently double-counted, inflating
        # that painpoint's score against genuine single-mention rivals in
        # long-form copy. Each distinct phrase counts once.
        # STAGED (bulk-ingested NHS/petMD) entries match ONLY on their full
        # name phrase + curated cues — NOT on split name tokens, whose generic
        # words ('shoulder', 'anxiety') would over-fire against the curated 49.
        # The curated 49 (no 'staged' flag) keep full name-term matching.
        if p.get("staged"):
            terms = list(dict.fromkeys(
                [c.lower() for c in p.get("cues", [])] + [p["name"].lower()]))
        else:
            terms = list(dict.fromkeys(
                [c.lower() for c in p.get("cues", [])] + _name_terms(p["name"])))
        hits = sum(1 for t in terms if _hit(t))
        # vision-brief labels count as one corroborating hit, however many
        # of this painpoint's terms they touch (don't let a verbose label
        # outvote the actual copy).
        if extra and any(len(t) >= 5 and re.search(
                r"\b" + re.escape(t) + r"\b", extra) for t in terms):
            hits += 1
        if hits:
            out.append((p, hits))
    out.sort(key=lambda t: -t[1])

    # ------------------------------------------------------------------ #
    # SEMANTIC ADJUDICATION — the decision rule (replaces raw hit-count). #
    # ------------------------------------------------------------------ #
    # Regex is a good CANDIDATE GENERATOR but a bad JUDGE: it counts cue hits, so
    # a bare word the copy is not ABOUT can out-count the real subject — 'skin'
    # on a dog ad -> human Wrinkles, a stray 'performance' -> Erectile
    # Dysfunction, 'foot' -> Footwear. Every one of those used to need its own
    # hand-written guard. Instead, let the embedding model decide which candidate
    # the copy is actually about: score each regex candidate by cosine to the
    # copy, VETO the ones that are semantically off (below _SEM_FLOOR, or far
    # below the best candidate), and rank survivors by relevance. One principled
    # rule subsumes the species / noise-painpoint guards and keeps full KB
    # coverage (candidates still come from the KB). If every candidate is off
    # (best < floor), out is emptied so the gap-filler re-searches the whole KB.
    qvec = None
    adjudicated = False
    _sem = {}                 # painpoint id -> best segment cosine (see below)
    if out and os.environ.get("V4_SEM_ADJUDICATE", "1") not in ("0", "", "false"):
        # skip the embed when regex is already confident: a single candidate with
        # >=2 independent cue hits is rarely a bare-word leak. Adjudicate every
        # multi-candidate case and every lone single-hit (the usual noise leak).
        if len(out) >= 2 or out[0][1] <= 1:
            try:
                segvecs = _segment_vecs(text)
                qvec = segvecs[0]                 # whole-text vec, reused below
                avecs = _anchor_vecs()
                s = {}
                for _p, _h in out:
                    _av = avecs.get(_p["id"])
                    if _av is not None:
                        s[_p["id"]] = max(_cos(_sv, _av) for _sv in segvecs)
                _sem = s
                if s:
                    smax = max(s.values())
                    # VETO weak (single-cue) candidates the copy is not about.
                    # A hits>=2 match is lexically specific and never vetoed; a
                    # candidate with NO anchor (KB changed since anchors were
                    # built) can't be judged and is kept on its cue-hits.
                    kept = [(p, h) for p, h in out
                            if h >= 2 or p["id"] not in s
                            or s[p["id"]] >= _SEM_FLOOR]
                    if not kept and smax >= _SEM_FLOOR:
                        # every candidate weak-and-off but the field IS relevant
                        kept = [max(out, key=lambda t: s.get(t[0]["id"], -1.0))]
                    # default to the regex (cue-hit) order, stable ...
                    kept.sort(key=lambda t: -t[1])
                    # ... then PROMOTE the most-relevant survivor to #1 only when
                    # the semantic signal is confident AND decisively above the
                    # current top (else near-ties keep the lexical order).
                    if kept:
                        best = max(kept, key=lambda t: s.get(t[0]["id"], -1.0))
                        bs = s.get(best[0]["id"], -1.0)
                        ts = s.get(kept[0][0]["id"], -1.0)
                        # A single-cue candidate must NOT override a candidate
                        # that out-hits it by 2+ specific cues on semantics
                        # alone: segment-MAX can score a vivid off-topic lede
                        # chunk high ("3am, couldn't sleep" -> Low Energy over a
                        # 3-hit "Delete Joint Pain" ad). Multi-cue promotions
                        # (gout 3 over joint 5) and ties (rosacea 1 over
                        # wrinkles 2) are still allowed.
                        _outhit = best[1] <= 1 and kept[0][1] >= best[1] + 2
                        if (best[0]["id"] != kept[0][0]["id"] and not _outhit
                                and bs >= _SEM_PROMOTE and bs - ts >= _SEM_RERANK):
                            kept.remove(best)
                            kept.insert(0, best)
                    out = kept
                    adjudicated = True
            except Exception:  # noqa: BLE001
                pass

    # SINGLE-HIT DENSITY GUARD. On LONG copy every candidate lands on exactly
    # one hit, so hit count ranks nothing and the painpoint card fills with
    # whatever incidental word appeared. Measured on three real Ad Library ads
    # (8-12k chars, 2026-08-27): 8-11 matches each, ALL at 1 hit —
    # "Pet Food & Nutrition" fired on `treats`, "Wearables & Smartwatches" on
    # `steps`, "Knitting, Sewing & Fiber" on `quilting`, "Hotels & Stays" on a
    # de-spaced OCR fragment. The absolute `_SEM_FLOOR` (0.3) cannot help: on
    # that copy EVERY cosine sits between 0.49 and 0.77, so nothing is ever
    # vetoed. The separation is there, but it is RELATIVE — the subject scores
    # 0.767 while the junk scores 0.495.
    #
    # So: one bare hit in a long document must also be semantically close to
    # the BEST candidate to survive. Multi-hit matches (a condition the ad
    # actually keeps talking about) are never touched, and short copy, where a
    # single hit is genuinely informative, is exempt.
    # 0.06 chosen on 209 out-of-sample ads, not on anecdotes. Cross-vertical
    # contamination (a painpoint list spanning 3+ unrelated parent categories
    # — the user's "randomly assigns stuff that makes no sense" made
    # countable) falls 18% -> 4% while painpoints/ad only moves 2.71 -> 2.16.
    # 0.04 and 0.03 buy nothing further (both 4%) and cut more real
    # painpoints, so 0.06 is the loosest margin that captures the whole gain.
    _LONG_COPY = int(os.environ.get("V4_LONG_COPY_CHARS", "1500"))
    _SEM_MARGIN = float(os.environ.get("V4_SEM_MARGIN", "0.06"))
    if _sem and len(out) > 1:
        _best = max(_sem.values())
        _top_h = max(h for _, h in out)

        def _near(p):
            return _sem.get(p["id"], 1.0) >= _best - _SEM_MARGIN

        if len(low) >= _LONG_COPY:
            # Flat evidence: hit count cannot discriminate, semantics must.
            _dense = [(p, h) for p, h in out if h >= 2 or _near(p)]
        else:
            # SHORT copy: a single hit IS informative, so semantics alone must
            # not cut. But when the subject is named repeatedly and a rival
            # caught one incidental word, BOTH signals are weak — drop it.
            # "Anxiety and chronic stress ... the racing thoughts stopped"
            # matched Anxiety (3 hits) and "Car Enthusiast & Mods" (1 hit, on
            # `racing`), and both were shown as what the buyer feels.
            _dense = [(p, h) for p, h in out
                      if h >= 2 or h >= _top_h or _near(p)]
        if _dense:                      # never empty the list
            out = _dense

    # TIE-BREAK THE RANKING BY SEMANTICS. `out` is ordered by hit count, and
    # split_angles treats rank 1 as THE SUBJECT that can never be filed as the
    # angle. On long copy every candidate has exactly one hit, so that rank was
    # decided by declaration order in v4_correlations.json — arbitrary. Live
    # consequence: a menopause ad (cosine 0.719) had "Menopause Symptoms"
    # filed as the ANGLE while "High Blood Pressure" (0.661, one incidental
    # mention) was shown as what the buyer feels. Exactly the inversion the
    # user reported. Hit count still leads; the embedder only breaks ties.
    if _sem and len(out) > 1:
        out.sort(key=lambda t: (-t[1], -_sem.get(t[0]["id"], 0.0)))

    # SPECIES AXIS — always applied, INCLUDING after adjudication. This is not a
    # lexical-noise leak (which adjudication handles); it is the human/animal
    # axis, which the sentence embedder genuinely cannot separate: "dog gut
    # health" sits right next to human "Gut Health & Digestion" in embedding
    # space, so semantic adjudication happily promotes the human painpoint on a
    # dog ad. Doubly gated (a pet marker AND a veterinary painpoint actually
    # matched), so it can't misfire on human ads: when both hold, keep only the
    # veterinary reads.
    # The guard was ONE-SIDED: a pet ad kept only veterinary rows, but a HUMAN
    # ad was never stopped from surfacing veterinary ones. That is how a human
    # beauty ad reported "Glaucoma in Dogs" and "Pyometra in Cats" as the
    # buyer's painpoints (seen live 2026-08-27) — the 399 staged petMD rows
    # match on their full name phrase, and a single incidental token is enough
    # when there is little copy to compete with. No pet subject in the copy =>
    # no veterinary painpoint. If that empties the list, the semantic
    # gap-filler below resolves a human painpoint instead, which is correct.
    out = _species_filter(out, low)

    _ntok = len(_topic_tokens(low))
    if not out:
        # SEMANTIC GAP-FILLER: exact-cue matching is brittle (misses paraphrase and
        # novel framings — "I replaced my salary from my laptop" is make-money but
        # names no cue; LASIK, compression socks, adaptogen). When regex finds
        # NOTHING, fall back to v4_semantic_painpoint (embeds copy vs each painpoint,
        # cosine). The anchors span the WHOLE marketing universe now — health +
        # wealth + business + relationships + beauty + self-growth + survival +
        # lifestyle.
        # EVERY DR AD SELLS TO A PAINPOINT, so we ALWAYS try to resolve one — the
        # old `_ntok >= 1` gate silently BLANKED euphemistic copy that names no
        # vertical word ("Rebuild Your Willy", disguised as fitness -> male sexual
        # health). The cosine bar scales with how much topical signal there is:
        # >=2 tokens clear at 0.55; a SINGLE token demands 0.62; ZERO tokens demand
        # 0.58 so true gibberish still resolves nothing while a confident disguised
        # ad does.
        try:
            import v4_semantic_painpoint as _sem
            pps = load_kb()["painpoints"]
            _by_id = {p["id"]: p for p in pps}
            # one-sided vertical restriction: when the copy trips >=2 tokens of
            # ONE side and none of the other, confine the semantic search to that
            # side's anchors. Health has 987 anchors vs 56 non-health, so without
            # this a wealth/dating/beauty ad's correct painpoint is out-scored by
            # some health anchor (brain-overload->focus_adhd, nest-egg->falls).
            h = len(set(_HEALTH_GATE.findall(low)))
            nh = len(set(_NONHEALTH_GATE.findall(low)))
            allow = None
            if nh >= 2 and h == 0:
                allow = {p["id"] for p in pps
                         if p.get("domain") in _NONHEALTH_DOMAINS}
            elif h >= 2 and nh == 0:
                allow = {p["id"] for p in pps
                         if p.get("domain") not in _NONHEALTH_DOMAINS}
            _thr = 0.55 if _ntok >= 2 else (0.62 if _ntok == 1 else 0.58)
            cands = _sem.match(text, topk=6, thresh=_thr, allow_ids=allow,
                               _qvec=qvec)
            # DR sells to a PROBLEM. When the copy names no vertical at all (0
            # topic tokens) the embedder often ranks a lifestyle/INTEREST anchor
            # (Home Gym, Toned Body, Building Habits) a hair above the health
            # CONDITION the ad disguises (Low-T under "Rebuild Your Willy"). So
            # among near-top matches, prefer a real condition painpoint over an
            # interest category.
            if cands and _ntok == 0 and allow is None:
                _top = cands[0][2]
                _cond = [c for c in cands if c[2] >= _top - 0.05
                         and _by_id.get(c[0], {}).get("domain")
                         not in _NONHEALTH_DOMAINS]
                if _cond:
                    cands = _cond + [c for c in cands if c not in _cond]
            for pid, _nm, _sc in cands[:2]:
                p = _by_id.get(pid)
                if p:
                    # SOFT match: a fuzzy semantic rescue is strong enough to ground
                    # the DESIRE (desire derives from the problem) but too weak to
                    # commit a person's age/gender — demographic_prior skips _soft so
                    # a gender-neutral device (bioblade microcurrent) stays 'unclear'
                    # instead of inheriting a beauty painpoint's female/25-34 skew.
                    sp = dict(p); sp["_soft"] = True
                    out.append((sp, 1))
        except Exception:  # noqa: BLE001
            pass

    if (not out and _ntok == 0 and len(low.strip()) >= 12
            and os.environ.get("V4_SEM_UNGATE", "1") not in ("0", "", "false")):
        # UNGATE — a DR ad ALWAYS targets a painpoint, so it must never blank.
        # Zero topical tokens means DISGUISED copy: "Rebuild Your Willy's in
        # Minutes a Day / Strong chest, strong arms" reads as generic fitness,
        # so the old gate (>=1 token) skipped it entirely -> empty result.
        # Score the whole KB by segment-max, then — because the surface is
        # misleading — prefer the top HEALTH CONDITION when a non-health
        # lifestyle interest (Building Habits, Home Gym) only NEAR-ties it: the
        # euphemism ("Willy") is the tell that this is the condition, not a gym
        # ad. Lands "Rebuild Your Willy" on Low Testosterone & Andropause.
        try:
            pps = load_kb()["painpoints"]
            _by_id = {p["id"]: p for p in pps}
            avecs = _anchor_vecs()
            segvecs = _segment_vecs(text)
            scored = sorted(
                ((pid, max(_cos(sv, av) for sv in segvecs))
                 for pid, av in avecs.items() if pid in _by_id),
                key=lambda t: -t[1])
            if scored:
                top_pid, top_sc = scored[0]
                if _by_id[top_pid].get("domain") in _NONHEALTH_DOMAINS:
                    for pid, sc in scored[:6]:
                        if (_by_id[pid].get("domain") not in _NONHEALTH_DOMAINS
                                and top_sc - sc <= 0.04):
                            top_pid = pid
                            top_sc = sc
                            break
                if top_sc >= 0.58:
                    sp = dict(_by_id[top_pid]); sp["_soft"] = True
                    out.append((sp, 1))
        except Exception:  # noqa: BLE001
            pass

    # Re-apply on the way OUT. Both fallbacks above (the semantic gap-filler
    # and the ungate) score the WHOLE KB and append straight to `out`, so they
    # re-introduced exactly what the guard had just removed: a hair-loss ad
    # (?id=1819239872392567, 11.5k chars, product Kiierr) came back with
    # "Pica in Cats", "Cancer in Cats", "Cat Asthma", "Bird Flu in Cats" as
    # the buyer's painpoints and "Shock in Cats" as the angle. Filtering once
    # in the middle of the function is not enough — filter at the exit.
    return _species_filter(out, low)


# ---------------------------------------------------------------------------
# DESIRE DERIVES FROM THE PROBLEM (BERNAY equation: "Desire is a derivative of a
# problem"). The avatar's desires are read deterministically off the DETECTED
# painpoint — NOT a learned classifier or motif cosine (both of which collapsed
# every ad to connection/comfort/love). Every one of the 987 painpoints resolves:
# per-id override first, else its domain's profile, else a safe health default.
# Desire vocab = the motif taxonomy's desire tags (chakra/maslow linked_desires).
# ---------------------------------------------------------------------------
_DOMAIN_DESIRES = {
    "chronic_health": ["survival", "safety", "control"],
    "hormonal": ["harmony", "control", "comfort"],
    "weight_metabolism": ["recognition", "control", "belonging"],
    "energy_fatigue_sleep": ["survival", "comfort", "mastery"],
    "hair_loss": ["recognition", "status", "expression"],
    "skin_aging": ["recognition", "status", "expression"],
    # visible marks others read off your face — an appearance problem, so it
    # derives the same wants as the other Beauty & Care domains
    "skin_pigment": ["recognition", "status", "expression"],
    "dental": ["recognition", "connection", "comfort"],
    "vision_hearing": ["safety", "control", "connection"],
    "mental_health": ["harmony", "safety", "comfort"],
    "financial": ["safety", "survival", "control"],
    "relationship_loneliness": ["connection", "belonging", "love"],
    "libido_sexual_health": ["power", "connection", "recognition"],
    "oncology": ["survival", "safety", "control"],
    "urologic": ["comfort", "control", "safety"],
    "digestive": ["comfort", "control", "harmony"],
    "rheumatic": ["comfort", "control", "mastery"],
    "neurologic": ["safety", "clarity", "control"],
    "autoimmune": ["safety", "comfort", "control"],
    "vascular": ["survival", "safety", "control"],
    "condition_nhs": ["safety", "comfort", "control"],
    "veterinary_dog": ["love", "safety", "comfort"],
    "veterinary_cat": ["love", "safety", "comfort"],
}
_PAINPOINT_DESIRES = {          # per-id refinements over the domain default
    "low_testosterone": ["power", "recognition", "connection"],
    "erectile_dysfunction": ["power", "connection", "recognition"],
    "low_libido_sexual_health": ["power", "connection", "recognition"],
    "prostate_cancer": ["survival", "safety", "recognition"],
    "prostate_enlargement": ["comfort", "control", "recognition"],
    "menopause_symptoms": ["harmony", "comfort", "recognition"],
    "postpartum_weight_hormones": ["harmony", "recognition", "control"],
    "joint_pain_arthritis": ["comfort", "control", "mastery"],
    "chronic_back_pain": ["comfort", "control", "mastery"],
    "osteoporosis_bone_loss": ["safety", "control", "mastery"],
    "anxiety_stress": ["harmony", "safety", "comfort"],
    "depression": ["harmony", "connection", "purpose"],
    "insomnia_poor_sleep": ["comfort", "harmony", "survival"],
    "focus_adhd": ["mastery", "clarity", "control"],
    "stubborn_weight_loss": ["recognition", "control", "belonging"],
    "loneliness_dating": ["connection", "belonging", "love"],
    "hair_loss_thinning": ["recognition", "status", "expression"],
    "wrinkles_skin_aging": ["recognition", "status", "expression"],
    "hyperpigmentation_dark_spots": ["recognition", "status", "expression"],
    "low_energy_fatigue": ["survival", "mastery", "comfort"],
    "hearing_loss_tinnitus": ["comfort", "connection", "control"],
    "vision_decline": ["safety", "control", "mastery"],
    "gut_health_digestion": ["comfort", "control", "harmony"],
    "high_blood_pressure": ["survival", "safety", "control"],
    "high_cholesterol": ["survival", "safety", "control"],
    "blood_sugar_diabetes": ["survival", "safety", "control"],
    "tooth_decay_gum_disease": ["recognition", "connection", "comfort"],
    "retirement_savings": ["safety", "survival", "control"],
    "cost_of_living": ["safety", "survival", "control"],
    "thyroid_dysfunction": ["survival", "control", "harmony"],
    "peripheral_neuropathy": ["comfort", "control", "safety"],
    "varicose_veins": ["recognition", "comfort", "control"],
    "ibs": ["comfort", "control", "harmony"],
    "gout": ["comfort", "control", "mastery"],
    "migraine": ["comfort", "control", "harmony"],
    "urinary_incontinence": ["control", "recognition", "comfort"],
    "fibromyalgia": ["comfort", "control", "harmony"],
    "rheumatoid_arthritis": ["comfort", "control", "mastery"],
    "lupus": ["safety", "comfort", "control"],
    "alzheimers": ["safety", "connection", "recognition"],
    "breast_cancer": ["survival", "safety", "connection"],
    "ovarian_cancer": ["survival", "safety", "connection"],
    "cervical_cancer": ["survival", "safety", "connection"],
    "uterine_cancer": ["survival", "safety", "connection"],
    "pcos": ["harmony", "recognition", "control"],
    "endometriosis": ["comfort", "harmony", "control"],
    "lymphatic_drainage_fluid_retention": ["comfort", "recognition", "control"],
}
_DESIRE_DEFAULT = ["safety", "comfort", "control"]


# ---------------------------------------------------------------------------
# ANGLE vs PAINPOINT — a DR distinction the decomposition was collapsing.
#
# A painpoint is what the buyer FEELS and wants gone ("I can't get hard").
# An ANGLE is the underlying cause/mechanism the ad INVOKES to explain that
# painpoint and make its product credible ("...because diabetes damaged the
# nerves and blood flow"). Diabetes / neuropathy / varicose veins are not what
# the reader is suffering from in that ad — they are the promotional angle.
# Reported as painpoints they corrupt BOTH the creative read and the avatar
# (a diabetes-prevalence skew is not this ad's buyer).
#
# Direction comes from the KB's OWN sourced mechanism text, not a guess: if
# condition X is named inside condition Y's mechanism/cause chain, X causes Y,
# so X is upstream (ANGLE) and Y is the terminal effect (PAINPOINT). Verified
# on the real ad: ED's mechanism names diabetes and nerve damage; neuropathy's
# names diabetes -> diabetes -> neuropathy -> ED, ED terminal.
# The copy's own causal framing ("caused by", "the real reason is") is the
# second, ad-specific signal.
# ---------------------------------------------------------------------------
_CAUSAL_FRAME = re.compile(
    r"\b(?:caused by|because of|due to|thanks to|linked to|blamed? on|"
    r"triggered by|comes down to|the (?:real |root |hidden |actual )?"
    r"(?:reason|cause)(?:\s+\w+){0,4}|root cause(?:\s+\w+){0,3}|"
    r"result of|stems? from|driven by|suffer(?:ing)? from|if you (?:have|are)|"
    r"when you have|starts? with|behind (?:your|the)|it'?s your)"
    r"(?:\s+\w+){0,4}\W*$", re.I)

# CAUSAL ROLE IS LEARNED, NOT LISTED.
#
# The first version of this enumerated causal verbs/prepositions. That is the
# wrong layer: a hand-written list ("caused by|destroyed|caramelizing|...") is
# an English-only re-implementation of something the corpus already taught —
# NHS/petMD condition prose is saturated with cause statements ("diabetes is
# the most common cause of peripheral neuropathy", "impaired vascular delivery
# produces erectile dysfunction"). It also breaks on the very next ad: the real
# OREVIA ad frames cause with VERBS ("your diabetes destroyed", "type 2
# diabetes is caramelizing your veins", "Diabetes-Related ED") and matched none
# of the prepositional patterns.
#
# So the ad-side signal is scored by SIMILARITY to how cause is expressed,
# using the same sentence embedder the rest of the pipeline judges with —
# generalising to unseen phrasing and (post-translation) to any language.
# The KB mechanism graph still supplies DIRECTION; this only answers "is THIS
# ad talking about the condition as a cause, or as the complaint?".
_CAUSE_ANCHORS = (
    "this condition is the underlying cause of the problem",
    "it damages and destroys the nerves and blood vessels",
    "that is the real reason this keeps happening to you",
    "the disease leads to and triggers the symptoms",
    "fixing the root cause behind the problem",
)
_COMPLAINT_ANCHORS = (
    "the problem I suffer from and want to get rid of",
    "the embarrassing symptom that ruins my life every day",
    "what I am struggling with and desperate to fix",
    "the relief and results this product finally gives me",
    "stop it for good and feel normal again",
)
_ROLE_VECS = None


def _role_vecs():
    """Cached anchor vectors for the cause-vs-complaint role judgement."""
    global _ROLE_VECS
    if _ROLE_VECS is None:
        import v4_embed as _emb
        v = _emb.encode(list(_CAUSE_ANCHORS) + list(_COMPLAINT_ANCHORS))
        n = len(_CAUSE_ANCHORS)
        _ROLE_VECS = (v[:n], v[n:])
    return _ROLE_VECS


def _spoken_as_cause(clauses, margin=0.02):
    """-> set of indices of `clauses` the ad speaks of as a CAUSE rather than
    as the complaint. Embedding-based, so novel phrasing still scores."""
    if not clauses:
        return set()
    import v4_embed as _emb
    cause_a, comp_a = _role_vecs()
    out = set()
    for i, v in enumerate(_emb.encode(list(clauses))):
        c = max(_cos(v, a) for a in cause_a)
        k = max(_cos(v, a) for a in comp_a)
        if c - k >= margin:
            out.add(i)
    return out

# Single anatomy/generic words that must NOT, on their own, link two conditions
# in the mechanism graph ("blood" appears in half the KB's mechanism prose).
_MECH_GENERIC = {"blood", "nerve", "nerves", "pain", "damage", "flow", "skin",
                 "heart", "weight", "sugar", "level", "levels", "health",
                 "vessel", "vessels", "tissue", "muscle", "hormone",
                 "hormones", "chronic", "disease", "condition", "symptom",
                 "symptoms", "system", "older", "aging", "ageing"}


def _mech_blob(p):
    """UPSTREAM-ONLY mechanism text for one painpoint, lowercased.

    Deliberately NOT the whole mechanism record. Free-form summaries describe a
    condition's complications and symptoms as well as its causes, so treating
    any co-mention as a causal edge points arrows backwards: cirrhosis prose
    names 'fluid retention' as something it CAUSES, which made fluid retention
    look upstream of cirrhosis and got the real painpoint filed as an angle
    (the lymphoria_beer_belly_liver regression).

    `cause_effect_chain` is the one field that is explicitly DIRECTIONAL —
    "A -> B -> C -> the condition" — so only its prefix (everything before the
    final arrow, i.e. the condition itself) counts as upstream."""
    m = p.get("mechanism")
    if not isinstance(m, dict):
        return ""
    chain = str(m.get("cause_effect_chain") or "")
    if "->" in chain:
        chain = "->".join(chain.split("->")[:-1])   # drop the terminal effect
    return chain.lower()


def _mech_terms(p):
    """Distinctive phrases that identify painpoint `p` inside another's
    mechanism text. Multi-word phrases and long single words only — generic
    anatomy words are excluded so 'blood' can't wire everything together."""
    terms = {p["name"].lower()}
    for c in p.get("cues", []):
        c = str(c).lower().strip()
        if " " in c or len(c) >= 7:
            terms.add(c)
    for w in re.split(r"[^a-z]+", p["name"].lower()):
        if len(w) >= 6:
            terms.add(w)
    return {t for t in terms
            if len(t) >= 5 and t not in _MECH_GENERIC and not t.isdigit()}


def split_angles(text, matched):
    """Split match_painpoints() output into (painpoints, angles).

    Both keep the [(painpoint, hits)] shape. A condition becomes an ANGLE when
    the KB's mechanism graph says it CAUSES another matched condition, or the
    copy names it in a causal frame. Never returns an empty painpoint list: if
    every candidate looks upstream, the most-supported one stays the painpoint
    (a DR ad always sells to a felt problem)."""
    hard = [(p, h) for p, h in (matched or []) if not p.get("_soft")]
    if len(hard) < 2:
        return list(matched or []), []
    low = (text or "").lower()

    # 1) mechanism graph: X causes Y  ->  X is upstream of Y
    causes_something, caused_by = set(), {}
    _h = {p["id"]: h for p, h in hard}
    for yi, (y, _) in enumerate(hard):
        blob = _mech_blob(y)
        if not blob:
            continue
        for xi, (x, _) in enumerate(hard):
            if xi == yi or x["id"] == y["id"]:
                continue
            if any(re.search(r"\b" + re.escape(t) + r"\b", blob)
                   for t in _mech_terms(x)):
                # X only counts as this ad's ANGLE if the effect it points at is
                # itself at least as well-supported by the copy. Otherwise the
                # ad isn't really arguing X -> Y, and a well-evidenced subject
                # (dog anxiety) gets demoted in favour of an incidental match.
                if _h.get(y["id"], 0) >= _h.get(x["id"], 0):
                    causes_something.add(x["id"])
                caused_by[y["id"]] = caused_by.get(y["id"], 0) + 1

    # 2) HOW THIS AD SPEAKS OF IT — learned, not listed. Collect the sentence
    # each condition is named in and ask the embedder whether that sentence
    # reads as a CAUSE statement or as the buyer's COMPLAINT. Replaces the old
    # causal-verb regex, which was English-only and missed the real ad's verb
    # framing ("your diabetes destroyed", "caramelizing your veins").
    _bounds = [0] + [m.end() for m in re.finditer(r"[.!?\n]+", low)] + [len(low)]
    sents, owner = [], []
    for p, _ in hard:
        seen_here = set()
        for t in _mech_terms(p):
            for m in re.finditer(r"\b" + re.escape(t) + r"\b", low):
                st = max((b for b in _bounds if b <= m.start()), default=0)
                en = min((b for b in _bounds if b > m.start()), default=len(low))
                frag = low[st:en].strip()
                if len(frag) >= 15 and frag not in seen_here:
                    seen_here.add(frag)
                    sents.append(frag)
                    owner.append(p["id"])
    framed = set()
    try:
        for i in _spoken_as_cause(sents):
            framed.add(owner[i])
    except Exception:  # noqa: BLE001 — embedder down: fall back to the frames
        for p, _ in hard:
            for t in _mech_terms(p):
                for m in re.finditer(r"\b" + re.escape(t) + r"\b", low):
                    if _CAUSAL_FRAME.search(low[max(0, m.start() - 70):
                                                m.start()]):
                        framed.add(p["id"])
                        break

    # 3) THE SUBJECT is protected. A condition that other matched conditions
    # cause, and which causes none of them, is the terminal effect = what the
    # ad is actually about. The copy's causal frame must not demote it: a
    # mechanism reveal ("...chokes blood flow") re-states the painpoint's own
    # vocabulary inside the cause clause, which otherwise tagged the subject
    # (ED) as its own angle. Most incoming edges wins, then cue support.
    _hits = {p["id"]: h for p, h in hard}
    terminal = [pid for pid in caused_by if pid not in causes_something]
    protected = max(terminal,
                    key=lambda i: (caused_by[i], _hits.get(i, 0))) \
        if terminal else None

    # THE RULE (user, 2026-08-21): "if it's a cause AND a symptom it's always
    # an angle; when it's the effect, like stubborn weight and belly fat, it's
    # always the painpoint." So position in the causal graph decides it:
    # anything with an OUTGOING edge (it causes another matched condition) is a
    # mediator the ad is using to explain — an ANGLE. Only pure SINKS, the
    # terminal effects, are what the buyer actually feels.
    # The ad-side "spoken as cause" read is kept as a widener for conditions the
    # KB has no chain for (bulk NHS entries carry prose, not cause_effect_chain),
    # not as a required second vote — requiring both made this under-fire and
    # report nothing on real ads.
    # THE #1 RANKED MATCH IS THE SUBJECT AND IS NEVER AN ANGLE. `matched` comes
    # out of the adjudication pipeline already ordered, and on flat-evidence ads
    # that ORDER is the only signal there is: a pet ad matches ~30 dog
    # conditions at exactly 1 hit each (the word "dog"), so hit counts cannot
    # discriminate and demoting by graph alone filed "Dog Anxiety" as the angle
    # while "Low White Blood Cell Count in Dogs" became the painpoint.
    _rank1 = next((p["id"] for p, _ in matched if not p.get("_soft")), None)
    _keep = {i for i in (protected, _rank1) if i}
    angle_ids = (causes_something | framed) - _keep

    # SUPPORT GUARD. Demoting to angle must not hand the ad to a weaker match:
    # 'athlete's foot' (3 cue hits) got filed as an angle and the painpoint card
    # was left showing 'Wearables & Smartwatches' (1 hit). A condition the copy
    # supports better than anything left standing is the subject, whatever the
    # graph says about it, so the best-supported match is never demoted alone.
    # No angle may out-evidence EVERY surviving painpoint. Restore any such
    # condition until the card that remains is at least as well-supported as
    # anything filed behind it — otherwise a 3-hit subject ends up as the angle
    # while a 1-hit incidental match ("Wearables & Smartwatches", "Tooth & Gum
    # Health") is presented as what the buyer feels. Terminates: each pass
    # strictly shrinks angle_ids.
    while angle_ids:
        top_rest = max((h for p, h in hard if p["id"] not in angle_ids),
                       default=0)
        over = [p["id"] for p, h in hard
                if p["id"] in angle_ids and h > top_rest]
        if not over:
            break
        angle_ids -= set(over)
    pains = [(p, h) for p, h in matched if p["id"] not in angle_ids]
    angles = [(p, h) for p, h in matched if p["id"] in angle_ids]
    if not [t for t in pains if not t[0].get("_soft")]:
        # everything read as upstream — keep the best-supported as the painpoint
        best = max(hard, key=lambda t: t[1])
        angles = [t for t in angles if t[0]["id"] != best[0]["id"]]
        pains = [best] + [t for t in pains if t[0]["id"] != best[0]["id"]]
    if protected:                       # the subject leads the painpoint list
        pains.sort(key=lambda t: (t[0]["id"] != protected, t[0].get("_soft",
                                                                    False)))
    return pains, angles


# The bulk petMD ingest kept each source page's ARTICLE HEADLINE as the
# painpoint label, so 43 KB rows read as questions rather than conditions:
# "Pyometra in Cats: What Is It and How Do Vets Treat It?", "IBS in Dogs:
# What Causes It?". Those rendered verbatim into the Painpoints card. The
# canonical name is left untouched (KB lookups and the anchors key on it) —
# this trims the headline apparatus for DISPLAY only.
_HEADLINE_TAIL = re.compile(
    r"\s*:\s*(?:what|why|how|types?|signs?|symptoms?|causes?|treatment|"
    r"everything|is it|are they|when to|can )\b.*$", re.I)


def _condition_label(name):
    """'Glaucoma in Dogs: What Is It, and What Are the Symptoms?' ->
    'Glaucoma in Dogs'."""
    s = str(name).strip()
    trimmed = _HEADLINE_TAIL.sub("", s)
    # Only accept the trim if something substantive survives — a label that is
    # ALL headline ("What Is It: ...") should stay as it is rather than vanish.
    if len(trimmed) >= 3:
        s = trimmed
    return s.rstrip(" ?:,").strip()


def display_name(name, text):
    """Show only the facet(s) of a BUNDLED KB label that the copy supports.

    Several KB entries bundle related complaints under one label — "Brain Fog,
    Focus & ADHD", "Hearing Loss & Tinnitus", "Low Testosterone & Andropause".
    Matching on one facet then printing the whole label ASSERTS things the ad
    never said: a menopause ad that mentions "brain fog" was reported as
    "Brain Fog, Focus & ADHD", claiming an ADHD angle that appears nowhere in
    the copy. Keep the canonical name for KB lookups; trim only for display,
    and only when at least one facet fired and at least one did not."""
    name = _condition_label(name)
    parts = [s.strip() for s in re.split(r"\s*(?:&|,)\s*", str(name))
             if s.strip()]
    if len(parts) < 2 or not text:
        return name
    low = (text or "").lower()

    def _seen(part):
        toks = [w for w in re.split(r"[^a-z0-9]+", part.lower()) if len(w) >= 3]
        if not toks:
            return False
        # the facet counts as present if its distinctive words appear
        return all(re.search(r"\b" + re.escape(w) + r"\b", low) for w in toks)

    hit = [p for p in parts if _seen(p)]
    if not hit or len(hit) == len(parts):
        return name
    # Only trim when the HEAD facet is the one the copy supports. The entry can
    # match through cues that belong to a facet whose words never appear
    # literally ("hip pain" -> Joint Pain & Arthritis), and trimming there would
    # replace the ad's own framing with a diagnosis it never made
    # ("Arthritis"). Head fired + trailing facets absent is the safe case, and
    # it is exactly the reported one: "brain fog" -> Brain Fog, not ADHD.
    if parts[0] not in hit:
        return name
    return " & ".join(hit)


def desires_for(matched):
    """Desire profile IMPLIED by the detected painpoint(s), weighted by cue hits.
    `matched` is match_painpoints() output: [(painpoint, hits)]. Returns
    [(desire, weight)] normalized (top 6), or [] if nothing matched — so the
    caller falls back to the motif read only when there's no problem to derive from.
    """
    if not matched:
        return []
    agg = {}
    for p, hits in matched:
        ds = (p.get("desires")                       # embedded (non-health KB)
              or _PAINPOINT_DESIRES.get(p.get("id"))
              or _DOMAIN_DESIRES.get(p.get("domain", ""))
              or _DESIRE_DEFAULT)
        w = float(hits) or 1.0
        for rank, d in enumerate(ds):        # first-listed desire is strongest
            agg[d] = agg.get(d, 0.0) + w * (1.0 - 0.2 * rank)
    top = sorted(agg.items(), key=lambda t: -t[1])[:6]
    tot = sum(v for _, v in top) or 1.0
    return [(d, v / tot) for d, v in top]


_MARKET_TYPE_BY_KEY = None


def market_type_for(name_or_id):
    """'aspiration' for aesthetic/community/identity categories (fashion, tribes,
    hobbies) where the desire is PRIMARY and there is no problem to fix;
    'problem' for everything else (health, debt, loneliness — a felt pain that
    creates the desire). Accepts a painpoint id OR display name. Defaults to
    'problem' (incl. copy-derived open-set problems never in the KB)."""
    global _MARKET_TYPE_BY_KEY
    if _MARKET_TYPE_BY_KEY is None:
        m = {}
        for p in load_kb().get("painpoints", []):
            mt = p.get("market_type", "problem")
            for k in (p.get("id"), p.get("name")):
                if k:
                    m[str(k).strip().lower()] = mt
        _MARKET_TYPE_BY_KEY = m
    if not name_or_id:
        return "problem"
    return _MARKET_TYPE_BY_KEY.get(str(name_or_id).strip().lower(), "problem")


# health/veterinary domains -> the 17-parent ad-intel taxonomy. Non-health
# painpoints carry parent_category inline (from v4_nonhealth_painpoints); the
# health KB (987 conditions) maps by domain here so we never touch every entry.
_HEALTH_DOMAIN_PARENT = {
    "condition_nhs": "Health & Supplements", "chronic_health": "Health & Supplements",
    "hormonal": "Health & Supplements", "oncology": "Health & Supplements",
    "mental_health": "Health & Supplements", "urologic": "Health & Supplements",
    "rheumatic": "Health & Supplements", "weight_metabolism": "Health & Supplements",
    "energy_fatigue_sleep": "Health & Supplements",
    "vision_hearing": "Health & Supplements", "neurologic": "Health & Supplements",
    "vascular": "Health & Supplements", "hair_loss": "Beauty & Care",
    "skin_aging": "Beauty & Care", "skin_pigment": "Beauty & Care",
    "dental": "Health & Supplements",
    "libido_sexual_health": "Health & Supplements", "digestive": "Health & Supplements",
    "autoimmune": "Health & Supplements",
    "veterinary_dog": "Pet Supplies", "veterinary_cat": "Pet Supplies",
}
_PARENT_BY_KEY = None


def parent_category_for(name_or_id):
    """-> one of the 17 ad-intel PARENT categories (Fashion, Health & Supplements,
    Automotive, …) for a painpoint id/name. Non-health painpoints carry an inline
    `parent_category`; health/vet map by domain; unknown -> 'Other'."""
    global _PARENT_BY_KEY
    if _PARENT_BY_KEY is None:
        m = {}
        for p in load_kb().get("painpoints", []):
            par = (p.get("parent_category")
                   or _HEALTH_DOMAIN_PARENT.get(p.get("domain", ""), "Other"))
            for k in (p.get("id"), p.get("name")):
                if k:
                    m[str(k).strip().lower()] = par
        _PARENT_BY_KEY = m
    if not name_or_id:
        return "Other"
    return _PARENT_BY_KEY.get(str(name_or_id).strip().lower(), "Other")


# ---------------------------------------------------------------------------
# OPEN-SET problem discernment — name the problem an ad addresses straight from
# its copy, even when it's NOT one of the catalogued KB conditions. This is the
# generalization layer: the KB (closed set of ~25 SOURCED conditions) grounds
# demographics; this names the problem for everything else. A copy-derived
# problem carries NO demographic prior — naming a problem is not the same as
# having sourced epidemiology for it, and we never fabricate the latter.
# ---------------------------------------------------------------------------
_PROBLEM_TRIGGERS = (
    r"struggl\w* with", r"suffer\w* from", r"tired of", r"sick of",
    r"fed up with", r"dealing with", r"plagued by", r"bothered by",
    r"embarrassed (?:by|about)", r"worried about", r"frustrated (?:with|by)",
    r"get rid of", r"say goodbye to", r"put an end to", r"banish",
    r"do you (?:have|suffer from|struggle with)", r"caused by",
    r"are you (?:struggling with|tired of|dealing with|bothered by)",
    r"the (?:real |root |hidden |number one |#?1 )?"
    r"(?:cause|reason|culprit) (?:of|behind|for)", r"end your", r"stop your",
)
_SYMPTOM = (r"pains?|aches?|discomfort|odou?rs?|itch\w*|dryness|cramps?|"
            r"swelling|inflammation|bloating|cravings?|breakouts?|wrinkles?|"
            r"sagging|stiffness|spasms?|flare[\- ]?ups?|tension|knots?|"
            r"tightness|soreness")
_PHRASE_STOP = {"and", "or", "but", "so", "to", "with", "for", "that", "this",
                "when", "because", "while", "your", "you", "the", "a", "an",
                "is", "are", "it", "of", "in", "on", "at", "by", "from", "now",
                "today", "again", "more", "less", "without", "like", "feel",
                "feeling", "my", "our", "their", "them", "all"}

_TRIG_RE = re.compile(
    r"(?:%s)\s+(?:your |that |the |a |an |this |their |from )?"
    r"([a-z][a-z',\- ]{2,45})" % "|".join(_PROBLEM_TRIGGERS))
_SYM_RE = re.compile(r"\b([a-z]{3,}\s+(?:%s))\b" % _SYMPTOM)

# "<part> pain" only names a problem when <part> is a PART. _SYM_RE captures
# exactly one word before the symptom noun, so a verb or pronoun in that slot
# is never anatomy — yet "the first thing she felt was pain" published the
# painpoint "Was Pain" on a real cataract ad, sitting on the card next to the
# genuine read. Anatomy and severity words ('lower', 'joint', 'chronic',
# 'constant', 'sharp') are deliberately absent from this set.
_NOT_A_PART = frozenset("""
was were are being been feel feels felt feeling get gets got getting
has have had having make makes made cause causes caused bring brings brought
this that these those they them their there here what when then than
your yours mine ours his her hers its own same such more most less least
any all some none very just still only ever never much many real actual
said says say know knew think thought saw see seen went come came took take
about with without from into onto over under after before while because
""".split())
_MED_RE = re.compile(
    r"\b([a-z]{4,}(?:itis|osis|emia|aemia|algia|pathy|plasia|trophy|oma))\b")


# Words that may appear INSIDE a problem phrase but can never end one.
_DANGLING = {"than", "as", "into", "onto", "over", "under", "about", "after",
             "before", "up", "down", "out", "off", "if", "then", "every",
             "any", "some", "not", "being", "been", "be", "too", "very",
             "just", "even", "still", "much", "many", "each", "both"}


def _clean_phrase(s):
    """Trim a captured span to a tight problem phrase (<=4 content words)."""
    words = re.findall(r"[a-z'\-]+", s.lower())
    while words and words[0] in _PHRASE_STOP:
        words.pop(0)
    kept = []
    for w in words[:6]:
        if w in _PHRASE_STOP and kept:        # stop at first connective
            break
        kept.append(w)
    while kept and kept[-1] in _PHRASE_STOP:
        kept.pop()
    # ...and never END on a word that cannot close a noun phrase. The break
    # above stops at the first CONNECTIVE in _PHRASE_STOP, but comparatives
    # and prepositions are not in that set, so they were kept as the last
    # word: "Tired of looking older than you feel" -> "Looking Older Than",
    # which is what the painpoint card actually displayed. Trim them.
    while kept and kept[-1] in _DANGLING:
        kept.pop()
    return " ".join(kept[:4])


# Body parts and bare objects: the thing the complaint is ABOUT, never the
# complaint. Alone they assert nothing ("Hair"), so they are not a painpoint;
# in a phrase they are fine ("Hair Loss", "Knee Pain").
_SOLO_STOP = {
    "hair", "hairline", "scalp", "skin", "face", "body", "teeth", "tooth",
    "gums", "eye", "eyes", "ear", "ears", "gut", "stomach", "belly", "hand",
    "hands", "foot", "feet", "leg", "legs", "arm", "arms", "knee", "knees",
    "hip", "hips", "shoulder", "back", "neck", "nail", "nails", "joint",
    "joints", "muscle", "muscles", "bone", "bones", "heart", "liver", "gums",
    "product", "products", "formula", "results", "science", "ingredients",
}


def extract_problem(text, min_len=4):
    """OPEN-SET: name the problem from arbitrary ad copy. Reads problem-framing
    language ('struggling with X', 'get rid of X'), '<part> pain/odor' symptom
    phrases, and clinical -itis/-osis terms. Returns a Title-Cased phrase or
    None. Carries NO demographic prior — naming != sourced epidemiology."""
    low = (text or "").lower()
    cand = {}                                  # phrase -> [score, first_pos]

    def add(span, pos, boost):
        ph = _clean_phrase(span)
        if len(ph) < min_len:
            return
        if all(w in _PHRASE_STOP or w in _STOP for w in ph.split()):
            return
        c = cand.setdefault(ph, [0, pos])
        c[0] += boost
        c[1] = min(c[1], pos)

    for m in _TRIG_RE.finditer(low):
        add(m.group(1), m.start(), 2)          # 'struggling with <X>'
    for m in _SYM_RE.finditer(low):
        if m.group(1).split()[0] in _NOT_A_PART:
            continue                           # "…felt was pain" -> "Was Pain"
        add(m.group(1), m.start(), 2)          # '<part> pain / odor'
    for m in _MED_RE.finditer(low):
        add(m.group(1), m.start(), 3)          # clinical -itis/-osis term
    if not cand:
        return None
    best = max(cand.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
    # A bare BODY PART is not a painpoint. "Hair" was reported as one on a
    # hair-loss ad (2026-08-27), sitting beside the real read. A single word
    # can still be a genuine complaint — "Tension" is a felt state and is a
    # locked expectation — so the cut is anatomy/objects, not word count.
    if best.lower() in _SOLO_STOP:
        return None
    return best.title()


# a brand token's TitleCase shape, and the DR launch verbs that precede it.
_BRAND = r"[A-Z][A-Za-z0-9][A-Za-z0-9'’&+\-]*(?:\s+[A-Z][A-Za-z0-9'’&+\-]+){0,2}"
_PROD_VERB = re.compile(
    r"\b(?i:introducing|introduce|meet|try|discover|switch to|say hello to|"
    r"upgrade to|order)\s+(" + _BRAND + r")")
_PROD_POSS = re.compile(r"\b(" + _BRAND + r")['’]s\b")
# The product-as-ANSWER reveal — DR/UGC ads name the product as the thing the
# prospect "just needed". Whisper transcripts often LOWERCASE the brand (sung /
# uncapitalized speech), which the TitleCase _BRAND patterns above miss, so this
# slot accepts a lowercase token — gated HARD by _looks_coined below so an
# everyday word in the slot ("you just needed rest") is rejected.
_PROD_ANSWER = re.compile(
    r"\b(?i:you (?:just |really |only )?(?:ever )?needed|"
    r"all you (?:ever )?need(?:ed)?(?: is| was)|"
    r"the (?:answer|secret)(?: is| was))\s+"
    r"(?:the |a |an |your )?([A-Za-z][A-Za-z0-9'’\-]{3,})\b")
# Frequent words that land in that slot but are NOT brands. A coined brand is,
# by construction, not an everyday word, so we reject the common ones rather
# than try to enumerate brands.
_COMMON_WORD = {
    "rest", "help", "time", "sleep", "relief", "healing", "support", "care",
    "love", "change", "more", "less", "truth", "future", "answer", "secret",
    "best", "thing", "things", "water", "food", "money", "energy", "focus",
    "balance", "peace", "calm", "results", "proof", "science", "nature",
    "magic", "system", "method", "formula", "solution", "product", "device",
    "patience", "consistency", "discipline", "yourself", "something",
    "everything", "nothing", "comfort", "freedom", "confidence", "strength",
    "power", "control", "clarity", "movement", "mobility", "hope", "trust",
    "routine", "habit", "break", "space", "boundaries", "therapy",
}


def _looks_coined(tok):
    """Brand-shaped token, or None. CamelCase ('BioBlade') is almost always a
    brand; otherwise accept a single uncommon alpha token >=5 chars and
    TitleCase it. Conservative — a wrong product name is worse than none."""
    t = (tok or "").strip(" '’\"-")
    low = t.lower()
    if low in _COMMON_WORD or low in _PROD_STOP or len(t) < 5:
        return None
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9'’\-]+", t):
        return None
    if re.search(r"[a-z][A-Z]", t):            # CamelCase -> keep as-is
        return t
    return t[:1].upper() + t[1:].lower()


# ---- LANDING-PAGE / URL -> brand --------------------------------------------
# The single most reliable product signal: a DR ad's brand is its OWN domain
# (try.wildmintcosmetics.com -> WildMint Cosmetics, tryfloralabs.com -> Flora
# Labs). The HOOK frame is a problem headline with no brand; the brand is at the
# CTA/landing reveal. So scan the copy for a landing URL and read the brand off
# the registrable domain.
_TLD2 = {"co.uk", "org.uk", "com.au", "net.au", "co.nz", "co.za", "com.br",
         "co.in", "co.jp", "com.mx", "com.sg", "co.kr", "com.tr", "co.il",
         "com.my", "com.ph", "com.tw", "com.hk"}
_LINK_SERVICE = {"app.link", "onelink.me", "go.link", "smart.link", "bit.ly",
                 "linktr.ee", "lnk.to", "rebrand.ly", "tinyurl.com"}
_DOMAIN_PREFIX = ("try", "get", "shop", "use", "join", "buy", "go", "my",
                  "order", "the", "go2", "join")
_NON_BRAND_HOST = {"facebook", "instagram", "google", "youtube", "tiktok",
                   "amazon", "linktr", "page", "paid", "app", "link", "click",
                   "track", "ads"}
# common word components for splitting a glued domain into a readable brand
_BRAND_WORDS = ("wellness", "cosmetics", "skincare", "nutrition", "naturals",
                "organics", "botanicals", "supplements", "vitamins", "health",
                "beauty", "labs", "lab", "nutra", "derma", "beam", "wild",
                "mint", "clean", "program", "flora", "bud", "skin", "care",
                "better", "vitals", "true", "life", "gut", "bio", "pure",
                "nourish", "co", "the", "my", "now", "daily")


def _segment_brand(sld):
    """Greedy split of a glued domain SLD into words for display
    ('floralabs'->'Flora Labs', 'wildmintcosmetics'->'Wild Mint Cosmetics').
    Falls back to TitleCasing the whole token."""
    s, out = sld.lower(), []
    words = sorted(_BRAND_WORDS, key=len, reverse=True)
    while s:
        for w in words:
            if s.startswith(w) and (len(s) == len(w) or len(s) - len(w) >= 2):
                out.append(w)
                s = s[len(w):]
                break
        else:
            return sld[:1].upper() + sld[1:]      # no clean split
    return " ".join(w.title() for w in out)


def _brand_from_url(url):
    """Brand string from a landing URL, or None for tracking/social domains."""
    m = re.search(r"https?://([^/?#\s'\"]+)", url or "")
    if not m:
        return None
    labels = m.group(1).lower().strip(".").split(".")
    if len(labels) >= 2 and ".".join(labels[-2:]) in _LINK_SERVICE:
        labels = labels[:-2]                       # brand is the subdomain
    if not labels:
        return None
    if len(labels) >= 3 and ".".join(labels[-2:]) in _TLD2:
        sld = labels[-3]
    else:
        sld = labels[-2] if len(labels) >= 2 else labels[0]
    while labels and labels[0] in _DOMAIN_PREFIX and labels[0] != sld:
        labels.pop(0)
    for p in _DOMAIN_PREFIX:                        # glued prefix: tryfloralabs
        if sld.startswith(p) and len(sld) - len(p) >= 4:
            sld = sld[len(p):]
            break
    if len(sld) < 3 or sld in _NON_BRAND_HOST:
        return None
    return _segment_brand(sld)


def brand_from_copy_urls(text):
    """Brand from the FIRST product URL in the copy (skips social/tracking)."""
    for u in re.findall(r"https?://[^\s'\"<>]+", text or ""):
        b = _brand_from_url(u)
        if b:
            return b
    return None


# A sales-page <title> NAMES the product — read it off the page like a human,
# not the URL. Pages format the title as 'Topic | Category | Store', so the
# brand is the trailing coined segment ('Constipation I Elderberry I Elvera' ->
# Elvera). Separators: |, dashes, bullets, and a lone capital-I some themes use.
_TITLE_SEP = re.compile(r"\s*(?:[|–—•·]|(?<=\s)I(?=\s))\s*")
_TITLE_GENERIC = {"home", "shop", "official", "store", "site", "buy", "order",
                  "sale", "products", "product", "collections", "page",
                  "welcome", "menu", "checkout", "cart", "best", "reviews"}


# READ THE PRODUCT FROM THE AD ITSELF. The advertised brand is a COINED word
# (rare in normal English — wordfreq tells us) that the ad uses in a SELLING
# context ("Elvera is selling out / offers a guarantee / try Elvera"), as
# opposed to a COMPETITOR it bashes ("Miralax keeps failing you") — which may
# actually be MORE frequent. So: rarity filter to find coined candidates, then
# score each by selling-vs-competitor context. This is reading the ad the way a
# person does, not parsing the URL or the <title> tag.
_COINED_MAX_FREQ = 1e-7        # above this = a real English word, not a brand

# ---- ROLE REASONING: judge what each candidate IS in the ad's logic ---------
# The advertised product is the one being SOLD. A drug/competitor is what the
# product REPLACES; a condition is what it TREATS. We don't list drug names —
# we read the ROLE each candidate plays. {role: pattern with X = candidate}.
# `{X}` is substituted with the escaped candidate at scan time.
# BROAD selling signal — any of these near the candidate means the ad is
# pitching it (window-scan, not adjacency, so 'ColonBroom ... 60-day guarantee'
# counts even with words between).
_SELL_CUE = re.compile(
    r"selling out|sold out|\boffers?\b|guarantee|money.?back|\btry\b|\border\b|"
    r"introduc\w*|what makes \w+ different|\d+%\s*off|delivers?|risk.?free|"
    r"\bbuy\b|\bshop\b|subscrib\w*|refund|restock\w*|free shipping|\bdaily\b|"
    r"\bformula\b|capsules?|gummies|supplement|results?|\bcreated\b|developed|"
    r"clinically", re.I)
# ROLE patterns that DEMOTE a candidate — they mark X as a drug the product
# replaces, a condition it treats, or a competitor it bashes ({X}=candidate).
_REF_ROLE = [
    r"alternative to\s+{X}\b", r"\b{X}\b\s+alternative",
    r"(?:like|similar to)\s+{X}\b\s+(?:but|without)",
    r"without\s+(?:the\s+)?{X}\b", r"instead of\s+{X}\b",
    r"(?:skip|ditch|drop|quit|stop taking|cancel|replace[sd]?|prescrib\w+)\s+"
    r"(?:the\s+|your\s+)?{X}\b",
    r"(?:cheaper|safer|better)\s+than\s+{X}\b", r"nature'?s\s+{X}\b",
    r"\b{X}\b\s+(?:side.?effects?|costs?|prescription|injection|needle|shot)",
    r"\b{X}\b\s+(?:relief|symptoms?|flares?|test\b)", r"(?:relief|free)\s+from\s+{X}\b",
    r"(?:treat|treats|fight|fights|beat|combat|cure|cures|tackle|end|ends|"
    r"reverse|soothe|ease|heal|relieve)\s+(?:your\s+)?{X}\b",
    r"(?:suffering from|struggling with|dealing with|diagnosed with)\s+{X}\b",
    r"symptoms? of\s+{X}\b", r"cause[ds]? of\s+{X}\b",
    r"unlike\s+{X}\b", r"the problem with\s+{X}\b",
    r"\b{X}\b\s+(?:fails?|failing|doesn'?t work|won'?t work|is a scam)",
    # GENERAL: a drug you're 'on' / a condition tested 'for' / replaced 'than|
    # like|without' is the OBJECT of a preposition — the product never is.
    r"\b(?:on|for|from|than|versus|vs)\s+{X}\b",
    r"\b{X}\s*(?:®|™)",          # a cited registered brand (drug/competitor)
    # PERSON / testimonial: a narrator or cited expert is not the product.
    # wordfreq filters common names (Kathy/Doug); this catches rare surnames
    # (Budoff, Bellingham) sitting in a person frame.
    r"(?:Dr\.?|Mr\.?|Mrs\.?|Ms\.?|Prof\.?)\s+{X}\b",
    r"\b{X},?\s+(?:MD|PhD|RN|DO|MPH)\b",
    r"(?:friend|sister|brother|mother|father|husband|wife|neighbou?r|"
    r"colleague|patient|named|called)\s+(?:named\s+)?{X}\b",
    r"\b{X}\b\s+(?:said|says|asked|told|smiled|noticed|realized|explained|"
    r"admitted|whispered|added|recalls?|shared)",
    # INGREDIENT / active compound listed, not the branded product itself:
    # 'with Astaxanthin', 'contains Berberine', 'Rhodiola extract', 'X and Y'.
    r"(?:with|contains?|containing|of|plus|including|features?)\s+{X}\b",
    r"\b{X}\b\s+(?:extract|root|powder|complex|blend|dosage|mg\b)",
    # …plus a CHEMICAL HEAD NOUN: "Tranexamic Acid", "Zinc Oxide" name a
    # compound the product contains, never the product.
    r"\b{X}\s+(?:acid|oxide|peptides?|hydrochloride|hcl|sulfate|sulphate|"
    r"citrate|glycol|ceramides?|retinoids?)\b",
]

# An ACTIVES LIST — three or more '+'-joined capitalized terms, e.g.
# "Kojic Acid + Glycolic + Tranexamic + Alpha-Arbutin + Niacinamide". Every
# item in such a run is an ingredient, so none of them is the brand; that ad
# was read as a product called "Tranexamic", and because brand_from_ad_copy
# runs ahead of the landing-URL reader it beat the real brand off
# trystrawberry.com. This has to be a LIST-level test, not a {X} pattern:
# the candidate token is whatever survives tokenisation ("Arbutin" out of
# "Alpha-Arbutin"), so a regex anchored on the bare token misses the very
# items it needs to catch. Three items are required so an offer bundle
# ("Elvera + free shipping" — lowercase, two terms) is never demoted.
_ACTIVES_RUN = re.compile(
    r"(?:\b[A-Z][A-Za-z]+(?:-[A-Z][A-Za-z]+)?(?:\s+(?:Acid|Oxide|Extract))?"
    r"\s*\+\s*){2,}[A-Z][A-Za-z]+(?:-[A-Z][A-Za-z]+)?")


def _in_actives_list(tok, text):
    """How many actives-list runs name this token (0 when it never appears in
    one). Matching on the raw run text catches hyphenated members."""
    return sum(1 for m in _ACTIVES_RUN.finditer(text or "")
               if re.search(r"\b" + re.escape(tok) + r"\b", m.group(0), re.I))


def _referenced_role(tok, text):
    """How strongly the ad casts X in a NON-product role (drug it replaces,
    condition it treats, competitor it bashes)."""
    esc = re.escape(tok)
    return (sum(len(re.findall(p.replace("{X}", esc), text, re.I))
                for p in _REF_ROLE)
            + 2 * _in_actives_list(tok, text))


def brand_from_ad_copy(text, title=""):
    """Read the advertised product by REASONING about each candidate's ROLE: is
    the ad SELLING it, or REPLACING / TREATING / BASHING it? The product is the
    coined word pitched in selling context whose referenced-role is weaker — so
    a drug ('alternative to Ozempic'), a condition ('SIBO relief'), or a
    competitor ('unlike Miralax') filter THEMSELVES out by their own role, no
    name list. Returns the brand or None."""
    if not text:
        return None
    try:
        import wordfreq as _wf
        freq = lambda w: _wf.word_frequency(w, "en")          # noqa: E731
    except Exception:  # noqa: BLE001
        freq = lambda w: 0.0 if w not in _COMMON_WORD else 1.0  # noqa: E731
    title_l = (title or "").lower()
    import collections
    counts = collections.Counter(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text))
    # a token in a PRODUCT-NOUN FRAME ("Elvera Serum", "the Nektar Complex")
    # is a brand even on a SINGLE mention — DR hooks name the product once.
    prod_head = (r"(?:serum|gummies|gummy|complex|formula|blend|drops?|"
                 r"capsules?|patch|patches|cream|oil|shot|booster|regime|"
                 r"supplement|collagen|pills?|powder|kit|system|device|"
                 r"brace|sleeves?|band|belt)")
    framed = set(re.findall(r"\b([A-Z][a-zA-Z]{3,})\s+" + prod_head, text))
    framed |= set(re.findall(r"\bthe\s+([A-Z][a-zA-Z]{3,})\b", text))
    # ATTRIBUTION FRAME. A DR creative names its brand ONCE, in an explicit
    # attribution rather than a product-noun phrase — "Only by MuscleTech.",
    # "...T&Cs apply MOSH", "STAR DENTALS On Rosedale". Those tokens are
    # neither repeated nor followed by a product noun, so the n<2 rule below
    # discarded every one of them. Measured on 42 real creatives: the brand was
    # readable in the OCR for 23, and the extractor returned NOTHING for 16 of
    # those 23. The name being present but unclaimed was the single biggest
    # source of missing products — not context length (failures were spread
    # evenly across 84..553 chars of available text).
    # NOTE deliberately NOT re.I — under re.I the [A-Z] class also matches
    # lowercase, so "by" would capture any following word. Capitalisation is
    # the signal here, so the prefix is spelled out in both casings instead.
    attrib = set(re.findall(
        r"(?:[Oo]nly|[Mm]ade|[Pp]owered|[Bb]rought to you)?\s*\b[Bb]y\s+"
        r"([A-Z][a-zA-Z]{3,})\b", text))
    attrib |= set(re.findall(r"\b([A-Z][a-zA-Z]{3,})(?:\.com|\.co\b|\.net)",
                             text))
    attrib |= set(re.findall(r"(?:©|\(c\))\s*(?:\d{4}\s*)?([A-Z][a-zA-Z]{3,})",
                             text))
    # An all-caps rule was tried here and REGRESSED the metric (19 wrong vs 5):
    # OCR emits headline copy as caps run-ons — "METABOLICHATIVATOR",
    # "WEIGHTLOSSPILLSTHAT", "MINZIONEFREQUENTE" — which are low-frequency and
    # therefore look exactly like coined brand names. Do not re-add it without
    # a word-boundary-aware segmenter.
    framed |= {t for t in attrib if t.lower() not in _PROD_STOP}
    best, best_score = None, 0.0
    for tok, n in counts.items():
        if tok.lower() in _PROD_STOP:
            continue
        if freq(tok.lower()) >= _COINED_MAX_FREQ:             # a real word
            continue
        # single mention is allowed only when it sits in a product frame; a
        # bare capitalized coined word appearing once is too weak to trust.
        if n < 2 and tok not in framed:
            continue
        sell = 0
        for m in re.finditer(r"\b" + re.escape(tok) + r"\b", text):
            sell += len(_SELL_CUE.findall(
                text[max(0, m.start() - 70):m.end() + 70]))
        ref = _referenced_role(tok, text)
        in_title = tok.lower() in title_l
        in_frame = tok in framed
        # REASONING: keep it only if the ad SELLS it more than it
        # replaces/treats/bashes it (title mention / product-noun frame are
        # themselves selling intent).
        if sell + (3 if in_title else 0) + (2 if in_frame else 0) <= ref:
            continue
        # the recurrence bonus rewards a SOLD brand, not a bashed competitor
        # the ad keeps naming: only credit repetition when role is clean.
        recur = 0.3 * n if ref == 0 else 0.0
        score = (sell - 1.5 * ref + recur
                 + (5 if in_title else 0) + (2 if in_frame else 0))
        if score > best_score:
            best, best_score = tok, score
    return best


def brand_from_title(title, corpus=""):
    """Brand from a page <title>. Scans segments from the END (the store name
    sits last) for a short COINED proper noun; prefers one that also recurs in
    the page copy `corpus`. Returns the brand or None."""
    if not title or not isinstance(title, str):
        return None
    segs = [s.strip() for s in _TITLE_SEP.split(title) if s.strip()]
    body = (corpus or "").lower()
    best = None
    for s in reversed(segs):
        words = s.split()
        if not (1 <= len(words) <= 2):
            continue
        if any(w.lower() in _TITLE_GENERIC for w in words):
            continue
        cand = _looks_coined(s) if len(words) == 1 else (
            s if s[0].isupper() and s.lower() not in _COMMON_WORD else None)
        if not cand:
            continue
        # a brand recurs in the page copy; if it does, take it immediately
        if body and body.count(cand.split()[0].lower()) >= 2:
            return cand
        best = best or cand                       # else remember the last-segment
    return best


# capitalized sentence-starts / screamers that are NOT brand names
_PROD_STOP = {
    "the", "this", "that", "your", "you", "my", "our", "we", "it", "its",
    "i", "a", "an", "and", "but", "so", "if", "when", "why", "how", "what",
    "imagine", "because", "new", "now", "today", "free", "stop", "finally",
    "no", "yes", "fight", "send", "master", "built", "made", "men", "women",
    "people", "everyone", "comment", "shop", "read", "learn", "whether",
    "there", "here", "let", "say", "get", "join", "men's", "women's",
    # PUBLISHER / DISCLOSURE CHROME — page furniture, never a product. A DR
    # advertorial is a sales page dressed as an article, so it carries the
    # label its format requires ("ADVERTORIAL", "SPONSORED", "PAID POST") in
    # large type at the very top, i.e. inside the densest selling language on
    # the page. Each of these is rarer in English than _COINED_MAX_FREQ, so
    # the coined-word filter reads it as an invented brand name and it then
    # scores maximally on every axis brand_from_ad_copy measures — high sell
    # context, zero competitor role. Measured: "advertorial" (freq 5.6e-08)
    # scored 6.6 and was the ONLY surviving candidate on a hair-loss ad whose
    # real brand ("Simply Revival") is made of ordinary words and never
    # reaches the scorer at all.
    "advertorial", "advertisement", "advertising", "advert", "sponsored",
    "sponsor", "promoted", "promotion", "disclosure", "disclaimer",
    "affiliate", "endorsement", "editorial", "testimonial", "transcript",
    "subscribe", "newsletter",
}


# Corporate brand suffixes — a word + one of these IS almost always the brand,
# even glued/all-caps from stylized-label OCR ('LUMANUTRITION' -> Luma
# Nutrition, 'Arctic Botanicals', 'BioRoot Labs'). Longest-first so 'nutrition'
# wins before 'nutra'.
_BRAND_SUFFIXES = ["nutraceuticals", "nutritionals", "supplements",
                   "botanicals", "nutrition", "naturals", "wellness",
                   "biotics", "formulas", "pharma", "nutra", "labs",
                   "health", "naturals", "lab"]


# generic descriptors that precede a suffix in ordinary copy ('healthy
# nutrition', 'advanced formula') — NOT brands.
_NOT_BRAND_PREFIX = {"healthy", "daily", "complete", "advanced", "total",
                     "essential", "optimal", "premium", "good", "better",
                     "best", "your", "our", "the", "real", "active", "balanced"}


def _brand_with_suffix(txt):
    """Recover a brand carrying a corporate suffix. Returns 'Word Suffix' or
    None. The brand word must be a real capitalized proper noun (case-SENSITIVE
    [A-Z] — no global re.I, which would match lowercase prose like 'healthy
    nutrition'); the suffix is case-insensitive. Handles spaced title-case AND
    glued all-caps OCR ('LUMANUTRITION' -> Luma Nutrition)."""
    suf = "((?i:" + "|".join(_BRAND_SUFFIXES) + "))"   # capturing + scoped flag

    def ok_pre(p):
        return p.lower() not in _PROD_STOP and p.lower() not in _NOT_BRAND_PREFIX

    m = re.search(r"\b([A-Z][A-Za-z'&]{2,14})\s+" + suf + r"\b", txt)
    if m and ok_pre(m.group(1)):
        return f"{m.group(1)[:1].upper()}{m.group(1)[1:]} {m.group(2).title()}"
    for run in re.findall(r"[A-Z]{6,}", txt):          # glued all-caps run
        low = run.lower()
        for s in _BRAND_SUFFIXES:
            if low.endswith(s) and len(low) - len(s) >= 2:
                pre = run[:len(run) - len(s)]
                if ok_pre(pre):
                    return f"{pre.title()} {s.title()}"
    return None


_TAGLINE_STARTERS = {"still", "why", "your", "our", "this", "that", "pay",
                     "get", "the", "what", "when", "how", "stop", "youre",
                     "were", "buy", "order", "dont", "are", "does", "is"}


def _tagline_camel(tok):
    """True if a CamelCase token is really a de-spaced TAGLINE fragment, not a
    brand: first segment is a verb (-ing) or a question/pronoun starter
    ('BattlingBlood', 'StillTired', 'WhyPay'). Coined brands (BioRoot, LeanBiome)
    start with a non-verb modifier, so they pass."""
    m = re.search(r"[a-z][A-Z]", tok)
    if not m:
        return False
    first = tok[:m.start() + 1].lower()
    return first.endswith("ing") or first in _TAGLINE_STARTERS


def extract_product(text):
    """Best-effort BRANDED product name from copy when no vision brief named
    it — DR ads launch with 'Introducing X' or name a possessive brand
    ('Revolt's silk boxers …'). Returns a short brand string or None.
    HIGH-PRECISION on purpose: a wrong product name is worse than none, so a
    bare capitalized sentence-start is rejected and possessive brands must
    actually recur in the copy."""
    txt = text or ""

    def ok(name):
        name = name.strip(" '’\"-")
        words = name.split()
        if len(name) < 3 or not words or words[0].lower() in _PROD_STOP:
            return None
        if name.isupper() and len(words) > 1:        # 'ORDER NOW' screamer
            return None
        if name.isupper() and len(name.replace(" ", "")) < 4:  # PCP/PSA/FDA/ED
            return None
        return name

    # A LANDING URL is the most reliable brand signal — a DR ad's brand is its
    # own domain. Checked FIRST: the brand is revealed at the CTA/landing, not
    # the hook frame.
    bu = brand_from_copy_urls(txt)
    if bu:
        return bu

    # A corporate-SUFFIX brand is the strongest signal and survives stylized /
    # glued OCR ('LUMANUTRITION' -> Luma Nutrition) — checked next.
    bs = _brand_with_suffix(txt)
    if bs:
        return bs

    # NB: a recurring-CamelCase heuristic was tried here and REMOVED — on
    # garbled stylized-label OCR it grabbed de-spaced fragments as fake brands
    # ('OilUltra', 'NutritionCold', 'AliveNourush'): 6% right / 86% wrong over
    # 50 ads. Coined brands that lack a corporate suffix (BioRoot/LeanBiome) are
    # still caught below by their launch verb / _brand_with_suffix; a wrong
    # brand is worse than none, so we no longer guess from CamelCase shape.
    for m in _PROD_VERB.finditer(txt):               # 'Introducing X' — strong
        n = ok(m.group(1))
        if n:
            return n
    for m in _PROD_POSS.finditer(txt):               # 'Revolt's' — if it recurs
        n = ok(m.group(1))
        if n and len(re.findall(r"\b" + re.escape(n) + r"\b", txt)) >= 2:
            return n
    for m in _PROD_ANSWER.finditer(txt):    # 'you just needed bioblade' (lower)
        n = _looks_coined(m.group(1))
        if n:
            return n
    return None


def _age_buckets_for(lo, hi):
    return [b for b, (blo, bhi) in _AGE_RANGES.items()
            if not (hi < blo or lo > bhi)]


def _parse_value(value):
    """Free-text correlate value -> {dimension: [buckets]}. e.g.
    'women 45-64' -> {'gender':['female'],'age':['45-54','55+']}."""
    v = value.lower()
    out = {}
    if re.search(r"\b(women|female|woman|girls?)\b", v):
        out.setdefault("gender", []).append("female")
    elif re.search(r"\b(men|male|man|boys?)\b", v):
        out.setdefault("gender", []).append("male")
    nums = [int(n) for n in re.findall(r"\d{2}", v)]
    if "65+" in v or "older" in v or "elderly" in v or "seniors" in v:
        out.setdefault("age", []).extend(["55+"])
    if nums:
        lo = min(nums)
        hi = max(nums) if len(nums) > 1 else (120 if "+" in v else lo + 9)
        out.setdefault("age", []).extend(_age_buckets_for(lo, hi))
    for kw, ls in [("retire", "empty nester / retiree"),
                   ("postmenopaus", "empty nester / retiree"),
                   ("toddler", "parent of young kids"),
                   ("new mother", "parent of young kids"),
                   ("teen", "parent of teens"),
                   ("student", "student"),
                   ("college", "student")]:
        if kw in v:
            out.setdefault("life_stage", []).append(ls)
    # dedup
    return {d: sorted(set(b)) for d, b in out.items()}


# --- Pet-ownership demographic overlay ---------------------------------------
# Marketers researching product-fit use the standard US pet-owner demographics
# (AVMA Pet Ownership & Demographics Sourcebook; APPA National Pet Owners
# Survey). A pet-CONDITION ad ("probiotic for cats", "senior dog joint chews")
# is bought by the OWNER, whose demographics the pet painpoint's veterinary
# study never carries — so a cat ad would otherwise read 'unclear' or borrow a
# human condition's skew. This overlay adds the OWNER's sourced demographics.
# Cat owners skew female and 35-64; dog owners are broader and slightly younger.
# Fires on explicit ownership language, not any 'cat'/'dog' token.
_CAT_OWNER = re.compile(
    r"\bcats?\b|\bkittens?\b|\bfeline|cat mom|cat dad|cat owner|cat parent|"
    r"cat lady|litter box|litter tray|\btabby\b|your cat|my cat|senior cat|"
    r"indoor cat|kitty|meow", re.I)
_DOG_OWNER = re.compile(
    r"\bdogs?\b|\bpupp(?:y|ies)\b|dog mom|dog dad|dog owner|dog parent|\bpup\b|"
    r"your dog|my dog|\bleash\b|\bcanine|senior dog|\bpooch|\bdoggo", re.I)


def pet_owner_correlates(text):
    """Sourced US pet-OWNER demographic correlates (same schema as the KB), so a
    pet-condition ad grounds the buyer's sex/age/life-stage — not just the
    animal's condition. Cat owners: female, 35-64. Dog owners: broader, 25-54.
    Returns [] when there is no clear pet-ownership signal."""
    low = text.lower()
    cat, dog = bool(_CAT_OWNER.search(low)), bool(_DOG_OWNER.search(low))
    out = []
    AVMA = "AVMA Pet Ownership & Demographics Sourcebook"
    AVMA_URL = ("https://www.avma.org/resources-tools/reports-statistics/"
                "us-pet-ownership-statistics")
    APPA = "APPA National Pet Owners Survey"
    APPA_URL = "https://www.americanpetproducts.org/industry-trends-and-stats"
    if cat:
        out += [
            {"factor": "sex (US cat-owning households)", "value": "women",
             "finding": "Women are the primary caregivers in most US cat-owning "
             "households.", "source_name": AVMA, "source_url": AVMA_URL,
             "confidence": "high"},
            {"factor": "age (US cat owners)", "value": "adults 35 to 64",
             "finding": "US cat ownership concentrates among adults 35-64.",
             "source_name": APPA, "source_url": APPA_URL, "confidence": "high"},
            {"factor": "age (US cat owners, modal band)", "value": "adults 45 to 54",
             "finding": "Middle-aged adults (45-54) are the largest single "
             "cat-owning age band.", "source_name": AVMA, "source_url": AVMA_URL,
             "confidence": "high"},
            {"factor": "life stage (cat owners)", "value": "empty-nest / retiree "
             "households (fewer young kids at home)",
             "finding": "Cat ownership skews toward established / empty-nest "
             "households rather than homes with young children.",
             "source_name": APPA, "source_url": APPA_URL, "confidence": "low"},
        ]
    if dog:
        out += [
            {"factor": "sex (US dog-owning households)", "value": "women "
             "(primary caregiver)",
             "finding": "Women are most often the primary caregiver in "
             "dog-owning households, though ownership is broad.",
             "source_name": AVMA, "source_url": AVMA_URL, "confidence": "medium"},
            {"factor": "age (US dog owners)", "value": "adults 25 to 54",
             "finding": "US dog ownership is highest among adults 25-54, "
             "including families.", "source_name": APPA, "source_url": APPA_URL,
             "confidence": "high"},
        ]
    return out


def pet_owner_prior_buckets(text):
    """CAT-owner targeting ASSERTS the buyer's demographic (female, established
    household) like a copy marker — cat ownership/purchasing is genuinely
    female-skewed (AVMA/APPA), NOT a disease prevalence skew, so it is exempt
    from the gender study-prior cap in infer_demographics. Dog ownership is
    gender-balanced, so it only corroborates via the (capped) study-prior.
    Returns {dim: {value: asserting_weight}}."""
    out = {"gender": {}, "life_stage": {}}
    if _CAT_OWNER.search(text.lower()):
        out["gender"]["female"] = 1.6
        out["life_stage"]["empty nester / retiree"] = 1.2
    return out


def demographic_prior(text, extra_painpoints=None):
    """Aggregate the matched painpoints' study correlates into weighted
    priors. Returns {'age':{bucket:w}, 'gender':{...}, 'life_stage':{...},
    'painpoints':[(name,summary)], 'sources':[(finding,name,url)]}.

    `extra_painpoints` (clean vision-brief labels) are folded into the
    match, so a condition Gemini read off the creative still drives the
    epidemiological age/gender prior. Pet-condition ads ALSO fold in the
    sourced pet-OWNER demographics (AVMA/APPA), since the buyer is the owner."""
    kb = load_kb()
    matched = match_painpoints(text, extra_painpoints=extra_painpoints)
    prior = {"age": {}, "gender": {}, "life_stage": {}}
    painpoints, sources = [], []
    # DOMINANCE weighting: a multi-benefit supplement ad mentions its PRIMARY
    # condition many times (joint pain) and secondary benefits once ("brain
    # fog", "feel like yourself again"). Scale each painpoint's correlate
    # weight by its hit share so the primary condition drives the audience and
    # an incidental one-cue match can't pull age/gender to its own skew.
    matched = [(p, h) for p, h in matched if not p.get("_soft")]  # soft=desire-only
    max_hits = matched[0][1] if matched else 1
    for p, hits in matched:
        corrs = kb["correlations"].get(p["id"], [])
        painpoints.append((p["name"], p.get("description", "")))
        hw = hits / max_hits
        for c in corrs:
            w = CONF_W.get(c.get("confidence", "medium"), 0.6) * hw
            for dim, buckets in _parse_value(c.get("value", "")).items():
                for b in buckets:
                    prior[dim][b] = prior[dim].get(b, 0.0) + w
            if c.get("source_url"):
                sources.append((c.get("finding", ""), c.get("source_name", ""),
                                c["source_url"]))
    # pet-OWNER demographics (AVMA/APPA) — the human who BUYS a pet-condition ad.
    # Weighted (1.8x) above the incidental human/vet correlate so a cat ad reads
    # the OWNER (female, 35-64) rather than the animal's condition epidemiology
    # or a spuriously-matched human painpoint's skew.
    for c in pet_owner_correlates(text):
        w = CONF_W.get(c.get("confidence", "medium"), 0.6) * 1.8
        for dim, buckets in _parse_value(c.get("value", "")).items():
            for b in buckets:
                prior[dim][b] = prior[dim].get(b, 0.0) + w
        sources.append((c.get("finding", ""), c.get("source_name", ""),
                        c["source_url"]))
    return {**prior, "painpoints": painpoints, "sources": sources[:8]}


def condition_choices():
    """[{id, name, description}] for every KB condition — the menu the vision
    model maps its OWN understanding of the ad onto (see v4_distill's
    canonical_conditions field). Lets the model do the taxonomy mapping
    instead of brittle keyword cue-matching after the fact."""
    return [{"id": p["id"], "name": p["name"],
             "description": p.get("description", "")}
            for p in load_kb().get("painpoints", [])]


def prior_for_ids(ids):
    """Cited demographic priors for an EXPLICIT set of canonical condition
    ids — the conditions the vision model judged genuinely present — rather
    than keyword-matching raw copy. Same shape as demographic_prior, plus
    'names'. An empty/none list yields an empty prior (correct: a novel
    angle the KB doesn't cover contributes no study skew)."""
    kb = load_kb()
    by_id = {p["id"]: p for p in kb.get("painpoints", [])}
    chosen = [i for i in (ids or []) if i in by_id]
    prior = {"age": {}, "gender": {}, "life_stage": {}}
    painpoints, sources = [], []
    for pid in chosen:
        p = by_id[pid]
        painpoints.append((p["name"], p.get("description", "")))
        for c in kb.get("correlations", {}).get(pid, []):
            w = CONF_W.get(c.get("confidence", "medium"), 0.6)
            for dim, buckets in _parse_value(c.get("value", "")).items():
                for b in buckets:
                    prior[dim][b] = prior[dim].get(b, 0.0) + w
            if c.get("source_url"):
                sources.append((c.get("finding", ""), c.get("source_name", ""),
                                c["source_url"]))
    return {**prior, "painpoints": painpoints, "sources": sources[:8],
            "names": [by_id[i]["name"] for i in chosen]}


# The biological AGENTS a DR ad blames. An ANGLE is the mechanism the ad
# points at ("it's not your age, it's DHT"), but painpoint_angles can only
# ever hold PAINPOINT ROWS — so when an ad blamed DHT the card showed the
# nearest matched CONDITION instead ("Menopause Symptoms" on a female
# hair-loss ad, 2026-08-27), which the KB does not even link to hair.
#
# These are surfaced ONLY when the term appears in the ad copy AND in the
# matched painpoint's own sourced KB mechanism — never inferred, never
# fabricated. Every one below is present in the KB's mechanism prose.
_MECH_AGENTS = {
    "dht": "DHT",
    "dihydrotestosterone": "DHT",
    "5-alpha reductase": "5-alpha reductase",
    "5 alpha reductase": "5-alpha reductase",
    "cortisol": "Cortisol",
    "insulin resistance": "Insulin resistance",
    "insulin": "Insulin",
    "estrogen": "Estrogen",
    "oestrogen": "Estrogen",
    "testosterone": "Testosterone",
    "thyroid": "Thyroid",
    "inflammation": "Inflammation",
    "collagen": "Collagen",
    "histamine": "Histamine",
    "serotonin": "Serotonin",
    "dopamine": "Dopamine",
}
_MECH_AGENT_RE = re.compile(
    r"\b(" + "|".join(sorted((re.escape(k) for k in _MECH_AGENTS),
                             key=len, reverse=True)) + r")\b", re.I)


def _mech_text(p):
    """All sourced mechanism prose on a painpoint row, lowercased."""
    bits = []
    for src in (p.get("mechanism") or {}, p.get("biological_mechanism") or {}):
        for f in ("summary", "cause_effect_chain", "copy_signal"):
            v = src.get(f)
            if isinstance(v, str):
                bits.append(v)
    return " ".join(bits).lower()


def mechanism_angles(text, matched, limit=3):
    """The MECHANISM this ad blames, named — grounded twice over.

    A term qualifies only if the ad copy says it AND the matched painpoint's
    own KB mechanism says it. That keeps the Angle card sourced: on a female
    hair-loss ad that argues DHT shrinks the follicle, it reports
    'DHT' — the agent the ad actually names — instead of an unrelated
    comorbid condition, and instead of a bundled label like 'Menopause
    Symptoms' that names no symptom at all.

    Returns [(label, painpoint_name)], most-supported first, or [].
    """
    low = (text or "").lower()
    said = {m.group(1).lower() for m in _MECH_AGENT_RE.finditer(low)}
    if not said:
        return []
    out, seen = [], set()
    for p, _h in (matched or []):
        mech = _mech_text(p)
        if not mech:
            continue
        for term in said:
            label = _MECH_AGENTS[term]
            if label in seen:
                continue
            if re.search(r"\b" + re.escape(term) + r"\b", mech):
                seen.add(label)
                out.append((label, p["name"]))
                if len(out) >= limit:
                    return out
    return out


def mechanisms_for(names, limit=3):
    """Sourced biological/psychological MECHANISM + copy-signal for the matched
    painpoint NAMES — the copy->mechanism correlation surfaced as report data.
    Only what's in the KB (never fabricated); empty for painpoints not yet
    enriched. This is what lets analyze() explain WHY an ad's problem maps to a
    condition via a real mechanism instead of a bare label."""
    if not names:
        return []
    want = [n.lower() for n in names]                 # preserve match order
    idx = {p["name"].lower(): p for p in load_kb()["painpoints"]}
    out = []
    for n in want:
        p = idx.get(n)
        if not p:
            continue
        entry = {"painpoint": p["name"]}
        bm = p.get("biological_mechanism")
        if bm and bm.get("summary"):
            entry["biological"] = {"summary": bm["summary"],
                                   "source": bm.get("source", ""),
                                   "url": bm.get("source_url", "")}
        m = p.get("mechanism")
        if m and (m.get("summary") or m.get("copy_signal")):
            entry["mechanism"] = {"domain": m.get("domain"),
                                  "summary": m.get("summary"),
                                  "cause_effect": m.get("cause_effect_chain"),
                                  "copy_signal": m.get("copy_signal"),
                                  "sources": m.get("sources", [])}
        if len(entry) > 1:
            out.append(entry)
        if len(out) >= limit:
            break
    return out


def kb_sentences():
    """Distil the correlations KB into clean factual sentences for the
    prime corpus (the 'train it in' half). Each verified finding becomes
    one sourced sentence; deterministic order, deduped, no slop."""
    kb = load_kb()
    out, seen = [], set()
    for p in kb.get("painpoints", []):
        for c in kb.get("correlations", {}).get(p["id"], []):
            finding = (c.get("finding") or "").strip()
            if not finding or finding in seen:
                continue
            seen.add(finding)
            src = c.get("source_name", "")
            yr = c.get("year", "")
            tail = f" ({src}{', ' + yr if yr else ''})" if src else ""
            sep = "" if finding.endswith(".") else "."
            out.append(f"{finding}{sep}{tail}")
    # Schwartz 4.5 — mechanism knowledge: each painpoint's biological/
    # psychological cause->effect summary (sourced) + its copy-signal, so the
    # model learns to CORRELATE ad copy with a mechanism, not a keyword.
    for p in kb.get("painpoints", []):
        bm = p.get("biological_mechanism")
        if bm and (bm.get("summary") or "").strip():
            bs = bm["summary"].strip()[:400]
            if bs not in seen:
                seen.add(bs)
                bsrc = bm.get("source") or ""
                out.append(f"{bs}{'' if bs.endswith('.') else '.'}"
                           f"{(' (' + bsrc + ')') if bsrc else ''}")
        m = p.get("mechanism")
        if not m:
            continue
        summ = (m.get("summary") or "").strip()
        if summ and summ not in seen:
            seen.add(summ)
            srcs = m.get("sources") or []
            dom = ""
            if srcs and srcs[0].get("url"):
                dom = re.sub(r"^www\.", "", re.sub(r"^https?://([^/]+).*", r"\1",
                             srcs[0]["url"]))
                dom = f" ({dom})"
            out.append(f"{summ}{'' if summ.endswith('.') else '.'}{dom}")
        sig = (m.get("copy_signal") or "").strip()
        if sig and sig not in seen:
            seen.add(sig)
            out.append(f"Marketing angle for {p['name']}: \"{sig}\""
                       f"{'' if sig.endswith(('.', '!', '?')) else '.'}")
    return "\n".join(out)


def format_painpoints(text):
    matched = match_painpoints(text)
    if not matched:
        return None
    return ", ".join(f"{p['name']}" for p, _ in matched[:5])


if __name__ == "__main__":
    kb = load_kb()
    print(f"KB: {len(kb['painpoints'])} painpoints, "
          f"{sum(len(v) for v in kb['correlations'].values())} correlates")
    if not kb["painpoints"]:
        print("(KB empty — run the v4-correlations-build workflow first)")
        raise SystemExit(0)
    for probe in [
        "MENOBELLY? hot flashes and night sweats keeping you up?",
        "thinning hair and a receding hairline got you worried?",
        "worried your blood pressure is creeping up with age?",
        "behind on retirement savings and stressed about money?",
    ]:
        print(f"\n> {probe}")
        print("  painpoints:", format_painpoints(probe))
        pr = demographic_prior(probe)
        for dim in ("gender", "age", "life_stage"):
            if pr[dim]:
                top = max(pr[dim], key=pr[dim].get)
                print(f"  {dim}: {top}  (from {len(pr['sources'])} cited "
                      "correlates)")
