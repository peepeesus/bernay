"""
V4 demographic avatar inference — Stage 1: deterministic marker layer.

From ad text alone, estimate WHO the ad is targeting: gender skew, age
bracket, life stage, income band. Ads leak their audience through
surface markers ("MENOBELLY?" -> female 45-60; "your toddler's tantrum"
-> parents 25-44), so this stage needs no training — the same
rules-over-weak-model approach that carries the awareness classifier.

Every inference returns its EVIDENCE (the phrases that earned it), and
abstains ("unclear") below a confidence floor instead of guessing.

Stage 2 (v4_meta_audience.py) trains a small head on Meta Ad Library
reach breakdowns to refine these priors with real data. Deliberately
out of scope at every stage: nationality/ethnicity inference.
"""

import re

# marker -> (dimension, value, weight). Weights: 2.0 = near-definitive
# (biology/legal status), 1.0 = strong convention, 0.5 = weak hint.
MARKERS = {
    "gender": {
        "female": [
            (r"\bmenopause|menobelly|perimenopaus", 2.0),
            (r"\bpregnan(t|cy)|postpartum|c.section\b", 2.0),
            (r"\bher husband\b|\bgirls?'? night\b", 1.0),
            (r"\bmascara|foundation shade|bra\b|leggings\b", 1.0),
            (r"\bmom friends\b|\bmama\b|\bgirl ?boss\b", 1.0),
            (r"\bskincare routine\b|\bhot flash(es)?\b", 1.0),
            # DTC female-targeting slang ('looking for girlies that…',
            # 'be that girl', 'hey besties')
            (r"\bgirlies?\b|\bthat girl\b|\bfor the girls\b|\bbesties?\b", 1.0),
            # SEX-SPECIFIC anatomy/conditions — biologically female, so they
            # pin gender even when no lifestyle marker fires (a 'breast health'
            # or 'PCOS' ad is unmistakably female — the exact read that was
            # missing). \bbreast\b can't match 'breastfeeding' (no word
            # boundary), so the nursing disclaimer stays unaffected.
            (r"\bbreasts?\b|\bovar(y|ian|ies)\b|\buter(us|ine)\b|\bvagina\w*"
             r"|\bvulva\w*|\bcervi(x|cal)\b|\bendometrios\w*|\bfibroids?\b"
             r"|\bpcos\b|\bpcod\b|\bmenstru\w*|\bperiod cramps?\b"
             r"|\byeast infection\b|\bpelvic floor\b|\bhormonal acne\b", 2.0),
            # female hormone therapy — strong, not definitive (men rarely)
            (r"\bestrogen\b|\bprogesterone\b|\bHRT\b", 1.5),
            # THE ADVERTISER NAMING ITS OWN AUDIENCE. Keyed on the POSSESSIVE
            # ("women's hair loss") and explicit targeting ("for women",
            # "women over 40") — never bare "women", which is ordinary social
            # proof ("thousands of women swear by it") and would pin gender on
            # a unisex ad. This is what a landing slug like
            # /pages/peptides-for-womens-hair-loss says outright, and the read
            # returned gender 'unclear' on exactly that ad.
            (r"\bwomen'?s\b|\bfor women\b|\bwomen over \d", 1.5),
        ],
        "male": [
            (r"\btestosterone|prostate|erectile\b", 2.0),
            # ED / erectile-dysfunction signals incl. the category's DTC
            # brands (BlueChew/Rugiet are ED-only) — a male-targeting tell an
            # ad rarely spells out as 'erectile'. 'ed meds' covers the glued
            # OCR form re-surfaced below.
            # NB: 'erectile' (incl. 'erectile dysfunction') is already covered
            # by the testosterone/prostate/erectile marker above — list ONLY
            # the non-overlapping ED signals here so the same phrase isn't
            # double-counted (that inflated the gender margin).
            (r"\bed meds?\b|\bblue ?chew\b|\brugiet\b", 2.0),
            (r"\bbeard|razor burn\b", 1.5),
            # mirror of the female audience-naming marker above
            (r"\bmen'?s\b|\bfor men\b|\bmen over \d", 1.5),
            (r"\bhis wife\b|\bguys('| are)?\b", 1.0),
            (r"\bgains\b|\bbench press\b|\blocker room\b", 1.0),
            (r"\bdad bod\b|\bgentlemen\b", 1.0),
            # Colloquial male body-image hooks — a DR staple for
            # gut/lymphatic/liver/hormone ads that never say 'men' outright
            # ('beer belly', 'skinny fat', a 'dad bod') but are unmistakably
            # talking to men. Strong convention, not 100%-biological, so 1.5
            # (same class as beard/razor burn) rather than 2.0.
            (r"\bbeer (?:belly|gut)\b|\bskinny fat\b", 1.5),
            # SEX-SPECIFIC anatomy/conditions — biologically male
            (r"\b(erections?)\b|\bbph\b|enlarged prostate|\bprostatectomy\b"
             r"|\bpsa (level|test|score)\b|\bsemen\b|\bsperm\b|\bejaculat\w*"
             r"|\blow.?t\b|\bman.?boobs?\b|\bgynecomastia\b", 2.0),
            (r"\bmade for men\b|\bfor men\b", 1.0),
        ],
    },
    "age": {
        "18-24": [
            (r"\b(?:dorm|college|campus)\b|\bstudent loan(?!s for your kid)",
             1.5),
            (r"\bfirst (job|apartment|paycheck)\b", 1.0),
            # unbounded 'no cap'/'lowkey' matched as bare substrings (no
            # trailing \b applies to interior alternatives in Python re) --
            # 'no cap' fired inside ordinary supplement copy ('No capsules.
            # No droppers.'), a real false positive on a gethookd board ad
            # whose narrator explicitly stated her age as 67.
            (r"\bbestie\b|\bno cap\b|\blowkey\b|\bfr fr\b", 1.0),
            # bare 'exam' fires on 'eye exam'/'pre-exam'/'exam chair' --
            # constant vocabulary in ANY doctor-narrated health ad, not just
            # student ones (caught on 4 real ophthalmology ads misread as
            # academic-exam season). Require the actual academic phrasing.
            (r"\bexam (week|season|schedule)|\bfinal exams?\b|\bsemester\b",
             1.0),
        ],
        "25-34": [
            (r"\bwedding|engagement|newlywed\b", 1.0),
            (r"\bstartup|side hustle|career ladder\b", 1.0),
            (r"\bfirst home|down payment\b", 1.0),
            (r"\bnewborn|baby registry\b", 1.0),
        ],
        "35-44": [
            (r"\b(?:toddler|school run|pta|screen.time)\b", 1.5),
            (r"\bmortgage|minivan|family suv\b", 1.0),
            (r"\bmetabolism (slow|isn'?t)\b", 1.0),
            (r"\bwork.life balance\b", 0.5),
        ],
        "45-54": [
            (r"\bmenopause|menobelly|perimenopaus", 2.0),
            (r"\bteenagers?\b|\bcollege fund\b", 1.0),
            (r"\breading glasses\b|\bknee pain\b", 1.0),
            (r"\bblood pressure|cholesterol\b", 1.0),
        ],
        "55+": [
            (r"\bretire(d|ment)\b|\bpension\b", 2.0),
            (r"\bgrandkids?|grandchildren\b", 2.0),
            (r"\bmedicare|social security\b", 2.0),
            (r"\bjoint replacement|arthritis|hearing aid\b", 1.5),
            (r"\bprescriptions? for (decades|years)\b", 1.0),
            (r"\b(thirty|30)\+? years (ago|of)\b", 0.5),
        ],
    },
    "life_stage": {
        # bare 'exam' dropped -- see the age/18-24 marker above for why: it
        # fires on 'eye exam'/'pre-exam' in ordinary doctor-narrated ad copy,
        # not just academic exams (4 real ophthalmology-ad false positives).
        "student": [(r"\b(?:campus|dorm|semester|student)\b"
                    r"|\bexam (week|season|schedule)\b|\bfinal exams?\b",
                    1.5)],
        "young professional": [
            (r"\bcareer ladder|side hustle|linkedin\b", 1.0),
            (r"\bcommute|9.to.5|standup meeting\b", 1.0)],
        "parent of young kids": [
            (r"\btoddler|diaper|nap time|daycare|stroller\b", 2.0),
            (r"\bschool run|bedtime battle|screen.time\b", 1.5),
            (r"\bmy (daughter|son)\b.{0,30}\b(is )?(\d|1[0-2])\b", 1.5)],
        "parent of teens": [
            (r"\bteenagers?|high school|driving lessons\b", 1.5),
            (r"\bcollege (fund|applications)\b", 1.5)],
        "empty nester / retiree": [
            (r"\bretire(d|ment)|grandkids?|downsiz(e|ing)\b", 2.0),
            (r"\bempty nest\b", 2.0)],
    },
    "income": {
        "budget-conscious": [
            (r"\bcoupon|affordable|budget|paycheck to paycheck\b", 1.5),
            (r"\bsave money|can'?t afford|cheap(er)?\b", 1.0),
            (r"\b(50|fifty) cents\b|\bunder \$?\d+\b", 1.0)],
        "mid-market": [
            (r"\bvalue for money|investment in yourself\b", 0.5),
            (r"\bfree shipping|bundle deal\b", 0.5)],
        "premium": [
            (r"\bbespoke|concierge|hand.?crafted|artisanal\b", 2.0),
            (r"\bexclusive|members.only|waitlist\b", 1.0),
            (r"\bportfolio|private banking|first class\b", 1.5)],
    },
}

_COMPILED = {dim: {val: [(re.compile(p, re.I), w) for p, w in pats]
                   for val, pats in vals.items()}
             for dim, vals in MARKERS.items()}

CONFIDENCE_FLOOR = 1.0          # below this evidence weight: "unclear"
# A near-tied top-2 (e.g. one 'male' marker vs one 'female' marker of equal
# weight) is GENUINELY AMBIGUOUS evidence, not a vote for whichever value
# happens to be declared first in MARKERS (dict order previously decided
# every exact tie, which silently defaulted every tie to 'female' since it's
# the first key under "gender" — caught on a men's liver/beer-belly ad whose
# only markers were 'dad bod'+'guys' (male, 2.0) vs a same-weight 'pregnant'
# false-positive). Below this margin, abstain instead of asserting either.
MARGIN_FLOOR = 0.5

# A first-person self-stated age ('I'm 67', 'I'm 58 years old') is LITERAL
# GROUND TRUTH from the ad's own narrator -- the standard testimonial voice
# in long-form DR copy -- yet nothing in MARKERS ever looked for it; age was
# inferred only from indirect lifestyle vocabulary. Caught on a real
# gethookd ad ("My name is Linda. I'm 67 years old.") that still read
# 18-24 because an unrelated false-positive marker outscored a complete
# absence of competing evidence. Excludes trailing duration/quantity units
# so "I'm 30 minutes early" / "I'm 100% sure" don't get read as an age.
_EXPLICIT_AGE = re.compile(
    r"\bi'?m\s+(\d{1,3})\b(?!\s*(?:%|percent|minutes?|mins?|hours?|hrs?"
    r"|days?|weeks?|months?))"
    r"|\bi\s+am\s+(\d{1,3})\b(?!\s*(?:%|percent|minutes?|mins?|hours?|hrs?"
    r"|days?|weeks?|months?))",
    re.I)


# "...works like I'm 30 AGAIN", "feels like I'm 25 all over" — a SIMILE about
# restored youth, which in DR copy means the speaker is emphatically NOT that
# age. Caught live on a real ad: a diabetic ED testimonial ("A1C dropped from
# 7.6 to 6.8 ... my equipment works like I'm 30 again") was read as a literal
# self-stated age, scoring 25-34 at +4.0 and burying the sourced condition
# prior (45-54 / 55+ at 2.2) — the avatar came out 25-34 on an ad whose own
# epidemiology says mid-40s+.
_AGE_ORDER = ["18-24", "25-34", "35-44", "45-54", "55+"]

_AGE_SIMILE_BEFORE = re.compile(
    r"\b(?:like|as if|as though|feels?|felt|works?|working|back to|"
    r"than (?:i|when))\s+(?:i\s+)?$", re.I)
_AGE_SIMILE_AFTER = re.compile(r"^\s*(?:again|all over|once more)\b", re.I)


# A LIFE STAGE IS AN AGE RANGE WEARING A DIFFERENT NAME. "empty nester /
# retiree" at 35-44 is not a close call for the margin logic to break — it is a
# combination that must never be REPRESENTABLE. age and life_stage were
# independent argmaxes over their own marker sets, so the very words that score
# 55+ at 2.0 under age ('retirement', 'grandkids' — the SAME regexes) could
# lose the age race and still win life_stage uncontested, because that
# dimension has far fewer competing values. Caught on a real gethookd ad that
# printed "age 35-44" and "life stage empty nester / retiree" on one card.
#
# These are the ages at which a stage is POSSIBLE, deliberately generous at the
# edges (a 39-year-old can have teenagers) — only genuine impossibility is
# filtered, never a merely unlikely pairing. AGE decides first and is never
# overruled here: it has four independent evidence channels (self-stated,
# markers, study prior, faces) to life_stage's one.
_LIFE_STAGE_AGE_RANGE = {
    "student": (18, 29),
    "young professional": (18, 39),
    "parent of young kids": (22, 49),
    "parent of teens": (32, 59),
    "empty nester / retiree": (50, 120),
}


def _age_span(value):
    """'35-44' -> (35, 44); '55+' -> (55, 120); 'unclear'/'' -> None."""
    v = (value or "").strip()
    m = re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", v)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{1,3})\s*\+", v)
    if m:
        return int(m.group(1)), 120
    return None


def _stage_possible(stage, span):
    """Can someone in `span` be in this life stage at all? Unknown stages are
    always possible — this filter exists to remove contradictions, not to
    police a vocabulary it doesn't own."""
    rng = _LIFE_STAGE_AGE_RANGE.get(stage)
    if not rng or not span:
        return True
    return rng[0] <= span[1] and span[0] <= rng[1]


def _explicit_age_bucket(text):
    """-> age bucket string, or None. See _EXPLICIT_AGE above. Skips SIMILES
    ('like I'm 30 again'), which assert the opposite of the stated number."""
    m = None
    for cand in _EXPLICIT_AGE.finditer(text):
        pre = text[max(0, cand.start() - 30):cand.start()]
        post = text[cand.end():cand.end() + 14]
        if _AGE_SIMILE_BEFORE.search(pre) or _AGE_SIMILE_AFTER.search(post):
            continue                      # comparison, not the speaker's age
        m = cand
        break
    if not m:
        return None
    n = int(m.group(1) or m.group(2))
    if n < 18 or n > 110:
        return None
    if n <= 24:
        return "18-24"
    if n <= 34:
        return "25-34"
    if n <= 44:
        return "35-44"
    if n <= 54:
        return "45-54"
    return "55+"


# Standard supplement/OTC SAFETY BOILERPLATE — "if pregnant or nursing,
# consult a doctor", "keep out of reach of children" — appears on virtually
# every supplement and is NOT an audience signal. Mask the pregnant/nursing
# mention so it can't skew gender to female. Built to survive OCR mangling
# (lost spaces/garbled words): 'PREGNANTORNURSINGWOEN', 'pregnant or
# nursing women', 'do not use if pregnant'. Real targeting copy ('pregnant
# moms', 'during your pregnancy') has no nursing/consult/keep-out adjacency,
# so it is left intact.
_DISCLAIMER_PREG = re.compile(
    r"pregnan\w*\W{0,4}(?:or|and|/|,)?\W{0,4}(?:nursing|breast.?feed|lactat)\w*"
    r"|(?:nursing|breast.?feed|lactat)\w*\W{0,4}(?:or|and|/|,)?\W{0,4}pregnan\w*"
    r"|pregnan\w*ornursing\w*"
    r"|(?:do not use|consult|ask your|before use)\W{0,30}pregnan\w*"
    r"|keep out of reach\w*\W{0,40}pregnan\w*",
    re.I)

# FIGURATIVE 'pregnant' — bloating/weight-loss copy routinely uses "look(s)
# X months pregnant" as a SIMILE for a swollen belly, aimed at ANY gender
# ("that gut makes you look 6 months pregnant" on a men's liver-health ad).
# It is not a literal pregnancy signal, so it must not fire the near-
# definitive (2.0) female marker. Real targeting copy ("pregnant moms",
# "during your pregnancy") has no look/like preamble and is left intact.
_SIMILE_PREG = re.compile(
    r"\b(?:look|looks|looking|feel|feels|feeling|like (?:you'?re|i'?m|she'?s"
    r"|he'?s))\b.{0,25}?pregnan\w*",
    re.I)


def _mask_disclaimers(text):
    """Blank out safety-disclaimer pregnant/nursing mentions AND figurative
    '(makes you) look pregnant' similes so demographic MARKERS scan only
    genuine audience-signalling copy (see _DISCLAIMER_PREG / _SIMILE_PREG)."""
    text = _DISCLAIMER_PREG.sub(" ", text)
    return _SIMILE_PREG.sub(" ", text)


def _decide(scores, evidence, dim=None):
    """Scored ballot -> {value, confidence, evidence, scores}.

    Extracted so the life_stage ballot can be decided a SECOND time after
    impossible values are struck from it (see the constraint pass at the end of
    infer_demographics) using identical floor/margin rules — a re-decision that
    silently used different thresholds than the first pass would be its own
    class of bug."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:                      # every candidate was struck
        return {"value": "unclear", "confidence": 0.0, "evidence": [],
                "scores": scores}
    top_val, top_s = ranked[0]
    runner_val, runner_s = ranked[1] if len(ranked) > 1 else (None, 0.0)
    margin = top_s - runner_s
    # Two GENUINELY COMPETING signals (both independently clear the
    # confidence floor) within MARGIN_FLOOR of each other is conflicting
    # evidence, not a vote for `top_val` — without this, an exact tie
    # silently resolved to whichever value is declared first in MARKERS
    # (always 'female' for gender; see MARGIN_FLOOR docstring above).
    tied = runner_s >= CONFIDENCE_FLOOR and margin < MARGIN_FLOOR
    # AGE is ORDINAL, unlike gender/life-stage: two ADJACENT buckets are one
    # coherent range, not conflicting evidence. A condition set whose studies
    # say 45-54 AND 55+ (ED + type-2 diabetes + neuropathy) was being scored
    # as a tie and thrown away as 'unclear' — discarding exactly the
    # epidemiology the KB exists to provide. Merge them into the span
    # instead, so named conditions alone can carry an honest age read.
    if dim == "age" and tied and top_val in _AGE_ORDER \
            and runner_val in _AGE_ORDER:
        i, j = sorted((_AGE_ORDER.index(top_val),
                       _AGE_ORDER.index(runner_val)))
        if j - i == 1:
            lo = _AGE_ORDER[i].split("-")[0]
            hi_b = _AGE_ORDER[j]
            span = f"{lo}+" if hi_b.endswith("+") else \
                f"{lo}-{hi_b.split('-')[1]}"
            ev = sorted(set(evidence[top_val] + evidence[runner_val]))
            return {"value": span, "confidence": round(top_s, 2),
                    "evidence": ev[:4], "scores": scores}
    if top_s < CONFIDENCE_FLOOR or tied:
        return {"value": "unclear", "confidence": 0.0, "evidence": [],
                "scores": scores}
    return {"value": top_val, "confidence": round(margin, 2),
            "evidence": evidence[top_val][:4], "scores": scores}


def infer_demographics(text, vision_prior=None, painpoints=None,
                       condition_ids=None):
    """-> {dimension: {"value", "confidence", "evidence", "scores"}}.

    confidence = winning weight minus runner-up weight (margin), so two
    conflicting signals yield low confidence, not false certainty.

    Blends THREE evidence sources: surface markers in the copy, the
    EPIDEMIOLOGICAL prior from v4_correlations (a named painpoint's study
    skew), and an optional VISION_PRIOR from the Gemini brief
    ({dim: {value: weight}}) — what the creative actually showed/said
    (gender, age, accent->income). Each contribution tags its evidence
    ('study-prior' / 'vision') so the source is visible. `painpoints` are
    clean vision-brief condition labels passed to the epidemiological prior
    so a condition Gemini SAW (joint pain -> older adults) grounds the age
    read even when the copy never used a KB cue phrase. No-op for the
    blended sources until their inputs exist."""
    # When the vision model supplied the conditions it judged present
    # (condition_ids, possibly []), the epidemiological prior comes ONLY from
    # those — no keyword scan of the copy, so a stray word can't inject an
    # unrelated study skew. Falls back to keyword matching for text-only ads.
    try:
        try:
            import v4_correlations
        except ImportError:
            # Bernay's packaged layout (communities_XX_YY/...) needs the
            # qualified path; the bare import above only resolves in the
            # flat Downloads dev tree. This was silently failing on Bernay
            # and swallowed by the except below, so the ENTIRE
            # epidemiological-prior evidence source -- one of the three this
            # function's docstring promises -- was dead on every ad the live
            # app ever analyzed. Found via a real gethookd-board case
            # (bioroot_turmeric) that scored 'age: unclear' on Bernay but
            # correctly '55+' on Downloads with byte-identical code.
            from communities_00_09.community_4_v4_correlations_v4_kb_probe \
                import v4_correlations
        if condition_ids is not None:
            prior = v4_correlations.prior_for_ids(condition_ids)
        else:
            prior = v4_correlations.demographic_prior(
                text, extra_painpoints=painpoints)
    except Exception:  # noqa: BLE001
        prior = {"age": {}, "gender": {}, "life_stage": {}, "sources": []}
    vp = vision_prior or {}
    # surface markers scan copy with safety-disclaimer boilerplate masked,
    # so a 'pregnant or nursing' warning can't masquerade as an audience cue.
    marker_text = _mask_disclaimers(text)
    # OCR space-loss re-surfacing: stylized ad text comes back de-spaced
    # ('FORGIRLIESTHAT'), so re-inject a few distinctive merged tokens with
    # spaces restored, letting the word-boundary markers above match them.
    _ns = re.sub(r"[^a-z0-9]", "", marker_text.lower())
    for _tok, _phrase in (("girlies", " girlies "), ("girlie", " girlie "),
                          ("thatgirl", " that girl "),
                          ("girlboss", " girl boss "),
                          ("edmeds", " ed meds "),       # 'EDMedsPerformance'
                          ("bluechew", " bluechew "),
                          ("rugiet", " rugiet ")):
        if _tok in _ns:
            marker_text += _phrase

    explicit_age = _explicit_age_bucket(marker_text)

    # Pet-OWNER targeting asserts the buyer's demographic (cat owners = female,
    # established household) — a purchasing/ownership fact from AVMA/APPA, not a
    # disease-prevalence skew, so unlike the gender study-prior it is NOT capped.
    try:
        pet_assert = v4_correlations.pet_owner_prior_buckets(text)
    except Exception:  # noqa: BLE001
        pet_assert = {}

    out, ev_all = {}, {}
    for dim, vals in _COMPILED.items():
        scores, evidence = {}, {}
        for val, pats in vals.items():
            s, ev = 0.0, []
            for p, w in pats:
                m = p.search(marker_text)
                if m:
                    s += w
                    ev.append(m.group(0).strip().lower())
            # a self-stated age is literal ground truth, weighted above any
            # single indirect marker so it wins on its own -- but still runs
            # through the normal confidence-floor/margin logic below, not a
            # hard bypass.
            if dim == "age" and val == explicit_age:
                s += 4.0
                ev.append("self-stated age")
            # epidemiological prior contribution for this dim/value
            kb_w = prior.get(dim, {}).get(val, 0.0)
            if dim == "gender":
                # A condition's gender PREVALENCE skew is NOT proof THIS ad
                # targets that gender — advertisers routinely target the
                # minority (women's hair-loss, men's skincare, oyster/fertility
                # supplements for women). So the gender study-prior only
                # CORROBORATES: capped below the confidence floor (1.0) it
                # cannot assert gender alone, but it crosses the floor alongside
                # a copy marker. Sex-SPECIFIC conditions still assert via their
                # anatomy MARKERS (breast/ovarian/prostate/menopause...), which
                # are unaffected. (Caught by gethookd validation: hair-loss and
                # oyster ads targeting women were being read 'male' from skew.)
                kb_w = min(kb_w * 0.4, 0.8)
            elif dim == "life_stage":
                # SAME argument as gender, and it had been missed. A condition's
                # epidemiology says at what AGE it is prevalent; it says nothing
                # about whether this buyer's kids have left home or whether she
                # has stopped working. Uncapped, a lone prior of exactly 1.0 sat
                # on an EMPTY ballot (life_stage has no competing values unless
                # the copy names one), cleared the floor unopposed, and printed
                # "empty nester / retiree" with `evidence: ['study-prior']` —
                # the real cause of the retiree label on the SHE-lajit ad, whose
                # copy is a 47-year-old perimenopause testimonial and never
                # mentions retirement, grandkids or an empty nest. Capped, the
                # prior can still corroborate a real copy marker; it can no
                # longer invent a life stage on its own.
                kb_w = min(kb_w * 0.4, 0.8)
            if kb_w:
                s += kb_w
                ev.append("study-prior")
            # vision (Gemini brief) contribution
            v_w = vp.get(dim, {}).get(val, 0.0)
            if v_w:
                s += v_w
                ev.append("vision")
            # pet-OWNER targeting (asserts, uncapped — it's who buys, not a skew)
            pw = pet_assert.get(dim, {}).get(val, 0.0)
            if pw:
                s += pw
                ev.append("pet-owner (AVMA/APPA)")
            scores[val] = s
            evidence[val] = ev
        ev_all[dim] = evidence
        out[dim] = _decide(scores, evidence, dim)

    # STRUCTURAL CONSTRAINT — not a tie-break. A life stage the resolved age
    # makes IMPOSSIBLE is struck from the ballot entirely and the remaining
    # candidates are re-decided; if nothing survives the floor, life_stage
    # abstains. Weighing the two against each other was the wrong shape: no
    # amount of 'retirement' evidence makes a 35-44 buyer a retiree, it makes
    # the AGE read suspect — and age, with four evidence channels to
    # life_stage's one, is not the side to overturn from here.
    _span = _age_span((out.get("age") or {}).get("value"))
    _ls = out.get("life_stage")
    if _span and _ls and _ls.get("scores"):
        _kept = {v: s for v, s in _ls["scores"].items()
                 if _stage_possible(v, _span)}
        if _kept != _ls["scores"]:
            _struck = sorted(set(_ls["scores"]) - set(_kept))
            _re = _decide(_kept, ev_all.get("life_stage", {}), "life_stage")
            # keep the FULL ballot visible so the strike is auditable, and say
            # plainly why a value that scored well isn't the answer
            _re["scores"] = _ls["scores"]
            _re["ruled_out"] = {s: f"impossible at age "
                                   f"{out['age']['value']}" for s in _struck}
            out["life_stage"] = _re
    return out


def vision_markers(caption_text, discount=0.5):
    """Scan a LOCAL VLM's scene-CAPTION prose (Florence's description of the
    creative -- 'a man showing his beer belly', not literal on-screen/spoken
    ad copy) through the SAME MARKERS taxonomy used for text, returning a
    vision_prior-shaped {dim: {value: weight}} dict.

    This is the reasoning step that lets local, offline vision contribute to
    avatar categorization using the research data the model already has
    (the sourced marker/KB taxonomy) instead of a raw uncalibrated softmax
    class: caption -> structured marker hits -> calibrated prior -> the same
    grounded decision logic as everything else in infer_demographics.

    DISCOUNTED (default x0.5, same rate as the existing face-derived
    vision_prior below the confidence floor): a caption is a MODEL'S
    paraphrase of pixels, not literal ad text, and can hallucinate (observed
    on a real ad: a stock selfie photo captioned as 'likely related to
    pregnancy test results'). So on its own a caption hit can corroborate a
    copy marker or study-prior across the confidence floor, but a WEAK cue
    never asserts a value alone -- only a near-definitive one (an explicit
    sex-specific anatomical mention, weight 2.0 pre-discount = 1.0 post) can
    still clear CONFIDENCE_FLOOR unaided, same as literal copy would need to
    for that class of marker.

    One MAX hit per (dim, value) pair, not a sum across patterns -- a short
    caption stacking multiple pattern hits into one inflated score would
    undercut the whole point of discounting a paraphrase."""
    out = {}
    if not caption_text:
        return out
    for dim, vals in _COMPILED.items():
        for val, pats in vals.items():
            w = 0.0
            for p, weight in pats:
                if p.search(caption_text):
                    w = max(w, weight)
            if w:
                out.setdefault(dim, {})[val] = round(w * discount, 2)
    return out


def format_demographics(demo):
    """ASCII block for the admix report."""
    lines = []
    label = {"gender": "gender skew", "age": "age bracket",
             "life_stage": "life stage", "income": "income band"}
    for dim in ("gender", "age", "life_stage", "income"):
        d = demo[dim]
        if d["value"] == "unclear":
            lines.append(f"{label[dim]:<22} unclear (no strong markers)")
        else:
            ev = ", ".join(f"'{e}'" for e in d["evidence"])
            lines.append(f"{label[dim]:<22} {d['value']}  "
                         f"(margin {d['confidence']}; {ev})")
    return "\n".join(lines)


if __name__ == "__main__":
    probes = [
        ("MENOBELLY? That stubborn belly that showed up with menopause. "
         "Hot flashes keeping you up too?",
         {"gender": "female", "age": "45-54"}),
        ("Tired of the school run chaos? Your toddler's screen-time "
         "fights end this week. Affordable for every family budget.",
         {"life_stage": "parent of young kids", "income":
          "budget-conscious"}),
        ("Retired and finally free - but your knees didn't get the "
         "memo. Grandkids deserve a grandpa who can keep up.",
         {"age": "55+", "life_stage": "empty nester / retiree"}),
        ("Testosterone drops 1% a year after 30. The gains you chase "
         "in the gym start in your hormones, guys.",
         {"gender": "male"}),
        ("Bespoke, hand-crafted, members-only. Join the waitlist.",
         {"income": "premium"}),
        ("This pen writes in three colors.", {}),     # no markers
    ]
    correct = total = 0
    for text, expect in probes:
        demo = infer_demographics(text)
        print(f"\n> {text[:64]}...")
        print(format_demographics(demo))
        for dim, want in expect.items():
            total += 1
            got = demo[dim]["value"]
            correct += got == want
            assert got == want, f"{dim}: wanted {want}, got {got}"
        if not expect:
            assert all(d["value"] == "unclear" for d in demo.values()), \
                "marker-free text must abstain everywhere"
    print(f"\nself-test: {correct}/{total} labeled dimensions OK, "
          "abstention OK")
