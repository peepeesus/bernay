"""
V4 AD-MIX ANALYZER — paste ONE blob of text; the engine decomposes it.

    python v4_admix.py        then paste anything: an ad transcript, a
    creative brief, avatar notes + copy mixed together. End with TWO
    empty lines (or Ctrl+Z then Enter).

What it figures out, automatically:
  1. ROLE SPLIT — paragraphs are classified avatar-ish vs copy-ish by
     deterministic voice markers (third-person persona language vs
     second-person/brand selling language). No labels needed.
  2. THE AD MIX of the copy — through the prime-trained motif scorer:
     angle archetype, Schwartz awareness stage, market sophistication
     level, and the desires the copy channels.
  3. THE IMPLIED AVATAR — if you pasted no avatar text, one is derived
     from the copy itself: per Schwartz, copy can only channel existing
     desire, so the desires it channels define its target. The implied
     avatar's motif vector is synthesized from those desire tags.
  4. PV = (I x D) x T against the detected or implied avatar.
"""

import json
import os
import re

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import torch

import v4_demographics as demo_mod
import v4_motif_scorer as ms
import v4_pv_engine as eng_mod
import v4_tokenizer as tok
from v4_motif_scorer import MotifScorer

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))

AVATAR_MARKERS = re.compile(
    r"\b(man|woman|male|female|parent|mother|father|aged|years old|"
    r"customer|audience|persona|avatar|demographic|income|worried about|"
    r"skeptical|struggles? with|wants proof|already tried|he is|she is|"
    r"in his|in her|notices|embarrassed)\b", re.I)
COPY_MARKERS = re.compile(
    r"\b(you|your|we|our|buy|order|guarantee|money back|introducing|"
    r"discover|unlock|free|today|new|stop|imagine|finally|claim|deserve|"
    r"proven|results)\b", re.I)

# The model's VISUAL product-category (v4_vision_head niche) -> KB condition
# NAME. Used ONLY as a text-blind fallback to name the problem domain when the
# copy yielded nothing (garbled-OCR / image-only creatives) — never to inject a
# demographic prior, since even the reliable classes below are a learned read,
# not certainty.
VISUAL_TO_CONDITION = {
    "joint pain": "Joint Pain & Arthritis",
    "prostate": "Prostate Enlargement",
    "menopause": "Menopause Symptoms",
    "hair loss": "Hair Loss & Thinning",
    "blood sugar": "Blood Sugar & Type 2 Diabetes",
    "gut health": "Gut Health & Digestion",
    "tinnitus": "Hearing Loss & Tinnitus",
    "eye vision": "Vision Decline & Eye Health",
    "teeth": "Tooth & Gum Health",
    "weight loss": "Stubborn Weight & Belly Fat",
    "anxiety sleep": "Anxiety & Chronic Stress",
    "energy fatigue": "Low Energy & Chronic Fatigue",
    "skincare": "Wrinkles & Skin Aging",
}
# The reliable-image-class set (RELIABLE_IMAGE_CLASSES) and the routing
# logic that uses it now live in v4_vision_head.py, imported lazily where
# needed below — it's a property of the image classifier itself, not a
# pipeline concern, and v4_admix.py already imports v4_vision_head lazily
# (best-effort: vision degrades gracefully if unavailable) rather than at
# module level, so it can't be a bare module-level constant here too
# without either duplicating it (drift risk) or forcing a top-level import.


def read_one_paste():
    print("Paste your text (ad transcript / brief / anything). "
          "Finish with TWO empty lines:")
    lines, blanks = [], 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            blanks += 1
            if blanks >= 2 and any(s.strip() for s in lines):
                break
        else:
            blanks = 0
        lines.append(line)
    return "\n".join(lines).strip()


def split_roles(text):
    """Paragraph-level avatar vs copy classification by voice markers."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) == 1:
        paras = [text]
    avatar_ps, copy_ps = [], []
    for p in paras:
        a = len(AVATAR_MARKERS.findall(p))
        c = len(COPY_MARKERS.findall(p))
        words = max(len(p.split()), 1)
        (avatar_ps if a / words > 0.04 and a > c * 0.6 else
         copy_ps).append(p)
    if not copy_ps:                      # everything looked avatar-ish:
        copy_ps, avatar_ps = avatar_ps, []   # treat it all as copy
    return " ".join(avatar_ps), " ".join(copy_ps)


# ---- deterministic awareness rules (ad-voice structural signals) ----------
# ads targeting each Schwartz stage have recognizable MOVES, not just
# vocabulary: problem-aware ads call out the symptom (often a question
# headline, "MENOBELLY?"); solution-aware ads sell the mechanism;
# product-aware ads stack proof and risk-reversal; most-aware ads just
# announce the deal. Regex per stage; each distinct rule hit adds a fixed
# boost on top of the motif z-score.
# Rules are CLUSTERS (lists of pattern-lists): one marketing ELEMENT
# scores once however many of its phrasings appear. "30-day money-back
# guarantee" is one risk-reversal element, not three product signals —
# triple-counting it is exactly how a story-lead advertorial got called
# product_aware.
AWARENESS_RULES = {
    "unaware": [
        [r"\bdid you know\b"],
        [r"\bmost (people|women|men)\b.{0,40}\bnever\b"],
        [r"\bhidden (reason|cause|link)\b"],
        [r"\bwithout (even )?knowing\b"],
        [r"\b(don'?t|never|didn'?t) realize\b"],
        [r"\bthe real reason\b"], [r"\bsilently\b"],
        [r"\b(he|she|they) (thought|figured|assumed)\b"],
        [r"\bchanged everything\b"], [r"\byou could be \w+ing\b"],
        [r"\bturns? out\b"],
        # the mystery/accidental-discovery beat of story advertorials
        [r"\b(didn'?t|never) know what\b",
         r"\bstumbled (on|across|upon)\b",
         r"\buntil (i|she|he|we) (found|saw|tried)\b"],
        # the EXPOSE lead: enter an unaware market through a shocking
        # revelation about an unseen danger/scam ("zero FDA regulation",
        # "nobody checks what's on the label")
        [r"\bzero (fda |epa )?(regulation|oversight)\b",
         r"\bno (regulation|oversight)\b", r"\bunregulated\b",
         r"\bnobody (checks|tests|verifies|approves)\b",
         r"\b(no one|nobody) (is )?(checking|testing|watching)\b"],
        [r"\bthe truth (about|is)\b",
         r"\bwhat'?s really (in|inside|behind)\b",
         r"\bonce you see (that|it)\b", r"\bask yourself\b",
         r"\bthey don'?t want you\b",
         r"\bknown for (decades|years)\b"]],
    "problem_aware": [
        [r"^[^.!\n]{0,80}\?"],                # question headline up front
        [r"\bsound familiar\b"], [r"\btired of\b"], [r"\bsick of\b"],
        [r"\bstruggling with\b"], [r"\bstubborn\b"], [r"\brefuses to\b"],
        [r"\bkeeps getting worse\b"],
        [r"\bnothing (you try|helps|works)\b"],
        [r"\byou('re| are) not (alone|imagining)\b"], [r"\bfed up\b"],
        # NARRATIVE/UGC symptom-decline framing — the problem shown as lived
        # exhaustion/depletion rather than a DR headline ("you're not tired,
        # you're depleted", "running on nothing", "your body's screaming for")
        [r"\b(depleted|drained|burned ?out|worn ?out|wiped out|run ?down)\b",
         r"\brunning on (nothing|empty|fumes)\b", r"\bno energy\b",
         r"\bcan'?t keep up\b",
         r"\byour body('?s| is| has been)? (screaming|crying|begging|"
         r"starving|aching) for\b",
         r"\byou'?re not \w+[.,]?\s+you'?re\b"],  # "you're not X, you're Y"
        # narrative reassurance that NAMES a felt problem ("you're not broken /
        # crazy / lazy / weak / imagining it") — UGC/story problem-agitation
        [r"\byou'?re not (broken|crazy|lazy|weak|failing|imagining|alone)\b",
         r"\bit'?s not your fault\b", r"\bnothing is wrong with you\b"],
        # the long-suffered problem: searching/desperate for relief, or a
        # symptom held too long ("tension locked too tight", "knots for years")
        # THE FAILED-SOLUTIONS LITANY — the definitive problem-aware beat: the
        # reader already knows the problem and has bought things that did not
        # fix it ("Bought a mattress. Didn't help. Maybe new pillows. I tried
        # that. Nothing."). Without this the hip-pain testimonial scored ZERO
        # problem_aware clusters and lost the stage to one generic tail phrase.
        [r"\b(did\s?n'?t|does\s?n'?t|would\s?n'?t|never) (help|work|change)",
         r"\bnothing (helped|worked|changed|made a difference)\b",
         r"\btried (everything|it all|them all|that|so many)\b",
         r"\bstill (aching|hurting|there|no better|the same)\b",
         r"\bwaste of (money|time)\b"],
        [r"\b(been )?searching for\b.{0,30}\b(relief|answer|cure|fix|help)\b",
         r"\b(relief|answer|cure|fix)\b.{0,30}\b(you'?ve|they'?ve)? ?"
         r"(been )?searching\b", r"\bdesperate for\b.{0,20}\b(relief|answer)\b",
         r"\b(tension|knots?|tightness|stiffness|soreness|pain|ache)\b"
         r".{0,40}\b(locked|stuck|trapped|too tight|won'?t (let go|release|"
         r"budge)|for (years|months|so long)|all day)\b"]],
    "solution_aware": [
        [r"\bunlike (other|anything)\b"], [r"\bhow .{0,40} works\b"],
        # the REFRAME — reveals the category of SOLUTION ("it's not candy,
        # it's a supplement", "that's not a diet, it's a system")
        [r"\b(it'?s|they'?re|that'?s) not (?:just )?(?:a |an )?\w+[.,!]?\s+"
         r"(?:it'?s|they'?re|that'?s) (?:a|an|actually|really)\b"],
        # 'process' removed: in a UGC story "the process" is ordinary narration
        # ("that's just part of the process"), not a mechanism reveal. On a
        # menopause/hip-pain testimonial it was the ONLY solution_aware hit in
        # the whole ad and it flipped the read to solution-aware.
        [r"\bthe (method|mechanism|formula|enzyme)\b"],
        [r"\bbetter way\b"], [r"\bworks (with|by|differently)\b"],
        [r"\bwhy \w+ beats?\b"], [r"\bcompare\b"],
        [r"\bthe difference\b"], [r"\bforget \w+ing\b"],
        [r"\bafter \d+ (hours?|days?|weeks?)\b"],
        [r"\bwhat happens (inside|when|to)\b"],
        [r"\b(at|from) the root\b"], [r"\bthe science (behind|of)\b"],
        [r"\bcompounds?\b"],
        # PRODUCT-AS-SOLUTION reveal (one element): "introducing X", "meet X",
        # and the UGC "turns out you just needed X / all you needed was X" beat
        # that names the product as the answer to the problem.
        [r"\bintroduc(e|ing)\b", r"\bmeet \w+\b", r"\bsay hello to\b",
         r"\bnew from\b", r"\bthat'?s (where|why) \w+ comes? in\b",
         r"\byou (?:just |really |only )?(?:ever )?needed\b",
         r"\ball you (?:ever )?need(?:ed)?(?: is| was)?\b"],
        # PHYSICAL / DEVICE mechanism acting on the body — the solution-aware
        # move for massagers, microcurrent / red-light / gua-sha tools and
        # topicals: describing HOW it works on the tissue (releases / melts /
        # breaks up the problem), not a supplement's biochemical mechanism.
        [r"\b(?:releas\w+|loosen\w+|melt\w+|dissolv\w+|scrap\w+|break\w+ up|"
         r"unlock\w+|flush\w+|soothe?s?|relax\w+)\b.{0,30}\b(?:tension|knots?|"
         r"tightness|muscles?|soreness|stiffness|buildup|toxins?|fascia)\b",
         r"\bmicro.?currents?\b", r"\bred.?light therapy\b",
         r"\bdeep tissue\b", r"\bpercussi\w+\b"],
        # the MECHANISM acting on the problem's root/balance — the core
        # solution-aware move ("this probiotic RESTORES your pH BALANCE",
        # "rebuilds the gut LINING", "targets the root CAUSE")
        [r"\b(restores?|rebalances?|replenish(es)?|repairs?|rebuilds?|"
         r"neutraliz(e|es)|targets?|attacks?|tackles?|resets?|optimiz(e|es))\b"
         r".{0,30}\b(root|cause|balance|ph|gut|level|levels|barrier|lining|"
         r"flora|microbiome|hormones?|collagen|circulation)\b"],
        # PROOF-OF-MECHANISM (one element): clinical / formulation language
        [r"\bclinically (proven|tested|shown|studied|dosed)\b",
         r"\bbacked by (science|research|studies|data)\b",
         r"\bdoctor.?(formulated|developed|recommended|approved)\b",
         r"\bformulated (to|with|for)\b", r"\bpowered by\b",
         r"\b(active|key|hero) ingredient\b", r"\bclinical(ly)? dose\b"]],
    "product_aware": [
        # ONE risk-reversal element, however it is phrased. The N-day
        # pattern requires offer context: "730 days" of garlic aging is
        # process language, not a trial offer.
        [r"\bmoney.?back\b", r"\bguarantee\b", r"\brisk.?free\b",
         r"\bfree trial\b", r"\brefund\b",
         r"\b\d+.day\b.{0,24}(guarantee|trial|money|return|refund)"],
        # ONE social-proof element, however it is phrased
        [r"\brated [\d.]+\b", r"\breviews?\b", r"\btestimonials?\b",
         r"\bbefore and after\b", r"\bloved by [\d,]+\b",
         r"\bjoin thousands\b", r"\bas seen in\b"]],
    "most_aware": [
        # restock element
        [r"\bback in stock\b", r"\brestock\b",
         r"\byour favorite \w+ (returns|is back)\b"],
        # deal element
        [r"\bdiscount code\b", r"\bsale ends\b", r"\bcoupon\b",
         r"\bprice drop\b", r"\bsubscribe (and|&) save\b"],
        # member/reorder element
        [r"\bat checkout\b", r"\b(re)?order again\b",
         r"\byour size is\b", r"\bmembers? (only|price|get)\b"]],
}
_AW_COMPILED = {sid: [[re.compile(p, re.I) for p in cluster]
                      for cluster in clusters]
                for sid, clusters in AWARENESS_RULES.items()}
AW_BOOST_PER_HIT = 1.2
AW_BOOST_CAP = 4.8
# Awareness decisions weigh structural rules over motif z: the motif
# channel confuses an ad's SUBJECT MATTER with its TARGETING (an expose
# about supplement ingredients reads lexically like product_aware), and
# every real-ad misfire this far traced to z, not rules.
AW_Z_WEIGHT = 0.5

_NARRATIVE = re.compile(r"\b(i|my|we|she|he|her|his|him)\b", re.I)
_DIRECT = re.compile(r"\b(you|your)\b", re.I)


AWARENESS_ORDER = ["unaware", "problem_aware", "solution_aware",
                   "product_aware", "most_aware"]


def awareness_distribution(scorer, z, boosts):
    """Weight across ALL FIVE Schwartz stages (not just the top 2): an ad
    can span several stages — open unaware, build through solution, close
    most-aware — and the spread should show that. Stages above ~15% are
    'present'; this surfaces multi-stage ads instead of flattening them
    to one label."""
    comb = z[:44] + z[44:]
    idx = {c["id"]: i for i, c in enumerate(scorer.cats)
           if c["family"] == "awareness"}
    raw = {s: AW_Z_WEIGHT * float(comb[idx[s]]) + boosts.get(s, 0.0)
           for s in AWARENESS_ORDER}
    lo = min(raw.values())
    shifted = {s: raw[s] - lo for s in AWARENESS_ORDER}
    tot = sum(shifted.values()) or 1.0
    return [(s, shifted[s] / tot) for s in AWARENESS_ORDER]


def awareness_boost(text):
    """Per-stage additive boost from distinct ELEMENT hits (clusters),
    POSITION-WEIGHTED: the lead (first 60%) decides who an ad targets —
    Schwartz's rule that the headline meets the prospect where he stands
    — while scarcity/guarantee language in the close is standard
    furniture on every direct-response ad, so tail hits count at 40%.

    STORY-LEAD detector: unaware markets are entered through story and
    identification, and story leads are written in narrative voice. A
    lead whose narrative pronouns swamp its 'you/your' count gets an
    unaware boost worth two elements."""
    cut = max(int(len(text) * 0.6), 200)
    lead, tail = text[:cut], text[cut:]
    boosts = {}
    for sid, clusters in _AW_COMPILED.items():
        score = 0.0
        for cluster in clusters:
            if any(p.search(lead) for p in cluster):
                score += AW_BOOST_PER_HIT
            elif any(p.search(tail) for p in cluster):
                score += 0.4 * AW_BOOST_PER_HIT
        boosts[sid] = min(score, AW_BOOST_CAP)

    nar = len(_NARRATIVE.findall(lead))
    direct = len(_DIRECT.findall(lead))
    if nar >= 8 and nar > 3 * direct:
        boosts["unaware"] = min(boosts["unaware"] + 2 * AW_BOOST_PER_HIT,
                                AW_BOOST_CAP)
    return boosts


def family_top(scorer, z, family, k=2, boosts=None):
    """Top categories within a family. Scores are reported RELATIVE to
    the family mean (positive = above this text's family average), so a
    text that is globally unusual doesn't print a wall of minuses.
    When boosts are given (awareness family), motif z is attenuated by
    AW_Z_WEIGHT — rules carry that decision."""
    comb = z[:44] + z[44:]
    zw = AW_Z_WEIGHT if boosts else 1.0
    idx = [(i, zw * float(comb[i]) +
            (boosts or {}).get(scorer.cats[i]["id"], 0.0))
           for i, c in enumerate(scorer.cats) if c["family"] == family]
    fam_mean = sum(v for _, v in idx) / len(idx)
    idx = [(i, v - fam_mean) for i, v in idx]
    idx.sort(key=lambda t: -t[1])
    return [(scorer.cats[i]["id"], v) for i, v in idx[:k]]


# desires come from the MOTIVATIONAL families only. awareness and
# sophistication categories describe copy STRATEGY (what stage the ad
# speaks to), and their linked tags (clarity/control/insight) were
# flooding the desire profile of every mechanism/expose ad — the
# emotional read belongs to the chakra/maslow/sephirot/archetype
# structure the prime model was trained around.
DESIRE_FAMILIES = ("chakra", "maslow", "sephira", "archetype")


# ---- Problem (pbm) grounding for PV = (Problem x Desire) x T --------------
# Per the redefinition, Intensity = Problem x Desire, and PROBLEM draws on
# the psychological hierarchy the model was trained around: LOWER, more
# primal needs (survival, sex, safety) are more emotionally intense than
# higher ones (esteem, transcendence). An ad that hits root/sacral chakra
# or physiological/safety Maslow therefore gets a higher problem_gain,
# raising Intensity and PV — the user's "lower levels like sex are stronger"
# logic, grounded in the existing taxonomy rather than a new free parameter.
PRIMALITY = {
    "physiological": 1.00, "safety": 0.85, "belonging": 0.60,
    "esteem": 0.45, "self_actualization": 0.30,
    "root_chakra": 1.00, "sacral_chakra": 0.95, "solar_plexus_chakra": 0.70,
    "heart_chakra": 0.55, "throat_chakra": 0.40, "third_eye_chakra": 0.30,
    "crown_chakra": 0.20,
}


def problem_grounding(scorer, z):
    """-> (gain, top_primal_category, primality). gain multiplies the
    Problem head: ~1.6x for survival/sex-led ads, ~0.8x for transcendent
    ones. primality = activation-weighted mean of PRIMALITY over the
    Maslow+chakra categories (weights = positive-shifted z activation)."""
    comb = z[:44] + z[44:]
    items = [(i, c) for i, c in enumerate(scorer.cats)
             if c["family"] in ("maslow", "chakra")]
    vals = [float(comb[i]) for i, _ in items]
    lo = min(vals) if vals else 0.0
    tot = acc = 0.0
    top, topw = None, -1e9
    for (i, c), v in zip(items, vals):
        w = v - lo
        acc += w * PRIMALITY.get(c["id"], 0.5)
        tot += w
        if w > topw:
            topw, top = w, c["id"]
    primality = acc / tot if tot > 0 else 0.5
    return 0.6 + primality, top, primality            # gain ~[0.8, 1.6]


def desire_profile(scorer, z, k=6):
    comb = z[:44] + z[44:]
    keep = [i for i, c in enumerate(scorer.cats)
            if c["family"] in DESIRE_FAMILIES]
    # always rank RELATIVE to the weakest desire-family category (shift so
    # min = 0). clamp(min=0) used to zero every category below the
    # short-ad-calibrated mean, so a huge VSL transcript — whose features
    # all sit below that mean — left a single survivor and collapsed to
    # "love 50% / abundance 50%". Relative shift preserves the full
    # ranking and never collapses, while top desires still dominate.
    sub = comb[keep]
    sub = sub - sub.min()
    w = {}
    for j, i in enumerate(keep):
        for tag in scorer.cats[i]["linked_desires"]:
            w[tag] = w.get(tag, 0.0) + float(sub[j])
    top = sorted(w.items(), key=lambda t: -t[1])[:k]
    total = sum(v for _, v in top) or 1.0
    return [(t, v / total) for t, v in top]


def implied_avatar_z(scorer, desires):
    """Synthesize an avatar motif z-vector from channeled desire tags."""
    w = dict(desires)
    z = torch.zeros(88)
    for i, c in enumerate(scorer.cats):
        score = sum(w.get(t, 0.0) for t in c["linked_desires"])
        z[i] = z[44 + i] = score
    z = z / (z.std() + 1e-8)
    return z.clamp(-2.5, 2.5)


# ---- vision brief -> inference inputs (the v4_distill override) ------------
def _is_face_avatar(brief):
    """True when the brief's avatar gender/age came from a FACE CLASSIFIER
    on the creative (the local, no-Gemini path), not a reasoned read of who
    the ad TARGETS. _local_brief sets `_local` and stamps `n_people` on the
    avatar when insightface fired — either flag means 'this is the person
    SHOWN, not necessarily the buyer'."""
    av = brief.get("avatar", {}) or {}
    return bool(brief.get("_local")) or bool(av.get("n_people"))


def _vision_prior_from_brief(brief):
    """Turn the Gemini brief into a demographics vision_prior:
    {dim: {value: weight}}.

    A REASONED Gemini avatar (it judged who the ad targets) is strong
    evidence, so gender/age land at near-marker weight (2.0 / 1.5). But a
    FACE-CLASSIFIER avatar from the local path only tells us who APPEARS
    on screen — a young demonstrator is not the target of a gender-neutral
    muscle-relief device. So a face-derived read drops to a weak hint (0.5),
    BELOW the demographics confidence floor (1.0): on its own it abstains
    ('unclear') instead of asserting a buyer, but it still CORROBORATES —
    crossing the floor — when the copy markers or the sourced study prior
    point the same way. Accent->income is added by analyze() via the
    sourced regional-income table when present."""
    import v4_correlations
    prior = {"gender": {}, "age": {}, "life_stage": {}, "income": {}}
    av = brief.get("avatar", {}) or {}
    if _is_face_avatar(brief):
        # A face read = who's SHOWN, not always who buys. A SINGLE clear face of a
        # PLAUSIBLE-TARGET AGE (35+) in a DR health/pet creative is almost always
        # the sufferer/testimonial, so let its GENDER commit (above the 1.0 floor).
        # A YOUNG single face (<35) is usually a model/presenter/demonstrator (the
        # bioblade lesson: a 25-34 woman fronting a gender-neutral muscle device is
        # NOT the buyer) -> corroborate-only. AGE never fully commits from a face
        # (before/after shots skew it). Multiple/mixed faces stay corroborate-only.
        one_face = ((av.get("n_people") or 0) == 1
                    and av.get("gender") in ("female", "male"))
        target_age = str(av.get("age_range", "")) in ("35-44", "45-54", "55+")
        g_w, a_w = (1.2, 0.7) if (one_face and target_age) else (0.5, 0.5)
    else:
        g_w, a_w = 2.0, 1.5
    g = av.get("gender")
    if g in ("female", "male"):
        prior["gender"][g] = g_w
    for b in v4_correlations._parse_value(av.get("age_range", "")).get(
            "age", []):
        prior["age"][b] = prior["age"].get(b, 0.0) + a_w
    # STEP 2 of local avatar reasoning: fold in the scene-caption's OWN
    # marker evidence (v4_demographics.vision_markers — discounted, may
    # corroborate but not assert alone) so an unmistakable visual cue the
    # local VLM DESCRIBES ('a man with a beer belly') can reach the decision
    # the same way an insightface-read face does, not just OCR'd text.
    cap = (brief.get("scene_caption") or "") if isinstance(brief, dict) else ""
    if cap:
        for dim, vals in demo_mod.vision_markers(cap).items():
            for val, w in vals.items():
                prior.setdefault(dim, {})
                prior[dim][val] = max(prior[dim].get(val, 0.0), w)
    return prior


def _brief_copy_text(brief):
    """The REAL ad copy Gemini transcribed — on-screen text + spoken
    words — which is what the awareness regexes and motif scorer were
    built for, unlike the freeform [VISUAL] narration that broke them."""
    parts = [brief.get("onscreen_text", ""), brief.get("spoken_transcript", "")]
    return "\n".join(p for p in parts if p).strip()


def load_stack():
    """Load everything once: motif scorer + stats + PV engine."""
    blob = torch.load(ms.CACHE_PATH, weights_only=False)
    assert blob["vocab_hash"] == tok.vocab_hash()
    scorer = MotifScorer()
    # load_engine reads the head's OWN arch/feature record out of the ckpt, so
    # a legacy 353-dim head and an honest-feature head both load correctly and
    # inference cannot feed either one a vector it was not fit on.
    engine, ck = eng_mod.load_engine(eng_mod.CKPT)
    return dict(scorer=scorer, mean=blob["mean"], std=blob["std"],
                engine=engine, lam=ck["lambda"])


# --- abstention -------------------------------------------------------------
# The model had NO floor: fed the empty string it returned 'Prostate Cancer'
# at PV 2.05, and fed the 73 characters left over from a failed Ad Library
# capture ("[ON-SCREEN TEXT]\n[scene] outdoors; packaged product shot; a
# beauty device") it returned a full, confident decomposition. With almost no
# copy the KB match is noise, so whichever of the 1,270 rows happens to score
# highest wins -- which is how a human beauty ad surfaced glaucoma in dogs and
# pyometra in cats. Below the floor the honest answer is "I couldn't read this
# ad", not a diagnosis.
MIN_EVIDENCE_CHARS = int(os.environ.get("V4_MIN_EVIDENCE_CHARS", "40"))

# Our own section tags, not the ad's language.
_SCAFFOLD = re.compile(
    r"^\s*\[(?:ON-SCREEN TEXT|VISUAL|TRANSCRIPT|AD COPY|SPOKEN)[^\]]*\]\s*$",
    re.I | re.M)
# The [scene] line is the vision arm's CAPTION of the creative -- a
# description WE generated, not words the ad said. It IS real evidence when
# there is a real creative behind it (a caption reading "a shirtless
# middle-aged man with man boobs" correctly types the avatar with no copy at
# all), so it counts -- at the same 0.5 discount v4_demographics already
# applies to vision, which is what keeps a thin caption of a failed capture
# ("outdoors; packaged product shot; a beauty device") under the floor.
_CAPTION = re.compile(r"^\s*\[scene\](.*)$", re.I | re.M)
CAPTION_WEIGHT = 0.5


def evidence_chars(copy_text, brief=None):
    """Letters/digits of ad evidence available to decompose, with the vision
    caption discounted against the ad's own words.

    The caption reaches analyze by two routes: inline as a `[scene]` line in
    the text, and as the brief's own `scene_caption` key (the v4_distill
    route, where copy_text holds only OCR + transcript). Both count."""
    body = _SCAFFOLD.sub(" ", copy_text or "")
    caption = "".join(_CAPTION.findall(body))
    body = _CAPTION.sub(" ", body)
    if brief:
        caption += " " + (brief.get("scene_caption") or "")
    n = sum(1 for c in body if c.isalnum())
    n += int(CAPTION_WEIGHT * sum(1 for c in caption if c.isalnum()))
    return n


def _abstain(reason, copy_text, brief=None, market=None):
    """A result that asserts nothing. Same keys as a real analysis so the
    deck / API / REPL render it without special-casing."""
    return {
        "insufficient_evidence": True,
        "abstain_reason": reason,
        "evidence_chars": evidence_chars(copy_text, brief),
        "product": None, "visual_category": None, "presenter": None,
        "painpoints": [], "painpoint_angles": [],
        "age": "unclear", "gender": "unclear", "income": "unclear",
        "income_by_age": None, "life_stage": "unclear",
        "awareness_journey": [], "desires": [], "problem": "unclear",
        "market_type": None, "parent_category": None, "angle_mode": None,
        "dsr": None, "T": None, "gain": None, "top_primal": None,
        "PV": None, "winner_prob": None, "ethnicity": None,
        "awareness_spread": [], "sophistication_spread": [],
        "angles": [], "sophistication": [], "psych_center": [],
        "maslow_level": [], "condition_epidemiology": [], "mechanisms": [],
        "market": market,
    }


def _read_open_set(copy_text, reader, pain_names, angle_names):
    """Run the reasoning loop and let its answer lead.

    READ -> GROUND -> CHECK -> REVISE (v4_reason_loop). A read that fails its
    own critics, or names nothing, leaves the KB answer exactly as it was —
    the reader is allowed to improve this call, never to break it.
    """
    try:
        try:
            import v4_reason_loop as RL
        except ImportError:
            from communities_00_09.community_9_v4_admix_v4_categorize_batch \
                import v4_reason_loop as RL
        got = RL.decode(copy_text, reader, max_rounds=1)
        name = (got.get("painpoint") or "").strip()
        if not name or not got.get("passed"):
            return pain_names, angle_names
        g = got.get("grounding") or {}
        # A grounded read gets the KB's canonical label (so the mechanism,
        # study prior and epidemiology all key off it); an ungrounded one
        # keeps the buyer's own words and simply carries no mechanism.
        lead = g.get("display") or g.get("kb_name") or name
        angle = (got.get("angle") or "").strip()
        # REPLACE the KB tail, do not lead it. Prepending kept the closed
        # matcher's list behind the reader's answer, so a CAT ad came back
        # "Bilious Vomiting Syndrome in Cats | Gut Health & Digestion | Upset
        # Stomach in Dogs" — the dog row still there, just demoted. The tail is
        # produced by exactly the vocabulary this layer exists to supersede,
        # and critic_singular already holds the reader to ONE problem, so an
        # answer that passed its critics stands alone.
        return [lead], ([angle] + [n for n in angle_names if n != angle])[:6] \
            if angle else angle_names
    except Exception:  # noqa: BLE001 — a reader must never fail an analysis
        return pain_names, angle_names


def analyze(text, stack, brief=None, return_result=False, reader=None):
    scorer, mean, std = stack["scorer"], stack["mean"], stack["std"]
    engine, lam = stack["engine"], stack["lam"]
    z = lambda t: (scorer.score(t) - mean) / std

    if brief is not None:
        # the v4_distill route: score the REAL copy Gemini transcribed
        # (on-screen + spoken), not the [VISUAL] narration that starved
        # the awareness/desire heuristics and produced "unaware 53%".
        avatar_text, copy_text = "", (_brief_copy_text(brief) or text)
    else:
        avatar_text, copy_text = split_roles(text)

    # Grounded canonical conditions from a distilled brief are hard evidence
    # regardless of how little raw copy came with them -- never abstain there.
    _grounded = bool(brief and (brief.get("canonical_conditions")
                                or brief.get("painpoints")))
    _ev = evidence_chars(copy_text, brief)
    if _ev < MIN_EVIDENCE_CHARS and not _grounded:
        res = _abstain(
            f"only {_ev} characters of readable ad language were recovered "
            f"(floor is {MIN_EVIDENCE_CHARS}) — nothing here identifies a "
            f"buyer or a problem, so no decomposition is reported",
            copy_text, brief)
        if return_result:
            return res
        print(f"Insufficient evidence — {res['abstain_reason']}.")
        return None


    # Non-English market: TRANSLATE to English before the English-trained
    # motif scorer / KB cue regex / demographic markers read the copy, so a
    # foreign-language lander decomposes accurately (awareness/archetype/
    # desires/painpoints) instead of producing noise. Detect on the ORIGINAL;
    # keep `market` for the income line (which must use that country's figure,
    # not the US ACS table). Local + offline (argos/CTranslate2, no API).
    market, translated = None, False
    try:
        import v4_stats
        market = v4_stats.detect_market(copy_text)
        if market:
            import v4_translate
            _eng = v4_translate.to_english(copy_text, region=market)
            if _eng:
                copy_text = _eng
                translated = True
    except Exception:  # noqa: BLE001
        market = None
    zg = z(copy_text)

    # AWARENESS arc lives in the spoken NARRATIVE (VSL transcript) or page copy
    # — NOT the OCR on-screen FURNITURE (prices, "ORDER NOW", ratings,
    # "guarantee", "subscribe & save") which is product/most-aware boilerplate
    # on every DR ad. When vision/OCR was added, that furniture got prepended
    # into copy_text and, landing in the position-weighted awareness LEAD, it
    # swamped the real problem/solution arc (the exact recognition regression).
    # So recognize awareness from the narrative when it's substantial; statics /
    # pasted text have no transcript and fall back to the full copy.
    narrative = ((brief.get("spoken_transcript") or "").strip()
                 if brief is not None else "")
    if narrative and translated:
        try:
            import v4_translate
            _n = v4_translate.to_english(narrative, region=market)
            if _n:
                narrative = _n
        except Exception:  # noqa: BLE001
            pass
    aware_src = narrative if len(narrative) >= 200 else copy_text
    zg_aware = zg if aware_src is copy_text else z(aware_src)

    # LOW-SIGNAL detector via the COLLAPSE signature: when the embedding
    # channel washes to the generic mean (a long non-ad — a marketing
    # lecture, a meeting transcript), the desire read collapses onto the
    # "no signal" attractor where only one category (chesed -> love +
    # abundance) survives and the 3rd-ranked desire is ~0. That exact
    # shape is the tell, regardless of why it happened.
    # LOW-SIGNAL detector via the COLLAPSE signature: when the embedding
    # channel washes to the generic mean (a long non-ad — a marketing
    # lecture, a meeting transcript), the desire read collapses onto the
    # "no signal" attractor where only one category (chesed -> love +
    # abundance) survives and the 3rd-ranked desire is ~0. That exact
    # shape is the tell, regardless of why it happened.
    desires = desire_profile(scorer, zg)
    low_signal = len(desires) >= 3 and desires[2][1] < 0.01
    # DESIRE DERIVES FROM THE PROBLEM (BERNAY: "Desire is a derivative of a
    # problem"): read the ad's desires off its DETECTED painpoint(s) via the KB,
    # deterministically — not the motif cosine (which collapsed every ad to
    # connection/comfort/love) and not a learned classifier (no real desire labels
    # exist; the synthetic avatar universe is off-domain). Falls back to the motif
    # profile only when no painpoint is detected. low_signal keeps the cosine shape.
    try:
        import v4_correlations as _corr          # imported locally later in analyze
        _dz = _corr.desires_for(_corr.match_painpoints(copy_text))
        if _dz:
            desires = _dz
    except Exception:  # noqa: BLE001
        pass

    # Style & Color Configuration
    try:  # Enable ANSI on legacy Windows consoles
        import ctypes
        h = ctypes.windll.kernel32
        h.SetConsoleMode(h.GetStdHandle(-11), 7)
    except Exception:
        pass

    def fg(r, g, b):
        return f"\033[38;2;{r};{g};{b}m"

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    C_HEADER = fg(79, 124, 201)    # Soft blue
    C_LABEL = fg(180, 180, 180)     # Soft white/gray
    C_HIGHLIGHT = fg(240, 240, 240) # Elegant bold silver/white (no orange)
    C_GREEN = fg(240, 240, 240)     # Elegant bold silver/white (no green)
    C_WARNING = fg(140, 140, 140)   # Muted silver/gray (no red)
    C_BORDER = fg(100, 100, 120)    # Muted slate border

    STAGE_COLORS = {
        "unaware": (216, 57, 47),   # Red
        "problem": (232, 99, 42),   # Dark Orange
        "solution": (244, 165, 42),  # Orange/Yellow
        "product": (63, 165, 82),   # Green
        "most": (79, 124, 201),     # Blue
    }

    print(f"\n{C_BORDER}╔══════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{C_BORDER}║{RESET}                     {BOLD}{C_HEADER}AD MIX DECOMPOSITION{RESET}                     {C_BORDER}║{RESET}")
    print(f"{C_BORDER}╚══════════════════════════════════════════════════════════════╝{RESET}")

    if low_signal and brief is None:
        print(f"{C_WARNING}{BOLD}⚠️  LOW AD SIGNAL — the desire read collapsed{RESET}")
        print(f"   This text isn't ad copy the engine can resolve (likely a lecture/transcript")
        print(f"   ABOUT ads, not an ad). Treat the decomposition as low-confidence.\n")
    elif low_signal:
        print(f"{C_HIGHLIGHT}{BOLD}ℹ️  long-form copy — motif desire read is low-confidence here{RESET}")
        print(f"   The [vision] desires below are the authoritative read.\n")

    if avatar_text:
        print(f"{DIM}Detected AVATAR text ({len(avatar_text)} chars):{RESET} {C_LABEL}{avatar_text[:70]}...{RESET}")
    print(f"{DIM}Detected COPY text ({len(copy_text)} chars):{RESET} {C_LABEL}{copy_text[:70]}...{RESET}\n")

    aw_boosts = awareness_boost(aware_src)
    for family, label in [("archetype", "ANGLE (archetype)"),
                          ("sophistication", "SOPHISTICATION level"),
                          ("chakra", "PSYCH center"),
                          ("maslow", "MASLOW level")]:
        tops = family_top(scorer, zg, family)
        styled_parts = []
        for rank, (n, v) in enumerate(tops):
            score_color = C_GREEN if v >= 0 else C_WARNING
            score_str = f"{score_color}{v:+.2f}{RESET}"
            if rank == 0:
                styled_parts.append(f"{C_HIGHLIGHT}{BOLD}{n}{RESET} ({score_str})")
            else:
                styled_parts.append(f"{DIM}{n}{RESET} ({score_str})")
        line = ", ".join(styled_parts)
        print(f"{BOLD}{C_LABEL}{label:<22}{RESET} {line}")

    # awareness as a full 5-stage spread — ads often span several stages.
    # Use the NARRATIVE motif (zg_aware), not the OCR-furniture-polluted copy.
    dist = awareness_distribution(scorer, zg_aware, aw_boosts)
    # LEARNED read (the model's OWN taxonomy-grounded recognition head, k-fold
    # CV ~93% vs 23% baseline) decides the DOMINANT stage; it's blended in as a
    # strong prior so the model — not the hand-written rules — drives the call.
    # The head is single-stage-peaked, so the rule/motif spread still supplies
    # the secondary stages of a multi-stage arc (see `present` below).
    learned_aw = None
    try:
        import v4_recognition_head as _rec
        learned_aw = _rec.predict_awareness(aware_src)
    except Exception:  # noqa: BLE001
        learned_aw = None
    if learned_aw:
        a = 0.55
        share = dict(dist)
        blended = {s: a * learned_aw.get(s, 0.0) + (1 - a) * share.get(s, 0.0)
                   for s in AWARENESS_ORDER}
        tot = sum(blended.values()) or 1.0
        dist = [(s, blended[s] / tot) for s in AWARENESS_ORDER]
        learned_top = max(learned_aw, key=learned_aw.get)
    else:
        learned_top = None

    spread_parts = []
    bar_segments = []
    total_bar_width = 20
    for s, p in dist:
        name = s.split('_')[0]
        rgb = STAGE_COLORS.get(name, (150, 150, 150))
        color_code = fg(*rgb)
        spread_parts.append(f"{color_code}{name} {100 * p:.0f}%{RESET}")
        
        blocks = int(p * total_bar_width + 0.5)
        if blocks > 0:
            bar_segments.append(f"{color_code}{'█' * blocks}{RESET}")
            
    spread = "  ".join(spread_parts)
    bar_visual = "".join(bar_segments)
    if bar_visual:
        bar_visual = f" {DIM}[{RESET}{bar_visual}{DIM}]{RESET}"

    # "present" is the targeting CLAIM (which stages the ad actually plays to),
    # kept distinct from the bar, which is a zero-sum motif texture: one strong
    # boost can crowd another genuine stage's share toward zero. A stage counts
    # as present if it holds a real share (>=15%) OR a structural rule element
    # fired for it (>=0.48) OR it is the LEARNED head's dominant call (the
    # model's own 93% read is always claimed). Listed in Schwartz funnel order
    # so it reads as the ad's arc, not by magnitude.
    _share = dict(dist)
    present = [s for s in AWARENESS_ORDER
               if _share.get(s, 0.0) >= 0.15 or aw_boosts.get(s, 0.0) >= 0.48
               or s == learned_top]
    # …but the funnel is an ORDERED LADDER, and those three tests fire per
    # stage independently, so the claim could come out with a HOLE in it —
    # "Solution Aware" and "Most Aware" lit with "Product Aware" dark between
    # them. That reads as a model that skipped a rung, and it isn't a claim
    # anyone would make: you cannot address a reader who already knows your
    # product without passing the reader who knows solutions exist. Fill the
    # span between the lowest and highest stage that fired, so the arc is
    # contiguous. The SPREAD above is untouched — that stays the raw texture,
    # holes and all, so the evidence behind the claim is still visible.
    if len(present) > 1:
        _idx = [AWARENESS_ORDER.index(s) for s in present]
        present = AWARENESS_ORDER[min(_idx):max(_idx) + 1]
    print(f"{BOLD}{C_LABEL}{'AWARENESS spread':<22}{RESET} {spread}{bar_visual}  {DIM}(motif texture){RESET}")
    if brief is not None and brief.get("awareness_journey"):
        journey = [e["stage"] for e in brief["awareness_journey"]]
        print(f"{BOLD}{C_LABEL}{'  stages present':<22}{RESET} {BOLD}{C_GREEN}" + " ➔ ".join(journey) + f"{RESET}  {DIM}[vision]{RESET}")
    else:
        styled_present = []
        for p_stage in present:
            name = p_stage.split('_')[0]
            rgb = STAGE_COLORS.get(name, (150, 150, 150))
            styled_present.append(f"{fg(*rgb)}{BOLD}{name}{RESET}")
        print(f"{BOLD}{C_LABEL}{'  stages present':<22}{RESET} "
              + (", ".join(styled_present) if styled_present else f"{DIM}(none dominant){RESET}"))
    if brief is not None and brief.get("selling_stages"):
        beats = [e["stage"] for e in brief["selling_stages"]]
        print(f"{BOLD}{C_LABEL}{'SELLING beats':<22}{RESET} {C_HIGHLIGHT}" + " ➔ ".join(beats) + f"{RESET}  {DIM}[vision]{RESET}")

    styled_desires = []
    for rank, (t, v) in enumerate(desires):
        if rank == 0:
            styled_desires.append(f"{C_HIGHLIGHT}{BOLD}{t} {100 * v:.0f}%{RESET}")
        elif rank < 3:
            styled_desires.append(f"{C_LABEL}{t} {100 * v:.0f}%{RESET}")
        else:
            styled_desires.append(f"{DIM}{t} {100 * v:.0f}%{RESET}")

    print(f"{BOLD}{C_LABEL}{'DESIRES channeled':<22}{RESET} "
          + ", ".join(styled_desires)
          + (f"  {DIM}(motif texture){RESET}" if brief is not None else ""))
    if brief is not None and brief.get("core_desires"):
        print(f"{BOLD}{C_LABEL}{'  desires':<22}{RESET} {C_HIGHLIGHT}"
              + ", ".join(brief["core_desires"][:6]) + f"{RESET}  {DIM}[vision]{RESET}")

    if avatar_text:
        za = z(avatar_text)
        src = f"{C_HIGHLIGHT}detected avatar text{RESET}"
    else:
        za = implied_avatar_z(scorer, desires)
        src = f"{DIM}IMPLIED avatar (derived from the copy's desire profile){RESET}"
    comb = za[:44] + za[44:]
    person = [i for i, c in enumerate(scorer.cats)
              if c["family"] in ("chakra", "maslow", "sephira")]
    av_tops = sorted(person, key=lambda i: -float(comb[i]))[:4]
    print(f"\n{BOLD}{C_LABEL}{'TARGET avatar':<22}{RESET} {src}")
    
    profile_items = []
    for rank, i in enumerate(av_tops):
        item_id = scorer.cats[i]["id"]
        if rank == 0:
            profile_items.append(f"{C_HIGHLIGHT}{BOLD}{item_id}{RESET}")
        else:
            profile_items.append(f"{C_LABEL}{item_id}{RESET}")
    print(f"{'':<22}{BOLD}profile:{RESET} " + ", ".join(profile_items))

    if brief is not None:
        av = brief.get("avatar", {}) or {}
        ar = brief.get("accent_region", {}) or {}
        cta = brief.get("cta", {}) or {}
        bits = [x for x in (av.get("gender"), av.get("age_range"),
                            av.get("role")) if x and x != "unclear"]
        if av.get("relationship_context"):
            bits.append(av["relationship_context"])
        # A face-derived (local) avatar is the PRESENTER, not a reasoned read of
        # the buyer — label it so, matching the AUDIENCE 'presenter' line, so
        # this doesn't contradict the abstained gender/age below.
        buyer_label = ("  [on-screen] shown" if _is_face_avatar(brief)
                       else "  [vision] buyer")
        if bits:
            print(f"{BOLD}{C_LABEL}{buyer_label:<22}{RESET} " + "; ".join(bits))
        if cta.get("occasion"):
            occ_line = cta["occasion"]
            try:
                import v4_stats
                o = v4_stats.occasion_audience(cta["occasion"])
                if o:
                    occ_line += (f"  {DIM}(~${o['avg_per_person']}/person, "
                                 f"{o['total']} total [{o['source_name']}]){RESET}")
            except Exception:  # noqa: BLE001
                pass
            print(f"{BOLD}{C_LABEL}{'  [vision] occasion':<22}{RESET} {occ_line}")
        if (ar.get("likely_region")
                and ar.get("confidence") not in (None, "none")):
            line = (f"{ar.get('accent', '')} -> {ar['likely_region']} "
                    f"({ar['confidence']})")
            try:                                   # sourced regional income
                import v4_stats
                inc = v4_stats.region_income(ar["likely_region"])
                if inc:
                    line += (f"; {DIM}{inc['measure']} {inc['display']} "
                             f"[{inc['source_name']}]{RESET}")
            except Exception:  # noqa: BLE001
                pass            # v4_stats arrives in Phase 1
            print(f"{BOLD}{C_LABEL}{'  [vision] voice':<22}{RESET} {line}")

    import v4_correlations
    # When the vision model gave us a brief, TRUST ITS UNDERSTANDING: it read
    # the painpoints straight off the creative, and `canonical_conditions` is
    # its OWN mapping of the ad onto the cited KB conditions it judged
    # genuinely present. We do NOT keyword-scan the raw copy in that case, so
    # a stray word ("worms LIVING inside you", "SUGAR cravings") can't inject
    # an unrelated condition and its demographic prior. Keyword matching stays
    # for the no-brief path — a pasted text ad with no creative to look at.
    # A LOCAL OCR brief (brief['_local']) has copy but NO structured vision
    # understanding, so it must be keyword-scanned like pasted text, not
    # trusted for its (empty) painpoints/conditions.
    weak_kb_match = False    # overwritten below in the _local/no-brief branch
    if brief is not None and not brief.get("_local"):
        brief_pains = brief.get("painpoints", []) or []
        # None (not []) when absent, so an empty list falls back to
        # canonicalizing the free-text painpoints rather than zeroing the prior.
        cond_ids = brief.get("canonical_conditions") or None
        # grounded canonical KB names lead (they carry the study-prior and
        # must survive the pain_names[:6] truncation); verbose free-text
        # brief painpoints follow.
        if cond_ids:
            prior = v4_correlations.prior_for_ids(cond_ids)
            pain_names = list(dict.fromkeys(
                list(prior.get("names", [])) + list(brief_pains)))
        else:
            # Gemini gave free-text painpoints but no canonical ids: map them
            # (and the copy) onto the KB so the canonical painpoint name + its
            # study-prior still ground age/gender ('Joint pain' -> 'Joint Pain
            # & Arthritis' -> older).
            matched = v4_correlations.match_painpoints(
                copy_text, extra_painpoints=brief_pains)
            pain_names = list(dict.fromkeys(
                [p["name"] for p, _ in matched[:5]] + list(brief_pains)))
            # see the matching tie-break comment in the _local branch below
            if (len(matched) > 1 and matched[0][1] == matched[1][1]
                    and not brief_pains):
                tie_prob = v4_correlations.extract_problem(copy_text)
                if tie_prob and tie_prob not in pain_names:
                    pain_names = [tie_prob] + pain_names
            prior = v4_correlations.demographic_prior(
                copy_text, extra_painpoints=brief_pains)
    else:
        cond_ids = None
        # Florence's SCENE CAPTION ("a bottle of prostate & bladder formula, an
        # older man") grounds painpoint/problem matching — what the model SAW,
        # not just OCR/transcript. Folded in HERE (not the awareness narrative)
        # so a description can't distort the targeting read.
        ground_text = (copy_text + "\n"
                       + ((brief or {}).get("scene_caption") or "")).strip()
        pains = v4_correlations.match_painpoints(ground_text)
        pain_names = [p["name"] for p, _ in pains[:5]]
        # A TIED top score (2+ painpoints matched equally) is genuinely
        # ambiguous evidence in long-form copy, not a vote for whichever
        # condition happens to be declared earlier in the KB JSON -- which
        # is what silently decided it before this check existed. Caught on
        # real gethookd-board ads: a tinnitus ad and a glaucoma ad each lost
        # a 1-1 tie to an incidental 'digestion'/'circulation' mention
        # purely because that painpoint sits earlier in v4_correlations.json.
        # On a tie, prefer the ad's own literal stated subject over an
        # arbitrary KB-order pick.
        #
        # A single, UN-tied hit is ALSO thin evidence -- it only takes one
        # incidental word to fire, and that word doesn't have to come from
        # the ad's own copy: Florence's scene_caption (folded into
        # ground_text above) can hallucinate. Caught live: an eye-health
        # supplement's bottle text ("AREDS2+ daily eye formula") was
        # captioned by Florence as "...helps to reduce the appearance of
        # wrinkles" -- a plausible-sounding but wrong guess -- and the bare
        # word 'wrinkles' alone scored one single, un-tied KB hit for
        # 'Wrinkles & Skin Aging', while both the image AND text trained
        # classifiers were >99% confident this is 'eye vision'. Requiring
        # 2+ hits before trusting a KB match outright lets that stronger,
        # dedicated signal below correct a single spurious word.
        weak_kb_match = (not pains) or (len(pains) > 1
                                        and pains[0][1] == pains[1][1]) \
            or (pains and pains[0][1] < 2)
        if weak_kb_match and pains:
            tie_prob = v4_correlations.extract_problem(ground_text)
            if tie_prob and tie_prob not in pain_names:
                pain_names = [tie_prob] + pain_names
        prior = v4_correlations.demographic_prior(ground_text)
        # STEP 3 of local avatar reasoning: thread these ground_text-matched
        # conditions (copy AND scene_caption) through to the infer_demographics
        # call below via `painpoints=`. Previously computed here for DISPLAY
        # only — brief_pains stayed hardcoded None for every local/no-Gemini
        # media ad, so a condition the scene caption named (e.g. Florence
        # reading 'an older man' + 'prostate' on the creative) could name the
        # PROBLEM but its sourced epidemiological correlate could never reach
        # the gender/age decision. No new inference channel: just stops
        # discarding a KB match this same code already made.
        # DEMOGRAPHICS get only HARD (exact-cue) painpoints: a _soft semantic
        # rescue grounds the DESIRE (desire derives from the problem) but must not
        # commit age/gender. Its NAME, threaded through extra_painpoints, would
        # otherwise re-match as a hard hit and smuggle its study skew back in
        # (bioblade microcurrent: soft 'Anxiety & Chronic Stress' -> female/young).
        brief_pains = [p["name"] for p, _ in pains[:5]
                       if not p.get("_soft")] or None

    # OPEN-SET fallback: the KB is a closed set of ~25 SOURCED conditions, so
    # an ad outside it (e.g. vaginal/intimate health) matches nothing. Rather
    # than show "(none)", read the problem straight off the copy. A copy-derived
    # problem is NAMED only — it carries no study, so it adds no demographic
    # prior (we never fabricate age/gender for a condition we have no data on).
    copy_derived = False
    if not pain_names:
        prob = v4_correlations.extract_problem(
            (copy_text + "\n" + ((brief or {}).get("scene_caption") or "")).strip())
        if prob:
            pain_names = [prob]
            copy_derived = True
            # a narrow open-set phrase-hunt guess ('Tension', 'Diagnosis') is
            # weaker evidence than a genuine multi-hit KB match -- let the
            # trained visual/text classifier below still override it.
            weak_kb_match = True

    # NATIVE VISUAL read — a ROUTED read (v4_vision_head.predict_category_
    # routed), not a blend: consult the image classifier first, trust it
    # outright ONLY on the ~9 niches its own honest per-class CV proved
    # genuinely visually distinctive (RELIABLE_IMAGE_CLASSES); otherwise
    # trust the companion text classifier instead (81% overall vs image's
    # 60.8%, and specifically the stronger tool everywhere outside that
    # reliable set). This picks a SOURCE per prediction, never averages two
    # sources — averaging (predict_category_fused, tried 2026-07-18) was
    # caught live dragging a confidently-correct image read ('prostate',
    # 0.998) to a wrong answer because a confidently-WRONG text guess
    # ('supplement', 0.979 on uninformative generic copy) got 0.7 weight in
    # the average; a fixed blend can't tell a confident right answer from a
    # confident wrong one. Routing sidesteps that by never averaging.
    visual_cat = None
    visual_cat_label = None
    visual_cat_source = None
    _frames = (brief or {}).get("_frames") if brief is not None else None
    if _frames:
        try:
            import v4_vision_head as _vh
            vc, visual_cat_source = _vh.predict_category_routed(_frames, copy_text)
            if vc:
                visual_cat = max(vc, key=vc.get)
                reliable = (visual_cat_source == "image"
                           and visual_cat in _vh.RELIABLE_IMAGE_CLASSES)
                if visual_cat_source == "metaclip":
                    caveat = " (zero-shot generalizing read)"
                else:
                    caveat = "" if reliable else " (low-confidence visual read)"
                visual_cat_label = visual_cat + caveat
                src_desc = {
                    "image": "the creative image by the model's trained vision head",
                    "metaclip": "the creative image by MetaCLIP zero-shot "
                                "(generalizing fallback beneath the head)",
                    "text": "the ad's own copy by the companion text head",
                }.get(visual_cat_source, "the creative")
                if visual_cat_source == "metaclip":
                    note = (f"{DIM} — head was unsure; this is the MetaCLIP "
                            f"zero-shot fallback, not used to infer audience{RESET}")
                else:
                    note = ("" if reliable else
                            f"{DIM} — this category is a near-coin-flip read for "
                            f"the CLIP head, not used to infer audience{RESET}")
                print(f"\n{BOLD}{C_LABEL}{'[visual] category':<22}{RESET} "
                      f"{C_HIGHLIGHT}{BOLD}{visual_cat}{RESET} "
                      f"{DIM}({100*vc[visual_cat]:.0f}% — read from "
                      f"{src_desc}){RESET}{note}")
        except Exception:  # noqa: BLE001 — vision is best-effort
            visual_cat = None
            visual_cat_label = None
            visual_cat_source = None

    # TRAINED-CLASSIFIER override: a DEDICATED classifier trained only on the
    # 13 niches that actually map to a KB condition (v4_condition_head.py),
    # not the general 26-niche product-category classifier used for the
    # display line above. Dropping 'supplement' and 12 near-empty ingredient
    # niches measurably improved this specific decision (5-fold CV: image
    # 60.8%->71%, text 81%->91%) and fixed a real over-attractor -- the
    # general classifier's 'menopause' class had 72% recall but only 60%
    # PRECISION, so on its own "reliable" (recall-only) list it could
    # confidently misroute an eye-product image (caught live: 49% and 99.8%
    # confidence menopause reads on two real eye-supplement ad photos).
    # RELIABLE_IMAGE_CLASSES here requires recall AND precision >=50%.
    #
    # Was consulted ONLY when pain_names was completely empty, but by then
    # the open-set extract_problem() fallback above had almost always
    # already filled SOMETHING in (even a vague fragment like 'Tension'), so
    # this stronger signal sat computed and unused on most real ads. Now it
    # also overrides a WEAK regex read (a KB tie, or extract_problem's narrow
    # phrase guess) -- promoted to the front, not just filling an empty list.
    # Still does NOT feed demographic_prior (that stays on brief_pains/
    # copy_text), so this can't over-assert age/gender even when it fires.
    visual_derived = False
    if not pain_names or weak_kb_match:
        try:
            import v4_condition_head as _ch
            import v4_correlations as _corr
            # v4_condition_head is a HEALTH-ONLY classifier: on non-health copy it
            # still confidently returns the nearest health condition (streetwear ->
            # Gut Health, a menswear ad -> Prostate). Gate its TEXT verdict behind
            # health-relevance so aesthetic / wealth / community copy is NOT force-
            # diagnosed. Reliable IMAGE classes stay (already precision-gated).
            _health_copy = len(set(_corr._HEALTH_GATE.findall(
                (copy_text or "").lower()))) >= 2
            cc, cc_source = _ch.predict_condition_routed(_frames, copy_text)
            if cc:
                cond_cat = max(cc, key=cc.get)
                _src_ok = ((cc_source == "text" and _health_copy)
                           or (cc_source != "text"
                               and cond_cat in _ch.RELIABLE_IMAGE_CLASSES))
                if cond_cat in VISUAL_TO_CONDITION and _src_ok:
                    mapped = VISUAL_TO_CONDITION[cond_cat]
                    pain_names = [mapped] + [p for p in pain_names
                                             if p != mapped]
                    visual_derived = True
        except Exception:  # noqa: BLE001 — best-effort, never blocks analyze
            pass
        visual_derived = True

    # PRODUCT — the vision brief names it; for a text/no-brief ad fall back to
    # the copy itself (DR ads launch with "Introducing X"). Surfaced so the
    # read isn't blank just because there was no creative to look at.
    product = (brief or {}).get("product") or ""
    from_copy = False
    if not product:
        try:
            b = brief or {}
            meta = b.get("_source_meta") if isinstance(b.get("_source_meta"),
                                                       str) else ""
            # READER-THAT-UNDERSTANDS FIRST: Florence read the HERO (what's
            # sold), not the body's drug/condition comparisons. So try naming
            # the product from its hero caption before the full copy — a clean
            # hero read beats the body where 'Ozempic'/'SIBO' live.
            hero = (b.get("scene_caption") or "").strip()
            product = ""
            if len(hero) >= 15:
                product = (v4_correlations.brand_from_ad_copy(hero)
                           or v4_correlations.extract_product(hero) or "")
            # THE TRANSCRIPT IS ALREADY INSIDE copy_text. brief_to_text()
            # renders "[TRANSCRIPT]\n<spoken_transcript>" into the very blob
            # handed to analyze() as copy_text, so appending it a second time
            # DOUBLED every token count in it. brand_from_ad_copy's "a bare
            # capitalized coined word appearing once is too weak to trust"
            # gate (n < 2) is a count test — the duplication silently promoted
            # every ONE-OFF word of the transcript to a "recurring" one. That
            # is how "ADVERTORIAL" — printed once at the head of a DR
            # advertorial lander, which is folded into this slot — cleared the
            # recurrence gate and beat the real brand for the product name.
            # Append it only when it is not already present (a caller with no
            # brief, or one that fills the slot after brief_to_text ran).
            spoken = (b.get("spoken_transcript") or "").strip()
            parts = [copy_text, hero]
            if spoken and spoken not in copy_text:
                parts.append(spoken)
            ad_text = "\n".join(parts).strip()
            # else READ THE AD COPY: the brand is the COINED word in a SELLING
            # context ('try Elvera'), not the competitor it bashes ('Miralax
            # keeps failing') — the role-reasoning in brand_from_ad_copy demotes
            # drugs/conditions/competitors by their role.
            if not product:
                product = v4_correlations.brand_from_ad_copy(ad_text,
                                                             title=meta) or ""
            # corroborators if the read was inconclusive: the page <title>, then
            # the copy's launch-verb/suffix brands, then the landing URL (last).
            if not product:
                product = v4_correlations.brand_from_title(meta,
                                                           corpus=copy_text) or ""
            if not product:
                prod_text = "\n".join([ad_text, b.get("cta_text") or "",
                                       b.get("landing_page") or b.get("landing")
                                       or ""]).strip()
                product = v4_correlations.extract_product(prod_text) or ""
            from_copy = bool(product)
        except Exception:  # noqa: BLE001
            product = ""

    # CROSS-SOURCE GROUNDING GUARD — a product is trusted only if it is
    # verifiably present in the ad's actual CONTENT (copy / creative caption /
    # spoken VO), not merely read off a URL, <title>, or aggregator chrome.
    # This is the check that blocks a gethookd / ad-library SHARE page from being
    # named as the 'product' (e.g. "Gethookd"): that name appears nowhere in the
    # real ad. When the current read is ungrounded, recover a grounded brand
    # from the content; failing that, abstain rather than assert a phantom. (A
    # domain-style name like "sundaysfordogs" is still grounded by a shorter
    # content brand token that is its prefix — "sundays" in the body.)
    if product:
        _b = brief or {}
        # the ADVERTISER's own landing domain IS a valid product source
        # (about.bugmd.com -> BugMD; sundaysfordogs.com -> Sundays) — that's the
        # ~95% naming signal. Only AGGREGATOR / social hosts are disqualified:
        # they HOST the ad, they are not the product (gethookd/trendtrack/fb…).
        _AGG = ("gethookd.", "trendtrack.", "facebook.", "fb.com", "instagram.",
                "tiktok.", "whatsapp", "app.link", "t.me", "messenger.",
                "linktr.ee", "youtu", "ads/library", "adlibrary")
        _land = (_b.get("landing_page") or _b.get("landing") or "")
        _host = _land.split("//", 1)[-1].split("/", 1)[0].lower() if _land else ""
        _landsrc = _host if (_host and not any(h in _host for h in _AGG)) else ""
        _content = "\n".join([copy_text or "", _b.get("scene_caption") or "",
                              _b.get("spoken_transcript") or "", _landsrc])
        _blob = re.sub(r"[^a-z0-9]", "", _content.lower())
        _toks = set(re.findall(r"[a-z0-9]{4,}", _content.lower()))

        def _grounded(p):
            npn = re.sub(r"[^a-z0-9]", "", (p or "").lower())
            if len(npn) < 4 or not _blob:
                return False
            if npn in _blob:
                return True
            return any(npn.startswith(t) or t.startswith(npn) for t in _toks)

        if not _grounded(product):
            recovered = v4_correlations.brand_from_ad_copy(_content) or ""
            if recovered and _grounded(recovered):
                print(f"{DIM}[grounding: '{product}' wasn't in the ad's "
                      f"content — named '{recovered}' from the copy instead]"
                      f"{RESET}")
                product, from_copy = recovered, True
            else:
                print(f"{DIM}[grounding: '{product}' isn't in the ad's "
                      f"copy/creative/VO — abstaining on product]{RESET}")
                product = ""

    # Show the product whenever we have one — including when a LOCAL brief left
    # it blank and we recovered the brand from the copy (was previously printed
    # only on the no-brief path, so media ads showed no product even when the
    # transcript named it).
    if product:
        plabel = "PRODUCT (from copy)" if (brief is None or from_copy) \
            else "PRODUCT"
        print(f"\n{BOLD}{C_LABEL}{plabel:<22}{RESET} "
              f"{C_HIGHLIGHT}{BOLD}{product}{RESET}")
    if brief is not None and brief.get("subject"):
        print(f"\n{BOLD}{C_LABEL}{'SUBJECT / CONDITION':<22}{RESET} "
              f"{C_HIGHLIGHT}{BOLD}{brief['subject']}{RESET}")
    if pain_names:
        styled_pains = [f"{C_WARNING}{BOLD}{pn}{RESET}" for pn in pain_names[:6]]
        # aesthetic/community categories are ASPIRATIONS, not problems — don't
        # mislabel a fashion/tribe read as a "problem"/"painpoint".
        _mt = v4_correlations.market_type_for(pain_names[0])
        if _mt == "aspiration":
            label = "ASPIRATION / IDENTITY"
        elif visual_derived:
            label = "DOMAIN ([visual] read)"
        elif copy_derived:
            label = "PROBLEM (from copy)"
        else:
            label = "PAINPOINTS detected"
        print(f"\n{BOLD}{C_LABEL}{label:<22}{RESET} " + ", ".join(styled_pains))
        _parent = v4_correlations.parent_category_for(pain_names[0])
        _angle = ("identity-affirming" if _mt == "aspiration"
                  else "problem-agitation")
        print(f"{DIM}{'  market / angle':<22}{RESET}{C_LABEL}{_parent}{RESET} "
              f"{DIM}·{RESET} {C_HIGHLIGHT}{_mt}{RESET} {DIM}→ frame the ad by "
              f"{_angle}{RESET}")
        if visual_derived:
            print(f"  {C_BORDER}- {RESET}{DIM}copy was blank/garbled; named from "
                  f"the CREATIVE IMAGE by the vision head (~48%) — adds no "
                  f"age/gender prior{RESET}")
        elif copy_derived:
            print(f"  {C_BORDER}- {RESET}{DIM}read from the ad's own wording; "
                  f"no catalogued study for it, so it adds no age/gender "
                  f"prior{RESET}")
        else:
            for finding, name, _ in prior["sources"][:3]:
                print(f"  {C_BORDER}- {RESET}{C_LABEL}{finding}{RESET} "
                      f"{DIM}[{name}]{RESET}")

    vp = _vision_prior_from_brief(brief) if brief is not None else None
    # THE ADVERTISER'S OWN LANDING SLUG is first-party targeting, and it was
    # being thrown away: `/pages/peptides-for-womens-hair-loss` says who the
    # ad is for more plainly than the creative does, yet demographics only
    # ever saw copy_text — so that ad read gender 'unclear'. Path words only
    # (never the query string, which carries signatures/ids), and never from
    # an aggregator that merely HOSTS the ad.
    demo_text = copy_text
    _lp = (brief or {}).get("landing_page") or (brief or {}).get("landing") or ""
    if _lp:
        _h = _lp.split("//", 1)[-1].split("/", 1)[0].lower()
        _AGG_D = ("gethookd.", "trendtrack.", "facebook.", "fb.com",
                  "instagram.", "tiktok.", "linktr.ee", "youtu", "metastatus.")
        if _h and not any(a in _h for a in _AGG_D):
            _path = _lp.split("//", 1)[-1].split("?", 1)[0]
            _slug = re.sub(r"[^a-z0-9]+", " ", _path.lower()).strip()
            if _slug:
                demo_text = f"{copy_text}\n{_slug}"
    demo_res = demo_mod.infer_demographics(
        demo_text, vision_prior=vp, painpoints=brief_pains,
        condition_ids=cond_ids)
    print(f"\n{BOLD}{C_LABEL}AUDIENCE{RESET} {DIM}(ad markers + study priors + [vision] from creative; "
          f"'unclear' = abstains; 'study-prior'/'vision' = grounded){RESET}")
    print(demo_mod.format_demographics(demo_res))
    # When the avatar was read off an on-screen FACE (local path), surface it
    # as the PRESENTER — explicitly NOT the target buyer — so the information
    # isn't lost now that it no longer pins the audience. Who demos a product
    # (often a young creator) is rarely who it's sold to.
    if brief is not None and _is_face_avatar(brief):
        fav = brief.get("avatar", {}) or {}
        who = ", ".join(x for x in (fav.get("gender"), fav.get("age_range"))
                        if x and x != "unclear")
        if who:
            n = fav.get("n_people")
            n_str = f"{n} on-screen; " if n else ""
            print(f"{'presenter (on-screen)':<22} {who}  {DIM}({n_str}who "
                  f"APPEARS in the creative, not necessarily the buyer){RESET}")

    # US ETHNICITY — reference income surfaced ONLY on an EXPLICIT copy signal
    # (a Spanish-language ad or an explicit community reference); NEVER inferred
    # from accent/appearance/name. Abstains (prints nothing) otherwise. European
    # ads key on nationality via the [vision] voice line above, not this axis.
    ethnicity = None
    try:
        import v4_stats
        sig = v4_stats.explicit_ethnic_signal(copy_text)
        if sig:
            grp, ev = sig
            inc_r = v4_stats.income_by_race(grp)
            ethnicity = {"group": grp, "evidence": ev,
                         "income": inc_r["display"] if inc_r else None}
            line = f"{grp} {DIM}({ev}){RESET}"
            if inc_r:
                line += (f"; {DIM}US median household income {inc_r['display']} "
                         f"[{inc_r['source_name']}]{RESET}")
            print(f"{BOLD}{C_LABEL}{'  [explicit] ethnicity':<22}{RESET} {line}")
    except Exception:  # noqa: BLE001
        pass

    # AGE-IMPLIED income baseline (sourced). The qualitative income BAND above
    # comes from lexical markers (budget/premium) and abstains on most ads.
    # When we DO know an age (markers / study-prior / face), surface the Census
    # median household income for that age cohort — a sourced POPULATION
    # baseline, clearly labelled as such (NOT a claim about this buyer's
    # wallet). This is the income 'range' the user asked for: it stops income
    # from being vacuously 'unclear' whenever age is grounded.
    income_age = None
    try:
        import v4_stats
        # Country-aware income: a lander WRITTEN in a European language is
        # targeted for THAT market, so its income baseline must be that
        # country's figure — not the US ACS age table. `market` was detected
        # on the ORIGINAL copy near the top (before translation).
        if market:
            rec = v4_stats.region_income(market)
            if rec:
                income_age = {"market": market, "display": rec["display"],
                              "measure": rec.get("measure"),
                              "source_name": rec["source_name"]}
                print(f"{BOLD}{C_LABEL}{'  [market] income':<22}{RESET} "
                      f"~{rec['display']} {DIM}{rec.get('measure', 'median income')}"
                      f" — the lander is written/targeted for {market} "
                      f"({rec.get('year', '')}) [{rec['source_name']}]{RESET}")
            else:
                # market identified but no sourced income figure in the KB
                # (e.g. India) — withhold, NEVER fall back to the US table.
                print(f"{BOLD}{C_LABEL}{'  [market] income':<22}{RESET} "
                      f"{DIM}withheld — market is {market}, no local income "
                      f"figure in the KB (NOT the US baseline){RESET}")
            # Say whether the motif reads above are trustworthy: if we
            # translated, they ran on the English text (reliable); if not,
            # they ran on foreign copy (low confidence).
            if translated:
                print(f"{BOLD}{C_LABEL}{'  [note]':<22}{RESET} {DIM}copy was "
                      f"non-English ({market}); translated to English locally "
                      f"(argos/CTranslate2) before scoring — ANGLE / AWARENESS "
                      f"/ DESIRES above are on the translation.{RESET}")
            else:
                print(f"{BOLD}{C_LABEL}{'  [note]':<22}{RESET} {DIM}copy is "
                      f"non-English ({market}) and translation was unavailable; "
                      f"ANGLE / AWARENESS / DESIRES are English-trained motif "
                      f"reads -> LOW CONFIDENCE.{RESET}")
        elif v4_stats.looks_english(copy_text):
            # US income ONLY on positive English evidence — never assume US by
            # default. A foreign / OCR-garbled (non-Latin script the OCR
            # couldn't read) creative must not inherit US figures.
            age_val = demo_res["age"]["value"]
            if age_val and age_val != "unclear":
                rec = v4_stats.income_by_age(age_val)
                if rec:
                    income_age = {"bucket": age_val, "display": rec["display"],
                                  "acs": rec["acs"],
                                  "source_name": rec["source_name"],
                                  "source_url": rec["source_url"]}
                    print(f"{BOLD}{C_LABEL}{'  [study-prior] income':<22}{RESET} "
                          f"~{rec['display']} {DIM}US median household income "
                          f"for householder age {rec['acs']} (population "
                          f"baseline for the {age_val} read, not the "
                          f"individual) [ACS 2019-2023 5-yr B19049]{RESET}")
        else:
            # neither a detected European market nor confidently English.
            # Distinguish the two sub-cases honestly: Latin-but-not-English
            # (a foreign language the market table missed — readable, just not
            # US) vs genuinely non-Latin / mangled OCR (truly unreadable).
            if v4_stats.is_mostly_latin(copy_text):
                print(f"{BOLD}{C_LABEL}{'  [note]':<22}{RESET} {DIM}copy is "
                      f"Latin-script but not confidently English and no market "
                      f"matched — possibly a foreign language not in the income "
                      f"table. Income withheld (no US default); text-derived "
                      f"reads are LOWER confidence.{RESET}")
            else:
                print(f"{BOLD}{C_LABEL}{'  [note]':<22}{RESET} {DIM}on-screen "
                      f"copy is a non-Latin script the OCR couldn't read. Income "
                      f"withheld (no US default); ANGLE / AWARENESS / DESIRES / "
                      f"painpoints are UNRELIABLE here — trust only face "
                      f"(gender/age) + scene.{RESET}")
    except Exception:  # noqa: BLE001
        pass

    gain, top_primal, primality = problem_grounding(scorer, zg)
    # FEATURES must match what the loaded head was fit on. The legacy layout
    # passed torch.zeros(88) for the trend block and torch.zeros(1) for
    # momentum because inference has no trend_id — but the head was TRAINED on
    # real values there, so 89 of its 353 inputs were a train/inference
    # mismatch. Measured cost: Spearman(PV^, PV*) 0.8785 trained-input ->
    # 0.7886 inference-input, i.e. the live head was running BELOW its own
    # 0.80 acceptance gate. The v2 builder uses only what inference can
    # actually compute (0.8836 held-out).
    if getattr(engine, "feats_ver", "v1") == "v2":
        feats = eng_mod.build_feats(za, zg).unsqueeze(0)
    else:
        feats = torch.cat([za, torch.zeros(88), zg, za * zg,
                           torch.zeros(1)]).unsqueeze(0)
    with torch.no_grad():
        p = engine(feats, torch.ones(1), problem_gain=gain)
    # PV = Desire x T. Desire derives from the problem (a problem creates the
    # emotion that creates the want), amplified by the problem's primality.
    dsr = float(p["Dsr"][0])
    # No KB match, no open-set extraction, no trained-classifier override =>
    # we genuinely could not identify a condition. Say so. The old fallback
    # emitted top_primal here, which is a Maslow/chakra EMOTIONAL TIER
    # ("safety", "heart chakra", "physiological") — not a medical condition —
    # so a cane ad and a price-drop ad both surfaced a psych label in the
    # problem field as if it were the diagnosis. The tier is still reported
    # on the Desire line below via top_primal; it just no longer masquerades
    # as the problem. Label-only: gain/dsr/PV are unchanged.
    # ANGLE vs PAINPOINT. A DR ad names the buyer's felt problem ("still going
    # soft") AND the mechanism it blames for it ("diabetes wrecked your nerves
    # and blood flow"). Both matched the KB, so both were reported as
    # painpoints — which is wrong twice over: the angle is not what the reader
    # suffers from, and it made the ad look like a diabetes ad. split_angles
    # separates them using the KB's own sourced mechanism graph (X named in
    # Y's cause chain => X is upstream) plus the copy's causal framing.
    # Reported separately; the demographic prior still sees BOTH, since a
    # named comorbidity legitimately informs the buyer's age.
    angle_names = []
    mech_angles = []
    try:
        _amatch = v4_correlations.match_painpoints(copy_text)
        # A NAMED MECHANISM OUTRANKS A COMORBID CONDITION. The Angle card says
        # "mechanism the ad blames", but angle_names can only ever hold
        # PAINPOINT ROWS — so an ad arguing that DHT shrinks the follicle got
        # "Menopause Symptoms" in that slot (a bundled label that names no
        # symptom, and which the KB never links to hair). mechanism_angles
        # returns the agent the copy ACTUALLY names, and only when the matched
        # painpoint's own sourced mechanism names it too.
        _apains, _aangles = v4_correlations.split_angles(copy_text, _amatch)
        _aset = {p["name"] for p, _ in _aangles}
        _kept = [n for n in pain_names if n not in _aset]
        if _aset and _kept:          # never strip the last painpoint
            angle_names = [n for n in pain_names if n in _aset]
            pain_names = _kept
        # Now drop the rows whose evidence is a rounding error next to the
        # leader's. Deliberately AFTER the split: a mechanism row can be
        # named far less often than the complaint it explains and still be
        # the ad's whole argument, so it must already be safe in angle_names
        # before this runs. (PCOS: 2 cue hits against hyperpigmentation's 17,
        # but it is exactly what that ad blames.)
        pain_names = v4_correlations.dominant_painpoints(pain_names, _amatch)
        # MASLOW'S READER — open-set naming, when a reader was supplied.
        #
        # Everything above picks the best row out of a closed KB. That has a
        # floor: the KB held 1,270 painpoints and no human pigmentation entry,
        # so a "no more dark marks" ad could only land on Wrinkles & Skin
        # Aging. The reader names the problem in the buyer's own words and the
        # KB is demoted to grounding it — and when it does NOT resolve, the
        # name stands on its own rather than borrowing the nearest row.
        #
        # Passed ONLY by the Maslow lens (audience & market). Schwartz asks a
        # different question — how the ad is built, what it ranks — and never
        # supplies one, so its path here is byte-for-byte what it was.
        if reader is not None:
            pain_names, angle_names = _read_open_set(
                copy_text, reader, pain_names, angle_names)
        # Read the mechanism off the RESOLVED painpoints, not the raw regex
        # hits. match_painpoints is cue-driven and misses ordinary phrasing:
        # on the real ad this came from, `Hair Loss & Thinning` fired on none
        # of its cues and none of its name terms, so _amatch held only
        # "Pet Grooming" — and the DHT the ad explicitly blames was invisible
        # even though that row's own sourced mechanism names it.
        mech_angles = v4_correlations.mechanism_angles(
            copy_text,
            v4_correlations.rows_for_names(pain_names + angle_names)
            or _amatch)
    except Exception:  # noqa: BLE001 — display split, never blocks analyze
        angle_names = []
        mech_angles = []
    problem = pain_names[0] if pain_names else "unclear"
    # market TYPE: 'aspiration' for aesthetic/community/identity categories
    # (fashion, tribes, hobbies) where the desire is primary and there is no
    # problem to fix; 'problem' for health/debt/loneliness/etc. Label-only —
    # the PV math (problem_grounding intensity x desire x T) is unchanged; this
    # just stops the read from calling a fashion aspiration a "problem".
    market_type = v4_correlations.market_type_for(problem)
    parent_category = v4_correlations.parent_category_for(problem)
    # ANGLE MODE: how the ad should MOVE the reader. Problem markets AGITATE the
    # pain then relieve it; aspiration markets AFFIRM the identity/belonging the
    # buyer is reaching for. This is the market-type distinction made actionable —
    # it flips the messaging posture, not the PV math.
    angle_mode = ("identity-affirming" if market_type == "aspiration"
                  else "problem-agitation")

    p_T = float(p['T'][0])
    p_PV = float(p['PV'][0])
    
    line_dsr = f" Desire      dsr = {dsr:.2f}  (from '{problem}'; x{gain:.2f} via {top_primal})"
    line_time = f" Time decay  T   = {p_T:.2f}  (t_ratio 1.0, lambda {lam:.3f})"
    
    def pad_line(content, color_prefix="", style_suffix=""):
        vis_len = len(content)
        pad = 60 - vis_len
        if pad < 0:
            content = content[:57] + "..."
            vis_len = 60
            pad = 0
        return f"{C_BORDER}║{RESET} {color_prefix}{content}{style_suffix}{' ' * pad} {C_BORDER}║{RESET}"

    print(f"\n{C_BORDER}╔══════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{C_BORDER}║{RESET}                    {BOLD}{C_GREEN}PERCEIVED VALUE ENGINE{RESET}                    {C_BORDER}║{RESET}")
    print(f"{C_BORDER}╠══════════════════════════════════════════════════════════════╣{RESET}")
    print(pad_line(line_dsr, color_prefix=C_LABEL))
    print(pad_line(line_time, color_prefix=C_LABEL))
    print(f"{C_BORDER}║{RESET}{' ' * 62}{C_BORDER}║{RESET}")
    
    pv_content = f" PV = Desire x T = ({dsr:.2f}dsr x {p_T:.2f}T) = "
    pv_score_str = f"{p_PV:.3f} PV"
    pv_line_styled = f"{C_BORDER}║{RESET}  {C_HIGHLIGHT}{BOLD}{pv_content}{RESET}{C_GREEN}{BOLD}{pv_score_str}{RESET}{' ' * (60 - len(pv_content) - len(pv_score_str) - 1)} {C_BORDER}║{RESET}"
    print(pv_line_styled)
    # winner-likelihood from the gethookd-trained head (copy -> win-prob,
    # held-out AUC ~0.76). Guarded: if the head isn't trained / errors, the
    # decomposition is unaffected.
    try:
        from v4_winner_score import win_prob as _winprob
        _wp = _winprob(text)
    except Exception:  # noqa: BLE001
        _wp = None
    if _wp is not None:
        wl_c = " Winner-likelihood (gethookd-graded copy) = "
        wl_s = f"{_wp:.0%}"
        print(f"{C_BORDER}║{RESET}  {C_HIGHLIGHT}{BOLD}{wl_c}{RESET}"
              f"{C_GREEN}{BOLD}{wl_s}{RESET}"
              f"{' ' * (60 - len(wl_c) - len(wl_s) - 1)} {C_BORDER}║{RESET}")
    print(f"{C_BORDER}╚══════════════════════════════════════════════════════════════╝{RESET}")
    if not return_result:
        return float(p["PV"][0])
    # structured decomposition for the evaluator (v4_autoresearch): every
    # field the correctness checks assert on.
    journey = ([e["stage"] for e in brief["awareness_journey"]]
               if brief is not None and brief.get("awareness_journey")
               else present)
    desire_tags = (brief.get("core_desires", [])[:6]
                   if brief is not None and brief.get("core_desires")
                   else [t for t, _ in desires])
    presenter = None
    if brief is not None and _is_face_avatar(brief):
        fav = brief.get("avatar", {}) or {}
        presenter = ", ".join(x for x in (fav.get("gender"),
                              fav.get("age_range")) if x and x != "unclear") \
            or None
    return {
        "product": product,
        "visual_category": visual_cat_label,
        "presenter": presenter,
        # trimmed to the facets the copy actually supports — a bundled KB label
        # must not assert a complaint the ad never made (see display_name)
        "painpoints": [v4_correlations.display_name(n, copy_text)
                       for n in pain_names[:6]],
        # The mechanism/cause the ad INVOKES to sell (not what the buyer feels).
        # NOT "angles" — in the packaged tree that key is the ARCHETYPE angle
        # (everyman/sage), and a duplicate key in one dict literal silently
        # kept the last one, rendering ('everyman', 0.63) in the Angle card.
        # A mechanism the ad NAMES leads; matched conditions only fill in
        # behind it. "DHT" beats "Menopause Symptoms" on a hair-loss ad
        # because it is what the copy actually blames — and unlike a bundled
        # condition label it names something specific.
        "painpoint_angles": ([lbl for lbl, _src in mech_angles]
                             + [v4_correlations.display_name(n, copy_text)
                                for n in angle_names])[:6],
        "age": demo_res["age"]["value"],
        "gender": demo_res["gender"]["value"],
        "income": demo_res["income"]["value"],
        "income_by_age": income_age,
        "life_stage": demo_res["life_stage"]["value"],
        "awareness_journey": journey,
        "desires": desire_tags,
        "problem": problem,
        "market_type": market_type,
        "parent_category": parent_category,
        "angle_mode": angle_mode,
        "dsr": dsr,
        "T": float(p["T"][0]),
        "gain": gain,
        "top_primal": top_primal,
        "PV": float(p["PV"][0]),
        "winner_prob": _wp,
        "ethnicity": ethnicity,
        # CONDITION EPIDEMIOLOGY — sourced facts about WHO the matched
        # condition affects (age/gender/ethnicity/geography disparities from
        # published studies), distinct from the privacy-guarded viewer
        # `ethnicity` inference above. This is "draw upon demographical studies
        # done on different groups" — a citation about the condition, never an
        # inference about the person viewing the ad.
        "condition_epidemiology": [
            {"finding": f, "source": n, "url": u}
            for (f, n, u) in (prior.get("sources", [])
                              if isinstance(prior, dict) else [])[:8]
            if f and u
        ],
        # MECHANISMS — the sourced biological/psychological cause->effect + the
        # marketing copy-signal for the matched painpoint(s). This is the
        # copy->mechanism correlation (Schwartz 4.5): explain WHY the ad's
        # problem maps to a condition via a real mechanism, not a bare label.
        # Report-only, sourced; empty for painpoints not yet enriched.
        # Painpoints AND angles: the ANGLE's mechanism is precisely what the ad
        # is explaining ("low testosterone is why the belly fat won't shift"),
        # so dropping it here would delete the sourced cause-chain the ad's
        # whole argument rests on.
        "mechanisms": v4_correlations.mechanisms_for(
            list(dict.fromkeys(pain_names + angle_names))),
    }


if __name__ == "__main__":
    text = read_one_paste()
    if not text:
        raise SystemExit("nothing pasted")
    print(f"\n[got {len(text):,} chars — analyzing, ~10 seconds...]\n",
          flush=True)
    analyze(text, load_stack())
