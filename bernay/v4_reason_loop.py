"""
v4_reason_loop — the ad decode as a REASONING LOOP, not a lookup.

The pipeline's shape until now: match the copy against a closed KB of painpoint
rows, then patch the wrong answers with filters. That shape has a floor. The KB
held 1,270 rows and not one human pigmentation entry, so a "no more dark marks"
ad could only ever land on `Wrinkles & Skin Aging` — the matcher had no way to
say "this problem isn't in my vocabulary", it could only pick the nearest row
and then explain a melanin problem with a collagen mechanism. Every guard added
since (species gate, evidence floor, density guard, dominance ratio) treats a
symptom of that shape.

So invert it. A READER names the problem in the buyer's own words, open set,
always. The KB stops being the vocabulary and becomes the GROUNDING TABLE — the
thing that attaches a sourced mechanism, a study prior, epidemiology to a name
the reader produced. When nothing resolves, the grounded fields are ABSENT, not
the nearest wrong row.

    READ    reader names {painpoint, angle, quote} from the copy
    GROUND  resolve that name onto the KB, for grounding only
    CHECK   run the existing guards as CRITICS on the reader's own answer
    REVISE  hand the failures back and let it answer again
    EMIT    grounded where it resolved; explicitly absent where it didn't

The critics are not new code. Each one is a guard that already exists, asked as
a question instead of applied as a filter:

    evidence   did the ad say enough to claim anything?   v4_admix.evidence_chars
    grounded   are these words actually in the copy?      quote span
    species    is this a human or an animal problem?      _is_pet_copy
    role       is that the problem, or the cause of it?   split_angles
    singular   is there really more than one problem?     dominant_painpoints
    facet      does the label assert an unmentioned facet? display_name

That reuse is the point twice over: the critics also SCORE traces, so the same
checks that make one answer honest generate the supervised data to fine-tune a
reader — pass/fail labels with no hand annotation.

Readers are pluggable and none is required to run the harness:

    RegexReader   the existing extract_problem/match_painpoints path, wrapped
                  as a reader — proves the loop with zero new models
    OllamaReader  any local instruct model over the Ollama HTTP API

    python v4_reason_loop.py --reader regex
    python v4_reason_loop.py --reader ollama --model qwen3.8-27b-iq3:8k
"""

import argparse
import io
import json
import os
import re
import sys
import urllib.request

try:                                    # flat Downloads tree
    import v4_correlations as C
except ImportError:                     # restructured live tree
    from communities_00_09.community_4_v4_correlations_v4_kb_probe \
        import v4_correlations as C

# The reader is allowed to name anything, but a name has to be SAID by the ad.
# Ungrounded naming is the one failure a fluent reader makes that the regex
# path never could, so it is checked first and hardest.
_STOP = {"the", "a", "an", "and", "or", "of", "your", "you", "my", "our",
         "with", "for", "in", "on", "to", "is", "are", "that", "this", "it"}


# ---------------------------------------------------------------------------
# readers
# ---------------------------------------------------------------------------
class RegexReader:
    """The CURRENT pipeline, wrapped as a reader.

    Deliberately included: it makes the harness runnable and measurable before
    any model exists, and it is the baseline every reader has to beat on the
    same cases.
    """

    name = "regex"

    def __call__(self, copy):
        matched = C.match_painpoints(copy)
        pains, angles = (matched, [])
        if matched:
            try:
                pains, angles = C.split_angles(copy, matched)
            except Exception:  # noqa: BLE001
                pains, angles = matched, []
        pp = pains[0][0]["name"] if pains else (C.extract_problem(copy) or "")
        an = angles[0][0]["name"] if angles else ""
        return {"painpoint": pp, "angle": an, "quote": ""}


class MaslowReader:
    """The reader served BY THE APP — Maslow's own GPU service (8799), the one
    the desktop app starts and reports in /api/health. No separate runtime to
    install, nothing that can be quietly down while the app claims ready."""

    name = "maslow"

    def __init__(self, url="http://127.0.0.1:8799"):
        self.url = url.rstrip("/")

    def status(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except Exception as e:  # noqa: BLE001
            return {"reader": f"unreachable: {type(e).__name__}"}

    def __call__(self, copy, correction=None):
        # correction rides its OWN field — folded into `copy` the model reads
        # it as more ad text and revises nothing.
        payload = {"copy": copy or "", "correction": correction or ""}
        req = urllib.request.Request(
            self.url + "/read", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            got = json.loads(r.read().decode("utf-8", "replace"))
        return {"painpoint": got.get("painpoint") or "",
                "angle": got.get("angle") or "", "quote": ""}


class OllamaReader:
    """Any local instruct model, over Ollama's HTTP API. Nothing leaves the
    machine and there is no API key."""

    SYS = (
        "You read direct-response ad copy and name what it sells against.\n"
        'Return STRICT JSON only: {"painpoint": "...", "angle": "...", '
        '"quote": "..."}\n'
        "painpoint = the problem the BUYER FEELS, in the buyer's own words, "
        "2-5 words. Never the product, the ingredient, or the mechanism.\n"
        "angle = the CAUSE the ad BLAMES for that problem. Empty if the ad "
        "names no cause.\n"
        "quote = the shortest span COPIED VERBATIM from the ad that shows the "
        "painpoint. It must appear in the ad character for character.\n"
        "If the ad is about a pet, say so (e.g. 'dog joint pain')."
    )

    def __init__(self, model="qwen3.8-27b-iq3:8k",
                 url="http://127.0.0.1:11434/api/chat"):
        self.model, self.url, self.name = model, url, "ollama:" + model

    def __call__(self, copy, correction=None):
        msgs = [{"role": "system", "content": self.SYS},
                {"role": "user", "content": copy[:12000]}]
        if correction:
            msgs.append({"role": "user", "content": correction})
        body = json.dumps({
            "model": self.model, "messages": msgs, "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 160},
        }).encode()
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode("utf-8", "replace"))
        txt = (out.get("message") or {}).get("content") or ""
        txt = re.sub(r"(?s)<think>.*?</think>", " ", txt)
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {"painpoint": "", "angle": "", "quote": ""}
        try:
            got = json.loads(m.group(0))
        except json.JSONDecodeError:
            return {"painpoint": "", "angle": "", "quote": ""}
        return {"painpoint": str(got.get("painpoint") or ""),
                "angle": str(got.get("angle") or ""),
                "quote": str(got.get("quote") or "")}


# ---------------------------------------------------------------------------
# critics — each is an existing guard, asked as a question
# ---------------------------------------------------------------------------
def _content_words(s):
    return [w for w in re.findall(r"[a-z]{3,}", (s or "").lower())
            if w not in _STOP]


def critic_evidence(copy, ans, ctx):
    """Did the ad say enough to claim anything at all? (v4_admix's floor.)"""
    try:
        try:
            import v4_admix
        except ImportError:
            from communities_00_09.community_9_v4_admix_v4_categorize_batch \
                import v4_admix
        n = v4_admix.evidence_chars(copy)
        floor = v4_admix.MIN_EVIDENCE_CHARS
    except Exception:  # noqa: BLE001
        n = sum(1 for c in (copy or "") if c.isalnum())
        floor = 40
    if n < floor:
        return False, ("The ad carries only %d characters of readable copy "
                       "(floor %d). Do not name a painpoint; answer with an "
                       "empty painpoint." % (n, floor))
    return True, None


def critic_grounded(copy, ans, ctx):
    """Is the named problem actually SAID by the ad?

    The one failure mode a fluent reader has that the regex path never did.
    A name passes if its content words appear in the copy, or if the verbatim
    quote it cited really is there.
    """
    pp = ans.get("painpoint") or ""
    if not pp:
        return True, None
    low = (copy or "").lower()
    q = (ans.get("quote") or "").strip().lower()
    if q and q in low:
        return True, None
    words = _content_words(pp)
    if not words:
        return False, ("'%s' names no problem. Give a concrete complaint the "
                       "ad makes." % pp)
    present = [w for w in words if w in low]
    if len(present) * 2 >= len(words):        # over half the content words
        return True, None
    return False, ("'%s' does not appear in the ad — the words %s are not in "
                   "the copy. Name the problem using the ad's own words, and "
                   "put the exact span you read it from in `quote`."
                   % (pp, [w for w in words if w not in present]))


def critic_species(copy, ans, ctx):
    """Human problem or animal problem? (the two-sided pet gate.)"""
    pp = (ans.get("painpoint") or "").lower()
    if not pp:
        return True, None
    is_pet_copy = C._is_pet_copy((copy or "").lower())
    says_pet = bool(re.search(r"\b(dog|cat|puppy|kitten|pet|canine|feline)\b",
                              pp))
    if is_pet_copy and not says_pet:
        # Name the species the COPY actually names. The example used to be
        # hardcoded "dog", so on a cat ad the critic was telling the reader to
        # answer "dog ..." — the exact dog/cat confusion this whole layer is
        # here to stop, reintroduced by the thing meant to catch it.
        low = (copy or "").lower()
        species = next((s for s in ("cat", "kitten", "feline", "dog", "puppy",
                                    "canine", "horse", "bird")
                        if re.search(r"\b%s" % s, low)), "pet")
        species = {"kitten": "cat", "feline": "cat", "puppy": "dog",
                   "canine": "dog"}.get(species, species)
        return False, ("This ad is about a %s, not a person. Say whose "
                       "problem it is — answer '%s %s'."
                       % (species, species, pp))
    if says_pet and not is_pet_copy:
        return False, ("This ad is about a person, not an animal. '%s' names "
                       "an animal." % pp)
    return True, None


def critic_role(copy, ans, ctx):
    """Is what you named the PROBLEM, or the CAUSE of it?

    The distinction the deck's two cards depend on: a felt state is the
    painpoint; it is only the angle when the ad blames it for something else.
    """
    pp, an = ans.get("painpoint") or "", ans.get("angle") or ""
    if not pp or not an:
        return True, None
    if pp.strip().lower() == an.strip().lower():
        return False, ("You gave '%s' as both the problem and its cause. If "
                       "the ad blames it for something else, that something "
                       "else is the painpoint." % pp)
    low = (copy or "").lower()
    # the ad casting the PAINPOINT as a cause of something = they are swapped
    for w in _content_words(pp)[:3]:
        if re.search(r"\b%s\b[^.]{0,40}\b(causes?|leads? to|triggers?|"
                     r"is why|behind|responsible for)\b" % re.escape(w), low):
            return False, ("The ad presents '%s' as a CAUSE, not the "
                           "complaint. Name what it is blamed FOR as the "
                           "painpoint, and put '%s' in `angle`." % (pp, pp))
    return True, None


def critic_singular(copy, ans, ctx):
    """Is there really more than one problem here? (the dominance ratio.)"""
    pp = ans.get("painpoint") or ""
    parts = [p for p in re.split(r"\s*(?:,| and | & |/)\s*", pp) if p.strip()]
    if len(parts) > 2:
        return False, ("'%s' lists %d complaints. An ad sells against ONE "
                       "problem. Name the single one it leads with."
                       % (pp, len(parts)))
    return True, None


CRITICS = [critic_evidence, critic_grounded, critic_species, critic_role,
           critic_singular]


# ---------------------------------------------------------------------------
# ground — the KB demoted from vocabulary to lookup
# ---------------------------------------------------------------------------
# Cosine floor for accepting a KB row as the grounding for a free-text name.
# 0.45 (the matcher's own gap-filler threshold) is far too loose HERE: an
# AirPods battery ad resolved to "Brain Fog, Focus & ADHD" at 0.557, which is
# the closed-vocabulary failure this module exists to end, just relocated into
# the grounding step. Measured on real reads — genuine grounds land high
# ("dark spots" -> Hyperpigmentation & Dark Spots 0.756, "digestion issues" ->
# Gut Health & Digestion 0.737) and the false one sits well below, so 0.70
# separates them with room on both sides.
def build_correction(msgs):
    """What the loop says back when the critics reject an answer. Shared with
    v4_reader_sft — a reader fine-tuned to act on a correction has to be
    trained on the correction it will actually be given."""
    return ("Your answer failed these checks. Fix them and answer again as "
            "JSON:\n" + "\n".join("- " + m for m in msgs))



GROUND_FLOOR = float(os.environ.get("V4_GROUND_FLOOR", "0.70"))


_GROUNDER_WARNED = []


def _species_ok(hits, copy):
    """Drop grounds on the wrong side of the human/animal line.

    `ground()` went straight to the embedder and so bypassed the two-sided pet
    gate the regex path has enforced for months — human copy about "empty
    supplement bottles" grounded to `Multiple Myeloma in Dogs` at 0.70, which
    clears any floor worth having. The gate already exists; this just stops
    the grounding step from being the one place that ignores it.
    """
    if not hits:
        return hits
    try:
        rows = {p.get("id"): p for p in C.load_kb()["painpoints"]}
        is_pet = C._is_pet_copy((copy or "").lower())

        def vet(h):
            r = rows.get(h[0]) or {}
            return str(r.get("domain", "")).startswith(("veterinary",
                                                        "pet_supplies"))
        kept = [h for h in hits if vet(h) == is_pet]
        # Never invent a match: if the gate empties the list the answer is
        # ungrounded, which is the honest outcome.
        return kept
    except Exception:  # noqa: BLE001 — gating must not break grounding
        return hits


def ground(name, copy, thresh=None):
    """Resolve a FREE-TEXT painpoint onto the KB, for grounding only.

    Returns the row plus its sourced mechanism / prior when the name resolves,
    and an explicit `grounded: False` when it does not — which is the whole
    point. An unresolved name keeps its own wording and simply carries no
    mechanism, rather than borrowing the nearest row's.
    """
    thresh = GROUND_FLOOR if thresh is None else thresh
    out = {"name": name, "grounded": False, "kb_id": None, "kb_name": None,
           "score": None, "mechanism": None, "epidemiology": None}
    if not name:
        return out
    try:
        try:
            import v4_semantic_painpoint as S
        except ImportError:
            from communities_20_29.community_27_v4_semantic_painpoint_v4_embed \
                import v4_semantic_painpoint as S
        # GROUND ON THE NAME *IN CONTEXT*, not the bare name.
        #
        # A painpoint is 2-5 words, so a polysemous one collides with whatever
        # KB row shares the word. Measured over 300 real reads, grounding the
        # bare name produced: 'delayed delivery' -> Stillbirth, 'part b
        # premium' -> Car Audio & Electronics, 'toxin removal' -> Diphtheria,
        # 'fake product' -> Fabricated or induced illness. The ad copy
        # disambiguates every one of those and was already being passed in
        # here unused: with context they become Flowers & Gift Delivery,
        # Medicare, High Cholesterol, Makeup & Cosmetics.
        #
        # The name is repeated so it still LEADS the query — on copy alone the
        # match drifts to whatever the ad is broadly about ('brain fog' ->
        # Menopause Symptoms, 'muscle cramps' -> Mastocytosis). Repeating it
        # recovers both while keeping the polysemy fixes.
        ctx = re.sub(r"\s+", " ", (copy or "")).strip()[:250]
        query = ("%s. %s. %s" % (name, name, ctx)) if ctx else name
        hits = S.match(query, topk=8, thresh=thresh)
        hits = _species_ok(hits, copy)
    except Exception as e:  # noqa: BLE001
        # NOT the same as "no row matched". A grounder that cannot run makes
        # every answer look unresolvable, and the caller emits a deck with no
        # mechanism while believing the KB simply had nothing to say.
        out["grounder_error"] = "%s: %s" % (type(e).__name__, e)
        if not _GROUNDER_WARNED:
            _GROUNDER_WARNED.append(1)
            print("[reason_loop] grounding UNAVAILABLE (%s) — every answer "
                  "will report ungrounded" % out["grounder_error"],
                  file=sys.stderr, flush=True)
        return out
    if not hits:
        return out
    kb_id, kb_name, score = hits[0]
    row = next((p for p in C.load_kb()["painpoints"] if p.get("id") == kb_id),
               None)
    if row is None:
        return out
    out.update({"grounded": True, "kb_id": kb_id, "kb_name": kb_name,
                "score": score, "mechanism": row.get("mechanism")})
    try:
        out["display"] = C.display_name(kb_name, copy)
    except Exception:  # noqa: BLE001
        out["display"] = kb_name
    return out


def decode(copy, reader, max_rounds=2, verbose=False):
    """READ -> GROUND -> CHECK -> REVISE -> EMIT. Returns the answer plus the
    full trace: which critics fired, what was said back, what changed."""
    trace = []
    correction = None
    ans = {"painpoint": "", "angle": "", "quote": ""}
    for rnd in range(max_rounds + 1):
        try:
            ans = (reader(copy, correction=correction)
                   if correction is not None and _takes_correction(reader)
                   else reader(copy))
        except TypeError:
            ans = reader(copy)
        fails = []
        for critic in CRITICS:
            ok, msg = critic(copy, ans, {})
            if not ok:
                fails.append((critic.__name__, msg))
        trace.append({"round": rnd, "answer": dict(ans),
                      "failed": [f for f, _ in fails]})
        if verbose:
            print("   round %d  %-28s %s"
                  % (rnd, (ans.get("painpoint") or "")[:28],
                     [f for f, _ in fails] or "OK"))
        if not fails:
            break
        correction = build_correction([m for _, m in fails])
        if not _takes_correction(reader):
            break                      # a non-revising reader gets one shot
    g = ground(ans.get("painpoint"), copy)
    return {"painpoint": ans.get("painpoint"), "angle": ans.get("angle"),
            "quote": ans.get("quote"), "grounding": g,
            "passed": not trace[-1]["failed"], "trace": trace}


def _takes_correction(reader):
    import inspect
    try:
        return "correction" in inspect.signature(reader.__call__).parameters
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reader", default="regex",
                    choices=("regex", "maslow", "ollama"))
    ap.add_argument("--model", default="qwen3.8-27b-iq3:8k")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("-n", type=int, default=0)
    args = ap.parse_args()

    reader = {"regex": RegexReader,
              "maslow": MaslowReader}.get(args.reader, None)
    reader = reader() if reader else OllamaReader(args.model)
    if isinstance(reader, MaslowReader):
        st = reader.status()
        print("maslow service: reader=%s  imagegen=%s"
              % (st.get("reader"), st.get("imagegen")), flush=True)
        if st.get("reader") != "ready":
            raise SystemExit(
                "Maslow's reader is not ready. Start the desktop app (it "
                "brings this service up itself), or run v4_maslow_server.py "
                "from .venv-bench.")
    here = os.path.dirname(os.path.abspath(__file__))
    rows = [json.loads(l) for l in
            io.open(os.path.join(here, "v4_regression_ads.jsonl"),
                    encoding="utf-8") if l.strip()]
    cases = [c for c in rows
             if (c.get("text") or c.get("transcript"))
             and "painpoint_re" in (c.get("expect") or {})]
    if args.n:
        cases = cases[:args.n]

    print("reader: %s   cases: %d   rounds: %d\n"
          % (reader.name, len(cases), args.rounds), flush=True)
    ok = ok_g = grounded = clean = 0
    for c in cases:
        copy = c.get("text") or c.get("transcript") or ""
        r = decode(copy, reader, max_rounds=args.rounds)
        hit = bool(re.search(c["expect"]["painpoint_re"],
                             r["painpoint"] or "", re.I))
        ok += hit
        # The deck emits the GROUNDED row where the name resolved, so score
        # that too: the locked regexes were written for KB names, and a reader
        # that says "belly fat" for `Stubborn Weight & Belly Fat` is not wrong.
        ok_g += bool(re.search(c["expect"]["painpoint_re"],
                               r["grounding"]["kb_name"] or r["painpoint"]
                               or "", re.I))
        grounded += bool(r["grounding"]["grounded"])
        clean += bool(r["passed"])
        print("  %s%s %-40s %-30s -> %s"
              % ("+" if hit else "-",
                 "" if r["passed"] else "!",
                 c["name"][:40], (r["painpoint"] or "")[:30],
                 r["grounding"]["kb_name"] or "(ungrounded)"), flush=True)
    n = len(cases)
    print("\npainpoint match   %d/%d  as named   %d/%d  as grounded"
          % (ok, n, ok_g, n))
    print("passed critics    %d/%d" % (clean, n))
    print("resolved to KB    %d/%d  (the rest keep their own wording and "
          "carry no mechanism)" % (grounded, n))


if __name__ == "__main__":
    main()
