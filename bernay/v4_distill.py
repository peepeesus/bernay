"""
v4_distill — Gemini-distilled STRUCTURED AD BRIEF.

The char backbone can't see a wife shopping for a Father's-Day gift, can't
hear a Southern accent, and can't read on-screen text. Gemini can. So
instead of dumping Gemini's freeform visual narration into the engine
(which is prose ABOUT the ad, not ad copy — the awareness/demographic
regexes find nothing in it and the tiny motif z fills the void with
noise), we ask Gemini in ONE constrained call to decompose the ad into a
machine-readable brief: who it targets, the accent/region, the awareness
journey, the CTA occasion, the desires. v4_admix then consumes those
fields directly, tagged [vision], overriding the copy-tuned heuristics.

Hard rails: infer only from what is seen/heard; abstain ('unclear' /
confidence 'none') rather than guess; NEVER infer ethnicity or
nationality — 'likely_region' is a broad linguistic/market region
(e.g. 'US South', 'UK') used only as an income/market signal, and only
when the accent/dialect is reasonably clear.
"""

import json
import os

import v4_correlations
import v4_vision

# v4_gemini is the OPTIONAL non-local backend and is imported WHERE IT IS USED
# (inside distill()), not here. The local path — _local_brief: OCR + CLIP +
# insightface — is the only one a model-only distribution ships, and a
# module-level import made the whole module unloadable without the external
# client present, in a project whose stated constraint is no external LLMs in
# the model.

# Sample creatives for the __main__ self-test only. BERNAY_ADS overrides;
# the default is a sibling folder of the source tree, so nothing here depends
# on one machine's layout.
_ADS = os.environ.get(
    "BERNAY_ADS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "ad_creatives"))


HERE = os.path.dirname(os.path.abspath(__file__))

# Gemini won't inline arbitrarily large audio; a ~8 MB wav (~4 min at
# 16 kHz mono) is the ceiling we send for accent judgement. Longer ads
# fall back to lexical accent cues from the transcript (lower confidence).
MAX_INLINE_AUDIO = 8_000_000

AWARENESS_STAGES = ["unaware", "problem_aware", "solution_aware",
                    "product_aware", "most_aware"]

# the user's selling framework — the beats a persuasive ad moves through.
SELLING_STAGES = ["problem", "victim", "solution", "applying", "result",
                  "urgency", "memorable", "call_to_action"]

# OpenAPI-subset schema Gemini is constrained to (responseSchema).
BRIEF_SCHEMA = {
    "type": "object",
    "properties": {
        "spoken_transcript": {"type": "string"},
        "onscreen_text": {"type": "string"},
        "product": {"type": "string"},
        "subject": {"type": "string"},
        "avatar": {
            "type": "object",
            "properties": {
                "gender": {"type": "string",
                           "enum": ["female", "male", "mixed", "unclear"]},
                "age_range": {"type": "string"},
                "role": {"type": "string"},
                "relationship_context": {"type": "string"},
            },
        },
        "accent_region": {
            "type": "object",
            "properties": {
                "accent": {"type": "string"},
                "likely_region": {"type": "string"},
                "confidence": {"type": "string",
                               "enum": ["high", "medium", "low", "none"]},
            },
        },
        "cta": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "occasion": {"type": "string"},
            },
        },
        "awareness_journey": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": AWARENESS_STAGES},
                    "evidence": {"type": "string"},
                },
                "required": ["stage"],
            },
        },
        "selling_stages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "enum": SELLING_STAGES},
                    "evidence": {"type": "string"},
                },
                "required": ["stage"],
            },
        },
        "angle": {"type": "string"},
        "sophistication_level": {"type": "integer"},
        "core_desires": {"type": "array", "items": {"type": "string"}},
        "painpoints": {"type": "array", "items": {"type": "string"}},
        "visible_claims": {"type": "array", "items": {"type": "string"}},
        "price": {"type": "string"},
        "social_proof": {"type": "string"},
    },
    "required": ["product", "subject", "avatar", "awareness_journey", "cta"],
}

PROMPT = """\
You are a senior direct-response strategist decomposing ONE advertisement
into a structured brief. Use BOTH the visuals (frames) and the audio.

Fill the brief from what is actually shown and said. Where you are not
reasonably sure, leave the field empty or set confidence/gender to the
'unclear'/'none' option — do NOT guess to fill a slot.

Specific guidance:
- product = the BRANDED product being sold, exactly as named on the
  packaging / logo / on-screen text / spoken brand mention (e.g.
  'BioRoot Turmeric', 'Prostadine'). If the brand name is never actually
  shown or said, do NOT leave this empty — name the product CATEGORY in
  plain words instead (e.g. 'a turmeric joint supplement', 'a tinnitus
  relief supplement', 'a breast-health supplement'). This field is ALWAYS
  filled: an empty product is a failure, a category is the correct fallback.
- subject = the core affliction/problem/topic the ad is fundamentally
  about, named as PLAINLY and SPECIFICALLY as the ad itself does — e.g.
  'intestinal parasites (worms)', 'toenail fungus', 'belly fat', 'type 2
  diabetes', 'tinnitus'. If the ad explicitly names it (the copy says
  "they're worms"), use that exact concrete name. NEVER abstract a named
  cause into a vague symptom ('something moving inside you'): if the ad
  says what it is, the subject says what it is. This field is always
  filled — it is the one-line answer to "what is this ad about?"
- avatar = WHO the ad is trying to persuade to BUY (the target/buyer),
  which may differ from who appears on screen. If a spouse is shopping
  for a partner (e.g. a wife buying her husband a gift), the avatar is
  the BUYER; capture that in role + relationship_context.
- accent_region: judge the speaker's accent/dialect from the AUDIO and
  give a broad linguistic/market region only (e.g. 'US South', 'US
  Midwest', 'UK', 'Australia'). Set confidence honestly; use 'none' if
  there is no clear spoken voice. NEVER state ethnicity or nationality
  beyond this broad region — it is used only as a market/income signal.
- awareness_journey: the ORDERED Schwartz stages the ad moves through
  (an ad usually spans several — e.g. open problem_aware, close
  product_aware). One entry per stage actually present, each with a
  short evidence phrase. Allowed stages: unaware, problem_aware,
  solution_aware, product_aware, most_aware.
- cta.occasion: any seasonal/gifting hook ('Father's Day', 'Black
  Friday', 'back to school') if present, else empty.
- selling_stages: the ORDERED selling beats the ad actually performs,
  one entry per beat present, each with a short evidence phrase. Beats:
  problem (names a problem the viewer wants solved),
  victim (portrays the target as the sufferer of that problem),
  solution (sells the SOLUTION inside the product, not the product),
  applying (shows how the solution is used/applied),
  result (the benefit/outcome, value greater than price),
  urgency (a reason to act now — limited offer/scarcity),
  memorable (a brand/identity hook that makes it stick),
  call_to_action (tells the viewer to buy/act).
- sophistication_level: Schwartz market sophistication 1-5.
- core_desires: the underlying human desires the copy channels.
- Transcribe spoken_transcript and onscreen_text verbatim.
"""


def _normalize(brief):
    """Defensive defaults so consumers never KeyError, and clamp the
    awareness stages to the canonical ids."""
    brief.setdefault("avatar", {})
    brief.setdefault("accent_region", {})
    brief.setdefault("cta", {})
    journey = []
    seen = set()
    for entry in brief.get("awareness_journey", []) or []:
        st = (entry or {}).get("stage")
        if st in AWARENESS_STAGES and st not in seen:
            seen.add(st)
            journey.append(entry)
    brief["awareness_journey"] = journey
    stages, seen2 = [], set()
    for entry in brief.get("selling_stages", []) or []:
        st = (entry or {}).get("stage")
        if st in SELLING_STAGES and st not in seen2:
            seen2.add(st)
            stages.append(entry)
    brief["selling_stages"] = stages
    for k in ("core_desires", "painpoints", "visible_claims"):
        if not isinstance(brief.get(k), list):
            brief[k] = []
    return brief


def _conditions_block():
    """(enum_ids, prompt_text) for the canonical-condition classifier. The
    model maps its OWN read of the ad onto the KB conditions, so the cited
    demographic priors downstream attach only to conditions actually present
    rather than to keyword collisions. Empty when the KB isn't built yet."""
    choices = v4_correlations.condition_choices()
    if not choices:
        return [], ""
    ids = [c["id"] for c in choices]
    menu = "\n".join(f"  - {c['id']}: {c['name']} — {c['description']}"
                     for c in choices)
    text = (
        "canonical_conditions: from the list below, return only the "
        "condition(s) that are the ad's PRIMARY SUBJECT — the core problem "
        "this product is sold to solve. A condition merely mentioned, "
        "implied, or adjacent does NOT count: a parasite-cleanse ad that "
        "happens to mention bloating or fatigue is NOT a gut-health or "
        "fatigue ad — its subject is parasites, which is not on the list, so "
        "the answer is []. Judge by what the ad is CENTRALLY about, not "
        "surface word overlap. Return [] whenever the ad's true subject "
        "isn't listed — that is the correct answer, not a failure. Never "
        "stretch: 'sugar cravings' is not diabetes; 'worms living inside "
        "you' is not cost of living.\n" + menu)
    return ids, text


def _local_brief(frames, transcript=None):
    """No-Gemini fallback brief from LOCAL signals only — OCR'd on-screen
    text (v4_vision) + the whisper transcript. The structured fields
    (avatar, awareness_journey, canonical_conditions) are deliberately left
    empty so the downstream text engine fills them from the actual words,
    rather than having no model guess them. brief_to_text then renders the
    on-screen + spoken copy, which is what the engine decomposes."""
    frame_list = list(frames or [])
    # FLORENCE-2 (the model's real eyes, run in the .venv-vlm subprocess) READS
    # the creative — clean OCR of the on-image ad text + a scene caption — far
    # better than RapidOCR, which mangled stylized creatives ('MICRO-CURRENTS'
    # -> garbage). Falls back to RapidOCR when the venv/Florence isn't present.
    onscreen, scene_caption = "", ""
    try:
        import v4_florence
        fr = v4_florence.read_via_subprocess(frame_list)
        if fr:
            onscreen = (fr.get("ocr") or "").strip()
            scene_caption = (fr.get("caption") or "").strip()
    except Exception:  # noqa: BLE001
        pass
    if not onscreen:
        onscreen = v4_vision.describe_images(frame_list) or ""
    # VISUAL demographics OCR can't read: detect the people in the creative
    # and classify gender/age (insightface) -> avatar, which v4_admix turns
    # into a vision_prior ('two women in their 30s' -> female). This is the
    # 'see the image' capability for gender/age.
    avatar = {}
    try:
        faces = v4_vision.face_demographics(frame_list)
        if faces.get("gender") in ("female", "male"):
            avatar["gender"] = faces["gender"]
        if faces.get("age_range"):
            avatar["age_range"] = faces["age_range"]
        if faces.get("n"):
            avatar["n_people"] = faces["n"]
    except Exception:  # noqa: BLE001
        pass

    return _normalize({
        "onscreen_text": onscreen or "",
        "spoken_transcript": transcript or "",
        "subject": "",
        "product": "",
        "avatar": avatar,
        # Florence's scene description — fed to PRODUCT/PAINPOINT grounding in
        # v4_admix (not the awareness narrative). What the model SEES in prose.
        "scene_caption": scene_caption,
        # the creative frame paths — so v4_admix can give the model its NATIVE
        # VISUAL read (v4_vision_head: CLIP image -> learned product category),
        # not just OCR'd text. This is the model SEEING the ad, offline.
        "_frames": frame_list,
        # marks a NON-Gemini brief: it carries copy but NO structured
        # understanding, so v4_admix must keyword-scan it (painpoints,
        # conditions, demographics) instead of trusting empty brief fields.
        "_local": True,
    })


def distill(frames, audio_path=None, transcript=None):
    """frames: list of image paths (video keyframes or a single image).
    transcript: the spoken words, transcribed LOCALLY by whisper upstream
    (see v4_media.transcribe_audio). Whisper at all times: we never inline
    audio to Gemini, so `audio_path` is accepted for backward compatibility
    but no longer sent — accent is inferred from the transcript wording only
    (lower confidence, as the prompt instructs).

    The model also fills `canonical_conditions`: its OWN mapping of the ad
    onto the KB condition taxonomy, so downstream demographic priors are
    grounded in the conditions it actually saw — not regex keyword matches.
    Returns the normalized brief dict.

    With NO Gemini backend (local mode), there is no strategist model to
    fill the structured fields, so we degrade to a LOCAL brief: the OCR'd
    on-screen text + the whisper transcript become the ad's copy and the
    downstream text engine derives avatar/awareness/conditions from the
    words — see _local_brief."""
    if v4_vision.backend() != "gemini":
        return _local_brief(frames, transcript)
    cond_ids, cond_text = _conditions_block()
    schema = BRIEF_SCHEMA
    if cond_ids:
        schema = dict(BRIEF_SCHEMA)
        schema["properties"] = dict(BRIEF_SCHEMA["properties"])
        schema["properties"]["canonical_conditions"] = {
            "type": "array", "items": {"type": "string", "enum": cond_ids}}
    import v4_gemini                     # optional backend; see module header
    parts = [{"text": PROMPT}]
    if cond_text:
        parts.append({"text": cond_text})
    n_context = len(parts)
    for f in frames or []:
        if f and os.path.exists(f):
            parts.append(v4_gemini._inline(f))
    if transcript:
        parts.append({"text": "Audio transcript (whisper; judge accent from "
                      "word choice/spelling only, lower your confidence "
                      f"accordingly):\n{transcript[:8000]}"})
    if len(parts) == n_context:
        raise ValueError("distill: no frames or transcript to send")
    brief = _normalize(v4_gemini.generate_json(parts, schema))
    if cond_ids:                                   # clamp to valid ids
        valid = set(cond_ids)
        brief["canonical_conditions"] = [
            c for c in brief.get("canonical_conditions", []) or []
            if c in valid]
    return brief


def brief_to_text(brief):
    """Render the brief as a readable blob — for the session log and as
    the text fallback that still feeds the motif scorer."""
    av = brief.get("avatar", {})
    ar = brief.get("accent_region", {})
    cta = brief.get("cta", {})
    lines = []
    if brief.get("subject"):
        lines.append(f"SUBJECT: {brief['subject']}")
    if brief.get("product"):
        lines.append(f"PRODUCT: {brief['product']}")
    role = av.get("role") or av.get("relationship_context")
    who = ", ".join(x for x in [av.get("gender"), av.get("age_range"),
                                role] if x and x != "unclear")
    if who:
        # A FACE-CLASSIFIER avatar (the local, no-Gemini path — `_local`, or
        # insightface stamping n_people) is the person SHOWN, not a reasoned
        # read of who the ad sells to. Calling it "TARGET BUYER" put a claim
        # at the head of the copy that every downstream reader — motif scorer,
        # painpoint matcher, demographics — treats as the audience: a
        # skincare ad opening "PCOS affects 1 in 10 women" was analysed under
        # "TARGET BUYER: male, 55+" because a 55-year-old man appeared in one
        # sampled frame. Label it the way v4_admix already does everywhere
        # else (the deck's "presenter — not necessarily the buyer" line).
        face_read = bool(brief.get("_local")) or bool(av.get("n_people"))
        lines.append(f"ON-SCREEN PRESENTER: {who} (shown in the creative, "
                     f"not necessarily the buyer)" if face_read
                     else f"TARGET BUYER: {who}")
    if ar.get("likely_region") and ar.get("confidence") not in (None, "none"):
        lines.append(f"VOICE/REGION: {ar.get('accent','')} "
                     f"-> {ar['likely_region']} ({ar.get('confidence')})")
    if cta.get("occasion"):
        lines.append(f"OCCASION: {cta['occasion']}")
    if cta.get("action"):
        lines.append(f"CTA: {cta['action']}")
    journey = " -> ".join(e["stage"] for e in brief.get("awareness_journey", []))
    if journey:
        lines.append(f"AWARENESS JOURNEY: {journey}")
    if brief.get("canonical_conditions"):
        lines.append("CONDITIONS: " + ", ".join(brief["canonical_conditions"]))
    if brief.get("onscreen_text"):
        lines.append("[ON-SCREEN TEXT]\n" + brief["onscreen_text"])
    if brief.get("spoken_transcript"):
        lines.append("[TRANSCRIPT]\n" + brief["spoken_transcript"])
    return "\n".join(lines)


if __name__ == "__main__":
    import glob
    import sys
    if len(sys.argv) > 1:
        frames = [a for a in sys.argv[1:] if os.path.exists(a)]
    else:
        frames = sorted(glob.glob(os.path.join(HERE, "*.png")))[:4] or \
            sorted(glob.glob(os.path.join(_ADS, "*.png")))[:4]
    if not frames:
        raise SystemExit("usage: python v4_distill.py <frame.png> [more...]")
    print(f"distilling {len(frames)} frame(s): "
          f"{[os.path.basename(f) for f in frames]}\n")
    b = distill(frames)
    print(json.dumps(b, indent=2)[:2000])
    print("\n--- brief_to_text ---")
    print(brief_to_text(b))
