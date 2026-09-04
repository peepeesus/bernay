"""
v4_vision — local, FREE on-screen-text reader for ad frames, the drop-in
replacement for the Gemini visual call when no GEMINI_API_KEY is set.

Whisper already gives the SPOKEN words locally (v4_media.transcribe_audio);
this gives the words PRINTED on the frame — headlines, captions, claims,
prices, CTAs. Together they are the ad's actual copy, which the existing
text engine (v4_admix / v4_demographics / v4_correlations) decomposes. The
engine predates Gemini, so feeding it the OCR'd + spoken text is exactly the
pre-vision path — no new reasoning model needed.

OCR is a tiny ONNX model (RapidOCR): ~110 MB RAM, no torch, no GPU, no
network, ~4 s/frame on CPU — so it runs on a low-RAM box and costs nothing
per call. It deliberately does NOT caption scenes or people (that needs a
real VLM and more RAM than this box has); avatar age/gender come from the
words the OCR surfaces + the existing heuristics.

Backend selection (describe_images):
  V4_VISION_BACKEND = gemini | local | off      (env override)
  default: 'gemini' iff a Gemini key is resolvable, else 'local'.

Invoked as a subprocess (`python v4_vision.py img1 img2 ...`) so the OCR
session's RAM is released by the OS on exit and never coexists with bernay's
resident model — the same shell-out pattern v4_media uses for ffmpeg /
whisper / yt-dlp.
"""

import os
import re
import subprocess
import sys

import v4_env  # loads Downloads/.env into os.environ (GEMINI_API_KEY etc.)

v4_env.load()
HERE = os.path.dirname(os.path.abspath(__file__))


def _gemini_available():
    if os.environ.get("GEMINI_API_KEY"):
        return True
    return os.path.exists(os.path.join(HERE, "v4_gemini_key.txt"))


def backend():
    """Resolved visual backend: env override wins, else gemini if a key
    exists, else local OCR.

      gemini   — Gemini vision (paid; needs a key)
      local    — RapidOCR on-screen text, free; AUTO-escalates to the VLM
                 only when OCR is sparse AND a face is present (image-only
                 ads — the one case OCR can't read)
      florence — always run the local VLM caption (free but heavy/RAM)
      off       — no visual read
    """
    b = (os.environ.get("V4_VISION_BACKEND") or "").strip().lower()
    if b in ("gemini", "local", "florence", "off"):
        return b
    return "gemini" if _gemini_available() else "local"


def _vlm_model():
    """HF id of the local caption VLM. SmolVLM-256M (Idefics3) uses standard
    transformers classes so it works on transformers 5.x — unlike Florence-2,
    whose bundled remote code breaks on 5.x. Override with V4_VLM_MODEL."""
    return os.environ.get(
        "V4_VLM_MODEL", "HuggingFaceTB/SmolVLM-256M-Instruct").strip()


def _ocr_min_chars():
    """Below this many OCR chars an ad is treated as 'image-driven' (no
    meaningful on-screen copy) and becomes eligible for VLM escalation."""
    try:
        return int(os.environ.get("V4_OCR_MIN_CHARS", "40"))
    except ValueError:
        return 40


def _vlm_available():
    """True if the scene-VLM deps are importable. We use CLIP zero-shot via
    fastembed/onnxruntime (v4_scene), NOT a torch caption model: torch's
    autoregressive generate() SEGFAULTS on this py3.14 CPU build (SmolVLM/
    Florence-2), whereas CLIP is a single forward pass on the stable
    onnxruntime stack. The model downloads lazily on first use."""
    try:
        import importlib.util
        return importlib.util.find_spec("fastembed") is not None
    except Exception:  # noqa: BLE001
        return False


_DET = {}


def _detector():
    """insightface det_10g, loaded once per process.

    Was cv2's Haar cascade. On the installed **opencv-python 5.0.0** that is
    dead twice over: OpenCV 5 removed `cv2.CascadeClassifier` from the
    top-level namespace (and ships no `cv2.objdetect` module), and it no
    longer bundles `haarcascade_frontalface_default.xml` either. The old
    try/except guarded only `import cv2`, so every call raised AttributeError
    — count_faces() never returned a number, it always threw. insightface is
    already a dependency, already downloaded (buffalo_l), and is what
    _faces_worker uses for gender/age, so share it.
    """
    if "det" in _DET:
        return _DET["det"]
    try:
        from insightface.model_zoo import model_zoo
        det = model_zoo.get_model(
            os.path.join(_buffalo_dir(), "det_10g.onnx"),
            providers=["CPUExecutionProvider"])
        det.prepare(ctx_id=-1, input_size=(640, 640))
    except Exception:  # noqa: BLE001
        det = None
    _DET["det"] = det
    return det


def count_faces(paths):
    """Max faces detected across frames. 0 on any failure — callers use this
    only as a gate, so a miss must degrade, never raise."""
    det = _detector()
    if det is None:
        return 0
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return 0
    best = 0
    for p in paths:
        if not (p and os.path.exists(p)):
            continue
        try:
            im = cv2.imread(p)
            if im is None:
                continue
            floor = max(56, im.shape[0] // 10)     # ignore tiny logo "faces"
            bboxes, _ = det.detect(im, max_num=0, metric="default")
            n = sum(1 for b in bboxes
                    if (b[3] - b[1]) >= floor and (b[2] - b[0]) >= floor)
            best = max(best, n)
        except Exception:  # noqa: BLE001
            continue
    return best


def available():
    """True if the local OCR backend can run (RapidOCR importable)."""
    try:
        import importlib.util
        return importlib.util.find_spec("rapidocr_onnxruntime") is not None
    except Exception:  # noqa: BLE001
        return False


_TILE_H, _TILE_OVERLAP, _TALL_RATIO = 1400, 150, 2.0


def _tiles(path):
    """Slice a TALL screenshot into overlapping horizontal strips.

    RapidOCR resizes its input to a fixed size, so a full-page capture
    (measured: 1280x7357, aspect 5.7:1) lands its text at ~2-3 px tall and
    comes back as pure character noise —
        "1 4SI-KL 3 / 0 D A  F / v- 0 6 F / raSi-Km.R9 ..."
    That noise then carries stray non-Latin glyphs, which trips the
    _has_nonlatin() re-read below into a multilingual OCR pass that LOOPS
    ("L'Oreal Paris, France, France. France, Paris..."), and the loop is what
    reaches brief["onscreen_text"]. Nothing raises; the report just fills with
    garbage. Tiling the same image recovers the real copy.

    Returns [path] unchanged for normally-proportioned frames, so video
    keyframes and standard creatives pay nothing.
    """
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return [path]
    try:
        im = Image.open(path).convert("RGB")
    except Exception:  # noqa: BLE001
        return [path]
    W, H = im.size
    if H <= _TILE_H or W <= 0 or H / W < _TALL_RATIO:
        return [path]
    import tempfile
    out, y, step = [], 0, _TILE_H - _TILE_OVERLAP
    while y < H:
        fd, tmp = tempfile.mkstemp(suffix=".png", prefix="v4ocr_")
        os.close(fd)
        try:
            im.crop((0, y, W, min(y + _TILE_H, H))).save(tmp)
            out.append(tmp)
        except Exception:  # noqa: BLE001
            break
        y += step
    return out or [path]


def _ocr(paths):
    """In-process OCR worker — on-screen text across all frames, de-duped in
    reading order (video keyframes repeat the same headline, so a set keeps
    one copy). Tall full-page shots are tiled first (see _tiles)."""
    from rapidocr_onnxruntime import RapidOCR
    eng = RapidOCR()
    seen, lines = set(), []
    for p in paths:
        if not (p and os.path.exists(p)):
            continue
        tiles = _tiles(p)
        for t_path in tiles:
            try:
                res, _ = eng(t_path)
            except Exception:  # noqa: BLE001
                res = None
            for box in res or []:
                t = (box[1] or "").strip()
                key = t.lower()
                if t and key not in seen:
                    seen.add(key)
                    lines.append(t)
            if t_path != p:                       # clean up our own temp tile
                try:
                    os.remove(t_path)
                except OSError:
                    pass
    return "\n".join(lines)


VLM_PROMPT = ("Describe this advertisement's image factually: the people "
              "(how many, apparent gender and age range), what they are "
              "doing, the product, and any visible setting. Be concise.")


def _vlm_caption(paths):
    """SmolVLM description of ONE representative frame — the scene/people read
    OCR can't give ('two women applying face cream'). The caption text then
    flows into the engine, whose markers turn 'women' -> female etc. Heavy
    (transformers + ~0.5GB model + RAM), so it runs in a subprocess that
    exits to reclaim memory. Returns '' if deps/model/RAM are unavailable so
    the caller keeps the OCR result."""
    real = [p for p in paths if p and os.path.exists(p)]
    if not real:
        return ""
    pick = next((p for p in real if count_faces([p]) > 0), real[0])
    try:
        import torch
        from PIL import Image
        try:                                       # transformers 5.x
            from transformers import AutoModelForImageTextToText as _VLM
        except ImportError:                        # older transformers
            from transformers import AutoModelForVision2Seq as _VLM
        from transformers import AutoProcessor
    except Exception:  # noqa: BLE001
        return ""
    name = _vlm_model()
    proc = AutoProcessor.from_pretrained(name)
    model = _VLM.from_pretrained(name, torch_dtype=torch.float32).eval()
    img = Image.open(pick).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": VLM_PROMPT}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True)
    inp = proc(text=prompt, images=[img], return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(**inp, max_new_tokens=160, do_sample=False)
    # generate() returns prompt + answer; decode only the NEW tokens.
    new = ids[:, inp["input_ids"].shape[1]:]
    return proc.batch_decode(new, skip_special_tokens=True)[0].strip()


def _run_ocr(real):
    """OCR the frames in a short-lived subprocess (RAM reclaimed on exit);
    fall back to in-process only if the subprocess itself failed."""
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__),
                              *real], capture_output=True, text=True,
                             timeout=300)
        if out.returncode == 0:
            return (out.stdout or "").strip()
    except Exception:  # noqa: BLE001
        pass
    try:
        return _ocr(real)
    except Exception as e:  # noqa: BLE001
        return f"(local OCR unavailable: {e})"


def _run_vlm(real):
    """Scene-tag the frames via CLIP zero-shot (v4_scene -> fastembed/
    onnxruntime, its own short-lived subprocess so the model RAM is reclaimed
    and never coexists with bernay's resident model). Returns a short scene
    phrase ('talking to camera; outdoors; an older adult') or '' on any
    failure — the caller then keeps just the OCR text.

    Replaces the old torch caption-VLM, which SEGFAULTED in generate() on this
    py3.14 build. CLIP is a single forward pass, so it can't hit that crash."""
    try:
        import v4_scene
        return v4_scene.caption(real, topk=3)
    except Exception:  # noqa: BLE001
        return ""


# Non-Latin script ranges (Arabic, Devanagari, Cyrillic, Thai, kana, CJK,
# Hangul). Latin-1 accents (ä ö é) are NOT here, so European copy never
# triggers the heavy multilingual OCR — only a genuinely non-Latin creative
# that RapidOCR's ch/en models mangle does.
_NONLATIN = re.compile(
    "[؀-ۿऀ-ॿЀ-ӿ฀-๿"
    "぀-ヿ㐀-鿿가-힯]")


def _has_nonlatin(text):
    return bool(text) and _NONLATIN.search(text) is not None


def _run_ocr_multi(real):
    """Multilingual OCR (EasyOCR, v4_ocr_multi — its own subprocess). Reads
    non-Latin scripts the default OCR can't. '' on any failure so the caller
    keeps the (mangled) default OCR rather than crashing."""
    try:
        import v4_ocr_multi
        return v4_ocr_multi.run(real)
    except Exception:  # noqa: BLE001
        return ""


def describe_images(paths, prompt=None):
    """Backend-routed visual read of frames -> text.
      gemini   -> Gemini prose (scene + on-screen text)
      florence -> local VLM caption (falls back to OCR if VLM unavailable)
      local    -> OCR on-screen text; auto-escalates to the VLM ONLY when
                  OCR is sparse AND a face is present (image-only ad)
    `prompt` is honored only by the Gemini backend (call-site parity)."""
    b = backend()
    if b == "off":
        return ""
    if b == "gemini":
        import v4_gemini
        return (v4_gemini.describe_images(paths, prompt) if prompt
                else v4_gemini.describe_images(paths))
    real = [p for p in (paths or []) if p and os.path.exists(p)]
    if not real:
        return ""
    if b == "florence":
        return _run_vlm(real) or _run_ocr(real)
    # local: OCR first (cheap), escalate to CLIP scene-tagging for ANY
    # image-driven ad (sparse on-screen text) — the case OCR can't read.
    # No face requirement: a no-face creative (product/logo comparison, pet,
    # packaged shot) is exactly where scene context adds the most, and the
    # vocab has 'packaged product shot'/'text-only graphic' so it degrades
    # gracefully on abstract graphics. Bounded to sparse OCR so text-heavy ads
    # never pay the CLIP cost.
    ocr = _run_ocr(real)
    # Non-Latin script the default ch/en OCR mangled (e.g. Devanagari -> CJK
    # garble) -> re-read with the multilingual OCR so the real copy is
    # captured (and can then be translated downstream). Only fires when the
    # default output actually contains non-Latin chars, so English/European
    # ads never pay EasyOCR's cost.
    if _has_nonlatin(ocr):
        multi = _run_ocr_multi(real)
        if multi:
            ocr = multi
    if len(ocr) < _ocr_min_chars() and _vlm_available():
        cap = _run_vlm(real)
        if cap:
            return (f"{ocr}\n[scene] {cap}".strip() if ocr
                    else f"[scene] {cap}")
    return ocr


_AGE_BUCKETS = [(0, 17, "<18"), (18, 24, "18-24"), (25, 34, "25-34"),
                (35, 44, "35-44"), (45, 54, "45-54"), (55, 200, "55+")]


def _age_bucket(a):
    for lo, hi, name in _AGE_BUCKETS:
        if lo <= a <= hi:
            return name
    return ""


def _buffalo_dir():
    return os.path.join(os.path.expanduser("~"), ".insightface",
                        "models", "buffalo_l")


def _faces_worker(paths):
    """In-process: detect faces and classify gender/age via insightface
    (onnxruntime — stable on this py3.14 build, unlike the torch VLM).
    Returns (genders[list of 'female'/'male'], ages[list of int]).

    We load ONLY the detection + genderage onnx files directly via model_zoo
    instead of FaceAnalysis. FaceAnalysis onnx.load()s every file in the
    buffalo_l folder before honoring allowed_modules, so a single corrupt
    model (e.g. a truncated w600k_r50.onnx recognition file) crashes the
    whole worker — and we never use recognition/landmark anyway. Loading just
    the two needed models also drops ~320MB of RAM on this box."""
    import cv2
    from insightface.app.common import Face
    from insightface.model_zoo import model_zoo

    root = _buffalo_dir()
    det = model_zoo.get_model(os.path.join(root, "det_10g.onnx"),
                              providers=["CPUExecutionProvider"])
    det.prepare(ctx_id=-1, input_size=(640, 640))
    ga = model_zoo.get_model(os.path.join(root, "genderage.onnx"),
                             providers=["CPUExecutionProvider"])
    ga.prepare(ctx_id=-1)
    genders, ages = [], []
    for p in paths:
        if not (p and os.path.exists(p)):
            continue
        im = cv2.imread(p)
        if im is None:
            continue
        bboxes, kpss = det.detect(im, max_num=0, metric="default")
        for i in range(bboxes.shape[0]):
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bboxes[i, 0:4], kps=kps,
                        det_score=bboxes[i, 4])
            ga.get(im, face)
            genders.append("female" if face.sex == "F" else "male")
            ages.append(int(face.age))
    return genders, ages


def face_demographics(paths):
    """Visual demographics of the people in the creative — what OCR can't
    read ('two women in their 30s'). Runs insightface in a short-lived
    subprocess (RAM reclaimed; never coexists with bernay's model). Returns
    {'n', 'gender', 'age_range'}: gender is the dominant one (>=60% of faces)
    else 'mixed'; age_range is the median face's bucket. Empty dict if no
    faces / insightface unavailable."""
    real = [p for p in (paths or []) if p and os.path.exists(p)]
    if not real:
        return {}
    genders, ages = [], []
    try:
        out = subprocess.run([sys.executable, os.path.abspath(__file__),
                              "--faces", *real], capture_output=True,
                             text=True, timeout=300)
        if out.returncode == 0 and out.stdout.strip():
            import json
            d = json.loads(out.stdout)
            genders, ages = d.get("genders", []), d.get("ages", [])
    except Exception:  # noqa: BLE001
        return {}
    if not genders:
        return {}
    nf = len(genders)
    nfem = genders.count("female")
    if nfem >= 0.6 * nf:
        gender = "female"
    elif nfem <= 0.4 * nf:
        gender = "male"
    else:
        gender = "mixed"
    age_range = ""
    if ages:
        med = sorted(ages)[len(ages) // 2]
        age_range = _age_bucket(med)
    return {"n": nf, "gender": gender, "age_range": age_range}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--vlm":
        sys.stdout.write(_vlm_caption(args[1:]))
    elif args and args[0] == "--faces":
        import contextlib
        import json
        # insightface prints model-load noise to stdout; send it to stderr so
        # stdout carries ONLY the JSON the parent parses.
        with contextlib.redirect_stdout(sys.stderr):
            g, a = _faces_worker(args[1:])
        sys.stdout.write(json.dumps({"genders": g, "ages": a}))
    else:
        sys.stdout.write(_ocr(args))
