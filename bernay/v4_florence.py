"""
Florence-2 captioner — runs in the .venv-vlm (Python 3.12) where autoregressive
generate() does NOT segfault (unlike the system py3.14 build). Reads each
creative with Florence's OCR + detailed-caption tasks; the resulting text feeds
the niche classifier. This is the "vision model that READS" the gist-only CLIP
couldn't be.

Run with the venv python:
    .venv-vlm/Scripts/python.exe v4_florence.py --smoke creatives_gh/<id>.jpg
    .venv-vlm/Scripts/python.exe v4_florence.py --caption       # all creatives
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "v4_creatives.jsonl")
OUT = os.path.join(HERE, "v4_creative_florence.json")
MODEL_ID = "microsoft/Florence-2-base"
_M = {}


def _load():
    if "model" in _M:
        return _M["model"], _M["proc"]
    import torch
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from transformers import AutoProcessor, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_imports
    from unittest.mock import patch

    def _fixed_imports(filename):                 # Florence-2 hard-imports
        imp = get_imports(filename)               # flash_attn even on CPU;
        return [i for i in imp if i != "flash_attn"]  # strip it so it loads

    with patch("transformers.dynamic_module_utils.get_imports", _fixed_imports):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32)
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model.eval()
    _M.update(model=model, proc=proc)
    return model, proc


def _run_task(image, task, max_new=200):
    import torch
    model, proc = _load()
    inp = proc(text=task, images=image, return_tensors="pt")
    with torch.no_grad():                         # greedy (num_beams=1) is
        ids = model.generate(input_ids=inp["input_ids"],  # ~3x faster than
                             pixel_values=inp["pixel_values"],  # beams=3 on CPU
                             max_new_tokens=max_new, num_beams=1,
                             do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    return proc.post_process_generation(txt, task=task,
                                        image_size=(image.width, image.height))


def read_creative(path):
    """Florence OCR + caption -> one text blob describing the ad."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    cap = _run_task(img, "<DETAILED_CAPTION>", 128).get("<DETAILED_CAPTION>", "")
    ocr = _run_task(img, "<OCR>", 200).get("<OCR>", "")
    return f"{cap}\n{ocr}".strip()


def read_via_subprocess(paths):
    """SAFE to call from the system Python 3.14 (where Florence segfaults):
    shells out to the .venv-vlm 3.12 interpreter running this file's --read,
    returns {'caption':..., 'ocr':...} or None. Imports nothing heavy here.
    This is the bridge that puts Florence's read into Bernay's pipeline."""
    import subprocess
    venv = os.path.join(HERE, ".venv-vlm", "Scripts", "python.exe")
    real = [p for p in (paths or []) if p and os.path.exists(p)]
    if not os.path.exists(venv) or not real:
        return None
    try:
        out = subprocess.run([venv, os.path.abspath(__file__), "--read", real[0]],
                             capture_output=True, text=True, timeout=180)
        for line in out.stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
    except Exception:  # noqa: BLE001
        return None
    return None


def _rows():
    return [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]


FRESH_NICHES = {"collagen", "probiotic", "magnesium", "ashwagandha",
                "sea moss", "liver detox", "nerve pain", "toenail fungus",
                "testosterone booster", "sleep gummies", "nad supplement",
                "bloating relief"}


def caption_all(per_niche=None, only_niches=None):
    out = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    rows = _rows()
    if only_niches:
        rows = [r for r in rows if r["niche"] in only_niches]
    if per_niche:                                 # balanced subset to prove it
        bucket, picked = {}, []
        for r in rows:
            n = r["niche"]
            if bucket.get(n, 0) < per_niche:
                bucket[n] = bucket.get(n, 0) + 1
                picked.append(r)
        rows = picked
    done = 0
    for r in rows:
        aid = str(r["ad_id"])
        if aid in out:
            continue
        p = os.path.join(HERE, r["image"])
        if not os.path.exists(p):
            continue
        try:
            out[aid] = read_creative(p)
        except Exception as e:  # noqa: BLE001
            out[aid] = ""
            print(f"  {aid}: {e!r}", flush=True)
        done += 1
        json.dump(out, open(OUT, "w", encoding="utf-8"))   # save EVERY image —
        if done % 3 == 0:                                   # runs get killed early
            print(f"  captioned {done} ...", flush=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"))
    print(f"done: {len(out)} creatives captioned -> {os.path.basename(OUT)}")


if __name__ == "__main__":
    if "--read" in sys.argv:                      # structured read for the bridge
        from PIL import Image
        img = Image.open(sys.argv[-1]).convert("RGB")
        cap = _run_task(img, "<DETAILED_CAPTION>", 128).get("<DETAILED_CAPTION>", "")
        ocr = _run_task(img, "<OCR>", 200).get("<OCR>", "")
        print(json.dumps({"caption": cap.strip(), "ocr": ocr.strip()}))
    elif "--smoke" in sys.argv:
        path = sys.argv[-1]
        print("loading Florence-2 (first run downloads ~0.5GB) ...", flush=True)
        print("CAPTION+OCR:\n", read_creative(path))
    elif "--caption" in sys.argv:
        pn = None
        for a in sys.argv:
            if a.startswith("--per-niche="):
                pn = int(a.split("=")[1])
        only = FRESH_NICHES if "--fresh" in sys.argv else None
        caption_all(per_niche=pn, only_niches=only)
