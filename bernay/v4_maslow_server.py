"""
MASLOW's GPU service — the audience-and-market half of the model.

Two things live here because both want the GPU and both are Maslow's:

  IMAGE   SDXL-base + the DR style LoRA (v4_maslow_imagegen), loaded once at
          startup, serving generations off the warm pipeline.
  READER  a small English language model that reads ad copy and names the
          problem the BUYER FEELS and the cause the ad BLAMES — open set, in
          the buyer's own words, with no KB vocabulary to pick from.

The split is deliberate. SCHWARTZ critiques the ad as built — awareness stage,
market sophistication, PV = Desire x T, winner likelihood — and keeps its own
char backbone and heads. MASLOW asks who the ad is for and what they lack. The
reader belongs on this side.

Served here rather than through a general LLM runner so the desktop app starts,
owns and reports every arm of the model itself (see server.py
_start_imagegen_bg / /api/health) — nothing to install alongside it, nothing
that can silently be down.

Reader defaults, all overridable by env:
  V4_MASLOW_READER        allenai/OLMo-2-0425-1B-DPO
                          Fully open weights, data, training code AND
                          intermediate checkpoints — which is why this is the
                          DPO stage rather than -Instruct. Instruct is
                          SFT + DPO + RLVR-MATH, and that final reinforcement
                          stage on math problems is the last thing that shaped
                          its weights. Measured on the 28 locked ad cases:
                          DPO 20/28, SFT 17/28, Instruct 16/28. Skipping the
                          math stage is free, and only possible because AI2
                          publishes the checkpoints before it.
  V4_MASLOW_READER_QUANT  nf4 | int8 | bf16 | off
                          nf4 measured at 1.64GB vs bf16's 3.21GB, same speed
                          and no measurable quality cost — which is what leaves
                          room for SDXL on one 16GB card.

Run:  <gpu venv>/bin/python v4_maslow_server.py
Then open http://127.0.0.1:8799
"""
import io
import json
import os
import re
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import v4_maslow_imagegen as maslow

app = FastAPI(title="Maslow — audience & market")

PIPE = None
DEV = None

# ---- the reader ------------------------------------------------------------
READER_ID = os.environ.get("V4_MASLOW_READER", "allenai/OLMo-2-0425-1B-DPO")
READER_QUANT = os.environ.get("V4_MASLOW_READER_QUANT", "nf4").lower()
# The LoRA that teaches the reader to pass the loop's own critics — trained
# by v4_reader_sft on nothing but critic verdicts. Resolved against THIS FILE,
# never the cwd: the app chdirs to BERNAY_ROOT before starting this service.
# ON-POLICY adapter: its revise targets were sampled from the adapter that was
# actually serving, not from the base model. Same-harness on 300 held-out ads —
# first pass 89.3% vs the off-policy r3's 86.3%, final 92.0% vs 90.7%, and 32
# rejections instead of 41. Note what did NOT improve: the revise FIX rate fell
# (25% vs 32%) on a smaller, harder residual. On-policy data made the reader
# wrong less often; it did not make it better at recovering.
READER_ADAPTER = os.environ.get("V4_MASLOW_READER_ADAPTER",
                                "v4_reader_lora_onpolicy")
if READER_ADAPTER and not os.path.isabs(READER_ADAPTER):
    READER_ADAPTER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  READER_ADAPTER)
READER = {"model": None, "tok": None, "state": "off", "id": READER_ID,
          "adapter": "none"}
READ_LOCK = threading.Lock()

# The prompt names its JSON keys explicitly. Without that both OLMo and Qwen
# answer the PROSE instead of the example and emit {"problem":..,"cause":..} —
# correct reads that a parser keyed only on painpoint/angle throws away.
READ_INSTR = (
    "Read the ad and fill in these two JSON keys, exactly as named:\n"
    '  "painpoint" = the problem the BUYER FEELS, 2-5 words, in the buyer\'s '
    "own words. Never the product or the ingredient.\n"
    '  "angle"     = the CAUSE the ad BLAMES for that problem. "" if none.\n'
    "Answer with one line of JSON and nothing else.\n\n"
    "EXAMPLE AD:\n"
    "Sick of knees that ache every morning? BioRoot's turmeric complex "
    "targets the joint inflammation behind stiffness.\n"
    "EXAMPLE ANSWER:\n"
    '{"painpoint": "aching stiff knees", "angle": "joint inflammation"}\n\n'
    "NOW THIS AD:\n")

def build_read_content(copy, correction="", previous=""):
    """The exact user turn the reader sees — shared by the /read endpoint and
    the SFT trainer (v4_reader_sft.py). Train on a string the server never
    sends and the adapter learns a prompt that does not occur in production."""
    content = READ_INSTR + (copy or "")[:5000]
    if (correction or "").strip():
        # Imperative and concrete. A 1B reader treats "say whose problem it
        # is" as commentary and returns the same answer; it needs to be told
        # that the painpoint VALUE is wrong and must be rewritten.
        content += ("\n\nSTOP. Your previous answer was WRONG and was "
                    "rejected for this reason:\n" + correction.strip())
        if (previous or "").strip():
            # Show the rejected answer back. "Do not repeat your previous
            # answer" is only actionable if the model can see what it was.
            content += ("\nYour rejected answer was: " + previous.strip())
        content += ("\n\nRewrite the \"painpoint\" value so it fixes that "
                    "exact problem. Do not repeat your previous answer. "
                    "Output one line of JSON only.")
    return content



_PP_KEYS = ("painpoint", "problem", "pain_point")
_AN_KEYS = ("angle", "cause", "mechanism")


def _pick(d, keys):
    for k in keys:
        for actual in d:
            if str(actual).strip().lower().replace("_", " ") == \
                    k.replace("_", " ") and d[actual]:
                return str(d[actual]).strip()
    return ""


def _parse_read(txt):
    """JSON first, then labelled lines. A model answering `problem: dry itchy
    skin` read the ad just as well as one that got the braces right."""
    txt = re.sub(r"(?s)<think>.*?</think>", " ", txt)
    m = re.search(r"\{[^{}]*\}", txt, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            if isinstance(d, dict):
                return _pick(d, _PP_KEYS), _pick(d, _AN_KEYS)
        except json.JSONDecodeError:
            pass
    pp = re.search(r'"?(?:%s)"?\s*[:=]\s*"?([^"\n,}]+)' % "|".join(_PP_KEYS),
                   txt, re.I)
    an = re.search(r'"?(?:%s)"?\s*[:=]\s*"?([^"\n,}]+)' % "|".join(_AN_KEYS),
                   txt, re.I)
    return (pp.group(1).strip() if pp else "",
            an.group(1).strip() if an else "")


def _load_reader():
    """Load the reader AFTER the image pipeline, on its own thread — a reader
    that fails to load must never cost Maslow its image half."""
    if READER_QUANT == "off":
        READER["state"] = "disabled"
        return
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        READER["state"] = "loading"
        kw = {"device_map": "cuda" if torch.cuda.is_available() else "cpu"}
        if READER_QUANT in ("nf4", "int8") and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = (
                BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_use_double_quant=True)
                if READER_QUANT == "nf4"
                else BitsAndBytesConfig(load_in_8bit=True))
        else:
            kw["dtype"] = torch.bfloat16
        READER["tok"] = AutoTokenizer.from_pretrained(READER_ID)
        model = AutoModelForCausalLM.from_pretrained(READER_ID, **kw)
        if READER_ADAPTER and READER_ADAPTER.lower() not in ("off", "none"):
            # A configured adapter that will not load must be VISIBLE. The
            # reader still comes up on the base weights — but /health says
            # which one is answering, so a silently untuned reader cannot be
            # mistaken for the tuned one.
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, READER_ADAPTER)
                READER["adapter"] = os.path.basename(READER_ADAPTER)
            except Exception as ae:  # noqa: BLE001
                READER["adapter"] = "error: %s: %s" % (type(ae).__name__, ae)
                print("[maslow-server] reader adapter FAILED: %s"
                      % READER["adapter"], flush=True)
        READER["model"] = model.eval()
        READER["state"] = "ready"
        print("[maslow-server] reader ready: %s (%s) adapter=%s"
              % (READER_ID, READER_QUANT, READER["adapter"]), flush=True)
    except Exception as e:  # noqa: BLE001 — image gen must survive this
        READER["state"] = f"error: {type(e).__name__}: {e}"
        print(f"[maslow-server] reader failed: {READER['state']}", flush=True)
# The diffusers pipeline's scheduler carries mutable step state across a
# generation (self.step_index etc.) — it is NOT thread-safe. FastAPI runs
# sync def endpoints in a threadpool, so two overlapping /generate calls
# (e.g. the automatic avatar+painpoint pair, or a retried request) corrupt
# each other's scheduler state mid-diffusion (IndexError: step out of
# bounds). One shared pipe -> one generation at a time, strictly serialized.
GEN_LOCK = threading.Lock()


@app.on_event("startup")
def _load():
    global PIPE, DEV
    print("[maslow-server] loading pipeline (this takes a while on first run)...", flush=True)
    PIPE, DEV = maslow._pipe()
    print(f"[maslow-server] ready on {DEV}", flush=True)
    # Reader comes up behind the pipeline, on its own thread: the app polls
    # this service for readiness, and image generation must not wait on a
    # language model — nor be taken down by one.
    threading.Thread(target=_load_reader, daemon=True).start()


class GenBody(BaseModel):
    painpoint: str = "joint pain arthritis"
    avatar: str = "smiling woman aged 55-64"
    product: str = "joint health collagen supplement"
    kind: str = "avatar"   # "avatar" (buyer portrait) | "painpoint" (problem scene)


PAGE = """<!doctype html>
<html><head><title>Maslow test</title>
<style>
body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
label{display:block;margin-top:12px;font-weight:600}
input{width:100%;padding:8px;font-size:14px;box-sizing:border-box}
button{margin-top:16px;padding:10px 20px;font-size:14px;cursor:pointer}
img{max-width:100%;margin-top:20px;border:1px solid #ccc}
#status{margin-top:12px;color:#666}
</style></head>
<body>
<h2>Maslow — DR creative generator</h2>
<label>Painpoint</label><input id="painpoint" value="joint pain arthritis">
<label>Avatar</label><input id="avatar" value="smiling woman aged 55-64">
<label>Product</label><input id="product" value="joint health collagen supplement">
<button onclick="gen()">Generate</button>
<div id="status"></div>
<img id="out" style="display:none">
<script>
async function gen(){
  const btn = document.querySelector('button');
  btn.disabled = true;
  document.getElementById('status').textContent = 'generating (~10-20s on GPU)...';
  const body = {
    painpoint: document.getElementById('painpoint').value,
    avatar: document.getElementById('avatar').value,
    product: document.getElementById('product').value,
  };
  const t0 = performance.now();
  const res = await fetch('/generate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  btn.disabled = false;
  if(!res.ok){ document.getElementById('status').textContent = 'error: ' + await res.text(); return; }
  const blob = await res.blob();
  const img = document.getElementById('out');
  img.src = URL.createObjectURL(blob);
  img.style.display = 'block';
  document.getElementById('status').textContent = `done in ${((performance.now()-t0)/1000).toFixed(1)}s`;
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/health")
def health():
    """Both arms, named. server.py's _imagegen_alive() used to fall back to
    'any answer means alive', which cannot tell a warm pipeline from one still
    loading — and now cannot see the reader at all."""
    return {"imagegen": "ready" if PIPE is not None else "loading",
            "device": str(DEV) if DEV else None,
            "reader": READER["state"], "reader_id": READER["id"],
            "reader_quant": READER_QUANT, "reader_adapter": READER["adapter"]}


class ReadBody(BaseModel):
    copy: str
    # What the critics rejected about the LAST answer. Carried as its own
    # field, not glued onto the ad — appended to the copy it reads as more ad,
    # which is exactly why the first revision attempt changed nothing.
    correction: str = ""
    # The rejected answer itself. "Do not repeat your previous answer"
    # is only actionable if the model can see what that answer was.
    previous: str = ""
    max_new_tokens: int = 220


@app.post("/read")
def read(body: ReadBody):
    """Name the problem the buyer FEELS and the cause the ad BLAMES.

    Open set: the answer is whatever the ad actually says, in the ad's own
    words, with no list of allowed painpoints to choose from. Grounding that
    name against the KB — and refusing to invent a mechanism when it does not
    resolve — is v4_reason_loop's job, not this endpoint's.
    """
    if READER["state"] != "ready":
        raise HTTPException(503, f"reader not ready: {READER['state']}")
    import torch
    tok, model = READER["tok"], READER["model"]
    content = build_read_content(body.copy, body.correction,
                                 body.previous)
    msgs = [{"role": "user", "content": content}]
    try:      # Qwen-family readers think by default and burn the whole budget
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         tokenize=False)
    enc = tok(prompt, return_tensors="pt").to(model.device)
    n_in = enc["input_ids"].shape[1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    with READ_LOCK, torch.no_grad():       # one GPU, one generation at a time
        out = model.generate(**enc, max_new_tokens=int(body.max_new_tokens),
                             do_sample=False, pad_token_id=pad)
    raw = tok.decode(out[0][n_in:], skip_special_tokens=True)
    painpoint, angle = _parse_read(raw)
    return {"painpoint": painpoint, "angle": angle, "raw": raw[:400],
            "seconds": round(time.time() - t0, 2), "model": READER["id"]}


@app.post("/generate")
def generate(body: GenBody):
    import torch
    decomp = {"painpoint": body.painpoint, "avatar": body.avatar, "product": body.product}
    prompt = (maslow.build_painpoint_prompt(decomp) if body.kind == "painpoint"
              else maslow.build_prompt(decomp))
    turbo = "turbo" in maslow.MODEL.lower()
    steps = int(os.environ.get("V4_MASLOW_STEPS", "4" if turbo else "28"))
    guidance = float(os.environ.get("V4_MASLOW_GUIDANCE", "0.0" if turbo else "6.0"))
    print(f"[maslow-server] {prompt}", flush=True)
    t0 = time.time()
    # Distinct seeds per kind: same painpoint scene text + same seed converges
    # to near-identical composition (pose/framing) regardless of prompt
    # wording differences — avatar and painpoint need to actually look like
    # two different photos, not a palette-swap of the same one.
    seed = {"avatar": 0, "painpoint": 1}.get(body.kind, 0)
    g = torch.Generator(DEV).manual_seed(seed)
    with GEN_LOCK:
        img = PIPE(prompt=prompt, negative_prompt=maslow.NEGATIVE_PROMPT,
                   num_inference_steps=steps, guidance_scale=guidance,
                   generator=g).images[0]
    # Hand the decode/attention workspace back to the driver. Torch's caching
    # allocator never releases freed blocks on its own, so without this the
    # process keeps its high-water mark reserved forever and sits on the whole
    # card while idle -- measured 15904MB reserved against 8138MB of live
    # weights, i.e. 7.6GB held but unused, with nothing left for the desktop.
    torch.cuda.empty_cache()
    print(f"[maslow-server] done in {time.time()-t0:.1f}s", flush=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8799)
