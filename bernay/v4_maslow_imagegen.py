"""
MASLOW image-gen — the visual-presentation model gains real text-to-image
generation. Turns a Bernay PV decomposition (avatar + painpoint + product +
archetype) into a mock ad-creative image via SDXL-Turbo (open-source, local,
GPU, 1-4 step fast inference — fits the 16GB RTX 5060 Ti). No external API.

  build_prompt(decomp) -> a DR-ad-creative prompt
  generate(decomp, out) -> PNG

Run: python v4_maslow_imagegen.py   (downloads SDXL-Turbo on first use, ~7GB)
"""
import os
import re
import sys

# Default to non-distilled SDXL-base: it renders far higher-quality DR-ad creatives
# than SDXL-Turbo AND is the only regime where our vanilla-loss style LoRA is valid
# (a Turbo-trained LoRA is a delta on Turbo's weights — garbage on base, haze on Turbo).
# Override with V4_MASLOW_MODEL=stabilityai/sdxl-turbo for fast 4-step drafts (no LoRA).
MODEL = os.environ.get("V4_MASLOW_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# painpoint -> visual scene shorthand — the RESOLVED state (this is the
# avatar/product-hero image: "buy this and be like this")
_SCENE = {
    "joint": "an active older adult walking pain-free outdoors",
    "skin": "a close-up of clear glowing healthy skin",
    "weight": "a fit confident person, flattering natural light",
    "energy": "an energetic person starting the day, bright morning",
    "menopause": "a serene confident woman in her 50s",
    "prostate": "a confident older man, relaxed at home",
    "blood sugar": "a healthy balanced meal, fresh vegetables",
    "gut": "a happy relaxed person, light and airy kitchen",
    "anxiety": "a calm relaxed person, soft warm lighting",
    "hair": "a person with full healthy shiny hair",
    "teeth": "a bright confident smile, clean white teeth",
    "dog": "a happy healthy energetic dog with its owner",
    "cat": "a content healthy cat at home",
}

# painpoint -> the PROBLEM itself (this is the painpoint-card image: the
# struggle a DR ad opens on, before the solution). Same keys as _SCENE on
# purpose — build_prompt and build_painpoint_prompt must describe opposite
# states of the SAME painpoint, not two versions of the same resolved scene.
_PROBLEM_SCENE = {
    "joint": ("an older adult wincing in discomfort, hand pressed against a "
             "sore knee, struggling to rise from a chair"),
    "skin": "a close-up of dry, irritated, blemished skin with visible redness",
    "weight": "a frustrated person tugging at ill-fitting clothes in front of a mirror",
    "energy": "an exhausted person slumped at a desk, head in hands, midday fatigue",
    "menopause": "a woman fanning herself, uncomfortable, restless, sweat on her brow",
    "prostate": "an older man standing anxiously outside a bathroom door at night",
    "blood sugar": "a person feeling dizzy and shaky, gripping a counter for support",
    "gut": "a person clutching their stomach in visible discomfort",
    "anxiety": "a person overwhelmed, head in hands, tense shoulders, restless",
    "hair": "a close-up of thinning hair with visible scalp and a hairbrush full of fallen hair",
    "teeth": "a close-up of someone wincing, hand on their jaw, tooth pain",
    "dog": "a lethargic, uncomfortable dog lying down, reluctant to move",
    "cat": "a listless, uncomfortable cat curled up, not eating",
}

# Reduces the base model's tendency to hallucinate stray malformed objects
# (canes, straps) and anatomy errors when nothing in the prompt steers away
# from them — applies to both images, not just the problem scene.
NEGATIVE_PROMPT = ("deformed, disfigured, extra limbs, extra fingers, "
                   "malformed hands, mutated hands, floating objects, "
                   "disconnected limbs, distorted anatomy, blurry, low "
                   "quality, watermark, text, logo")


def build_prompt(decomp):
    pain = (decomp.get("painpoint") or "").lower()
    scene = next((v for k, v in _SCENE.items() if k in pain), "a happy healthy person")
    avatar = decomp.get("avatar", "")
    product = decomp.get("product", "supplement")
    return (f"professional direct-response advertisement creative, {scene}, "
            f"{avatar}, promoting {product}, bright clean commercial product "
            f"photography, high detail, marketing hero image")


# The _PROBLEM_SCENE actor is written GENERICALLY ("a person", "an older
# adult") so one scene serves every avatar — which meant the painpoint image
# carried no gender or age at all and SDXL supplied its own prior: "an
# exhausted person slumped at a desk" renders a young man in an office. That
# card is captioned "what the buyer feels", so a female 35-44 analysis
# published a picture of a man as its own buyer. build_prompt already reads
# decomp['avatar'] and server.py sends the same body to both — this arm simply
# threw it away.
#
# Only the GENERIC nouns are substituted. The sex-specific scenes ("a woman
# fanning herself" for menopause, "an older man" for prostate) are anatomy, not
# avatar, and are left alone: a prostate ad bought by a wife must not render
# her outside the bathroom door. Scenes with no human actor (close-ups of skin
# or hair, the dog and cat scenes) match nothing and are untouched.
_ACTOR = re.compile(
    r"\b(an?)\s+((?:[a-z]+,\s+|[a-z]+\s+)*?)(?:older\s+)?(person|adult)\b",
    re.I)


def _article(text):
    """a/an for whatever actually follows — which is the scene's own adjective
    when it kept one ("an exhausted 39-year-old woman", not "a exhausted..."),
    so this reads the composed phrase rather than the age alone. Numbers go by
    how they are SAID: 'an 80-year-old', 'a 39-year-old'."""
    w = (str(text).strip().split() or [""])[0].lstrip("(,")
    if w[:1].isdigit():
        return "an" if w[0] == "8" or re.match(r"^(11|18)\b", w) else "a"
    return "an" if w[:1].lower() in "aeiou" else "a"


def avatar_actor(avatar):
    """'female aged 35-44' -> '39-year-old woman', or None when the analysis
    abstained on both dimensions (then the generic scene stands, which is
    honest: an unknown buyer should not be drawn as a specific one). 'unclear'
    is an abstention, not a description — never let it reach the prompt."""
    a = (avatar or "").lower()
    gender = ("woman" if "female" in a else
              "man" if re.search(r"\bmale\b", a) else None)
    span = re.search(r"(\d{1,3})\s*-\s*(\d{1,3})", a)
    if span:
        age = (int(span.group(1)) + int(span.group(2))) // 2
    else:
        open_ = re.search(r"(\d{1,3})\s*\+", a)
        age = int(open_.group(1)) + 5 if open_ else None
    if not gender and age is None:
        return None
    noun = gender or "person"
    return noun if age is None else f"{age}-year-old {noun}"


def build_painpoint_prompt(decomp):
    """A second, visually distinct image for the SAME analysis: not the buyer
    at their best (that's build_prompt's avatar portrait, the AFTER), but the
    problem's real-world moment — the BEFORE a DR ad opens on. Uses
    _PROBLEM_SCENE, the struggle-framed counterpart to _SCENE, so the two
    images describe opposite states of the same painpoint instead of two
    versions of the same resolved scene — and casts THIS analysis's buyer in
    it, so the sufferer shown is the person the ad is actually for."""
    pain = (decomp.get("painpoint") or "").lower()
    scene = next((v for k, v in _PROBLEM_SCENE.items() if k in pain),
                "a person struggling with an unresolved health concern")
    who = avatar_actor(decomp.get("avatar", ""))
    if who:
        def _cast(m):
            rest = f"{m.group(2)}{who}"
            return f"{_article(rest)} {rest}"
        scene = _ACTOR.sub(_cast, scene, count=1)
    return (f"editorial lifestyle photography, {scene}, candid authentic "
            f"moment, soft natural daylight, documentary style, shallow "
            f"depth of field, no text, no logos")


def _pipe():
    import torch
    from diffusers import AutoPipelineForText2Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32,
        variant="fp16" if dev == "cuda" else None)
    # SDXL's stock VAE ships force_upcast=True: diffusers casts it to fp32 for
    # every decode, so the 1024x1024 decoder runs in fp32 and its first upsample
    # stage alone allocates 512ch x 1024 x 1024 x 4B = 2.1GB per tensor. That
    # single spike is what took the card to 16310/16311MB and starved the
    # desktop compositor (monitor flicker). The fp16-fix VAE is the same
    # decoder rescaled to stay in fp16 range -- measured against the fp32
    # decode on an identical latent: PSNR 46.3dB, mean pixel diff 0.72/255,
    # zero NaNs. Visually lossless, and it removes the upcast entirely.
    if dev == "cuda":
        try:
            from diffusers import AutoencoderKL
            pipe.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
            pipe.vae.config.force_upcast = False
            print("[maslow] fp16-fix VAE (no fp32 upcast)", flush=True)
        except Exception as e:  # noqa: BLE001 -- stock VAE still works, just costs more
            print(f"[maslow] fp16-fix VAE unavailable ({e}); stock fp32-upcast VAE",
                  flush=True)
    pipe = pipe.to(dev)
    # trained-on-our-DR-creatives style LoRA (v4_maslow_train_lora.py). Two load notes:
    #  1) It's a peft adapter (get_peft_model().save_pretrained()), so it must be loaded
    #     via PeftModel.from_pretrained onto the unet. pipe.load_lora_weights() silently
    #     no-ops on this format (keys don't map) -> generation stays base.
    #  2) The LoRA is trained with vanilla diffusion loss, which is CORRECT for
    #     non-distilled SDXL-base but WRONG for SDXL-Turbo (distilled few-step): applied
    #     to Turbo it hazes the output. So it's OFF by default; enable it with
    #     V4_MASLOW_LORA_SCALE>0 when running on an SDXL-base checkpoint (V4_MASLOW_MODEL).
    # LoRA is valid ONLY on the base it was trained on (SDXL-base). Default it ON for
    # base, OFF for Turbo (a base-trained LoRA is garbage on Turbo's different weights).
    default_scale = "0" if "turbo" in MODEL.lower() else "1.0"
    scale = float(os.environ.get("V4_MASLOW_LORA_SCALE", default_scale))
    lora = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maslow_lora")
    if scale > 0 and os.path.isdir(lora):
        try:
            from peft import PeftModel
            from peft.tuners.lora import LoraLayer
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora).to(
                dev, torch.float16 if dev == "cuda" else torch.float32)
            for mod in pipe.unet.modules():
                if isinstance(mod, LoraLayer):
                    for a in list(mod.scaling):
                        mod.scaling[a] = scale
            print(f"[maslow] loaded DR-creative LoRA (scale={scale})", flush=True)
        except Exception as e:  # noqa: BLE001 — fall back to base
            print(f"[maslow] LoRA load failed ({e}); base model", flush=True)
    else:
        print("[maslow] base model (LoRA off; set V4_MASLOW_LORA_SCALE>0 on SDXL-base)",
              flush=True)
    if dev == "cuda":
        # Decode the latent in tiles instead of one 1024x1024 pass. Bounds the
        # decoder's peak regardless of output size. NOT enable_vae_slicing():
        # slicing only splits along the batch axis and we generate one image at
        # a time, so it is a no-op here.
        pipe.enable_vae_tiling()
        # ON by default. Keeps only the module currently running on the GPU, so
        # between calls the whole 6.6GB of SDXL sits in host RAM and the card
        # holds just the CUDA context + the nf4 reader. Measured on the live
        # service: 2,416MB idle and flat across generations, against 9,432MB
        # resident -- for +3.5s/image (17-18s vs 14s). The 4.9GB UNet crossing
        # PCIe once per call is the entire cost.
        # Set V4_MASLOW_OFFLOAD=0 to keep SDXL resident and buy the seconds back.
        #
        # Note: offloading ONLY the text encoders (they run once, then idle for
        # all 28 denoise steps) looks like the better trade and is NOT -- tried
        # and measured. It saves 1.5GB at idle, but moving them across PCIe every
        # call leaves ~1.6GB of live CUDA tensors behind and the process climbs
        # past the resident config (9,613 -> 10,481MB over two generations).
        if os.environ.get("V4_MASLOW_OFFLOAD", "1") == "1":
            pipe.enable_model_cpu_offload()
            print("[maslow] model CPU offload ON (low VRAM, slower)", flush=True)
    return pipe, dev


def generate(decomp, out="maslow_creative.png"):
    import torch
    pipe, dev = _pipe()
    prompt = build_prompt(decomp)
    turbo = "turbo" in MODEL.lower()               # Turbo: 4 steps / no CFG; base: 28 / CFG 6
    steps = int(os.environ.get("V4_MASLOW_STEPS", "4" if turbo else "28"))
    guidance = float(os.environ.get("V4_MASLOW_GUIDANCE", "0.0" if turbo else "6.0"))
    print(f"[{dev}] {MODEL} steps={steps} g={guidance} :: {prompt}", flush=True)
    g = torch.Generator(dev).manual_seed(0)
    img = pipe(prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
               generator=g).images[0]
    img.save(out)
    print(f"saved {out} ({img.size})", flush=True)
    return out


if __name__ == "__main__":
    demo = {"painpoint": "joint pain arthritis",
            "avatar": "smiling woman aged 55-64",
            "product": "joint health collagen supplement"}
    generate(demo, "maslow_creative_demo.png")
