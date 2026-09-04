"""v4_vision_siglip — SigLIP as an alternative local vision encoder to try
against the documented CLIP-ViT-B-32 ceiling (~44-48% on ad-creative
product-category classification; see BERNAY.md /
bernay-native-visual-cognition memory). Same image_vector() contract as
v4_vision_cognition.py so it plugs into v4_vision_head.py's CV harness with
a different embedding cache, nothing else changed.

SigLIP (google/siglip-base-patch16-224) uses a sigmoid pairwise loss instead
of CLIP's softmax contrastive loss — trains on noisier web data at scale and
has shown stronger zero-shot/embedding quality than same-size CLIP on many
benchmarks. Worth testing head-to-head on THIS task since nothing about
sigmoid-vs-softmax loss obviously fixes the specific failure mode already
diagnosed (generic bottle shots where the category isn't visually present) —
this is an honest experiment, not an assumed win.
"""
import os

import v4_quiet  # noqa: F401 — silence HF logging before it loads

_MODEL_NAME = os.environ.get("V4_SIGLIP_MODEL", "google/siglip-base-patch16-224")
_M = {}


def _model():
    if "model" not in _M or _M.get("name") != _MODEL_NAME:
        import torch
        from transformers import SiglipModel, SiglipProcessor
        _M["proc"] = SiglipProcessor.from_pretrained(_MODEL_NAME)
        _M["model"] = SiglipModel.from_pretrained(_MODEL_NAME)
        _M["model"].eval()
        _M["name"] = _MODEL_NAME
        torch.set_num_threads(max(2, os.cpu_count() or 2))
    return _M["model"], _M["proc"]


def available():
    try:
        import importlib.util
        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)
    except Exception:  # noqa: BLE001
        return False


def image_vector(paths):
    """Mean-pooled, L2-normalized SigLIP image embedding over up to 4 frames.
    Same contract as v4_vision_cognition.image_vector — a torch.FloatTensor,
    or None if it can't run / no images."""
    import torch
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    paths = [p for p in (paths or []) if p and os.path.exists(p)][:4]
    if not paths:
        return None
    try:
        model, proc = _model()
        imgs = [Image.open(p).convert("RGB") for p in paths]
        inputs = proc(images=imgs, return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
        v = feats.mean(dim=0)
        return v / (v.norm() + 1e-8)
    except Exception:  # noqa: BLE001
        return None


if __name__ == "__main__":
    import sys
    imgs = sys.argv[1:]
    print("available:", available())
    if imgs:
        v = image_vector(imgs)
        print("image_vector:", None if v is None else tuple(v.shape))
