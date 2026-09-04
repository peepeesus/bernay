"""
v4_embed — small CPU sentence-embedding encoder. Runs on fastembed/onnxruntime
(the SAME stable runtime as RapidOCR + insightface on this py3.14 box — no
torch, no GPU). Subprocess-isolated like v4_vision so the model RAM is
reclaimed on exit and never coexists with bernay's resident model.

This is the option-#3 "representation upgrade": a real 384-dim semantic
sentence embedding to replace/augment the 134k char-LM's weak motif vector as
the feature the recognition head learns from.

  encode(texts) -> list[list[float]]   (384-dim, BAAI/bge-small-en-v1.5)
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# The RETRIEVAL NET, and it is a ladder — bge-small is the SMALLEST rung
# fastembed offers (384-dim, 67 MB). Everything is env-tunable so the size can
# be measured (v4_retrieval_bench.py) rather than assumed; the defaults below
# are the long-standing behaviour and change nothing until a bigger rung is
# shown to win on the real-ad gate.
#
#   BAAI/bge-small-en-v1.5                384   67 MB   512 tok   (default)
#   BAAI/bge-base-en-v1.5                 768  210 MB   512 tok
#   BAAI/bge-large-en-v1.5               1024  1.2 GB   512 tok
#   snowflake/snowflake-arctic-embed-m-long
#                                         768  540 MB  8192 tok   <- long VSLs
#   thenlper/gte-large / mxbai-embed-large-v1
#                                        1024        512 tok
#
# SWAPPING THE MODEL INVALIDATES THE CACHED ANCHORS. v4_painpoint_anchors.json
# stores raw vectors with no model stamp, so a changed encoder silently
# compares new query vectors against old anchor vectors — cosine against a
# different space, which degrades quietly instead of erroring. Always rerun
# v4_semantic_painpoint.build_anchors() after changing V4_EMBED_MODEL.
MODEL = os.environ.get("V4_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# How much of an ad the encoder is allowed to see. 1800 chars is roughly the
# 512-token limit of the bge family; a long-context model can be given far
# more, which matters because a 7,000-char advertorial currently reaches the
# retriever with 74% of it cut off.
MAX_CHARS = int(os.environ.get("V4_EMBED_CHARS", "1800"))
BATCH = int(os.environ.get("V4_EMBED_BATCH", "4"))

_DIMS = {"BAAI/bge-small-en-v1.5": 384, "BAAI/bge-small-en": 384,
         "BAAI/bge-base-en-v1.5": 768, "BAAI/bge-base-en": 768,
         "BAAI/bge-large-en-v1.5": 1024,
         "snowflake/snowflake-arctic-embed-m-long": 768,
         "snowflake/snowflake-arctic-embed-l": 1024,
         "thenlper/gte-base": 768, "thenlper/gte-large": 1024,
         "mixedbread-ai/mxbai-embed-large-v1": 1024,
         "jinaai/jina-embeddings-v2-base-en": 768,
         "nomic-ai/nomic-embed-text-v1.5": 768}
DIM = _DIMS.get(MODEL, 384)


def _worker(texts):
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name=MODEL)
    # Truncate long VSLs (the targeting signal is up front) and embed in small
    # batches: one big batch of mixed-length texts padded a huge attention
    # matrix and OOM'd this box (~1.8GB alloc). Batch=4 keeps it tiny.
    texts = [(t or "")[:MAX_CHARS] for t in texts]
    out = []
    for v in model.embed(texts, batch_size=BATCH):
        out.append(list(map(float, v)))
    return out


def encode(texts):
    """Embed a list of texts in a short-lived subprocess (RAM reclaimed)."""
    texts = list(texts)
    if not texts:
        return []
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--encode"],
                       input=json.dumps(texts), capture_output=True,
                       text=True, timeout=900)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"embed failed: {(p.stderr or '')[-800:]}")
    return json.loads(p.stdout)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--encode":
        import contextlib
        payload = json.loads(sys.stdin.read())
        # fastembed/loguru print progress to stdout -> send to stderr so stdout
        # carries ONLY the JSON the parent parses.
        with contextlib.redirect_stdout(sys.stderr):
            vecs = _worker(payload)
        sys.stdout.write(json.dumps(vecs))
    else:
        v = encode(["joint pain when I climb the stairs",
                    "back to school sale ends tonight",
                    "MENOBELLY? that stubborn midsection after menopause"])
        print(f"encoded {len(v)} texts, dim {len(v[0]) if v else 0}")
