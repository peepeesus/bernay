"""
v4_quiet — silence noisy third-party logging so it never bleeds into the bernay
TUI box. fastembed logs via LOGURU ('... WARNING | fastembed... Local file sizes
do not match the metadata'), and huggingface_hub emits download warnings; both
write to stderr mid-render and corrupt the fixed input box / splash. Import this
FIRST (before fastembed/transformers/HF) in any process that draws the TUI.
Importing it has the side effect of quieting; safe and idempotent.
"""

import logging
import os
import warnings

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore")

# loguru is fastembed's logger — disabling the 'fastembed' namespace kills the
# 'Local file sizes do not match the metadata' WARNING that lands in the box.
try:
    from loguru import logger as _loguru
    _loguru.disable("fastembed")
    _loguru.disable("")            # any other loguru-using dep, e.g. on download
except Exception:  # noqa: BLE001 — loguru not installed in this process: fine
    pass

# stdlib loggers used by HF / transformers / onnxruntime
for _name in ("fastembed", "huggingface_hub", "transformers", "onnxruntime",
              "PIL", "urllib3"):
    try:
        logging.getLogger(_name).setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001
        pass

try:
    from huggingface_hub.utils import logging as _hfl
    _hfl.set_verbosity_error()
except Exception:  # noqa: BLE001
    pass


def silence():
    """Re-apply (call after a lib re-enables its logger)."""
    try:
        from loguru import logger as _l
        _l.disable("fastembed")
    except Exception:  # noqa: BLE001
        pass
