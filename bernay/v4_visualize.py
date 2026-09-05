"""
v4_visualize — MASLOW, Bernay's visual-presentation model.

Maslow is the visual counterpart to Schwartz-3: where Schwartz-3 DECOMPOSES an
ad (PV = Desire x T, v4_pv_engine), Maslow PRESENTS that decomposition. It takes
the structured result dict that `v4_admix.analyze(text, stack, return_result=True)`
returns and emits either an editable `.excalidraw` board or a self-contained HTML
slide deck: the equation with its live values, the Schwartz awareness funnel (the
targeted stages highlighted), the desire / chakra ranking, an avatar card (age /
gender / income with the sourced-or-withheld flag), and the painpoints. Open the
board at excalidraw.com or the VS Code Excalidraw extension — every shape stays
hand-drawn and drag-editable; open the deck in any browser.

No third-party deps (stdlib json/os/sys/math/argparse). The Excalidraw file
format is a plain JSON scene; we build the element dicts directly.

  to_excalidraw(result) -> dict          # the scene object
  write(result, path)                    # dump a .excalidraw file
  CLI:
    python v4_visualize.py result.json -o board.excalidraw
    cat result.json | python v4_visualize.py -o board.excalidraw
    python v4_visualize.py --demo -o demo.excalidraw     # built-in sample
"""
import argparse
import json
import math
import os
import sys

# ---- Excalidraw default palette --------------------------------------------
DARK = "#1e1e1e"
TRANSP = "transparent"
PAL = {  # name -> (background, stroke)
    "blue":   ("#a5d8ff", "#1971c2"),
    "green":  ("#b2f2bb", "#2f9e44"),
    "yellow": ("#ffec99", "#f08c00"),
    "red":    ("#ffc9c9", "#e03131"),
    "orange": ("#ffd8a8", "#e8590c"),
    "violet": ("#d0bfff", "#6741d9"),
    "indigo": ("#bac8ff", "#3b5bdb"),
    "teal":   ("#96f2d7", "#0ca678"),
    "gray":   ("#e9ecef", "#495057"),
}

# desire tag -> chakra colour band (root=primal/red ... crown=transcendent/violet)
_DESIRE_BAND = {
    "survival": "red", "safety": "red", "comfort": "red", "material": "red",
    "pleasure": "orange", "novelty": "orange", "connection": "orange",
    "power": "yellow", "status": "yellow", "control": "yellow",
    "ambition": "yellow", "recognition": "yellow",
    "love": "green", "belonging": "green", "harmony": "green",
    "expression": "blue", "clarity": "blue", "freedom": "blue",
    "insight": "indigo", "intellect": "indigo",
    "purpose": "violet", "transcendence": "violet", "growth": "violet",
    "mastery": "violet", "abundance": "teal", "discipline": "teal",
}

# canonical Schwartz funnel (matches v4_taxonomy awareness family)
STAGES = ["Unaware", "Problem", "Solution", "Product", "Most Aware"]
_STAGE_KEY = {  # substring -> canonical index, for matching journey strings
    "unaware": 0, "problem": 1, "solution": 2, "product": 3, "most": 4,
}

FONT_HAND = 1     # Virgil (hand-drawn)
CHAR_W = 0.55     # rough advance width as a fraction of font size


def _band(tag):
    return _DESIRE_BAND.get(str(tag).lower().strip(), "gray")


class Scene:
    """Accumulates Excalidraw elements with auto ids + deterministic seeds."""

    def __init__(self):
        self.els = []
        self._n = 0

    def _next(self):
        self._n += 1
        return self._n

    def _base(self, etype, x, y, w, h, stroke, bg, fill="solid",
              roundness=None, sw=2, rough=1):
        i = self._next()
        return {
            "id": f"el{i}", "type": etype,
            "x": float(x), "y": float(y),
            "width": float(w), "height": float(h),
            "angle": 0, "strokeColor": stroke, "backgroundColor": bg,
            "fillStyle": fill, "strokeWidth": sw, "strokeStyle": "solid",
            "roughness": rough, "opacity": 100, "groupIds": [],
            "frameId": None, "roundness": roundness,
            "seed": 100000 + i * 137, "version": 1,
            "versionNonce": 200000 + i * 251, "isDeleted": False,
            "boundElements": None, "updated": 1, "link": None,
            "locked": False,
        }

    def rect(self, x, y, w, h, color="gray", filled=False, sw=2, rough=1):
        bg, stroke = PAL.get(color, PAL["gray"])
        e = self._base("rectangle", x, y, w, h, stroke,
                       bg if filled else TRANSP, "solid",
                       {"type": 3}, sw, rough)
        self.els.append(e)
        return e

    def ellipse(self, x, y, w, h, color="green", filled=True, sw=2):
        bg, stroke = PAL.get(color, PAL["green"])
        e = self._base("ellipse", x, y, w, h, stroke,
                       bg if filled else TRANSP, "solid", None, sw)
        self.els.append(e)
        return e

    def text(self, x, y, s, size=20, color=DARK, align="left"):
        s = str(s)
        lines = s.split("\n")
        w = max((len(ln) for ln in lines), default=1) * size * CHAR_W
        h = len(lines) * size * 1.25
        e = self._base("text", x, y, w, h, color, TRANSP)
        e.update({
            "text": s, "fontSize": size, "fontFamily": FONT_HAND,
            "textAlign": align, "verticalAlign": "top", "containerId": None,
            "originalText": s, "lineHeight": 1.25,
            "baseline": int(size * 0.8),
        })
        self.els.append(e)
        return e

    def centered(self, cx, y, s, size=20, color=DARK):
        s = str(s)
        w = len(s) * size * CHAR_W
        return self.text(cx - w / 2, y, s, size, color, "center")

    def arrow(self, x, y, dx, dy, color=DARK, sw=2):
        e = self._base("arrow", x, y, abs(dx), abs(dy), color, TRANSP, "solid")
        e.update({
            "points": [[0, 0], [float(dx), float(dy)]],
            "lastCommittedPoint": None, "startBinding": None,
            "endBinding": None, "startArrowhead": None, "endArrowhead": "arrow",
        })
        self.els.append(e)
        return e

    def scene(self):
        return {
            "type": "excalidraw", "version": 2,
            "source": "bernay-v4_visualize",
            "elements": self.els,
            "appState": {"gridSize": None, "viewBackgroundColor": "#ffffff"},
            "files": {},
        }


# ---- value helpers ----------------------------------------------------------
def _pct(v):
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "--"


def _pvfmt(v):
    # PV is shown at 3 decimals to match v4_admix's analytical box exactly
    # (f"{PV:.3f}"), so the same ad reads identically in both models.
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "--"


def _journey_hits(journey):
    """Map free-form journey stage strings onto the 5 canonical indices."""
    hits = set()
    for stg in journey or []:
        low = str(stg).lower()
        for key, idx in _STAGE_KEY.items():
            if key in low:
                hits.add(idx)
    return hits


def _gender_symbol(g):
    low = str(g).lower()
    if low.startswith("m") and "male" in low and "fe" not in low:
        return "M"
    if "female" in low or low.startswith("f"):
        return "F"
    return "?"


# ---- the board --------------------------------------------------------------
def to_excalidraw(result):
    """Build an Excalidraw scene dict from a v4_admix.analyze result dict."""
    r = result or {}
    s = Scene()

    product = r.get("product") or "(product unknown)"
    pv, dsr, t = r.get("PV"), r.get("dsr"), r.get("T")
    gain = r.get("gain")

    # --- title ---------------------------------------------------------------
    s.text(40, 24, "Maslow", 28, DARK)
    s.text(40, 64, f"visual decomposition:  {product}", 18,
           PAL["gray"][1])


    # --- PV hero gauge -------------------------------------------------------
    gx, gy, gd = 60, 110, 150
    s.ellipse(gx, gy, gd, gd, "green", filled=True, sw=3)
    s.centered(gx + gd / 2, gy + gd / 2 - 30, _pvfmt(pv), 40)
    s.centered(gx + gd / 2, gy + gd / 2 + 18, "PV", 20, PAL["green"][1])

    # --- equation:  = Desire x T --------------------------------------------
    ey = gy + 35
    s.text(gx + gd + 24, ey + 18, "=", 40)
    dx0 = gx + gd + 70
    s.rect(dx0, ey, 130, 96, "blue", filled=True)
    s.centered(dx0 + 65, ey + 12, "Desire", 18, PAL["blue"][1])
    s.centered(dx0 + 65, ey + 40, _pct(dsr), 34)
    s.text(dx0 + 150, ey + 18, "x", 40)
    tx0 = dx0 + 196
    s.rect(tx0, ey, 130, 96, "yellow", filled=True)
    s.centered(tx0 + 65, ey + 12, "T (time)", 18, PAL["yellow"][1])
    s.centered(tx0 + 65, ey + 40, _pct(t), 34)
    if gain is not None:
        s.text(dx0, ey + 104, f"primal gain  x{_pct(gain)}   "
               f"(problem: {r.get('problem') or 'n/a'})", 14, PAL["gray"][1])

    # --- awareness funnel ----------------------------------------------------
    fy = 320
    s.text(60, fy - 30, "AWARENESS  (funnel - targeted stages filled)",
           18, DARK)
    hits = _journey_hits(r.get("awareness_journey"))
    bx, bw, bh, step = 60, 150, 58, 182
    for i, name in enumerate(STAGES):
        x = bx + i * step
        s.rect(x, fy, bw, bh, "green" if i in hits else "gray",
               filled=i in hits, sw=3 if i in hits else 2)
        s.centered(x + bw / 2, fy + bh / 2 - 11, name, 16,
                   DARK if i in hits else PAL["gray"][1])
        if i < len(STAGES) - 1:
            s.arrow(x + bw + 6, fy + bh / 2, step - bw - 12, 0, PAL["gray"][1])

    # --- desires / chakra ranking -------------------------------------------
    dyt = fy + bh + 60
    s.text(60, dyt - 30, "DESIRES channeled  (root -> crown)", 18, DARK)
    desires = (r.get("desires") or [])[:6]
    if not desires:
        s.text(80, dyt, "(none surfaced)", 16, PAL["gray"][1])
    for i, tag in enumerate(desires):
        ry = dyt + i * 34
        w = max(60, 280 * (1.0 - 0.13 * i))
        s.rect(80, ry, w, 24, _band(tag), filled=True, sw=1.5)
        s.text(80 + w + 12, ry + 1, str(tag), 16, DARK)

    # --- avatar card ---------------------------------------------------------
    ax, ay = 650, 110
    s.rect(ax, ay, 330, 180, "violet", filled=False, sw=2)
    s.text(ax + 16, ay + 12, "AVATAR", 18, PAL["violet"][1])
    g = _gender_symbol(r.get("gender"))
    s.text(ax + 16, ay + 46, f"sex     {g}   ({r.get('gender') or 'unclear'})",
           16)
    s.text(ax + 16, ay + 76, f"age     {r.get('age') or 'unclear'}", 16)
    s.text(ax + 16, ay + 106, f"stage   {r.get('life_stage') or 'unclear'}", 16)
    income = r.get("income") or "withheld"
    flag = "sourced" if r.get("income_by_age") or "$" in str(income) \
        else "withheld / no local figure"
    s.text(ax + 16, ay + 136, f"income  {income}", 16)
    s.text(ax + 16, ay + 158, f"        [{flag}]", 13, PAL["gray"][1])
    if r.get("presenter"):                  # who APPEARS on screen != the buyer
        s.text(ax + 16, ay + 190, f"presenter (on-screen): {r['presenter']}",
               13, PAL["gray"][1])

    # --- painpoints ----------------------------------------------------------
    px, py = 650, 320
    pains = (r.get("painpoints") or [])[:6]
    ph = 44 + max(1, len(pains)) * 26
    s.rect(px, py, 330, ph, "red", filled=False, sw=2)
    s.text(px + 16, py + 12, "PAINPOINTS", 18, PAL["red"][1])
    if not pains:
        s.text(px + 16, py + 44, "(none surfaced)", 15, PAL["gray"][1])
    for i, p in enumerate(pains):
        s.text(px + 16, py + 44 + i * 26, f"- {p}", 15)

    # --- footer --------------------------------------------------------------
    s.text(60, dyt + max(len(desires), 1) * 34 + 30,
           "PV = Desire x T  -  Desire is problem-grounded want x primal gain; "
           "T decays for slower-than-average solves.  Stats sourced or withheld, "
           "never fabricated.", 13, PAL["gray"][1])
    return s.scene()


def write(result, path):
    scene = to_excalidraw(result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2)
    return path


# ============================================================================
# HTML slide deck — self-contained, offline (no CDN), arrow-key navigable
# ============================================================================
def _esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


_HTML_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,Avenir,sans-serif;background:#0f1419;
 color:#e7ecf0;overflow:hidden}
.deck{height:100vh;width:100vw;position:relative}
.slide{position:absolute;inset:0;display:none;flex-direction:column;
 justify-content:center;align-items:center;padding:7vh 9vw;text-align:center}
.slide.active{display:flex;animation:fade .35s ease}
@keyframes fade{from{opacity:0;transform:translateY(12px)}to{opacity:1}}
.kicker{letter-spacing:.28em;text-transform:uppercase;font-size:.8rem;
 color:#7c8a99;margin-bottom:1.2rem}
h1{font-size:3.2rem;line-height:1.1;margin-bottom:.6rem}
h2{font-size:2.1rem;margin-bottom:2rem;font-weight:600}
.sub{color:#9aa7b2;font-size:1.15rem}
.gauge{width:230px;height:230px;border-radius:50%;display:flex;
 flex-direction:column;justify-content:center;align-items:center;margin:1rem 0;
 border:10px solid #2f9e44;background:#13351d;box-shadow:0 0 60px #2f9e4433}
.gauge .v{font-size:3.4rem;font-weight:700;line-height:1}
.gauge .l{color:#9fe3b0;letter-spacing:.2em;margin-top:.3rem}
.eq{display:flex;align-items:center;gap:1.4rem;flex-wrap:wrap;
 justify-content:center}
.eq .op{font-size:2.6rem;color:#7c8a99}
.tile{min-width:170px;padding:1.4rem 1.6rem;border-radius:16px;
 background:#172029;border:2px solid #2a3744}
.tile .tl{font-size:.85rem;letter-spacing:.15em;text-transform:uppercase;
 color:#8aa0b2;margin-bottom:.5rem}
.tile .tv{font-size:2.6rem;font-weight:700}
.note{margin-top:2rem;color:#7c8a99;font-size:.95rem;max-width:760px}
.funnel{display:flex;flex-direction:column;gap:12px;align-items:center;
 width:100%;max-width:760px}
.band{padding:1rem;border-radius:12px;font-size:1.2rem;font-weight:600;
 border:2px solid #2a3744;color:#8aa0b2;background:#141d25;transition:.2s}
.band.on{color:#0f1419;font-weight:700}
.bars{display:flex;flex-direction:column;gap:14px;width:100%;max-width:720px}
.row{display:flex;align-items:center;gap:14px}
.bar{height:30px;border-radius:8px}
.row .tag{font-size:1.15rem}
.cards{display:flex;gap:1.6rem;flex-wrap:wrap;justify-content:center;
 width:100%;max-width:1200px}
.card{flex:1;min-width:270px;background:#172029;border:2px solid #2a3744;
 border-radius:16px;padding:1.5rem;text-align:left}
.card h3{font-size:.9rem;letter-spacing:.18em;text-transform:uppercase;
 margin-bottom:.35rem}
/* one-line gloss under a card title — the Painpoint/Angle distinction is the
   whole point of the third card, so name it rather than assume it reads. */
.cardsub{color:#7c8a99;font-size:.78rem;margin-bottom:.8rem;font-style:italic}
.card.angle li:before{color:#f08c00}
.kv{display:flex;justify-content:space-between;padding:.45rem 0;
 border-bottom:1px solid #222d38;font-size:1.1rem}
.kv span:first-child{color:#8aa0b2}
.flag{color:#7c8a99;font-size:.85rem;margin-top:.4rem}
.card ul{list-style:none}
.card li{padding:.5rem 0;border-bottom:1px solid #222d38;font-size:1.1rem}
.card li:before{content:'\\25B8  ';color:#e03131}
.nav{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);
 display:flex;gap:10px;align-items:center;color:#5c6b78;font-size:.85rem}
.dot{width:9px;height:9px;border-radius:50%;background:#2a3744;cursor:pointer}
.dot.on{background:#2f9e44}
.hint{position:fixed;bottom:22px;right:26px;color:#3d4854;font-size:.8rem}
"""

_HTML_JS = """
let i=0;const sl=[...document.querySelectorAll('.slide')];
const dots=[...document.querySelectorAll('.dot')];
function go(n){i=Math.max(0,Math.min(sl.length-1,n));
 sl.forEach((s,k)=>s.classList.toggle('active',k===i));
 dots.forEach((d,k)=>d.classList.toggle('on',k===i));
 document.getElementById('cnt').textContent=(i+1)+' / '+sl.length;}
document.addEventListener('keydown',e=>{
 if(['ArrowRight','ArrowDown',' ','PageDown'].includes(e.key))go(i+1);
 if(['ArrowLeft','ArrowUp','PageUp'].includes(e.key))go(i-1);});
document.addEventListener('click',e=>{if(!e.target.closest('.nav'))go(i+1);});
dots.forEach((d,k)=>d.addEventListener('click',ev=>{ev.stopPropagation();go(k);}));
go(0);
"""


def to_html(result):
    """Self-contained HTML slide deck from a v4_admix.analyze result dict."""
    r = result or {}
    product = _esc(r.get("product") or "(product unknown)")
    model = _esc(r.get("model") or "Schwartz-4")
    pv, dsr, t, gain = r.get("PV"), r.get("dsr"), r.get("T"), r.get("gain")

    # slide 1 — title + PV gauge
    s1 = f"""<section class="slide"><div class="kicker">Maslow &middot;
 visual decomposition &middot; {model}</div>
 <h1>{product}</h1>
 <div class="gauge"><div class="v">{_pvfmt(pv)}</div>
 <div class="l">PV</div></div>
 <div class="sub">Perceived Value &nbsp;=&nbsp; Desire &times; T</div></section>"""

    # slide 2 — the equation
    gnote = (f"primal gain &times;{_pct(gain)} &nbsp;&middot;&nbsp; problem: "
             f"{_esc(r.get('problem') or 'n/a')}" if gain is not None else "")
    s2 = f"""<section class="slide"><div class="kicker">the equation</div>
 <div class="eq">
  <div class="tile" style="border-color:#2f9e44"><div class="tl">PV</div>
   <div class="tv" style="color:#69db7c">{_pvfmt(pv)}</div></div>
  <div class="op">=</div>
  <div class="tile" style="border-color:#1971c2"><div class="tl">Desire</div>
   <div class="tv" style="color:#74c0fc">{_pct(dsr)}</div></div>
  <div class="op">&times;</div>
  <div class="tile" style="border-color:#f08c00"><div class="tl">T (time)</div>
   <div class="tv" style="color:#ffd43b">{_pct(t)}</div></div>
 </div>
 <div class="note">{gnote}<br>Desire is a problem-grounded want amplified by
 the psychological hierarchy; T decays below 1 for slower-than-average
 solves.</div></section>"""

    # slide 3 — awareness funnel
    hits = _journey_hits(r.get("awareness_journey"))
    bands = []
    for k, name in enumerate(STAGES):
        w = 100 - k * 14
        if k in hits:
            bg, st = PAL["green"]
            style = f"width:{w}%;background:{bg};border-color:{st}"
            cls = "band on"
        else:
            style = f"width:{w}%"
            cls = "band"
        bands.append(f'<div class="{cls}" style="{style}">{name}</div>')
    s3 = f"""<section class="slide"><div class="kicker">awareness
 funnel &mdash; targeted stages highlighted</div>
 <h2>Awareness</h2><div class="funnel">{''.join(bands)}</div></section>"""

    # slide 4 — desires / chakra
    desires = (r.get("desires") or [])[:6]
    rows = []
    for k, tag in enumerate(desires):
        bg, st = PAL[_band(tag)]
        w = max(25, 100 - k * 13)
        rows.append(f'<div class="row"><div class="bar" '
                    f'style="width:{w}%;background:{bg};border:2px solid {st}">'
                    f'</div><div class="tag">{_esc(tag)}</div></div>')
    body = "".join(rows) or '<div class="sub">(none surfaced)</div>'
    s4 = f"""<section class="slide"><div class="kicker">desires channeled
 &mdash; root &rarr; crown</div>
 <h2>Desire / chakra ranking</h2><div class="bars">{body}</div></section>"""

    # slide 5 — avatar + painpoints
    income = r.get("income") or "withheld"
    sourced = bool(r.get("income_by_age")) or "$" in str(income)
    flag = "sourced (ACS)" if sourced else "withheld / no local figure"
    g = _gender_symbol(r.get("gender"))
    pains = (r.get("painpoints") or [])[:6]
    plist = "".join(f"<li>{_esc(p)}</li>" for p in pains) or "<li>(none)</li>"
    # ANGLE — the mechanism/cause the ad blames for the painpoint (diabetes,
    # neuropathy...). Its own card so it stops masquerading as what the buyer
    # suffers from. Only rendered when the split actually found one.
    angles = (r.get("painpoint_angles") or [])[:6]
    angle_card = ""
    if angles:
        alist = "".join(f"<li>{_esc(a)}</li>" for a in angles)
        angle_card = (
            '<div class="card angle" style="border-color:#f08c00">'
            '<h3 style="color:#ffc078">Angle</h3>'
            '<div class="cardsub">mechanism the ad blames</div>'
            f"<ul>{alist}</ul></div>")
    s5 = f"""<section class="slide"><div class="kicker">who &amp; why</div>
 <div class="cards">
  <div class="card" style="border-color:#6741d9"><h3 style="color:#b197fc">
   Avatar</h3>
   <div class="kv"><span>sex</span><span>{g} ({_esc(r.get('gender')
     or 'unclear')})</span></div>
   <div class="kv"><span>age</span><span>{_esc(r.get('age')
     or 'unclear')}</span></div>
   <div class="kv"><span>life stage</span><span>{_esc(r.get('life_stage')
     or 'unclear')}</span></div>
   <div class="kv"><span>income</span><span>{_esc(income)}</span></div>
   <div class="flag">[{flag}]</div>{(
   '<div class="kv"><span>presenter</span><span>' + _esc(r.get('presenter'))
   + '</span></div><div class="flag">on-screen &mdash; not necessarily '
   'the buyer</div>') if r.get('presenter') else ''}</div>
  <div class="card" style="border-color:#e03131"><h3 style="color:#ff8787">
   Painpoints</h3><div class="cardsub">what the buyer feels</div>
   <ul>{plist}</ul></div>{angle_card}
 </div></section>"""

    slides = [s1, s2, s3, s4, s5]

    dots = "".join('<div class="dot"></div>' for _ in slides)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Maslow &middot; {product}</title><style>{_HTML_CSS}</style></head>
<body><div class="deck">{''.join(slides)}</div>
<div class="nav"><span id="cnt"></span>{dots}</div>
<div class="hint">&larr; &rarr; or click to navigate</div>
<script>{_HTML_JS}</script></body></html>"""


def write_html(result, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_html(result))
    return path


# ============================================================================
# Terminal presentation — Maslow's IN-TERMINAL view. This is what makes Maslow
# DISTINCT from Schwartz at the console: Schwartz prints the analytical AD-MIX
# DECOMPOSITION (v4_admix), Maslow PRESENTS the same result dict as a clean
# visual summary (PV hero, the equation, the Schwartz funnel, the desire
# ranking, avatar + painpoint cards). Switching models visibly changes the
# screen, not just whether a browser tab opens.
# ============================================================================
_BAND_RGB = {  # PAL name -> truecolor (stroke tones, for the desire bars)
    "red": (224, 49, 49), "orange": (232, 89, 12), "yellow": (240, 140, 0),
    "green": (47, 158, 68), "teal": (12, 166, 120), "blue": (25, 113, 194),
    "indigo": (59, 91, 219), "violet": (103, 65, 217), "gray": (134, 140, 150),
}
_STAGE_RGB = {  # Schwartz awareness stages, root->crown feel
    "Unaware": (216, 57, 47), "Problem": (232, 99, 42),
    "Solution": (244, 165, 42), "Product": (63, 165, 82),
    "Most Aware": (79, 124, 201),
}


def render_terminal(result, model=None):
    """Print Maslow's terminal PRESENTATION of a v4_admix result dict. Kept
    deliberately distinct (violet accent, PV hero, funnel + bars) from the
    blue analytical decomposition so the two models read differently."""
    r = result or {}
    model = model or r.get("model") or "Schwartz-4"

    try:  # enable ANSI on legacy Windows consoles
        import ctypes
        h = ctypes.windll.kernel32
        h.SetConsoleMode(h.GetStdHandle(-11), 7)
    except Exception:  # noqa: BLE001
        pass

    def fg(rr, gg, bb):
        return f"\033[38;2;{rr};{gg};{bb}m"
    RST, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
    ACC = fg(177, 151, 252)        # Maslow violet (vs Schwartz blue)
    LBL = fg(154, 167, 178)
    HI = fg(240, 240, 240)
    RULE = "─" * 64

    product = r.get("product") or "(product unknown)"
    pv = _pvfmt(r.get("PV"))
    dsr, t = _pct(r.get("dsr")), _pct(r.get("T"))
    gain = r.get("gain")

    print()
    print(f"{ACC}{RULE}{RST}")
    print(f"  {ACC}{BOLD}MASLOW{RST}  {DIM}· visual presentation{RST}")
    print(f"  {HI}{BOLD}{product}{RST}")
    print(f"  {DIM}presenting the {model} decomposition{RST}")
    print()
    # PV hero + the equation, on one strong line
    print(f"   {ACC}{BOLD}{pv}{RST} {HI}PV{RST}   {LBL}=  Desire × T   =  "
          f"{HI}{dsr}{LBL} × {HI}{t}{RST}")
    if gain is not None:
        print(f"   {DIM}primal gain ×{_pct(gain)} · problem: "
              f"{r.get('problem') or 'n/a'}{RST}")
    print()
    # awareness funnel — targeted stages filled
    hits = _journey_hits(r.get("awareness_journey"))
    print(f"  {LBL}{BOLD}AWARENESS{RST} {DIM}(funnel — filled = "
          f"targeted){RST}")
    cells = []
    for i, name in enumerate(STAGES):
        rgb = _STAGE_RGB.get(name, (150, 150, 150))
        cells.append(f"{fg(*rgb)}{BOLD}▆ {name}{RST}" if i in hits
                     else f"{DIM}░ {name}{RST}")
    print("    " + "   ".join(cells))
    print()
    # desire ranking — coloured bars, root->crown
    desires = (r.get("desires") or [])[:6]
    print(f"  {LBL}{BOLD}DESIRES{RST} {DIM}(root → crown){RST}")
    if not desires:
        print(f"    {DIM}(none surfaced){RST}")
    for k, tag in enumerate(desires):
        rgb = _BAND_RGB.get(_band(tag), (134, 140, 150))
        bar = "█" * max(3, int(14 * (1.0 - 0.12 * k)))
        nm = f"{HI}{BOLD}{tag}{RST}" if k == 0 else f"{LBL}{tag}{RST}"
        print(f"    {fg(*rgb)}{bar}{RST}  {nm}")
    print()
    # avatar + painpoint cards, side by side (plain-width padding for ANSI)
    g = _gender_symbol(r.get("gender"))
    income = r.get("income") or "withheld"
    sourced = bool(r.get("income_by_age")) or "$" in str(income)
    flag = "sourced" if sourced else "withheld"
    av = [("sex", f"{g} ({r.get('gender') or 'unclear'})"),
          ("age", str(r.get("age") or "unclear")),
          ("stage", str(r.get("life_stage") or "unclear")),
          ("income", f"{income} [{flag}]")]
    if r.get("presenter"):       # who APPEARS on screen, not the target buyer
        av.append(("shown", f"{r['presenter']} [on-screen, not the buyer]"))
    pains = (r.get("painpoints") or [])[:6]
    print(f"  {ACC}{BOLD}AVATAR{RST}{' ' * 28}{fg(224, 49, 49)}{BOLD}"
          f"PAINPOINTS{RST}")
    for i in range(max(len(av), len(pains))):
        if i < len(av):
            k, v = av[i]
            left_plain = f"{k:<7}{v}"
            left = f"{LBL}{k:<7}{RST}{HI}{v}{RST}"
        else:
            left_plain, left = "", ""
        pad = max(2, 34 - len(left_plain))
        right = (f"{fg(224, 49, 49)}-{RST} {LBL}{pains[i]}{RST}"
                 if i < len(pains) else "")
        print(f"    {left}{' ' * pad}{right}")
    print()
    print(f"  {DIM}PV = Desire × T — Desire is a problem-grounded want × "
          f"primal gain; T decays for slow solves.{RST}")
    print(f"{ACC}{RULE}{RST}")


# ---- demo sample (BioRoot turmeric, the engine's reference creative) --------
DEMO = {
    "product": "BioRoot Turmeric (joint mobility)",
    "painpoints": ["Joint Pain & Arthritis", "Morning stiffness",
                   "Reduced mobility"],
    "age": "55+", "gender": "male", "income": "$57,108",
    "income_by_age": True, "life_stage": "older adult",
    "awareness_journey": ["Problem Aware", "Solution Aware"],
    "desires": ["survival", "comfort", "freedom", "harmony"],
    "problem": "chronic joint pain", "dsr": 0.55, "T": 0.74,
    "gain": 1.30, "top_primal": "survival", "PV": 0.41,
    "ethnicity": None,
}


def _load(args):
    if args.demo:
        return DEMO
    if args.input and args.input != "-":
        with open(args.input, encoding="utf-8") as f:
            return json.load(f)
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("no result JSON on stdin (use --demo or a file path)")
    return json.loads(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Bernay decomposition -> .excalidraw board or .html deck")
    ap.add_argument("input", nargs="?", help="result JSON file ('-' = stdin)")
    ap.add_argument("-o", "--out", default=None,
                    help="output path; format inferred from .html/.excalidraw")
    ap.add_argument("-f", "--format", choices=["excalidraw", "html"],
                    default=None, help="override output format")
    ap.add_argument("--html", action="store_true",
                    help="shortcut for --format html")
    ap.add_argument("--demo", action="store_true",
                    help="use the built-in BioRoot sample")
    args = ap.parse_args()

    # resolve format + output path
    fmt = args.format or ("html" if args.html else None)
    if fmt is None and args.out:
        fmt = "html" if args.out.lower().endswith(".html") else "excalidraw"
    fmt = fmt or "excalidraw"
    out = args.out or (f"bernay_board.{'html' if fmt == 'html' else 'excalidraw'}")

    result = _load(args)
    if fmt == "html":
        write_html(result, out)
        # self-validation: non-empty, well-formed-ish single HTML document
        doc = open(out, encoding="utf-8").read()
        assert doc.startswith("<!doctype html") and "</html>" in doc
        assert doc.count('class="slide"') == 5, "expected 5 slides"
        print(f"wrote {out}  (5-slide HTML deck) — open it in any browser "
              f"(arrow keys / click to navigate).")
    else:
        write(result, out)
        with open(out, encoding="utf-8") as f:
            scene = json.load(f)
        assert scene["type"] == "excalidraw"
        req = {"id", "type", "x", "y", "width", "height", "seed", "version"}
        for el in scene["elements"]:
            miss = req - el.keys()
            assert not miss, f"element {el.get('id')} missing {miss}"
        print(f"wrote {out}  ({len(scene['elements'])} elements)")
        print("open at excalidraw.com (File > Open) or the VS Code Excalidraw "
              "ext.")
