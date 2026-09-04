# Bernay

A from-scratch marketing-psychology model. It reads an ad — copy, creative, or
a landing page — and decomposes it into the things that decide whether the ad
works: the painpoint it sells against, the mechanism it blames, the awareness
stage it addresses, the buyer it implies, and a perceived-value score.

Bernay runs on a CPU. There is no API call in the inference path.

## What it produces

For one ad:

| field | what it is |
| --- | --- |
| painpoint | the problem the ad sells against |
| angle | the mechanism the ad blames for that problem |
| awareness | where the reader sits on the Schwartz ladder, as a journey |
| avatar | sex, age, life stage, income — each able to abstain |
| PV | perceived value, `PV = Desire × T` |

Every field can come back empty. Abstention is a designed output: an ad with no
evidence for an age returns `unclear` rather than a confident guess, and a
painpoint the knowledge base cannot ground is reported as ungrounded rather
than snapped to the nearest catalogued row.

## Architecture

```
             ┌─ copy ──────────────┐
 ad input ───┼─ creative frames ───┼──► perception ──┐
             └─ landing page ──────┘                 │
                                                     ▼
                                   ┌──────── decomposition ────────┐
                                   │  painpoint · angle · awareness │
                                   │  avatar · desires · PV         │
                                   └───────────────┬───────────────┘
                                                   ▼
                              READ → GROUND → CHECK → REVISE → EMIT
```

**Backbone** — `gpt2_pv_v4.py`, a character-level GPT-2 variant trained from
scratch on a domain corpus. `v4_tokenizer.py` is its tokenizer.

**PV head** — `v4_pv_engine.py`. `PV = Dsr × T`, where desire is a learned
function of the decomposition and `T` decays with how long the ad has been
running relative to its cohort.

**Heads** — condition (`v4_condition_head.py`), awareness recognition
(`v4_recognition_head.py`), visual product category (`v4_vision_head.py`),
motif scoring (`v4_motif_scorer.py`), winner probability
(`v4_winner_score.py`).

**Decomposition** — `v4_admix.py` is the entry point. `v4_correlations.py`
holds the knowledge-base interface (painpoints, mechanisms, domains),
`v4_demographics.py` the audience inference, `v4_semantic_painpoint.py` the
grounding search.

**Reasoning loop** — `v4_reason_loop.py`. An open-set read, grounded against
the knowledge base, checked by critics (evidence, grounding, species, role,
singularity), revised, and only then emitted. A read that fails its critics is
discarded rather than published.

**Perception** — `v4_vision.py` reads creatives offline: OCR for on-screen
text, CLIP/SigLIP zero-shot scene tags, and face detection for the people
shown. `v4_distill.py` turns frames plus transcript into a structured brief.

### Two design rules worth stating

**Open-set, closed-grounding.** The painpoint is read from the copy in open
vocabulary, then *grounded* against the knowledge base. The knowledge base is a
grounding table, not the vocabulary. An ad about something uncatalogued is
named honestly and reported as ungrounded.

**Who is shown is not who buys.** A face detected in the creative is recorded
as the on-screen presenter, deliberately not the target buyer — a young
demonstrator is not the audience for the product they are demonstrating.
Faces on the advertiser's own landing page are treated as the stronger signal,
because that is the brand depicting its own customer.

## Performance

Measured on a Ryzen 5 7500F (6c/12t), 32 GB DDR5-6000, **no GPU**:

```
import + model load          1.6 s + 3.0 s   one-time
short pasted ad (1.1 KB)     1.6 s
20 KB advertorial            11.7 s
resident memory              211 MB loaded, 278 MB peak
```

It is single-threaded: 2 threads and 6 threads measure the same, so it behaves
the same on modest hardware.

## What is and is not in this repository

**Included** — the architecture, the heads, the decomposition, the reasoning
loop, and the perception path. 21 modules, ~430 KB of source.

**Not included** — the trained weights and the sourced knowledge base. As
published this is the model's code, not its parameters; you can read exactly
how it decides, and train your own. This mirrors how open-weight releases ship
without their training data.

Also deliberately absent: the desktop application, the ad-library and
landing-page scrapers, the training and evaluation harnesses, and the
regression corpus (which embeds third-party ad copy).

## Running it

```bash
pip install numpy onnxruntime rapidocr-onnxruntime fastembed
python -c "import bernay.v4_admix as a; s = a.load_stack(); a.analyze(open('ad.txt').read(), s)"
```

`load_stack()` expects a knowledge base and checkpoints, neither of which ships
here. Without them the code imports and the architecture is fully readable, but
it will not produce a decomposition until you supply your own.

Paths resolve from the module location and can be overridden:

| variable | meaning |
| --- | --- |
| `BERNAY_SRC` | the source tree |
| `BERNAY_ROOT` | the data/checkpoint tree |
| `V4_VISION_BACKEND` | `local` (default), `florence`, `off` |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
