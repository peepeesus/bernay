"""
Fetch transcripts for the goal's 7 YouTube ad videos and save each to
transcripts/<id>.txt. Robust across youtube-transcript-api versions
(the API surface changed across 0.6 -> 1.x); tries the known call
shapes and records failures instead of crashing the batch.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "transcripts")
os.makedirs(OUT, exist_ok=True)

URLS = [
    "https://www.youtube.com/watch?v=XlCQPnZ2gfA",
    "https://youtu.be/Wv10V5nUtzQ",
    "https://youtu.be/jkdv7obmdLQ",
    "https://youtu.be/-UPilLn_Gyo",
    "https://youtu.be/zASihbLpFNE",
    "https://www.youtube.com/watch?v=CV2M7oJjdRw",
    "https://www.youtube.com/watch?v=VXN7yll-b0c",
]


def video_id(url):
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else url


def fetch(vid):
    from youtube_transcript_api import YouTubeTranscriptApi as Y
    # try the 1.x instance API first, then the 0.6 classmethod API
    try:
        api = Y()
        for getter in ("fetch", "get_transcript"):
            if hasattr(api, getter):
                data = getattr(api, getter)(vid)
                return " ".join(s["text"] if isinstance(s, dict)
                                else s.text for s in data)
    except Exception:
        pass
    data = Y.get_transcript(vid)          # classmethod fallback
    return " ".join(s["text"] for s in data)


if __name__ == "__main__":
    results = {}
    for url in URLS:
        vid = video_id(url)
        try:
            text = fetch(vid).strip()
            path = os.path.join(OUT, f"{vid}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            results[vid] = len(text)
            print(f"OK   {vid}  {len(text):>6} chars", flush=True)
        except Exception as e:
            results[vid] = f"FAIL: {type(e).__name__}: {e}"
            print(f"FAIL {vid}  {type(e).__name__}: "
                  f"{str(e)[:80]}", flush=True)
    ok = sum(1 for v in results.values() if isinstance(v, int))
    print(f"\n{ok}/{len(URLS)} transcripts fetched -> {OUT}")
