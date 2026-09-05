import type { AnalyzeResult, Health } from './types'

// Same-origin when served by the Bernay API / desktop shell; explicit host
// when running under `vite dev` (proxy also covers this, belt & braces).
//
// Keyed on the dev port, NOT on 8756: sandbox instances (server/sandbox.py)
// serve this same bundle from 8757+, and pinning same-origin to 8756 made
// their UI call the REAL app instead of the sandbox it was served from.
const BASE =
  location.port === '5173' ? 'http://127.0.0.1:8756' : ''

export async function getHealth(): Promise<Health> {
  const r = await fetch(`${BASE}/api/health`)
  if (!r.ok) throw new Error(`health ${r.status}`)
  return r.json()
}

/** Maslow — the visual-presentation model. `html` (default) is a
 * self-contained slide deck meant to be opened inline. `excalidraw` is the
 * editable board — it can't auto-render in a webview, so the server sends
 * it as a download for excalidraw.com / a local Excalidraw instance.
 *
 * `token` MUST change whenever a new analysis lands. The deck is shown in an
 * <iframe>, and this URL used to be a constant — so React re-rendered with a
 * byte-identical `src`, the element was never remounted, and the browser
 * never re-requested it. The result: every later analysis left the FIRST
 * one's deck on screen ("I pasted in the link but it didn't update the
 * response", 2026-08-27). */
export function vizUrl(format: 'html' | 'excalidraw' = 'html',
                       token?: number | string): string {
  // `id` selects which analysis to render; `t` additionally defeats the
  // webview's HTTP cache (the URL is otherwise identical between runs).
  const t = token === undefined
    ? ''
    : `&id=${encodeURIComponent(String(token))}&t=${encodeURIComponent(String(token))}`
  return `${BASE}/api/viz?format=${format}${t}`
}

/** The interactive pipeline map — a static page in ui/public, so it ships in
 * the bundle and is served by whichever instance served this UI. */
export function architectureUrl(): string {
  return `${BASE}/architecture.html`
}

/** True when the input should stream progress (URL / local media path —
 * scrape + transcribe + distill can take minutes). */
function isMediaish(input: string): boolean {
  const s = input.trim()
  if (/\n/.test(s)) return false
  return (
    /^https?:\/\//i.test(s) ||
    /^[a-z]:[\\/]/i.test(s.replace(/^"|"$/g, '')) ||
    /\.(mp4|mov|webm|jpe?g|png|gif)$/i.test(s)
  )
}

/** `model` is the lens selected when Analyze was clicked ('Schwartz-4.5' |
 * 'Maslow') — it gates whether the backend also runs Maslow image-gen.
 * Schwartz and Maslow are separate models; analyzing under one must never
 * silently also produce the other's output. */
export function analyze(
  input: string,
  model: string,
  onProgress: (msg: string) => void,
): Promise<AnalyzeResult> {
  const s = input.trim()
  if (isMediaish(s)) {
    return new Promise((resolve, reject) => {
      const es = new EventSource(
        `${BASE}/api/analyze/stream?input=${encodeURIComponent(s)}&model=${encodeURIComponent(model)}`,
      )
      es.addEventListener('progress', (e) => {
        try {
          onProgress(JSON.parse((e as MessageEvent).data).message)
        } catch {
          /* ignore malformed progress */
        }
      })
      es.addEventListener('result', (e) => {
        es.close()
        resolve(JSON.parse((e as MessageEvent).data))
      })
      // Two very different failures arrive on the same event. A server-sent
      // `event: error` carries a payload describing what the pipeline hit. A
      // connection-level failure (non-200 — 503 while the model reloads, 422,
      // or a restarted server) arrives with NO data, because EventSource never
      // exposes the response body. Guessing there produced the useless
      // "is the Bernay API running?" — so ask /api/health instead of guessing.
      const fail = (e: Event) => {
        es.close()
        const data = (e as MessageEvent).data
        if (data) {
          try {
            reject(new Error(JSON.parse(data).message))
          } catch {
            reject(new Error(String(data)))
          }
          return
        }
        getHealth().then(
          (h) => reject(new Error(
            h.status === 'loading'
              ? 'the engine is still loading — try again in a moment'
              : h.status === 'error'
                ? `the engine failed to load: ${h.error ?? 'unknown error'}`
                : 'the analysis stream dropped before finishing — retry',
          )),
          () => reject(new Error('the Bernay API is not responding')),
        )
      }
      es.addEventListener('error', fail)
    })
  }
  onProgress('decomposing…')
  return fetch(`${BASE}/api/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: s, model }),
  }).then(async (r) => {
    if (!r.ok) {
      const body = await r.json().catch(() => null)
      throw new Error(body?.detail ?? `analyze failed (${r.status})`)
    }
    return r.json()
  })
}
