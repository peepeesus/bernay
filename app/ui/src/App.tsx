import type { CSSProperties } from 'react'
import { useCallback, useEffect, useRef, useState } from 'react'
import { analyze, architectureUrl, getHealth, vizUrl } from './api'
import { desireMeta } from './desires'
import { applyTheme, currentTheme, type ThemeChoice } from './theme'
import {
  AvatarCard,
  AwarenessSpread,
  FamilyChips,
  Painpoints,
  PVHero,
  SellingBeats,
  SophisticationSpread,
} from './components'
import { Button } from './design-system/components/Button'
import { Caption } from './design-system/components/Caption'
import { Card } from './design-system/components/Card'
import { Dropdown } from './design-system/components/Dropdown'
import { PasteZone } from './design-system/components/PasteZone'
import { SketchAccent } from './design-system/components/SketchAccent'
import { Wordmark } from './design-system/components/Wordmark'
import type { AnalyzeResult, Health, HistoryEntry } from './types'

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [progress, setProgress] = useState('')
  const [error, setError] = useState('')
  const [result, setResult] = useState<AnalyzeResult | null>(null)
  // Which analysis the deck should show. The Maslow deck is an <iframe>, and
  // `result` changing does NOT reload it — only a changed src/key does. Using
  // the server's result_id (rather than a bare counter) also makes restoring
  // a History entry restore ITS deck.
  const [vizSeq, setVizSeq] = useState(0)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const [modelLens, setModelLens] = useState('Schwartz-4.5')
  const [showArch, setShowArch] = useState(false)
  const [theme, setTheme] = useState<ThemeChoice>(currentTheme)
  const nextId = useRef(1)

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next: ThemeChoice = t === 'dark' ? 'light' : 'dark'
      applyTheme(next)
      return next
    })
  }, [])

  // Close the architecture overlay on Escape.
  useEffect(() => {
    if (!showArch) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowArch(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [showArch])

  // F11 -> native window fullscreen inside the desktop shell. A real browser
  // tab already handles F11 itself, so only intercept when the pywebview
  // bridge (window.pywebview.api, wired in desktop.py) is actually present —
  // otherwise this would break F11 for `vite dev` in an ordinary tab.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'F11') return
      const api = (window as unknown as {
        pywebview?: { api?: { toggle_fullscreen?: () => void } }
      }).pywebview?.api
      if (!api?.toggle_fullscreen) return
      e.preventDefault()
      api.toggle_fullscreen()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  // Health poll. Keeps running after the model is ready, slowly: the server
  // restarts underneath the window whenever the Python modules are edited, and
  // a frozen "ready" badge would leave Analyze enabled against a loading
  // server — which fails as an opaque "stream failed" instead of "loading".
  useEffect(() => {
    let stop = false
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      try {
        const h = await getHealth()
        if (stop) return
        setHealth(h)
        timer = setTimeout(tick, h.status === 'ready' ? 10000 : 2000)
      } catch {
        if (stop) return
        setHealth({
          ok: false, status: 'error', model: '?', params: 0,
          error: 'API unreachable',
        })
        timer = setTimeout(tick, 4000)
      }
    }
    tick()
    return () => { stop = true; clearTimeout(timer) }
  }, [])

  const run = useCallback(async () => {
    const s = input.trim()
    if (!s || busy) return
    setBusy(true)
    setError('')
    setProgress('sending…')
    try {
      const r = await analyze(s, modelLens, setProgress)
      setResult(r)
      // force the deck iframe to refetch, and point it at THIS analysis
      setVizSeq((n) => r.result_id ?? n + 1)
      const label =
        r.input_kind !== 'text'
          ? `[${r.input_kind}] ${s.slice(0, 40)}`
          : s.replace(/\s+/g, ' ').slice(0, 48)
      setHistory((h) => [
        { id: nextId.current++, label, at: new Date().toLocaleTimeString(), result: r },
        ...h,
      ])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      setProgress('')
    }
  }, [input, busy, modelLens])

  // The real filesystem path of a dropped file is resolved natively in
  // desktop.py (WebView2 never hands it to page JS) and pushed back in through
  // window.__bernaySetInput. Expose that hook here so the drop lands in the
  // input as authoritative React state.
  useEffect(() => {
    const w = window as unknown as { __bernaySetInput?: (p: string) => void }
    w.__bernaySetInput = (p: string) => setInput(p)
    return () => { delete w.__bernaySetInput }
  }, [])

  // Browser-side drop handler. In the desktop shell the path arrives via the
  // native bridge above; this only catches the pywebviewFullPath case for
  // completeness and, under `vite dev` in a plain browser, keeps the drop from
  // navigating away. A plain browser can't read the path, so nothing to set.
  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files?.[0] as (File & { pywebviewFullPath?: string }) | undefined
    if (f?.pywebviewFullPath) setInput(f.pywebviewFullPath)
  }, [])

  // Pasting a file (Ctrl+V after copying it in Explorer) can't work: no webview
  // exposes the file's path to a paste event, only its bytes. Turn that silent
  // no-op into guidance instead of leaving the user staring at an empty box.
  const onPaste = useCallback((e: React.ClipboardEvent) => {
    const dt = e.clipboardData
    if (!dt) return
    const hasFile = dt.files.length > 0 ||
      Array.from(dt.items).some((it) => it.kind === 'file')
    const hasText = dt.getData('text').trim().length > 0
    if (hasFile && !hasText) {
      e.preventDefault()
      setError(
        "Can't read a pasted file's path — drag the file into the box " +
        "instead (that works), or in Explorer use Shift+right-click → " +
        '"Copy as path" and paste that.',
      )
    }
  }, [])

  const ready = health?.status === 'ready'

  return (
    <div className="app" onDragOver={(e) => e.preventDefault()} onDrop={onDrop}>
      <header className="topbar">
        <Wordmark mark size={28} color="var(--text)" />
        <Button
          as="span"
          size="sm"
          title={health && health.params > 0
            ? `char backbone: ${(health.params / 1000).toFixed(1)}k trainable params. `
              + 'This is ONE component, not the size of the stack — the pretrained '
              + 'encoders it runs with (MetaCLIP-H, NLLB, whisper, Florence, CLIP-text, '
              + 'bge) dominate the real parameter count.'
            : undefined}
          style={{ cursor: 'default' }}
        >
          {health ? (modelLens === 'Maslow' ? 'Maslow' : health.model) : '…'}
        </Button>
        {/* Was "769.1k params", which reads as the size of the whole model.
            It is the char backbone alone (server.py sums scorer.model only);
            the stack as it RUNS is ~2.17B. Label what the number is. */}
        {health && health.params > 0 && (
          <span className="dim params-badge">
            backbone {(health.params / 1000).toFixed(1)}k
          </span>
        )}
        {/* takes the topbar's free space, so it sits right-aligned just left
            of the status badge (which keeps its own margin-left:auto). */}
        <Button
          variant="pencil"
          size="sm"
          onClick={() => setShowArch(true)}
          title="The layers an ad link passes through"
          style={{ marginLeft: 'auto' }}
        >
          View architecture
        </Button>
        <Button
          variant="pencil"
          size="sm"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          style={theme === 'dark' ? { color: '#ffffff' } : undefined}
        >
          {theme === 'dark' ? '☀' : '☾'}
        </Button>
        <span
          className={`status status-${health?.status ?? 'loading'}`}
          role="status"
          aria-live="polite"
        >
          {health?.status === 'ready' && '● ready'}
          {health?.status === 'loading' && '● loading engine…'}
          {(!health || health.status === 'error') && `● ${health?.error ?? 'connecting…'}`}
        </span>
      </header>

      <div className="layout">
        <aside className="side">
          <div className="input-panel">
            <PasteZone
              value={input}
              onChange={setInput}
              onDrop={onDrop}
              placeholder="Paste…"
              height={200}
              textareaStyle={{
                font: '400 13px/1.5 var(--font-sans)',
                textAlign: 'left',
                padding: '12px',
              }}
              textareaProps={{
                'aria-label': 'Ad copy, URL, or file path to analyze',
                spellCheck: false,
                onPaste,
                onKeyDown: (e) => {
                  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                    e.preventDefault()
                    run()
                  }
                },
              }}
            />
            <Caption style={{ marginTop: 10, fontSize: 12.5, lineHeight: 1.45 }}>
              Paste ad copy, a URL (YouTube / Meta Ad Library / TrendTrack /
              landing page), or a local image/video path. Drag &amp; drop a
              file works too.
            </Caption>
            <Button
              variant="ballpoint"
              onClick={run}
              disabled={!ready || busy || !input.trim()}
              style={{ width: '100%', marginTop: 12 }}
            >
              {busy ? 'working…' : 'Analyze'}
              {!busy && <span className="dim kbd-hint"> (Ctrl+Enter)</span>}
            </Button>
            {busy && progress && (
              <div className="progress-line" role="status" aria-live="polite">
                {progress}
              </div>
            )}
            {error && <div className="error" role="alert">{error}</div>}

            <Dropdown
              label="Model"
              options={['Schwartz-4.5', 'Maslow']}
              value={modelLens}
              onChange={setModelLens}
              style={{ marginTop: 16, width: '100%' }}
            />
            {modelLens === 'Maslow' && result && (
              <Caption
                as="a"
                href={vizUrl('excalidraw', vizSeq)}
                download="bernay_board.excalidraw"
                style={{ marginTop: 6, fontSize: 12, textDecoration: 'underline', cursor: 'pointer' }}
              >
                Download editable board (.excalidraw, open at excalidraw.com)
              </Caption>
            )}
          </div>

          {history.length > 0 && (
            <Card tone="screen" padding={14}>
              <div className="panel-title">History</div>
              {history.map((h) => (
                <button
                  key={h.id}
                  className="hist-item"
                  onClick={() => {
                    setResult(h.result)
                    // ...and bring its deck back with it (see vizSeq).
                    if (h.result.result_id) setVizSeq(h.result.result_id)
                  }}
                >
                  <span className="dim">{h.at}</span> {h.label}
                </button>
              ))}
            </Card>
          )}
        </aside>

        <main className="main">
          {!result && (
            <div className="empty">
              <div className="empty-title">
                Ad mix decomposition
                <SketchAccent
                  variant="underline"
                  width={260}
                  style={{ margin: '2px auto 0' }}
                />
              </div>
              <Caption align="center" style={{ marginTop: 16 }}>
                Paste an ad on the left — Bernay decomposes it into archetype
                angle, Schwartz awareness &amp; sophistication, desires, implied
                avatar and PV = Desire × T.
              </Caption>
            </div>
          )}
          {/* Maslow PRESENTS the same analysis instead of decomposing it —
              swap the whole main panel for its slide deck instead of the
              Schwartz result view, rather than popping it up over/beside it.
              The deck already contains a generated avatar image: the backend
              runs Maslow image-gen automatically as the last step of every
              analysis and splices it into the deck server-side — nothing to
              trigger here. */}
          {result && modelLens === 'Maslow' && (
            /* key AND a changing src: the key remounts the element, the src
               defeats the webview's HTTP cache. Either alone has been enough
               to leave a stale deck on screen. */
            <iframe
              key={vizSeq}
              className="arch-frame"
              src={vizUrl('html', vizSeq)}
              title="Maslow presentation"
            />
          )}
          {result && modelLens !== 'Maslow' && (
            <>
              {/* PV and the avatar are both short reads — pairing them puts
                  two half-width cards where two near-empty full-width ones
                  used to be. Awareness then gets the full width to itself. */}
              <div className="grid2">
                <PVHero pv={result.pv} winProb={result.win_prob} />
                <AvatarCard audience={result.audience} vision={result.vision} />
              </div>
              <AwarenessSpread
                spread={result.awareness_spread}
                journey={result.awareness_journey}
              />
              <SophisticationSpread spread={result.sophistication_spread} />
              <div className="grid2">
                <Card tone="screen" border="orchid" padding={14}>
                  <div className="panel-title">Angle &amp; motif families</div>
                  <FamilyChips title="archetype" cats={result.archetype_angles} />
                  <FamilyChips title="psych center" cats={result.psych_center} />
                  <FamilyChips title="maslow" cats={result.maslow_level} />
                  {result.desires.length > 0 && (
                    <div className="fam-row">
                      <span className="fam-title">desires</span>
                      {result.desires.map((d) => {
                        const { emoji, color, chakra } = desireMeta(d)
                        return (
                          <span
                            key={d}
                            className={color ? 'chip chip-desire' : 'chip'}
                            style={color
                              ? ({ '--chip-color': color } as CSSProperties)
                              : undefined}
                            title={chakra ? `${chakra} centre` : undefined}
                          >
                            {d.replaceAll('_', ' ')}
                            {emoji && (
                              <span className="chip-emoji" aria-hidden="true">
                                {emoji}
                              </span>
                            )}
                          </span>
                        )
                      })}
                    </div>
                  )}
                  {result.problem && (
                    <div className="kv" style={{ marginTop: 8 }}>
                      <span className="kv-k">problem</span>
                      <span className="kv-v">{result.problem}</span>
                    </div>
                  )}
                  {result.product && (
                    <div className="kv">
                      <span className="kv-k">product</span>
                      <span className="kv-v">{result.product}</span>
                    </div>
                  )}
                  {result.visual_category && (
                    <div className="kv">
                      <span className="kv-k">visual category</span>
                      <span className="kv-v">{result.visual_category}</span>
                    </div>
                  )}
                </Card>
                <Painpoints
                  pains={result.painpoints}
                  angles={result.painpoint_angles}
                />
              </div>
              <SellingBeats beats={result.selling_beats} />
              <details className="raw">
                <summary>terminal output</summary>
                <div className="raw-body">
                  <pre>{result.pretty_text}</pre>
                </div>
              </details>
            </>
          )}
        </main>
      </div>

      {/* Rendered INSIDE the app on purpose. window.open() escapes the
          pywebview/WebView2 shell — it has no new-window handler, so the OS
          opens the map in the default browser instead of the app window. An
          iframe keeps it in-window and behaves the same under `vite dev`. */}
      {showArch && (
        <div
          className="arch-overlay"
          role="dialog"
          aria-modal="true"
          aria-label="Bernay architecture"
        >
          <button
            type="button"
            className="arch-close arch-close-float"
            onClick={() => setShowArch(false)}
            title="Close (Esc)"
          >
            ✕
          </button>
          <iframe className="arch-frame" src={architectureUrl()} title="Bernay architecture" />
        </div>
      )}
    </div>
  )
}
