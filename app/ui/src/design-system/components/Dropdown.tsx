import { useEffect, useRef, useState, type CSSProperties } from 'react'
import type { SegmentOption } from './SegmentedControl'

/**
 * Collapsible model picker — a highlighter-green trigger pill (labelled with
 * the current selection, so it's obvious a change stuck) that reveals the
 * option list (same green gradient / white divider look as SegmentedControl)
 * directly beneath it when clicked, instead of showing all options at once.
 *
 * Picking an option updates the selection in place and keeps the list open
 * (no vanish-on-click) — dismiss via the trigger, an outside click, or Esc.
 */
export interface DropdownProps {
  /** Prefix shown before the current value, e.g. "Model" -> "Model: Maslow". */
  label?: string
  /** Options as strings or {value,label} objects. */
  options: SegmentOption[]
  /** Currently selected value. */
  value?: string
  onChange?: (value: string) => void
  style?: CSSProperties
}

function optionLabel(opt: SegmentOption): string {
  return typeof opt === 'string' ? opt : opt.label
}
function optionValue(opt: SegmentOption): string {
  return typeof opt === 'string' ? opt : opt.value
}

export function Dropdown({
  label = 'Model',
  options = [],
  value,
  onChange,
  style,
}: DropdownProps) {
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  // close on outside click / Escape
  useEffect(() => {
    if (!open) return
    const onDocMouseDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocMouseDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('mousedown', onDocMouseDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [open])

  const current = options.find((o) => optionValue(o) === value)
  const triggerText = current ? `${label}: ${optionLabel(current)}` : `Select ${label}`

  return (
    <div ref={rootRef} style={style}>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        style={{
          width: '100%',
          appearance: 'none',
          display: 'inline-flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 10,
          font: 'var(--weight-hand-light) 24px/1 var(--font-hand)',
          color: 'var(--action-text)',
          background: 'var(--grad-highlighter)',
          border: 'none',
          borderRadius: 'var(--radius-pill)',
          padding: '9px 26px',
          cursor: 'pointer',
          boxShadow: open
            ? 'var(--shadow-pill), inset 0 0 0 var(--stroke-marker) var(--marker-blue)'
            : 'var(--shadow-pill)',
        }}
      >
        {triggerText}
        {/* CSS-drawn triangle instead of a font glyph — the Kalam glyph for
            "▾" rendered near-invisibly small; this stays a fixed, legible
            size regardless of font metrics. */}
        <span
          aria-hidden="true"
          style={{
            display: 'inline-block',
            width: 0,
            height: 0,
            borderLeft: '7px solid transparent',
            borderRight: '7px solid transparent',
            borderTop: '9px solid currentColor',
            transform: open ? 'rotate(180deg)' : 'none',
            transition: 'transform .15s ease',
            flexShrink: 0,
          }}
        />
      </button>

      {open && (
        <div
          role="listbox"
          style={{
            // In normal flow (not position:absolute) so it pushes whatever
            // comes after it (e.g. the History card) down instead of
            // floating on top and covering it.
            marginTop: 6,
            display: 'flex',
            flexDirection: 'column',
            background: 'var(--grad-highlighter)',
            borderRadius: 'var(--radius-pill)',
            overflow: 'hidden',
            boxShadow: 'var(--shadow-pill)',
          }}
        >
          {options.map((opt, i) => {
            const val = optionValue(opt)
            const active = val === value
            return (
              <div key={val}>
                {i > 0 && (
                  <div
                    aria-hidden="true"
                    style={{
                      height: 2,
                      margin: '0 22px',
                      background: 'rgba(255,255,255,0.9)',
                      borderRadius: 1,
                    }}
                  />
                )}
                <button
                  type="button"
                  role="option"
                  aria-selected={active}
                  onClick={() => onChange?.(val)}
                  style={{
                    width: '100%',
                    appearance: 'none',
                    background: 'transparent',
                    color: 'var(--ink)',
                    font: 'var(--weight-hand-light) 24px/1 var(--font-hand)',
                    border: 'none',
                    cursor: 'pointer',
                    padding: '9px 26px',
                    textAlign: 'center',
                    whiteSpace: 'nowrap',
                    display: 'inline-flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 8,
                  }}
                >
                  {optionLabel(opt)}
                  {active && <span aria-hidden="true">&mdash;Selected</span>}
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
