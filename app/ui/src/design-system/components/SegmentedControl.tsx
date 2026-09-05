import type { CSSProperties } from 'react'

/**
 * Stacked highlighter selector — the source's "Schwartz-4 / Maslow" control:
 * green gradient segments divided by a thin white marker rule, one active at
 * a time.
 *
 * Ported from bernay-design-system/components/controls/SegmentedControl.jsx.
 */
export type SegmentOption = string | { value: string; label: string }

export interface SegmentedControlProps {
  /** Options as strings or {value,label} objects. */
  options: SegmentOption[]
  /** Currently selected value. */
  value?: string
  onChange?: (value: string) => void
  /** Source stacks vertically; horizontal is also supported. */
  orientation?: 'vertical' | 'horizontal'
  style?: CSSProperties
}

export function SegmentedControl({
  options = [],
  value,
  onChange,
  orientation = 'vertical',
  style,
}: SegmentedControlProps) {
  const isVertical = orientation === 'vertical'
  return (
    <div
      role="radiogroup"
      style={{
        display: 'inline-flex',
        flexDirection: isVertical ? 'column' : 'row',
        background: 'var(--grad-highlighter)',
        borderRadius: 'var(--radius-pill)',
        overflow: 'hidden',
        boxShadow: 'var(--shadow-pill)',
        ...style,
      }}
    >
      {options.map((opt, i) => {
        const val = typeof opt === 'string' ? opt : opt.value
        const label = typeof opt === 'string' ? opt : opt.label
        const active = val === value
        return (
          <button
            key={val}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange && onChange(val)}
            style={{
              appearance: 'none',
              background: active ? 'rgba(255,255,255,0.16)' : 'transparent',
              color: 'var(--ink)',
              font: 'var(--weight-hand-light) 24px/1 var(--font-hand)',
              border: 'none',
              cursor: 'pointer',
              padding: '9px 26px',
              textAlign: 'center',
              whiteSpace: 'nowrap',
              boxShadow:
                i > 0
                  ? isVertical
                    ? 'inset 0 1px 0 rgba(255,255,255,0.85)'
                    : 'inset 1px 0 0 rgba(255,255,255,0.85)'
                  : 'none',
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 8,
            }}
          >
            {label}
            {active && (
              <span aria-hidden="true" style={{ fontSize: '0.72em' }}>
                &#10003;
              </span>
            )}
          </button>
        )
      })}
    </div>
  )
}
