import { useState } from 'react'
import type { CSSProperties, DragEvent, TextareaHTMLAttributes } from 'react'

/**
 * The Bernay input - the source's hero "Paste..." zone. A purple bezel
 * around a black canvas that accepts ad copy, a URL, or a dropped file.
 *
 * Ported from bernay-design-system/components/surfaces/PasteZone.jsx.
 * Adapted for real use: the source renders both the placeholder AND the
 * live value in giant centred 64px Kalam (fine for a short "Paste..."
 * demo, unusable once real multi-paragraph ad copy / long URLs land in
 * it). Two additions preserve the source's placeholder gesture while
 * making the field actually usable for real content:
 *   - `textareaStyle` lets a caller restyle just the *typed* text (this
 *     app sets it to small left-aligned sans) without touching the big
 *     centred placeholder, which still renders in the source's hero style.
 *   - `textareaProps` passes through native attributes (onKeyDown,
 *     spellCheck, aria-label, autoFocus, ...) that the source component
 *     had no hook for, since this app needs Ctrl+Enter-to-submit and an
 *     accessible name on the field.
 */
export interface PasteZoneProps {
  value?: string
  onChange?: (value: string) => void
  /** Placeholder shown centred in Kalam. Source uses "Paste...". */
  placeholder?: string
  /** Fires on file drop; you handle the DataTransfer. */
  onDrop?: (e: DragEvent) => void
  /** Canvas height, px or CSS length. */
  height?: number | string
  /** Styles on the outer purple bezel. */
  style?: CSSProperties
  /** Styles on the inner black canvas. */
  canvasStyle?: CSSProperties
  /** Styles merged onto the live textarea only (not the placeholder). */
  textareaStyle?: CSSProperties
  /** Passthrough native attributes for the textarea (onKeyDown, spellCheck, aria-label, ...). */
  textareaProps?: Omit<
    TextareaHTMLAttributes<HTMLTextAreaElement>,
    'value' | 'onChange' | 'style'
  >
}

export function PasteZone({
  value = '',
  onChange,
  placeholder = 'Paste...',
  onDrop,
  height = 320,
  style,
  canvasStyle,
  textareaStyle,
  textareaProps,
  ...rest
}: PasteZoneProps) {
  const [dragging, setDragging] = useState(false)
  return (
    <div
      className="ds-bezel ds-bezel-purple"
      style={{
        borderRadius: 'var(--radius-screen)',
        padding: 'var(--space-3)',
        boxShadow: 'var(--shadow-screen)',
        ...style,
      }}
      {...rest}
    >
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          onDrop?.(e)
        }}
        style={{
          position: 'relative',
          background: 'var(--black)',
          borderRadius: 'var(--radius-screen)',
          height: typeof height === 'number' ? `${height}px` : height,
          boxShadow: dragging
            ? 'inset 0 0 0 var(--stroke-marker) var(--marker-blue)'
            : 'none',
          transition: 'box-shadow .12s ease',
          ...canvasStyle,
        }}
      >
        <textarea
          value={value}
          onChange={(e) => onChange?.(e.target.value)}
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            resize: 'none',
            border: 'none',
            outline: 'none',
            background: 'transparent',
            color: 'var(--paper)',
            textAlign: 'center',
            font: 'var(--weight-hand) 64px/1.1 var(--font-hand)',
            padding: '24px',
            boxSizing: 'border-box',
            ...textareaStyle,
          }}
          {...textareaProps}
        />
        {!value && (
          <div
            aria-hidden="true"
            style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
              color: 'var(--graphite)',
              font: 'var(--weight-hand) 64px/1.1 var(--font-hand)',
            }}
          >
            {placeholder}
          </div>
        )}
      </div>
    </div>
  )
}
