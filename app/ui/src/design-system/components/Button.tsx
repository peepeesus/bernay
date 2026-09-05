import type { CSSProperties, ElementType, MouseEvent, ReactNode } from 'react'

/**
 * Bernay's signature action: a highlighter-green gradient pill with a
 * hand-drawn Kalam label. The green gradient (top #00ff4d -> bottom #00992e)
 * and 9px radius come straight from the source.
 *
 * Ported from bernay-design-system/components/controls/Button.jsx.
 */
export interface ButtonProps {
  children?: ReactNode
  /** highlighter = green gradient pill (default); ballpoint = blue gradient
   *  pill; pencil = graphite gradient pill; marker = blue outline; ghost = bare. */
  variant?: 'highlighter' | 'ballpoint' | 'pencil' | 'marker' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  /** Adds a marker-pen selection ring and a check mark - the source's "selected" state. */
  selected?: boolean
  disabled?: boolean
  onClick?: (e: MouseEvent) => void
  as?: ElementType
  style?: CSSProperties
  title?: string
  [key: `data-${string}`]: unknown
  [key: `aria-${string}`]: unknown
}

const SIZES: Record<NonNullable<ButtonProps['size']>, CSSProperties> = {
  sm: { padding: '6px 16px', fontSize: 18 },
  md: { padding: '8px 22px', fontSize: 24 },
  lg: { padding: '12px 30px', fontSize: 28 },
}

const VARIANTS: Record<NonNullable<ButtonProps['variant']>, CSSProperties> = {
  highlighter: {
    background: 'var(--grad-highlighter)',
    boxShadow: 'var(--shadow-pill)',
  },
  // filled ballpoint-blue pill — same geometry as highlighter, cool ink
  ballpoint: {
    background: 'var(--grad-ballpoint)',
    boxShadow: 'var(--shadow-pill-ballpoint)',
  },
  // graphite pencil pill — neutral chrome action, doesn't compete with the
  // green Analyze pill for attention
  pencil: {
    background: 'var(--grad-pencil)',
    boxShadow: 'var(--shadow-pill-pencil)',
  },
  // outlined marker-pen pill for secondary actions
  marker: {
    background: 'transparent',
    color: 'var(--marker-blue)',
    boxShadow: 'inset 0 0 0 var(--stroke-marker) var(--marker-blue)',
  },
  ghost: {
    background: 'transparent',
    color: 'var(--ink)',
    boxShadow: 'none',
  },
}

export function Button({
  children,
  variant = 'highlighter',
  size = 'md',
  selected = false,
  disabled = false,
  onClick,
  as = 'button',
  style,
  ...rest
}: ButtonProps) {
  const Tag = as as ElementType
  const base: CSSProperties = {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    font: 'var(--weight-hand-light) 1em/1 var(--font-hand)',
    color: 'var(--action-text)',
    border: 'none',
    borderRadius: 'var(--radius-pill)',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.5 : 1,
    textAlign: 'center',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
    transition: 'transform .08s ease, filter .12s ease',
    ...SIZES[size],
  }
  const selectedRing: CSSProperties = selected
    ? { boxShadow: 'inset 0 0 0 var(--stroke-marker) var(--marker-blue)' }
    : {}
  return (
    <Tag
      onClick={disabled ? undefined : onClick}
      disabled={as === 'button' ? disabled : undefined}
      style={{ ...base, ...VARIANTS[variant], ...selectedRing, ...style }}
      {...rest}
    >
      {/* text-overflow doesn't apply to a flex container (centered overflow
          clips BOTH edges mid-glyph) — ellipsize inside a shrinkable child. */}
      <span
        style={{
          minWidth: 0,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {children}
      </span>
      {selected && (
        <span aria-hidden="true" style={{ fontSize: '0.72em' }}>
          &#10003;
        </span>
      )}
    </Tag>
  )
}
