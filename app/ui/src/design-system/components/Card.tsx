import type { CSSProperties, HTMLAttributes, ReactNode } from 'react'

/**
 * The Bernay "screen" - a purple gradient bezel (top #968def -> bottom
 * #4a4672) wrapping a dark canvas, exactly as in the source. Use as the
 * primary content surface: the thing the user is looking "into".
 *
 * Ported from bernay-design-system/components/surfaces/Card.jsx.
 */
/** Bezel gradient family. Width/radius/padding are identical across all of
 *  them — only the colour changes. Each drifts slowly (tie-dye). */
export type CardBorder =
  | 'purple'
  | 'rainbow'
  | 'sunset'
  | 'orchid'
  | 'gold'
  | 'steel'
  | 'crimson'

export interface CardProps extends Omit<HTMLAttributes<HTMLDivElement>, 'style'> {
  children?: ReactNode
  /** screen = dark canvas inside the bezel (default); paper = light canvas. */
  tone?: 'screen' | 'paper'
  /** Bezel colour family — geometry is unchanged, only the gradient. */
  border?: CardBorder
  /** Inner padding, px or CSS length. */
  padding?: number | string
  /** Styles on the outer bezel. */
  style?: CSSProperties
  /** Styles on the inner canvas. */
  bodyStyle?: CSSProperties
}

export function Card({
  children,
  tone = 'screen',
  border = 'purple',
  padding = 22,
  style,
  bodyStyle,
  className,
  ...rest
}: CardProps) {
  // `screen` is the black canvas — literal black + white text in BOTH themes,
  // so it reads the un-themed primitives. `paper` follows the theme.
  const bodyBg = tone === 'screen' ? 'var(--surface-screen)' : 'var(--surface-raised)'
  const bodyColor = tone === 'screen' ? 'var(--text-on-screen)' : 'var(--text-primary)'
  return (
    <div
      // background comes from the .ds-bezel-* class so the drift keyframes
      // can own background-position; inline background would outrank them.
      className={['ds-bezel', `ds-bezel-${border}`, className]
        .filter(Boolean)
        .join(' ')}
      style={{
        borderRadius: 'var(--radius-screen)',
        padding: 'var(--space-3)',
        boxShadow: 'var(--shadow-screen)',
        ...style,
      }}
      {...rest}
    >
      <div
        style={{
          background: bodyBg,
          color: bodyColor,
          borderRadius: 'var(--radius-screen)',
          padding: typeof padding === 'number' ? `${padding}px` : padding,
          height: '100%',
          boxSizing: 'border-box',
          ...bodyStyle,
        }}
      >
        {children}
      </div>
    </div>
  )
}
