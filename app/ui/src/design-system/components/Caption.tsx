import type { CSSProperties, ElementType, HTMLAttributes, ReactNode } from 'react'

/**
 * Muted helper text - the source's grey Instrument Sans note ("Paste ad
 * copy, a URL..."). Clean sans, calm grey, for instructions and captions
 * beneath the hand-drawn elements. The calm counterpoint to Bernay's
 * hand-drawn headings (per the "two voices, on purpose" content rule:
 * marker for the gesture, sans for the instruction).
 *
 * Ported from bernay-design-system/components/content/Caption.jsx.
 */
export interface CaptionProps extends Omit<HTMLAttributes<HTMLElement>, 'style' | 'color'> {
  children?: ReactNode
  align?: 'left' | 'center' | 'right'
  /** Defaults to --text-muted (the source grey). */
  color?: string
  as?: ElementType
  style?: CSSProperties
  /** Only meaningful when `as="a"`. */
  href?: string
  download?: string | boolean
}

export function Caption({
  children,
  align = 'left',
  color = 'var(--text-muted)',
  as = 'p',
  style,
  ...rest
}: CaptionProps) {
  const Tag = as as ElementType
  return (
    <Tag
      style={{
        font: 'var(--weight-sans) var(--text-md)/1.2 var(--font-sans)',
        color,
        textAlign: align,
        margin: 0,
        ...style,
      }}
      {...rest}
    >
      {children}
    </Tag>
  )
}
