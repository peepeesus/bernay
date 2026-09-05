import type { CSSProperties, ElementType, HTMLAttributes } from 'react'
import mustacheLogo from '../assets/logo.svg'

/**
 * Bernay wordmark — the product name set in the Junge serif, optionally
 * prefixed with the pixelated brown mustache mark. Uses the real
 * `assets/logo.svg` file via a Vite asset import (matches how `SketchAccent`
 * pulls its strokes) rather than a hand-rolled inline data URI — the
 * original `encodeURIComponent`-built data URI rendered as a broken-image
 * glyph in the packaged desktop app's webview, even though it was valid SVG.
 *
 * Ported from bernay-design-system/components/brand/Wordmark.jsx.
 */

export interface WordmarkProps extends Omit<HTMLAttributes<HTMLElement>, 'color'> {
  /** Font size in px (number) or any CSS length string. Source wordmark is 48px. */
  size?: number | string
  /** Text colour. Defaults to --ink; use --paper on dark screens. */
  color?: string
  /** Show the pixelated mustache mark before the word. Default false. */
  mark?: boolean
  /** Element tag to render. */
  as?: ElementType
  style?: CSSProperties
}

/**
 * The Bernay wordmark, set in Junge serif. Pass `mark` to prefix the
 * pixelated brown mustache logo.
 */
export function Wordmark({
  size = 48,
  color = 'var(--ink)',
  mark = false,
  as = 'span',
  style,
  ...rest
}: WordmarkProps) {
  const Tag = as as ElementType
  const px = typeof size === 'number' ? size : parseFloat(size) || 48
  return (
    <Tag
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: `${Math.round(px * 0.22)}px`,
        ...style,
      }}
      {...rest}
    >
      {mark && (
        <img
          src={mustacheLogo}
          alt=""
          aria-hidden="true"
          style={{
            height: `${Math.round(px * 0.62)}px`,
            width: 'auto',
            display: 'block',
            imageRendering: 'pixelated',
          }}
        />
      )}
      <span
        style={{
          font: `var(--weight-hand) 1em/1 var(--font-display)`,
          fontSize: typeof size === 'number' ? `${size}px` : size,
          color,
          letterSpacing: '0.01em',
        }}
      >
        Bernay
      </span>
    </Tag>
  )
}
