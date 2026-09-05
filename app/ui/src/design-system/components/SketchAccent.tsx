import type { CSSProperties } from 'react'
import underlineStroke from '../assets/sketch-stroke-underline.svg?raw'
import cornerStroke from '../assets/sketch-stroke-corner.svg?raw'

/**
 * Decorative hand-drawn marker stroke, copied verbatim from the source
 * sketch - the single most recognisable Bernay motif. Purely ornamental;
 * per the design system's rule, use ONE gesture per view (not per panel).
 *
 * Ported from bernay-design-system/components/brand/SketchAccent.jsx.
 * Adapted: the source references absolute `/assets/*.svg` paths (a static
 * public/ folder in the Figma export host); this app has no public/ dir,
 * so the strokes are bundled as local assets.
 *
 * Imported `?raw` and rendered INLINE (not `<img src=...>`): the source
 * SVGs use `fill="currentColor"` so they can be tinted/themed via CSS, but
 * `currentColor` inside an externally-referenced `<img>` document always
 * resolves to black — it has no access to the host page's `color`. Loading
 * it as an img made the stroke invisible against a dark background once
 * dark mode landed. Inlining the markup lets `color` (below, defaulting to
 * the theme-aware `--text-primary`) actually reach it.
 */
export interface SketchAccentProps {
  /** Which copied marker stroke to show. */
  variant?: 'underline' | 'corner'
  /** Defaults to --text-primary, which already flips with dark mode. */
  color?: string
  /** Width in px or CSS length; height scales automatically. */
  width?: number | string
  style?: CSSProperties
}

export function SketchAccent({
  variant = 'underline',
  color = 'var(--text-primary)',
  width,
  style,
  ...rest
}: SketchAccentProps) {
  const svg = variant === 'corner' ? cornerStroke : underlineStroke
  return (
    <div
      aria-hidden="true"
      className="sketch-accent"
      dangerouslySetInnerHTML={{ __html: svg }}
      style={{
        display: 'block',
        width: width ?? (variant === 'corner' ? 257 : 470),
        color,
        lineHeight: 0,
        ...style,
      }}
      {...rest}
    />
  )
}
