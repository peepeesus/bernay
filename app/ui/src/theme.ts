/**
 * Theme selection.
 *
 * No stored choice → follow the OS (`prefers-color-scheme`), handled purely in
 * CSS so it works before any JS runs. Toggling writes an explicit
 * `data-theme` on <html>, which outranks the media query, and persists it.
 *
 * The matching pre-paint script in index.html replays the stored choice before
 * first paint — without it the window flashes the other theme on every launch.
 */
export type ThemeChoice = 'light' | 'dark'

const KEY = 'bernay-theme'

export function systemTheme(): ThemeChoice {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches
    ? 'dark'
    : 'light'
}

export function storedTheme(): ThemeChoice | null {
  try {
    const v = localStorage.getItem(KEY)
    return v === 'dark' || v === 'light' ? v : null
  } catch {
    return null // storage can throw in restricted webview contexts
  }
}

/** What the page is showing right now. */
export function currentTheme(): ThemeChoice {
  return storedTheme() ?? systemTheme()
}

export function applyTheme(t: ThemeChoice): void {
  document.documentElement.dataset.theme = t
  try {
    localStorage.setItem(KEY, t)
  } catch {
    /* not fatal — the theme still applies for this session */
  }
}
