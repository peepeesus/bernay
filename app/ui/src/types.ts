// Shapes mirror bernay-app/example_analyze_response.json (authoritative).

export interface Citation {
  factor: string
  magnitude: string
  finding: string
  source: string
  url: string
}

export interface Painpoint {
  name: string
  citations: Citation[]
}

export interface ScoredCat {
  id: string
  score: number
}

export interface StageShare {
  stage: string
  share: number
}

export interface PV {
  desire: number
  t: number
  problem_gain: number
  top_primal: string | null
  pv: number
  equation_string: string
}

export interface IncomeByAge {
  bucket?: string
  display?: string
  acs?: string
  source_name?: string
  source_url?: string
}

export interface Audience {
  age?: string | null
  gender?: string | null
  income?: string | null
  income_by_age?: IncomeByAge | null
  life_stage?: string | null
  ethnicity?: string | null
  presenter?: string | null
}

export interface Vision {
  avatar?: Record<string, unknown> | null
  accent_region?: Record<string, unknown> | null
  cta?: Record<string, unknown> | null
  core_desires?: string[] | null
}

export interface AnalyzeResult {
  model: string
  input_kind: string
  /** Server-side id for THIS analysis. Pass it to /api/viz so the deck shows
   *  the ad currently on screen — restoring a History entry restores its own
   *  deck, instead of whichever analysis ran most recently. */
  result_id?: number
  insufficient_evidence?: boolean
  abstain_reason?: string | null
  pv: PV
  awareness_spread: StageShare[]
  awareness_journey: string[]
  sophistication_spread: StageShare[]
  archetype_angles: ScoredCat[]
  sophistication: ScoredCat[]
  psych_center: ScoredCat[]
  maslow_level: ScoredCat[]
  desires: string[]
  problem?: string | null
  painpoints: Painpoint[]
  /** The mechanism the ad blames ("DHT", "Cortisol") — NOT archetype_angles. */
  painpoint_angles?: string[]
  audience: Audience
  product?: string | null
  visual_category?: string | null
  selling_beats: string[]
  win_prob?: number | null
  vision?: Vision | null
  pretty_text: string
  text_analyzed_chars: number
}

export interface Health {
  ok: boolean
  status: 'loading' | 'ready' | 'error'
  model: string
  params: number
  error?: string | null
}

export interface HistoryEntry {
  id: number
  label: string
  at: string
  result: AnalyzeResult
}
