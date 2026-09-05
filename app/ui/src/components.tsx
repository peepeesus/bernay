import { Card } from './design-system/components/Card'
import { SketchAccent } from './design-system/components/SketchAccent'
import type {
  Audience,
  Painpoint,
  PV,
  ScoredCat,
  StageShare,
  Vision,
} from './types'

// Schwartz stage palette — mirrors v4_admix STAGE_COLORS in the REPL.
const STAGE_COLOR: Record<string, string> = {
  unaware: '#d8392f',
  problem_aware: '#e8632a',
  solution_aware: '#f4a52a',
  product_aware: '#3fa552',
  most_aware: '#4f7cc9',
}

const STAGE_LABEL: Record<string, string> = {
  unaware: 'unaware',
  problem_aware: 'problem',
  solution_aware: 'solution',
  product_aware: 'product',
  most_aware: 'most aware',
}

// Market-sophistication ladder — mirrors v4_admix SOPH_STAGE_COLORS/LABELS.
// A violet ramp (low → high sophistication), distinct from the awareness rainbow.
const SOPH_COLOR: Record<string, string> = {
  sophistication_1: '#b0a1e6',
  sophistication_2: '#9179dd',
  sophistication_3: '#7355d1',
  sophistication_4: '#5a3bb8',
  sophistication_5: '#422a8f',
}

const SOPH_LABEL: Record<string, string> = {
  sophistication_1: 'virgin',
  sophistication_2: 'enlarged claim',
  sophistication_3: 'mechanism',
  sophistication_4: 'elaborated',
  sophistication_5: 'identity',
}

export function PVHero({ pv, winProb }: { pv: PV; winProb?: number | null }) {
  return (
    <Card tone="screen" border="sunset" padding={14} className="pv-hero">
      <div className="panel-title">
        PV = Desire × T
        <SketchAccent variant="underline" width={140} style={{ marginTop: 2 }} />
      </div>
      <div className="pv-equation">
        <span className="pv-num">{pv.desire.toFixed(2)}</span>
        <span className="pv-var">dsr</span>
        <span className="pv-op">×</span>
        <span className="pv-num">{pv.t.toFixed(2)}</span>
        <span className="pv-var">T</span>
        <span className="pv-op">=</span>
        <span className="pv-total">{pv.pv.toFixed(2)}</span>
        <span className="pv-var">PV</span>
      </div>
      <div className="pv-meta">
        emotion ×{pv.problem_gain.toFixed(2)}
        {pv.top_primal ? ` via ${pv.top_primal.replaceAll('_', ' ')}` : ''}
        {typeof winProb === 'number' && (
          <span className="win-badge" title="winner-recognizer head (AUC ~0.76)">
            win {Math.round(winProb * 100)}%
          </span>
        )}
      </div>
    </Card>
  )
}

export function AwarenessSpread({
  spread,
  journey,
}: {
  spread: StageShare[]
  journey: string[]
}) {
  return (
    <Card tone="screen" border="rainbow" padding={14}>
      <div className="panel-title">Awareness — 5-stage spread</div>
      {spread.map((s) => (
        <div className="aw-row" key={s.stage}>
          <span className="aw-label">{STAGE_LABEL[s.stage] ?? s.stage}</span>
          <div className="aw-track">
            <div
              className="aw-bar"
              style={{
                width: `${Math.max(1, Math.round(s.share * 100))}%`,
                background: STAGE_COLOR[s.stage] ?? '#888',
              }}
            />
          </div>
          <span className="aw-pct">{Math.round(s.share * 100)}%</span>
        </div>
      ))}
      {journey.length > 0 && (
        <div className="aw-journey">
          journey:{' '}
          {journey.map((s, i) => (
            <span key={i}>
              {i > 0 && <span className="dim"> → </span>}
              <span style={{ color: STAGE_COLOR[s] ?? '#d8d8e0' }}>
                {STAGE_LABEL[s] ?? s}
              </span>
            </span>
          ))}
        </div>
      )}
    </Card>
  )
}

export function SophisticationSpread({ spread }: { spread: StageShare[] }) {
  if (!spread?.length) return null
  const lead = spread.reduce((a, b) => (b.share > a.share ? b : a), spread[0])
  return (
    <Card tone="screen" border="steel" padding={14}>
      <div className="panel-title">Market sophistication — 5-stage spread</div>
      {spread.map((s) => (
        <div className="aw-row" key={s.stage}>
          <span className="aw-label">{SOPH_LABEL[s.stage] ?? s.stage}</span>
          <div className="aw-track">
            <div
              className="aw-bar"
              style={{
                width: `${Math.max(1, Math.round(s.share * 100))}%`,
                background: SOPH_COLOR[s.stage] ?? '#888',
              }}
            />
          </div>
          <span className="aw-pct">{Math.round(s.share * 100)}%</span>
        </div>
      ))}
      <div className="aw-journey">
        dominant stage:{' '}
        <span style={{ color: SOPH_COLOR[lead.stage] ?? '#d8d8e0' }}>
          {SOPH_LABEL[lead.stage] ?? lead.stage}
        </span>
      </div>
    </Card>
  )
}

export function FamilyChips({
  title,
  cats,
}: {
  title: string
  cats: ScoredCat[]
}) {
  if (!cats?.length) return null
  return (
    <div className="fam-row">
      <span className="fam-title">{title}</span>
      {cats.map((c, i) => (
        <span key={c.id} className={`chip ${i === 0 ? 'chip-lead' : ''}`}>
          {c.id.replaceAll('_', ' ')}
          <span className={c.score >= 0 ? 'chip-pos' : 'chip-neg'}>
            {c.score >= 0 ? '+' : ''}
            {c.score.toFixed(2)}
          </span>
        </span>
      ))}
    </div>
  )
}

export function Painpoints({ pains, angles = [] }: {
  pains: Painpoint[]
  /** The MECHANISM the ad blames — "DHT", "Cortisol". Distinct from the
   *  archetype angle (everyman/sage) shown in the families panel. The Maslow
   *  deck has always rendered this; the Schwartz view never received it. */
  angles?: string[]
}) {
  if (!pains?.length) return null
  return (
    <Card tone="screen" border="crimson" padding={14}>
      <div className="panel-title">Painpoints</div>
      {angles.length > 0 && (
        <div className="pain" style={{ marginBottom: 10 }}>
          <div className="cite-finding" style={{ opacity: 0.75 }}>
            mechanism the ad blames
          </div>
          <div className="pain-name" style={{ color: 'var(--amber, #ffc078)' }}>
            {angles.join(' · ')}
          </div>
        </div>
      )}
      {pains.map((p) => (
        <div key={p.name} className="pain">
          <div className="pain-name">{p.name}</div>
          {p.citations.slice(0, 3).map((c, i) => (
            <div key={i} className="cite">
              <span className="cite-finding">{c.finding}</span>{' '}
              <a
                className="cite-src"
                href={c.url || undefined}
                target="_blank"
                rel="noreferrer"
              >
                [{c.source || 'source'}]
              </a>
            </div>
          ))}
        </div>
      ))}
    </Card>
  )
}

export function AvatarCard({
  audience,
  vision,
}: {
  audience: Audience
  vision?: Vision | null
}) {
  const rows: Array<[string, string | null | undefined]> = [
    ['gender', audience.gender],
    ['age', audience.age],
    ['life stage', audience.life_stage],
    ['income', audience.income_by_age?.display ?? audience.income],
    ['presenter', audience.presenter],
    ['ethnicity', audience.ethnicity],
  ]
  const visible = rows.filter(([, v]) => v && v !== 'unclear')
  const unclear = rows.filter(([, v]) => !v || v === 'unclear').map(([k]) => k)
  return (
    <Card tone="screen" border="gold" padding={14}>
      <div className="panel-title">Implied avatar</div>
      {visible.length === 0 && <div className="dim">unclear — evidence below floor</div>}
      {visible.map(([k, v]) => (
        <div className="kv" key={k}>
          <span className="kv-k">{k}</span>
          <span className="kv-v">{v}</span>
        </div>
      ))}
      {audience.income_by_age?.source_name && (
        <div className="cite" style={{ marginTop: 6 }}>
          <a
            className="cite-src"
            href={audience.income_by_age.source_url || undefined}
            target="_blank"
            rel="noreferrer"
          >
            [{audience.income_by_age.source_name}]
          </a>
        </div>
      )}
      {vision?.core_desires && vision.core_desires.length > 0 && (
        <div className="kv">
          <span className="kv-k">[vision] desires</span>
          <span className="kv-v">{vision.core_desires.join(', ')}</span>
        </div>
      )}
      {unclear.length > 0 && visible.length > 0 && (
        <div className="dim small">unclear: {unclear.join(', ')}</div>
      )}
    </Card>
  )
}

const BEAT_ORDER = [
  'problem',
  'victim',
  'solution',
  'applying',
  'result',
  'urgency',
  'memorable',
  'call_to_action',
]

export function SellingBeats({ beats }: { beats: string[] }) {
  if (!beats?.length) return null
  const present = new Set(beats)
  return (
    <Card tone="screen" padding={14}>
      <div className="panel-title">Selling beats</div>
      <div className="beats">
        {BEAT_ORDER.map((b, i) => (
          <span key={b}>
            {i > 0 && <span className="dim"> → </span>}
            <span className={present.has(b) ? 'beat-on' : 'beat-off'}>
              {b.replaceAll('_', ' ')}
            </span>
          </span>
        ))}
      </div>
    </Card>
  )
}
