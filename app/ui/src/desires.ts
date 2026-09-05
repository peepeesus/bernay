/**
 * Desire → colour + emoji.
 *
 * The colour is NOT decoration: v4_taxonomy.json's `chakra` family links each
 * desire to one of the seven centres, and those centres have canonical hues.
 * So two chips sharing a colour share a motivational centre — the row reads as
 * a profile at a glance ("all heart/throat" vs "all solar plexus").
 *
 * 20 of the 26 desires are mapped directly by that family. The other six
 * (abundance, discipline, freedom, growth, mastery, material) have no chakra
 * row; each was placed by co-occurrence across the other taxonomy families,
 * with the note below where that was a tie broken by meaning.
 *
 * Keep in sync with the `linked_desires` vocabulary in v4_taxonomy.json — a
 * desire missing here still renders, just uncoloured and without an emoji.
 */

export type Chakra =
  | 'root' | 'sacral' | 'solar' | 'heart' | 'throat' | 'brow' | 'crown'

export interface DesireMeta {
  chakra: Chakra
  emoji: string
}

export const DESIRES: Record<string, DesireMeta> = {
  // Root — Muladhara
  survival: { chakra: 'root', emoji: '🛟' },
  safety: { chakra: 'root', emoji: '🛡️' },
  comfort: { chakra: 'root', emoji: '🛋️' },
  material: { chakra: 'root', emoji: '💎' }, // Malkuth, the material kingdom

  // Sacral — Svadhisthana
  pleasure: { chakra: 'sacral', emoji: '🍯' },
  novelty: { chakra: 'sacral', emoji: '✨' },
  connection: { chakra: 'sacral', emoji: '🤝' },
  freedom: { chakra: 'sacral', emoji: '🕊️' }, // tie w/ solar; Explorer + Rebel

  // Solar plexus — Manipura (will, power, control)
  power: { chakra: 'solar', emoji: '⚡' },
  status: { chakra: 'solar', emoji: '👑' },
  control: { chakra: 'solar', emoji: '🕹️' },
  ambition: { chakra: 'solar', emoji: '🚀' },
  abundance: { chakra: 'solar', emoji: '🌾' }, // clear by co-occurrence
  discipline: { chakra: 'solar', emoji: '⏱️' }, // tie w/ brow; Gevurah = control
  mastery: { chakra: 'solar', emoji: '🎯' }, // tie w/ throat; personal will

  // Heart — Anahata
  love: { chakra: 'heart', emoji: '❤️' },
  belonging: { chakra: 'heart', emoji: '🫂' },
  harmony: { chakra: 'heart', emoji: '⚖️' }, // ☯️ renders as a flat disc at 13px

  // Throat — Vishuddha
  expression: { chakra: 'throat', emoji: '🎨' },
  recognition: { chakra: 'throat', emoji: '🏆' },

  // Brow — Ajna
  insight: { chakra: 'brow', emoji: '💡' },
  clarity: { chakra: 'brow', emoji: '🔍' },
  intellect: { chakra: 'brow', emoji: '🧠' },

  // Crown — Sahasrara
  transcendence: { chakra: 'crown', emoji: '🪷' }, // 🌌 muddies to a smear
  purpose: { chakra: 'crown', emoji: '🧭' },
  growth: { chakra: 'crown', emoji: '🌱' }, // Self-Actualization: growth+purpose
}

/** Colour + emoji for a desire tag, or nulls when it is not in the vocabulary. */
export function desireMeta(tag: string) {
  const meta = DESIRES[tag]
  return {
    emoji: meta?.emoji ?? null,
    color: meta ? `var(--chakra-${meta.chakra})` : null,
    chakra: meta?.chakra ?? null,
  }
}
