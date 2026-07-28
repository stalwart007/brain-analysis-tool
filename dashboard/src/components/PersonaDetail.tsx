"use client";

import { PersonaSeed, Trait } from "@/lib/api";

type TraitKey = "openness" | "conscientiousness" | "extraversion" | "agreeableness" | "neuroticism";

const TRAITS: { key: TraitKey; label: string }[] = [
  { key: "openness", label: "Openness" },
  { key: "conscientiousness", label: "Conscientiousness" },
  { key: "extraversion", label: "Extraversion" },
  { key: "agreeableness", label: "Agreeableness" },
  { key: "neuroticism", label: "Neuroticism" },
];

/** A trait bar: fill = score (0–1, 0.5 baseline marked), opacity = confidence. */
function TraitBar({ label, score, confidence }: { label: string; score: number; confidence: number }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-32 shrink-0 text-xs text-muted">{label}</span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
        {/* baseline marker at 0.5 */}
        <span className="absolute left-1/2 top-0 h-full w-px bg-white/20" />
        <div
          className="h-full rounded-full bg-accent transition-all"
          style={{ width: `${score * 100}%`, opacity: 0.35 + confidence * 0.65 }}
        />
      </div>
      <span className="w-24 shrink-0 text-right text-xs tabular-nums text-muted">
        {score.toFixed(2)} · c{confidence.toFixed(2)}
      </span>
    </div>
  );
}

export default function PersonaDetail({ persona }: { persona: PersonaSeed }) {
  const s = persona.signal;
  return (
    <div className="space-y-4 px-3 py-4">
      <div>
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">
          Behavioral signal
        </div>
        <div className="flex flex-wrap gap-2">
          {[
            ["deliberation", s.deliberation],
            ["frustration", s.frustration_signal],
            ["exploration", s.exploration_style],
            ["price sensitivity", s.price_sensitivity_signal],
          ].map(([k, v]) => (
            <span
              key={k}
              className="rounded-full border border-hairline px-2.5 py-0.5 text-xs text-ink-2"
            >
              {k}: <span className="text-ink">{v}</span>
            </span>
          ))}
          <span className="rounded-full border border-hairline px-2.5 py-0.5 text-xs text-ink-2">
            confidence: <span className="text-accent">{s.confidence.toFixed(2)}</span>
          </span>
        </div>
        <p className="mt-2 text-sm text-ink-2">{s.evidence}</p>
        <p className="mt-1 text-sm italic text-muted">“{s.likely_mindset}”</p>
      </div>

      <div>
        <div className="mb-2 text-xs font-medium uppercase tracking-wider text-muted">
          Soft trait seed{" "}
          <span className="normal-case text-muted/70">
            (bar = score, opacity = confidence, tick = population average)
          </span>
        </div>
        <div className="space-y-2">
          {TRAITS.map((t) => {
            const trait = persona.traits[t.key] as Trait;
            return (
              <TraitBar
                key={t.key}
                label={t.label}
                score={trait.score}
                confidence={trait.confidence}
              />
            );
          })}
        </div>
        <p className="mt-2 text-xs text-muted">{persona.traits.disclaimer}</p>
      </div>
    </div>
  );
}
