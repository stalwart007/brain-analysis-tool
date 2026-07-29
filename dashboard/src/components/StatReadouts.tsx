"use client";

/**
 * HUD readouts for the statistical layer. These exist to make the *uncertainty*
 * as visible as the point estimate — a mean with no interval is a guess wearing
 * a lab coat.
 */

import { BayesianResult, Econometrics, SegmentReport, SurvivalResult } from "@/lib/api";

/** A value with its 95% interval drawn as a track + whisker. */
export function CIStat({
  label,
  value,
  ci,
  accent,
}: {
  label: string;
  value: number;
  ci?: [number, number] | null;
  accent?: boolean;
}) {
  const lo = ci?.[0] ?? value;
  const hi = ci?.[1] ?? value;
  return (
    <div>
      <div className="hud-label">{label}</div>
      <div
        className={`font-display text-3xl font-medium tabular-nums tracking-tight ${
          accent ? "text-accent" : "text-ink"
        }`}
      >
        {value.toFixed(2)}
      </div>
      {ci && (
        <>
          <div className="relative mt-1 h-1 w-28 rounded-full bg-white/[0.07]">
            <div
              className="absolute h-1 rounded-full bg-accent/50"
              style={{ left: `${lo * 100}%`, width: `${Math.max(1, (hi - lo) * 100)}%` }}
            />
            <div
              className="neon-dot absolute -top-0.5 h-2 w-0.5 rounded bg-accent"
              style={{ left: `${value * 100}%` }}
            />
          </div>
          <div className="mt-0.5 font-mono text-[10px] tabular-nums text-muted">
            95% CI {lo.toFixed(2)}–{hi.toFixed(2)}
          </div>
        </>
      )}
    </div>
  );
}

/** Consensus / polarisation badge — surfaces a split audience a mean would hide. */
export function ConsensusBadge({
  consensus,
  polarization,
  bimodality,
}: {
  consensus?: string;
  polarization?: number;
  bimodality?: number | null;
}) {
  if (!consensus) return null;
  const polarised = consensus === "polarised";
  return (
    <div
      className={`rounded-lg border px-3 py-2 ${
        polarised ? "border-critical/40 bg-critical/[0.07]" : "border-hairline bg-white/[0.03]"
      }`}
    >
      <div className="hud-label">Audience</div>
      <div
        className={`font-display text-sm font-semibold ${
          polarised ? "text-critical" : "text-ink-2"
        }`}
      >
        {consensus}
      </div>
      <div className="mt-0.5 font-mono text-[10px] tabular-nums text-muted">
        entropy {(polarization ?? 0).toFixed(2)}
        {bimodality != null && ` · BC ${bimodality.toFixed(2)}`}
      </div>
      {polarised && (
        <div className="mt-1 max-w-52 text-[11px] leading-tight text-critical/90">
          Two camps — the average is misleading. Segment before acting.
        </div>
      )}
    </div>
  );
}

/** Bayesian A/B verdict: P(best) bars, credible intervals, expected loss. */
/* Lanczos log-gamma so we can draw the actual Beta(α,β) posterior densities —
   the same distributions the backend samples from, rendered exactly. */
const _G = [
  676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012,
  9.9843695780195716e-6, 1.5056327351493116e-7,
];
function lnGamma(z: number): number {
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lnGamma(1 - z);
  z -= 1;
  let a = 0.99999999999980993;
  for (let i = 0; i < _G.length; i++) a += _G[i] / (z + i + 1);
  const t = z + _G.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
}
function betaLogPdf(x: number, a: number, b: number): number {
  return (
    (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) -
    (lnGamma(a) + lnGamma(b) - lnGamma(a + b))
  );
}

const ARM_COLORS = ["#4fb6ff", "#f2ad1f", "#16d016", "#f078c8", "#b07ff0", "#38c8c8"];

/** The posteriors themselves, drawn as density curves — overlap = uncertainty. */
function PosteriorDensities({ bayes }: { bayes: BayesianResult }) {
  const W = 560, H = 110, pad = 26;
  const N = 160;
  const curves = bayes.arms.map((arm) => {
    const a = arm.successes + 1, b = arm.trials - arm.successes + 1;
    const ys: number[] = [];
    for (let i = 1; i < N; i++) {
      const x = i / N;
      ys.push(Math.exp(betaLogPdf(x, a, b)));
    }
    return ys;
  });
  const peak = Math.max(...curves.flat(), 1e-9);
  const px = (i: number) => pad + (i / (N - 2)) * (W - pad * 2);
  const py = (v: number) => H - 16 - (v / peak) * (H - 30);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0, 0.25, 0.5, 0.75, 1].map((v) => (
        <g key={v}>
          <line x1={pad + v * (W - pad * 2)} y1={10} x2={pad + v * (W - pad * 2)} y2={H - 16} stroke="rgba(120,165,230,0.08)" />
          <text x={pad + v * (W - pad * 2)} y={H - 5} fontSize="7.5" fill="#5f6b7d" textAnchor="middle" fontFamily="ui-monospace, monospace">
            {v}
          </text>
        </g>
      ))}
      {curves.map((ys, k) => {
        const color = ARM_COLORS[k % ARM_COLORS.length];
        const line = ys.map((v, i) => `${px(i)},${py(v)}`).join(" ");
        return (
          <g key={k}>
            <path
              d={`M ${px(0)},${H - 16} L ${line.split(" ").join(" L ")} L ${px(ys.length - 1)},${H - 16} Z`}
              fill={color} opacity="0.10"
            />
            <polyline points={line} fill="none" stroke={color} strokeWidth="1.5" opacity="0.9" />
            <text x={pad + 4} y={20 + k * 10} fontSize="7.5" fill={color} fontFamily="ui-monospace, monospace">
              — {bayes.arms[k].name} · Beta({bayes.arms[k].successes + 1}, {bayes.arms[k].trials - bayes.arms[k].successes + 1})
            </text>
          </g>
        );
      })}
      <text x={W - pad} y={H - 5} fontSize="7.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">
        conversion rate θ
      </text>
    </svg>
  );
}

export function BayesianPanel({ bayes }: { bayes: BayesianResult }) {
  return (
    <div
      className={`rounded-xl border p-4 ${
        bayes.conclusive ? "border-good/35 bg-good/[0.05]" : "border-accent-2/40 bg-accent-2/[0.05]"
      }`}
    >
      <div className="mb-2 flex items-center gap-2">
        <span className="hud-label">Bayesian verdict</span>
        <span
          className={`rounded-full px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider ${
            bayes.conclusive ? "bg-good/20 text-good" : "bg-accent-2/20 text-accent-2"
          }`}
        >
          {bayes.conclusive ? "conclusive" : "inconclusive"}
        </span>
      </div>
      <p className={`mb-3 text-sm ${bayes.conclusive ? "text-ink-2" : "text-accent-2"}`}>
        {bayes.verdict}
      </p>

      {/* the actual posterior distributions the verdict is computed from */}
      <div className="mb-3 rounded-lg border border-hairline/60 bg-black/30 p-2">
        <PosteriorDensities bayes={bayes} />
      </div>

      <div className="space-y-2.5">
        {bayes.arms.map((a) => (
          <div key={a.name}>
            <div className="mb-1 flex items-baseline justify-between text-xs">
              <span className="text-ink-2">{a.name}</span>
              <span className="font-mono tabular-nums text-muted">
                {a.successes}/{a.trials} conv · P(best){" "}
                <span className="text-accent">{(a.p_best * 100).toFixed(0)}%</span>{" "}
                <span className="text-[10px] text-muted">
                  [{(a.ci_low * 100).toFixed(0)}–{(a.ci_high * 100).toFixed(0)}%]
                </span>{" "}
                · E[loss] {a.expected_loss.toFixed(3)}
              </span>
            </div>
            {/* P(best) bar with the 95% credible interval drawn beneath */}
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className="h-full rounded-full bg-accent transition-all"
                style={{ width: `${a.p_best * 100}%` }}
              />
            </div>
            <div className="relative mt-1 h-1 w-full rounded-full bg-white/[0.04]">
              <div
                className="absolute h-1 rounded-full bg-neon/40"
                style={{
                  left: `${a.ci_low * 100}%`,
                  width: `${Math.max(1, (a.ci_high - a.ci_low) * 100)}%`,
                }}
                aria-label={`95% credible interval ${a.ci_low.toFixed(2)} to ${a.ci_high.toFixed(2)}`}
              />
            </div>
          </div>
        ))}
      </div>
      <p className="mt-3 font-mono text-[10px] text-muted">
        Beta-Binomial posteriors · {bayes.draws.toLocaleString()} Monte-Carlo draws · ship when
        E[loss] &lt; {bayes.rope}
      </p>
    </div>
  );
}

/** Kaplan-Meier survival curve + hazard bars for a funnel. */
export function SurvivalPanel({
  survival,
  steps,
}: {
  survival: SurvivalResult;
  steps: { step: string }[];
}) {
  const worst = survival.worst_step_index;
  return (
    <div className="rounded-xl border border-hairline bg-white/[0.02] p-4">
      <div className="mb-1 flex items-center gap-2">
        <span className="hud-label">Kaplan–Meier survival</span>
        <span className="font-mono text-[10px] text-muted">
          overall S = {(survival.overall_survival * 100).toFixed(0)}%
        </span>
      </div>
      <p className="mb-3 text-[11px] leading-tight text-muted">
        Hazard is the conditional drop risk <em>given</em> an agent reached the step — it finds the
        real killer step, which raw drop counts miss.
      </p>
      <div className="space-y-2">
        {survival.steps.map((s, i) => (
          <div key={i}>
            <div className="mb-0.5 flex items-baseline justify-between text-[11px]">
              <span className={i === worst ? "font-medium text-critical" : "text-ink-2"}>
                {i + 1}. {steps[i]?.step?.slice(0, 40) ?? `step ${i + 1}`}
                {i === worst && " ← killer step"}
              </span>
              <span className="font-mono tabular-nums text-muted">
                hazard {(s.hazard * 100).toFixed(0)}% · S {(s.survival * 100).toFixed(0)}%
                <span className="ml-1 text-[10px] text-muted">
                  {s.survival_ci_high - s.survival_ci_low <= 0
                    ? "· no spread"
                    : `· [${(s.survival_ci_low * 100).toFixed(0)}–${(s.survival_ci_high * 100).toFixed(0)}%]`}
                </span>
              </span>
            </div>
            <div className="flex gap-1">
              {/* The band is DRAWN now. This bar carried a `title` promising a
                  survival band and rendered only the point estimate, so the
                  interval existed in the payload, was named in a tooltip
                  nobody waits for, and was invisible in the one place it
                  mattered. A survival curve without its band is the single
                  most over-read chart in analytics. */}
              <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className="absolute inset-y-0 rounded-full bg-accent/25"
                  style={{
                    left: `${s.survival_ci_low * 100}%`,
                    width: `${Math.max(0, s.survival_ci_high - s.survival_ci_low) * 100}%`,
                  }}
                />
                <div
                  className="absolute inset-y-0 left-0 rounded-full bg-accent/70"
                  style={{ width: `${s.survival * 100}%` }}
                />
                {/* the point estimate, so it stays readable inside the band */}
                <div
                  className="absolute inset-y-0 w-px bg-ink"
                  style={{ left: `${s.survival * 100}%` }}
                />
              </div>
              <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/[0.06]">
                <div
                  className={`h-full rounded-full ${i === worst ? "bg-critical" : "bg-accent-2/70"}`}
                  style={{ width: `${s.hazard * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Elasticity readout for the price study. */
export function ElasticityPanel({ econ }: { econ: Econometrics }) {
  if (!econ.ok) return null;
  const e = econ.elasticity;
  return (
    <div className="rounded-xl border border-hairline bg-white/[0.02] p-4">
      <div className="hud-label mb-2">Econometrics</div>
      <div className="flex flex-wrap gap-6">
        <div>
          <div className="font-mono text-[10px] text-muted">price elasticity ε</div>
          <div className="font-display text-2xl font-medium tabular-nums text-accent">
            {e != null ? e.toFixed(2) : "—"}
          </div>
          {econ.elasticity_r2 != null && (
            <div className="font-mono text-[10px] text-muted">
              log-log fit R² {econ.elasticity_r2.toFixed(2)}
            </div>
          )}
        </div>
        <div>
          <div className="font-mono text-[10px] text-muted">continuous optimum</div>
          <div className="font-display text-2xl font-medium tabular-nums text-good">
            {econ.continuous_optimal_price != null ? `$${econ.continuous_optimal_price}` : "—"}
          </div>
          {econ.demand_fit_r2 != null && (
            <div className="font-mono text-[10px] text-muted">
              demand fit R² {econ.demand_fit_r2.toFixed(2)}
            </div>
          )}
        </div>
      </div>
      {econ.classification && (
        <p className="mt-2 text-[11px] leading-tight text-ink-2">{econ.classification}</p>
      )}
    </div>
  );
}

const ACTION_COLOR: Record<string, string> = {
  convert: "#16d016",
  continue: "#3f8ff0",
  hesitate: "#f2ad1f",
  abandon: "#f0605f",
};

/** Audience segments discovered by k-means++ (k chosen by silhouette). Only
 *  rendered when real structure exists — the backend returns null otherwise. */
export function SegmentsPanel({ segments }: { segments: SegmentReport }) {
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <span className="hud-label">DISCOVERED SEGMENTS</span>
        <span className="border border-hairline px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-ink-2">
          k-means++ · k={segments.k}
        </span>
        <span className="font-mono text-[10px] text-muted">
          silhouette {segments.silhouette.toFixed(2)}
        </span>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {segments.groups.map((g, gi) => {
          const color = ACTION_COLOR[g.dominant_action] ?? "#3f8ff0";
          return (
            <div
              key={gi}
              className="rounded-lg border border-hairline bg-black/30 p-3"
              style={{ borderLeftColor: color, borderLeftWidth: 2 }}
            >
              <div className="flex items-baseline justify-between">
                <span className="font-display text-sm font-semibold capitalize" style={{ color }}>
                  {g.label}
                </span>
                <span className="font-mono text-[10px] tabular-nums text-muted">
                  {g.size} twins · {Math.round(g.share * 100)}%
                </span>
              </div>
              {/* share bar */}
              <div className="mt-2 h-1 w-full rounded-full bg-white/[0.06]">
                <div
                  className="h-1 rounded-full"
                  style={{ width: `${g.share * 100}%`, background: color, opacity: 0.75 }}
                />
              </div>
              <div className="mt-2 flex gap-4 font-mono text-[10px] tabular-nums text-ink-2">
                <span>intent {g.mean_intent.toFixed(2)}</span>
                <span>engage {g.mean_engagement.toFixed(2)}</span>
                <span className="text-muted">→ {g.dominant_action}</span>
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-[10px] leading-tight text-muted">
        Segments found by clustering each twin&apos;s (intent, engagement) — k picked by
        silhouette score, reported only when structure genuinely exists.
      </p>
    </div>
  );
}
