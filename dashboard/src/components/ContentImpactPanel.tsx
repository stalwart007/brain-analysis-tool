"use client";

/**
 * NEURO-IMPACT STUDY — feed content (script / ad / article), the swarm
 * experiences it beat by beat, and the analysis stack answers "what does this
 * do to an audience": attention synchrony (ISC), structural drop-off points
 * (change-point detection), what memory will keep (peak-end + serial
 * position), the emotional trajectory (circumplex), and a functional brain
 * atlas per beat. Every number on screen is a model output.
 */

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CognitiveLoad } from "@/lib/api";
import { streamRun } from "@/lib/stream";
import { LoadToggle } from "./LoadToggle";
import BrainMap, { SystemScores } from "./BrainMap";
import SimStage from "./sim/SimStage";
import { ChartCursorProvider, useChartCursor } from "./charts/cursor";
import ChartFrame from "./charts/ChartFrame";
import TensorSim from "./sim/TensorSim";
import TensorExplorer from "./TensorExplorer";
import FindingsPanel, { ResearchQuestion, type FindingsBlock } from "./FindingsPanel";
import { AssetInput, assetReady, type AssetKind, type ContentAsset } from "./AssetInput";
import { clock, type YouTubeManifest } from "./content/YouTubeIngest";
import VideoTimeline from "./content/VideoTimeline";
import BeatDiagnostic from "./content/BeatDiagnostic";
import SynchronyNull, { type IscBlock } from "./content/SynchronyNull";

interface RetentionStep {
  beat: number;
  entered: number;
  dropped: number;
  /** null once the risk set is empty — UNDEFINED, not zero. */
  hazard: number | null;
  survival: number;
}

interface ContentResult {
  segments: string[];
  /** temporal | sequential | spatial — decides which statistics below exist. */
  axis?: string | null;
  beat_source?: string | null;
  beat_notes?: string | null;
  modality?: string;
  /** statistic -> why it was not computed for this modality. */
  withheld?: Record<string, string>;
  retention?: {
    threshold: number;
    steps: RetentionStep[];
    completion_rate: number;
    worst_beat: number | null;
  } | null;
  twin_count: number;
  curves: Record<string, number[]>;
  /** SIMULTANEOUS (sup-t) bands — valid to read across the whole curve. */
  curve_cis: Record<string, ([number, number] | null)[] | null>;
  /** Per-beat percentile intervals, valid only for a single pre-registered beat. */
  curve_pointwise_cis?: Record<string, ([number, number] | null)[] | null>;
  band_method?: string;
  findings?: FindingsBlock | null;
  /** Where each beat sits on the clock, for content that has one. Null for a
   *  deck or an article, which have an order and no duration. */
  timestamps_ms?: number[] | null;
  isc:
    | (IscBlock & { leave_one_out: (number | null)[] })
    | null;
  change_points: number[] | null;
  memory: {
    remembered_affect: number;
    peak_index: number;
    peak_index_distribution: number[];
    peak_agreement: number;
    encoding_strength: number;
    n_twins: number;
    model: string;
  } | null;
  trajectory: { dynamism: number; net_valence_shift: number; quadrant_occupancy: Record<string, number>; model: string } | null;
  systems: SystemScores;
  per_segment_systems: SystemScores[];
  share_intent_mean: number;
  gists: string[];
  verdict: string;
  disclaimer: string;
}

const SAMPLE = `Scene 1 (0:00): Black screen. A phone buzzes at 3:12 AM. A founder stares at a wall of unread Slack messages.
Scene 2 (0:08): Quick cuts — 12 browser tabs, 4 dashboards, a calendar with no gaps. Voiceover: "Your tools were supposed to help."
Scene 3 (0:18): Everything freezes. One clean interface slides in: Atlas. The clutter dissolves into a single feed.
Scene 4 (0:26): The founder types one sentence; Atlas drafts the plan, books the meetings, files the tickets.
Scene 5 (0:34): Morning. The founder is asleep. Atlas's status: "11 hours of busywork handled."
Scene 6 (0:40): Logo. "Atlas — get your week back." Free 14-day trial.`;

/** The reductions the backend runs once every twin has reported. Named here
 *  so the wait after the last arrival reads as work rather than a stall — ISC
 *  alone runs 500 permutations plus a twin-level bootstrap. Ids match the
 *  `stage` field on the wire; axis-inapplicable passes simply never fire. */
const ANALYSIS_PASSES = [
  { id: "isc", label: "synchrony" },
  { id: "changepoints", label: "change points" },
  { id: "memory", label: "peak-end" },
  { id: "retention", label: "retention" },
  { id: "systems", label: "systems" },
] as const;


export default function ContentImpactPanel(props: { personaCount: number }) {
  return (
    <ChartCursorProvider>
      <ContentImpactInner {...props} />
    </ChartCursorProvider>
  );
}


function ContentImpactInner({ personaCount }: { personaCount: number }) {
  const [kind, setKind] = useState<AssetKind>("text");
  const [asset, setAsset] = useState<ContentAsset>({ kind: "text" });
  const [question, setQuestion] = useState("");
  const [twins, setTwins] = useState(3);
  const [load, setLoad] = useState<CognitiveLoad>("low");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const [segments, setSegments] = useState<string[]>([]);
  const [traces, setTraces] = useState<number[][]>([]); // live attention curves
  const [expected, setExpected] = useState(0);   // twins dispatched, from `start`
  const [passes, setPasses] = useState<string[]>([]); // analysis stages seen
  const [result, setResult] = useState<ContentResult | null>(null);
  // Kept beside the result because it carries what the RESULT cannot: the real
  // runtime, the chapter titles, and the still frames the beats were read from.
  // A beat labelled "B3" is a bar on a chart; the same beat labelled "1:24 —
  // The fix" with its frame under it is a moment in a video someone can go and
  // watch.
  const [youtube, setYoutube] = useState<YouTubeManifest | null>(null);
  // ONE cursor for the beat axis, shared by the attention field, the
  // peak-index distribution, the retention curve and the brain-map scrubber.
  // The scrubber used to own a separate `beat` state, so clicking a beat there
  // highlighted nothing anywhere else — four views of one tensor behaving as
  // four unrelated pictures.
  const beatCursor = useChartCursor(
    "beats",
    segments.length,
    "Content beats",
    (i) => `beat ${i + 1} of ${segments.length}`
  );
  const beat = beatCursor.index;
  const setBeat = (i: number | null) => (i === null ? beatCursor.clear() : beatCursor.togglePin(i));
  // Twins are a different axis from beats — a separate domain, so hovering a
  // twin never lights up "beat 3" somewhere else and imply a link that does
  // not exist.
  async function run() {
    setBusy(true); setError(null); setResult(null);
    setTraces([]); setSegments([]); beatCursor.clear(); setPasses([]); setExpected(0);
    setStage("segmenting content…");
    try {
      await streamRun(
        "/studies/content/stream",
        {
          asset: { ...asset, kind },
          twins_per_persona: twins,
          cognitive_load: load,
          research_question: question,
        },
        (evt) => {
          if (evt.type === "stage") {
            setStage(String(evt.detail));
            // Analysis passes arrive AFTER the last twin. Without them the
            // stage sat silent through ISC's 500 permutations and read as a
            // hang at exactly the moment the heaviest work is happening.
            const id = String(evt.stage ?? "");
            if (id && id !== "segmenting") setPasses((p) => (p.includes(id) ? p : [...p, id]));
          }
          else if (evt.type === "start") {
            setSegments(evt.segments as string[]);
            setExpected(Number(evt.total ?? 0) || 0);
            setStage(`${(evt.segments as string[]).length} beats — audience experiencing content…`);
          } else if (evt.type === "agent") setTraces((t) => [...t, evt.attention as number[]]);
          else if (evt.type === "done") setResult(evt.result as unknown as ContentResult);
          else if (evt.type === "error") setError(String(evt.detail));
        }
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "Study failed");
    } finally {
      setBusy(false); setStage(null);
    }
  }

  return (
    <div className="card p-6">
      <div className="mb-1 flex items-baseline justify-between">
        <h2 className="font-display text-lg font-medium tracking-tight">Neuro-Impact Study</h2>
        <span className="text-xs text-muted">what the content does to an audience, beat by beat</span>
      </div>
      <p className="lede mb-4">
        Paste a link to anything — a YouTube video, a podcast, a landing page, a
        deck, an image — or paste the copy itself. We work out what it is; you
        do not classify it. The swarm then experiences it beat by beat and you
        get attention synchrony, drop-off, what memory keeps, the emotional arc,
        and a functional brain atlas. Statistics the format cannot support are
        withheld with their reason rather than computed anyway.
      </p>

      {/* The modality picker used to live here, ABOVE the input, as a gate:
          choose what your content is, then get the box that accepts it. It now
          lives inside `AssetInput` underneath the one box, because it reports
          what the server detected rather than asking the researcher to
          classify their own URL. Each kind still reaches the study through a
          different adapter and lands on a different AXIS — that part was always
          right, and is what decides whether peak-end, change points and
          retention are meaningful at all. */}
      <AssetInput
        kind={kind}
        asset={asset}
        onChange={setAsset}
        onKindChange={(k) => {
          setKind(k);
          setResult(null);
        }}
        disabled={busy}
        onYouTube={setYoutube}
      />

      {/* The one field that changes what the REPORT is about rather than what
          the run measures. Placed after the asset because the useful questions
          are the ones you think of once you are looking at the thing. */}
      <div className="mt-4">
        <ResearchQuestion
          value={question}
          onChange={setQuestion}
          disabled={busy}
          examples={[
            "e.g. where do we lose people, and why there?",
            "does the opening earn the next 10 seconds?",
            "is the offer the weakest beat?",
            "does this read as threatening to a first-time buyer?",
          ]}
        />
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        {kind === "text" && (
          <button
            onClick={() => setAsset({ kind: "text", text: SAMPLE })}
            className="rounded-lg border border-dashed border-hairline px-3 py-1.5 text-xs text-muted transition hover:border-accent/50 hover:text-ink-2"
          >
            use sample script
          </button>
        )}
        <label className="flex items-center gap-2 text-xs text-muted">
          Twins / persona
          <input
            type="number" min={1} max={20} value={twins}
            onChange={(e) => setTwins(parseInt(e.target.value, 10) || 1)}
            className="w-16 rounded-lg border border-hairline bg-surface-2 px-2 py-1.5 text-sm text-ink outline-none tabular-nums"
          />
        </label>
        <LoadToggle value={load} onChange={setLoad} id="content-load" />
        <motion.button
          whileTap={{ scale: 0.97 }}
          disabled={busy || !!assetReady(kind, asset)}
          onClick={run}
          title={assetReady(kind, asset) ?? undefined}
          className="ml-auto rounded-xl bg-accent px-5 py-2 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
        >
          {busy ? "Screening…" : "Run neuro-impact study"}
        </motion.button>
      </div>

      {/* The appraisal tensor filling in. This panel was the only one of the
          six with a live stage that showed NOTHING on Run, while starting the
          heaviest computation in the product. The tensor is the honest subject:
          ISC, change points, peak-end and retention are all reductions over it,
          so watching it populate is watching the actual work. */}
      <SimStage
        show={busy || !!result}
        busy={busy}
        runningLabel={
          segments.length
            ? `${traces.length}/${expected || "…"} twins × ${segments.length} beats`
            : "reading content"
        }
        doneLabel={`tensor complete · ${traces.length} × ${segments.length} × 9 dims`}
        progress={stage ?? undefined}
        height={250}
      >
        <TensorSim
          rows={traces.map((attention) => ({ attention }))}
          beats={segments.length}
          expected={expected}
          stages={ANALYSIS_PASSES.map((p) => ({
            id: p.id,
            label: p.label,
            done: passes.includes(p.id) && !busy,
            active: busy && passes[passes.length - 1] === p.id,
          }))}
        />
      </SimStage>

      {stage && (
        <p className="mt-3 font-mono text-[11px] text-accent">
          <span className="pulse-dot mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-accent align-middle" />
          {stage} {traces.length > 0 && `· ${traces.length} twins reported`}
        </p>
      )}
      {error && <p className="mt-3 text-sm text-critical">{error}</p>}

      {/* live ghost traces: every twin's attention curve as it arrives */}
      {(busy || result) && traces.length > 0 && segments.length > 1 && (
        <AttentionField traces={traces} result={result} segments={segments} />
      )}

      {result && (
        <div className="mt-5 space-y-5">
          {/* THE ANSWER LEADS. The template verdict below is a one-line
              summary of the statistics; this is what the study means for what
              was actually asked, with every claim wired to the measurement
              that licenses it. */}
          <FindingsPanel findings={result.findings} />

          <p className="verdict rounded-xl border border-accent/25 bg-accent/5 px-3 py-2 text-ink-2">
            <span className="hud-label mr-2">VERDICT</span>{result.verdict}
          </p>

          {/* How the beats were derived, and what this format cannot support.
              An absent statistic and an inapplicable one look identical in a
              UI, and only one of them is a finding — so the reasons are shown
              rather than the rows silently missing. */}
          {(result.axis || result.beat_notes) && (
            <p className="mt-2 font-mono text-[10px] leading-relaxed text-muted">
              {result.modality && <>{result.modality} · </>}
              {result.axis && <>{result.axis} axis · </>}
              {result.beat_source === "paragraph_fallback" ? (
                <span className="text-critical">
                  beats are a mechanical split, not model-identified structure
                </span>
              ) : (
                <>beats via {result.beat_source}</>
              )}
              {result.beat_notes && <> · {result.beat_notes}</>}
            </p>
          )}
          {result.withheld && Object.keys(result.withheld).length > 0 && (
            <div className="mt-2 rounded-xl border border-hairline bg-surface-2/40 px-3 py-2">
              <span className="hud-label text-muted">NOT COMPUTED FOR THIS FORMAT</span>
              <ul className="mt-1 space-y-1">
                {Object.entries(result.withheld).map(([stat, reason]) => (
                  <li key={stat} className="text-[10px] leading-relaxed text-muted">
                    <b className="text-ink-2">{stat.replace(/_/g, " ")}</b> — {reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* On a real clock, with the frames the beats were read from. Only
              for content that HAS a clock — `timestamps_ms` is absent for a
              deck or an article, and drawing those against elapsed time would
              invent a duration they do not have. */}
          {asset.frames && asset.frames.length > 0 && result.timestamps_ms && (
            <VideoTimeline
              frames={asset.frames}
              timestampsMs={result.timestamps_ms}
              attention={result.curves.attention}
              segments={result.segments}
              durationS={youtube?.duration_s ?? Math.max(...result.timestamps_ms) / 1000}
              chapters={youtube?.chapters}
              changePoints={result.change_points}
              cursor={beatCursor}
            />
          )}

          {/* The whole tensor, not one ninth of it. Reading DOWN a column —
              everything that happened at one beat — is the view that was
              impossible before, and it is usually what explains a weak beat:
              attention falls AND cognitive effort spikes means confusing, not
              boring, and those want opposite fixes. */}
          <TensorExplorer
            curves={result.curves}
            bands={result.curve_cis}
            segments={result.segments}
            cursor={beatCursor}
            twinCurves={traces}
          />

          {/* …and that same reading, performed rather than left to the eye. */}
          {result.curves.cognitive_effort && (
            <BeatDiagnostic
              attention={result.curves.attention}
              effort={result.curves.cognitive_effort}
              segments={result.segments}
              cursor={beatCursor}
              timeLabels={
                result.timestamps_ms?.map((t) => clock(t / 1000)) ?? undefined
              }
            />
          )}

          <div className="grid gap-5 lg:grid-cols-2">
            {/* brain atlas with beat scrubber */}
            <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
              <div className="mb-2 flex flex-wrap items-center gap-1.5">
                <span className="hud-label mr-1">FUNCTIONAL BRAIN ATLAS</span>
                <button
                  onClick={() => setBeat(null)}
                  className={`rounded-full border px-2 py-0.5 font-mono text-[9px] ${beat === null ? "border-accent text-accent" : "border-hairline text-muted"}`}
                >
                  OVERALL
                </button>
                {result.segments.map((_, i) => (
                  <button
                    key={i}
                    onClick={() => setBeat(i)}
                    className={`rounded-full border px-2 py-0.5 font-mono text-[9px] ${beat === i ? "border-accent text-accent" : "border-hairline text-muted"}`}
                  >
                    B{i + 1}
                  </button>
                ))}
              </div>
              <BrainMap
                scores={beat === null ? result.systems : result.per_segment_systems[beat]}
                caption={
                  beat === null
                    ? undefined
                    : `Beat ${beat + 1}: ${result.segments[beat].slice(0, 70)}`
                }
              />
            </div>

            <div className="space-y-4">
              {/* ISC */}
              {result.isc && <IscSynchrony isc={result.isc} />}

              {/* The permutation null behind the p-value beside it. 500 full
                  re-runs of the estimator were already being spent to produce
                  that number and then reported as three decimal places. */}
              {result.isc?.null_r_distribution?.length ? (
                <SynchronyNull isc={result.isc} />
              ) : null}

              {/* memory — the peak-index DISTRIBUTION, not a softmax.
                  Peak-end is a within-subject phenomenon, so this is where
                  each twin individually peaked. One tall bar = the audience
                  shares a peak; several = the audience is split, which the old
                  mean-curve version silently averaged away. */}
              {result.memory && (
                <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
                  <span className="hud-label">WHERE EACH TWIN PEAKED</span>
                  <div className="mt-2 flex items-end gap-1">
                    {result.memory.peak_index_distribution.map((count, i) => {
                      const top = Math.max(...result.memory!.peak_index_distribution, 1);
                      // Shares the "beats" domain with the attention field and
                      // the retention curve, so hovering a beat in any of them
                      // lights it up in all three. These are three reductions
                      // of one tensor; reading across them used to mean
                      // counting bars with a finger.
                      const on = beatCursor.isMarked(i);
                      return (
                        <div key={i} className="flex-1 text-center">
                          <div
                            className="mx-auto w-full rounded-t-sm transition-opacity"
                            style={{
                              height: `${4 + (count / top) * 94}px`,
                              background: i === result.memory!.peak_index ? "#f2ad1f" : "#3f8ff0",
                              opacity: on ? 1 : count ? 0.85 : 0.18,
                              outline: on ? "1px solid rgba(255,255,255,0.55)" : "none",
                            }}
                          />
                          <span className={`font-mono text-[8px] ${on ? "text-ink" : "text-muted"}`}>
                            B{i + 1}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <p className="mt-1 text-[9px] text-muted">
                    peak agreement {(result.memory.peak_agreement * 100).toFixed(0)}%
                    {result.memory.peak_agreement < 0.5 && (
                      <span className="text-critical">
                        {" "}— the audience does not share a peak
                      </span>
                    )}
                    {" "}· remembered affect {result.memory.remembered_affect > 0 ? "+" : ""}
                    {result.memory.remembered_affect.toFixed(2)} · encoding{" "}
                    {result.memory.encoding_strength.toFixed(2)}
                  </p>
                </div>
              )}

              {/* retention — the commercially load-bearing number for anything
                  with a running order: where do people stop. */}
              {result.retention && result.retention.steps.length > 0 && (
                <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
                  <span className="hud-label">RETENTION</span>
                  <div className="mt-2 flex items-end gap-1">
                    {result.retention.steps.map((st) => (
                      <div key={st.beat} className="flex-1 text-center">
                        <div
                          className="mx-auto w-full rounded-t-sm transition-opacity"
                          style={{
                            height: `${4 + st.survival * 94}px`,
                            // An undefined hazard (empty risk set) is drawn
                            // hollow, never as a zero-hazard bar — "nobody left
                            // here" and "nobody was left to leave" are
                            // different findings.
                            background: st.hazard === null ? "transparent" : "#4fb477",
                            border: st.hazard === null ? "1px dashed rgba(255,255,255,0.25)" : "none",
                            opacity: beatCursor.isMarked(st.beat) ? 1 : 0.85,
                            outline: beatCursor.isMarked(st.beat)
                              ? "1px solid rgba(255,255,255,0.55)"
                              : "none",
                          }}
                        />
                        <span
                          className={`font-mono text-[8px] ${
                            beatCursor.isMarked(st.beat) ? "text-ink" : "text-muted"
                          }`}
                        >
                          B{st.beat + 1}
                        </span>
                      </div>
                    ))}
                  </div>
                  <p className="mt-1 text-[9px] text-muted">
                    {(result.retention.completion_rate * 100).toFixed(0)}% reached the end
                    {result.retention.worst_beat !== null && (
                      <> · steepest drop at beat {result.retention.worst_beat + 1}</>
                    )}
                  </p>
                </div>
              )}

              {/* trajectory */}
              {result.trajectory && <Circumplex result={result} />}
            </div>
          </div>

          <div className="flex flex-wrap gap-6 font-mono text-[11px] text-ink-2">
            <span>share intent <b className="text-accent">{Math.round(result.share_intent_mean * 100)}%</b></span>
            <span>emotional dynamism <b className="text-accent">{result.trajectory?.dynamism.toFixed(2)}</b></span>
            <span>{result.twin_count} twins · load {load}</span>
          </div>

          {result.gists.length > 0 && (
            <details className="text-xs text-muted">
              <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider hover:text-ink-2">what twins took away</summary>
              <ul className="mt-1.5 space-y-1 border-l border-hairline pl-3">
                {result.gists.map((g, i) => <li key={i} className="text-[11px] text-ink-2">{g}</li>)}
              </ul>
            </details>
          )}
          <p className="text-[9px] leading-tight text-muted">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}

/* ── attention field: ghost curve per twin + mean + CI + change points ── */

/** Attention synchrony, with its per-twin leave-one-out strip.
 *
 * Extracted from the panel body so the cursor it needs lives with the only
 * subtree that ever used it, and so the readout's reserved height is
 * measurable in isolation rather than only behind a completed study.
 */
export function IscSynchrony({
  isc,
}: {
  isc: { overall: number; grip: string; leave_one_out: (number | null)[]; method: string };
}) {
  const looCursor = useChartCursor(
    "twins",
    isc.leave_one_out.length,
    "Per-twin leave-one-out synchrony",
    (i) => {
      const r = isc.leave_one_out[i];
      return `twin ${i + 1} left out, synchrony ${r == null ? "unresolvable" : r.toFixed(2)}`;
    }
  );

  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <div className="flex items-baseline justify-between">
        <span className="hud-label">ATTENTION SYNCHRONY (ISC)</span>
        {/* Colour follows the backend's `grip`, which is now gated on a
            permutation test. Re-deriving it here from raw r cuts was a
            third threshold ladder that could contradict the label
            printed beside it. */}
        <span
          className={`font-display text-xl font-semibold tabular-nums ${
            isc.grip === "locked-in"
              ? "text-good"
              : isc.grip === "engaged"
                ? "text-accent"
                : "text-critical"
          }`}
        >
          {isc.overall.toFixed(2)} · {isc.grip}
        </span>
      </div>
      {/* Leave-one-out ISC: each bar is the synchrony of the group WITHOUT that
          twin, so a conspicuously tall bar marks the twin who was dragging
          agreement down. That is the whole diagnostic, and it was previously a
          `title` away. */}
      <div
        {...looCursor.surfaceProps}
        className="mt-2 flex h-6 items-end gap-0.5 focus-visible:ring-1 focus-visible:ring-accent/60"
      >
        {isc.leave_one_out.map((r, i) => (
          <div
            key={i}
            className="flex-1 rounded-t-sm bg-accent transition-opacity"
            style={{
              height: `${Math.max(6, ((r ?? 0) + 1) * 50)}%`,
              opacity:
                looCursor.index === null
                  ? 0.35 + ((r ?? 0) + 1) * 0.3
                  : looCursor.isMarked(i)
                    ? 1
                    : 0.18,
              outline: looCursor.isPinned(i) ? "1px solid rgb(var(--region-rgb))" : undefined,
            }}
          />
        ))}
      </div>
      {/* Reserved height. `min-h-3` was 12px — a single line's worth of guard
          for three chained clauses that wrap to two or three lines inside this
          `lg:grid-cols-2` column, so sweeping the leave-one-out strip pushed the
          memory distribution and every panel below it down and back. The idle
          branch carries `isc.method`, a server-authored string, so BOTH states
          are variable-length and both are clamped. */}
      <p className="mt-1 min-h-[2.6rem] font-mono text-[9px] leading-relaxed tabular-nums text-muted">
        {looCursor.index !== null ? (
          <span className="line-clamp-3 text-ink-2">
            twin {looCursor.index + 1} left out · r ={" "}
            <span className="text-accent">
              {isc.leave_one_out[looCursor.index] == null
                ? "unresolvable"
                : isc.leave_one_out[looCursor.index]!.toFixed(3)}
            </span>
            <span className="text-muted">
              {" "}
              {(isc.leave_one_out[looCursor.index] ?? 0) > isc.overall
                ? "· synchrony rises without them — this twin dissents"
                : "· synchrony falls without them — this twin was with the group"}
            </span>
            {looCursor.pinned !== null && (
              <button
                onClick={looCursor.clear}
                className="ml-2 text-accent transition hover:text-ink"
              >
                pinned ✕
              </button>
            )}
          </span>
        ) : (
          <span className="line-clamp-3">
            {isc.method} · per-twin leave-one-out synchrony
          </span>
        )}
      </p>
    </div>
  );
}

function AttentionField({
  traces, result, segments,
}: {
  traces: number[][];
  result: ContentResult | null;
  segments: string[];
}) {
  const W = 720, H = 150, padX = 34, padY = 16;
  const n = segments.length;
  const x = (i: number) => padX + (i / Math.max(1, n - 1)) * (W - padX * 2);
  const y = (v: number) => H - padY - v * (H - padY * 2);
  const mean = useMemo(() => {
    if (result) return result.curves.attention;
    if (!traces.length) return [];
    return Array.from({ length: n }, (_, i) =>
      traces.reduce((s, t) => s + (t[i] ?? 0), 0) / traces.length
    );
  }, [traces, result, n]);
  const cis = result?.curve_cis.attention;
  const band = cis && cis.every(Boolean)
    ? `M ${cis.map((c, i) => `${x(i)},${y((c as [number, number])[1])}`).join(" L ")} L ${[...cis].reverse().map((c, ri) => `${x(cis.length - 1 - ri)},${y((c as [number, number])[0])}`).join(" L ")} Z`
    : null;
  const cursor = useChartCursor(
    "beats",
    n,
    "Audience attention across beats",
    (i) => `beat ${i + 1}: mean attention ${(mean[i] ?? 0).toFixed(2)}`,
    // Both options were missing, and each was its own drift.
    //
    // `x(i) = padX + (i/(n-1))·(W − 2·padX)` puts beat i at an exact vertex, so
    // this is a line chart over POINTS, not bands: the default band mapping
    // hit-tested the midpoints between beats and drew a crosshair whose edges
    // land where no datum is. And with no inset the pointer was mapped across
    // the whole element, ignoring the 34/720 gutter at each end — so the error
    // grew toward the edges, which is exactly where a reader hunts for the
    // first and last beat. `mapping` and `inset` are per-consumer, not
    // per-domain, so this does not disturb the other charts sharing "beats".
    { inset: [padX / W, padX / W], mapping: "point" }
  );
  const [muted, setMuted] = useState<Set<string>>(new Set());

  return (
    <div className="mt-4 rounded-xl border border-hairline bg-black/40 p-3">
      <ChartFrame
        cursor={cursor}
        title="AUDIENCE ATTENTION FIELD"
        categories={segments.map((_, i) => `Beat ${i + 1}`)}
        hidden={muted}
        onToggleSeries={(k) =>
          setMuted((m) => {
            const next = new Set(m);
            next.has(k) ? next.delete(k) : next.add(k);
            return next;
          })
        }
        series={[
          {
            key: "attention",
            label: "attention",
            color: "#4fb6ff",
            values: mean,
            intervals: (cis ?? undefined) as ([number, number] | null)[] | undefined,
          },
          ...(result
            ? [
                { key: "arousal", label: "arousal", color: "#f2ad1f", values: result.curves.arousal },
                { key: "valence", label: "valence", color: "#b07ff0", values: result.curves.valence },
              ]
            : []),
        ]}
        footnote="one trace per twin · bold = population mean · dashed = detected change points · click to pin a beat, arrow keys to step"
        height={150}
      >
      <svg viewBox={`0 0 ${W} ${H}`} className="h-full w-full">
        {[0, 0.5, 1].map((v) => (
          <g key={v}>
            <line x1={padX} y1={y(v)} x2={W - padX} y2={y(v)} stroke="rgba(120,165,230,0.10)" />
            <text x={padX - 6} y={y(v) + 2.5} fontSize="7" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">{v}</text>
          </g>
        ))}
        {band && <path d={band} fill="rgba(79,182,255,0.12)" />}
        {traces.map((t, k) => (
          <polyline key={k} fill="none" stroke="rgba(120,195,255,0.16)" strokeWidth="1"
            points={t.map((v, i) => `${x(i)},${y(Math.max(0, Math.min(1, v)))}`).join(" ")} />
        ))}
        {mean.length > 1 && (
          <polyline fill="none" stroke="#4fb6ff" strokeWidth="2"
            points={mean.map((v, i) => `${x(i)},${y(v)}`).join(" ")} />
        )}
        {result?.change_points?.map((cp) => (
          <g key={cp}>
            <line x1={x(cp)} y1={padY} x2={x(cp)} y2={H - padY} stroke="#f0605f" strokeWidth="1" strokeDasharray="3 3" />
            <text x={x(cp) + 3} y={padY + 7} fontSize="7" fill="#f0605f" fontFamily="ui-monospace, monospace">regime shift</text>
          </g>
        ))}
        {segments.map((_, i) => (
          <text key={i} x={x(i)} y={H - 3} fontSize="7"
            fill={cursor.isMarked(i) ? "#dfd9d9" : "#5f6b7d"}
            textAnchor="middle" fontFamily="ui-monospace, monospace">B{i + 1}</text>
        ))}
        {/* the cursor's own marker on the mean series — the crosshair band
            locates the beat, this locates the VALUE at that beat */}
        {cursor.index !== null && mean[cursor.index] != null && (
          <circle cx={x(cursor.index)} cy={y(mean[cursor.index])} r={3.5}
            fill={cursor.isPinned(cursor.index) ? "rgb(var(--region-rgb))" : "#dfd9d9"} />
        )}
      </svg>
      </ChartFrame>
    </div>
  );
}

/* ── circumplex: the emotional arc through valence × arousal space ────── */

function Circumplex({ result }: { result: ContentResult }) {
  const S = 170, pad = 20;
  const vx = (v: number) => pad + ((v + 1) / 2) * (S - pad * 2);
  const ay = (a: number) => S - pad - a * (S - pad * 2);
  const V = result.curves.valence, A = result.curves.arousal;
  return (
    <div className="rounded-xl border border-hairline bg-surface-2/60 p-3">
      <span className="hud-label">EMOTIONAL TRAJECTORY (CIRCUMPLEX)</span>
      <div className="mt-2 flex gap-3">
        <svg viewBox={`0 0 ${S} ${S}`} className="h-40 w-40 shrink-0">
          <rect x={pad} y={pad} width={S - pad * 2} height={S - pad * 2} fill="none" stroke="rgba(120,165,230,0.15)" />
          <line x1={S / 2} y1={pad} x2={S / 2} y2={S - pad} stroke="rgba(120,165,230,0.12)" />
          <line x1={pad} y1={S / 2} x2={S - pad} y2={S / 2} stroke="rgba(120,165,230,0.12)" />
          <text x={S - pad} y={12} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">arousal ↑</text>
          <text x={S - pad} y={S - 4} fontSize="6.5" fill="#5f6b7d" textAnchor="end" fontFamily="ui-monospace, monospace">valence →</text>
          <polyline fill="none" stroke="rgba(240,120,200,0.7)" strokeWidth="1.5"
            points={V.map((v, i) => `${vx(v)},${ay(A[i])}`).join(" ")} />
          {V.map((v, i) => (
            <g key={i}>
              <circle cx={vx(v)} cy={ay(A[i])} r={i === 0 ? 2 : 2.6} fill={i === V.length - 1 ? "#b07ff0" : "#f078c8"} opacity={0.5 + (i / V.length) * 0.5} />
              <text x={vx(v) + 4} y={ay(A[i]) - 3} fontSize="6" fill="#8fa3c0" fontFamily="ui-monospace, monospace">B{i + 1}</text>
            </g>
          ))}
        </svg>
        <div className="space-y-1 font-mono text-[10px] text-ink-2">
          <div>net valence shift <b className="text-accent">{result.trajectory!.net_valence_shift > 0 ? "+" : ""}{result.trajectory!.net_valence_shift.toFixed(2)}</b></div>
          <div>dynamism <b className="text-accent">{result.trajectory!.dynamism.toFixed(2)}</b></div>
          <div className="pt-1 text-[9px] text-muted">audience time share:</div>
          {Object.entries(result.trajectory!.quadrant_occupancy).map(([q, f]) => (
            <div key={q} className="flex items-center gap-1.5">
              <span className="w-16 text-muted">{q}</span>
              <div className="h-1.5 w-20 rounded bg-white/[0.07]">
                <div className="h-1.5 rounded bg-accent" style={{ width: `${f * 100}%`, opacity: 0.8 }} />
              </div>
              <span className="text-muted">{Math.round(f * 100)}%</span>
            </div>
          ))}
          <p className="pt-1 text-[9px] text-muted">{result.trajectory!.model}</p>
        </div>
      </div>
    </div>
  );
}
