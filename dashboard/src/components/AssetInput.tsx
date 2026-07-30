"use client";

/**
 * Get content into the study. One box, whatever form the content is in.
 *
 * WHAT THIS USED TO BE, and why it changed. Six modalities each owned an
 * input, behind a content-type radio that had to be set correctly BEFORE
 * anything would be read: a YouTube box, a video-link box, a page-link box, a
 * PDF-link box, an audio-link box, a file picker per kind. Six ways to paste a
 * link into the same product, and a picker that acted as a gate — classify
 * your own URL first, and be told off if you got it wrong.
 *
 * The classification was never the researcher's job, and they are worse at it
 * than the server is: `…/promo.mp4` served as HTML is a player page, `…/x?id=9`
 * served as image/png is an image, and looking at either URL tells you
 * nothing. So `UniversalInput` takes anything — a link of any kind, pasted
 * copy, or a file — the server says what it actually is, and the picker below
 * becomes a READOUT of that answer rather than a question asked in advance.
 *
 * WHAT SURVIVES PER MODALITY. Only the genuine extras, and only once the kind
 * is known: a transcript box for audio and video (nothing in this pipeline
 * transcribes, so for audio it is the actual input), a pages box for a deck,
 * and a brief. These are REFINEMENTS to an asset that already exists, not
 * alternative front doors to the product.
 */

import { useState, type ReactNode } from "react";
import UniversalInput, { type ResolvedAsset } from "./content/UniversalInput";
import { VideoReceipt, type YouTubeManifest } from "./content/YouTubeIngest";

export type AssetKind = "text" | "image" | "video" | "audio" | "page" | "document";

export interface TranscriptCue {
  t_ms: number;
  text: string;
}

export interface ContentAsset {
  kind: AssetKind;
  text?: string;
  content_type?: string;
  brief?: string;
  image_b64?: string;
  media_type?: string;
  frames?: { t_ms: number; image_b64: string; media_type?: string }[];
  transcript_cues?: TranscriptCue[];
  pages?: string[];
}

export const KINDS: { id: AssetKind; label: string; hint: string }[] = [
  { id: "text", label: "Script / copy", hint: "a video script, ad copy, an article" },
  { id: "image", label: "Image", hint: "a static ad, poster, banner" },
  { id: "video", label: "Video", hint: "keyframes are read in your browser" },
  { id: "audio", label: "Audio", hint: "a transcript, with timings if you have them" },
  { id: "page", label: "Landing page", hint: "prose read in the order a scroller meets it" },
  { id: "document", label: "Deck / document", hint: "one beat per page" },
];

/** `00:12 text` or `12.5 text` per line. Lines without a stamp become cues
 *  with no timing, which downgrades the axis from temporal to sequential
 *  rather than inventing timestamps. */
export function parseCues(raw: string): { cues: TranscriptCue[]; timed: boolean } {
  const cues: TranscriptCue[] = [];
  let timed = false;
  for (const line of raw.split("\n")) {
    const t = line.match(/^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s+(.*)$/);
    if (t && t[3].trim()) {
      const mins = t[1] ? parseInt(t[1], 10) : 0;
      cues.push({ t_ms: Math.round((mins * 60 + parseFloat(t[2])) * 1000), text: t[3].trim() });
      timed = true;
    } else if (line.trim()) {
      cues.push({ t_ms: -1, text: line.trim() });
    }
  }
  return { cues: timed ? cues.filter((c) => c.t_ms >= 0) : [], timed };
}

/** Provenance strip: what was actually retrieved, before any twins are spent.
 *
 *  A study that silently ran against a cookie wall is worse than one that
 *  refused, because it produces numbers that look fine. */
function Receipt({ receipt, kind }: { receipt: ResolvedAsset["receipt"]; kind: AssetKind }) {
  const redirected = receipt.hops.length > 1;
  return (
    <div className="rounded-lg border border-hairline/60 bg-black/20 p-2.5">
      <div className="flex flex-wrap items-baseline gap-x-2 font-mono text-[10px]">
        <span className="text-good">✓ {kind}</span>
        {receipt.bytes > 0 && <span className="text-muted">{(receipt.bytes / 1024).toFixed(0)} kB</span>}
        {receipt.contentType && <span className="text-muted">{receipt.contentType}</span>}
        {/* Where we ENDED UP, which is not always where they pointed — an apex
            that lands on a regional or consent URL is a different asset from
            the one they meant to study. */}
        <span className="truncate text-muted">{receipt.finalUrl}</span>
        {redirected && <span className="text-accent-2">· redirected {receipt.hops.length - 1}×</span>}
      </div>
      {/* The rung, when the server had to settle for less than the content
          itself. This is the difference between a study of a video and a study
          of its thumbnail, and it is never allowed to be implicit. */}
      {receipt.rung && receipt.rung !== "video" && (
        <p className="mt-1.5 rounded border border-[#f2ad1f]/40 bg-[#f2ad1f]/[0.07] px-2 py-1 font-mono text-[9px] leading-relaxed text-[#f2ad1f]">
          <b className="uppercase tracking-wider">{receipt.rung} study</b> — this reads what the
          page publishes about itself, not the content behind it.
        </p>
      )}
      {receipt.note && (
        <p className="mt-1 font-mono text-[9px] leading-relaxed text-muted">{receipt.note}</p>
      )}
      {receipt.sections && receipt.sections.length > 0 && (
        <ol className="mt-1.5 space-y-0.5">
          {receipt.sections.slice(0, 4).map((s, i) => (
            <li key={i} className="truncate text-[11px] leading-snug text-ink-2">
              <span className="mr-1 font-mono text-[9px] text-muted">{i + 1}</span>
              {s}
            </li>
          ))}
          {receipt.sections.length > 4 && (
            <li className="font-mono text-[9px] text-muted">
              +{receipt.sections.length - 4} more
            </li>
          )}
        </ol>
      )}
    </div>
  );
}

export function AssetInput({
  kind,
  asset,
  onChange,
  onKindChange,
  disabled,
  onYouTube,
}: {
  kind: AssetKind;
  asset: ContentAsset;
  onChange: (a: ContentAsset) => void;
  /** The picker is a readout, so the component that owns `kind` has to accept
   *  it changing as a RESULT of ingest rather than only as a click. */
  onKindChange: (k: AssetKind) => void;
  disabled?: boolean;
  onYouTube?: (manifest: YouTubeManifest | null) => void;
}) {
  const [receipt, setReceipt] = useState<ResolvedAsset["receipt"] | null>(null);
  const [manifest, setManifest] = useState<YouTubeManifest | null>(null);
  const [detected, setDetected] = useState(false);

  // `direct` means the server read the content itself and there is nothing to
  // warn about; anything else means it settled for less than what was asked
  // for, and the picker says so rather than only the receipt.
  const rung = manifest?.rung ?? receipt?.rung;
  const degradedRung = rung && rung !== "video" && rung !== "direct" ? rung : null;

  const textarea = (
    placeholder: string,
    value: string,
    set: (v: string) => void,
    rows = 4
  ) => (
    <textarea
      value={value}
      onChange={(e) => set(e.target.value)}
      rows={rows}
      disabled={disabled}
      placeholder={placeholder}
      className="w-full resize-y rounded-xl border border-hairline bg-surface-2 p-3.5 font-mono text-xs leading-relaxed outline-none placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
    />
  );

  return (
    <div className="space-y-3">
      <UniversalInput
        disabled={disabled}
        onText={(text) => {
          // Prose could be a script, an article or ad copy, and no inspection
          // distinguishes them — this is the one case the picker is genuinely
          // the researcher's to set, so a CHOSEN kind is respected.
          //
          // A DETECTED one is not, and the difference matters. After resolving
          // a link the picker holds an answer the server produced; pasting a
          // script next inherited it, so three scenes of copy were studied as a
          // "Landing page" because the previous thing in the box happened to be
          // one. Detection is about the last asset, not this one, so it is
          // dropped rather than carried forward.
          const next: AssetKind = !detected && (kind === "text" || kind === "page") ? kind : "text";
          setReceipt(null);
          setManifest(null);
          setDetected(false);
          onKindChange(next);
          onChange({ kind: next, text });
          onYouTube?.(null);
        }}
        onResolved={(r) => {
          setReceipt(r.receipt);
          setManifest(r.manifest ?? null);
          setDetected(true);
          onKindChange(r.kind);
          // REPLACED, not merged. Spreading the previous asset carried its
          // fields into the new one — resolve a video, then an image, and the
          // image asset still had eight keyframes hanging off it. The kind
          // decides which field the server reads, so the stale ones were
          // invisible rather than harmless: they travelled with every
          // subsequent request and would be picked up the moment the picker
          // was clicked back.
          onChange({ ...r.asset, kind: r.kind } as ContentAsset);
          onYouTube?.(r.manifest ?? null);
        }}
      />

      {/* DETECTED, not selected. Still clickable, because a resolved asset can
          legitimately be studied two ways — page prose is also just text — and
          because the pasted-prose case has no detectable answer. */}
      <div className="flex flex-wrap items-center gap-1.5">
        {/* The label carries the warning, not just the receipt below it.
            When a YouTube link degrades to its thumbnail the picker jumps from
            Script/copy to Image on its own, and that jump is what the eye is
            on — reported as "feels weird" by someone watching it happen, with
            the explanation sitting three lines lower in 9px amber. A picker
            that moves silently is the problem; one that says why it moved is
            the product doing the work. */}
        {detected && degradedRung ? (
          <span className="hud-label mr-1 text-[#f2ad1f]">
            DETECTED · {degradedRung === "metadata" ? "PREVIEW ONLY" : `${degradedRung} ONLY`}
          </span>
        ) : (
          <span className="hud-label mr-1 text-muted">{detected ? "DETECTED" : "STUDY AS"}</span>
        )}
        {KINDS.map((k) => {
          const on = kind === k.id;
          return (
            <button
              key={k.id}
              type="button"
              role="radio"
              aria-checked={on}
              title={k.hint}
              disabled={disabled}
              onClick={() => onKindChange(k.id)}
              className={`rounded-lg border px-2.5 py-1 text-[11px] transition ${
                on
                  ? "border-accent/60 bg-accent/[0.10] text-ink"
                  : "border-hairline text-muted hover:border-accent/40 hover:text-ink-2"
              }`}
            >
              {k.label}
            </button>
          );
        })}
      </div>

      {manifest ? <VideoReceipt manifest={manifest} /> : receipt && <Receipt receipt={receipt} kind={kind} />}

      {/* ── refinements, per kind, only once there is something to refine ── */}

      {kind === "audio" && (
        <div className="space-y-1.5">
          {textarea(
            "Transcript. Prefix each line with a timestamp to get a timeline:\n0:00 Welcome back to the show\n0:14 Today we are talking about…",
            asset.transcript_cues?.length
              ? asset.transcript_cues.map((c) => `${(c.t_ms / 1000).toFixed(1)} ${c.text}`).join("\n")
              : asset.text ?? "",
            (v) => {
              const { cues, timed } = parseCues(v);
              // Untimed transcripts stay as plain text so the server treats
              // them as a sequence rather than fabricating a clock.
              onChange(
                timed
                  ? { ...asset, transcript_cues: cues, text: undefined }
                  : { ...asset, text: v, transcript_cues: [] }
              );
            },
            8
          )}
          <p className="font-mono text-[10px] text-muted">
            Nothing here transcribes audio, so this is the actual input — with
            timestamps → a timeline (retention in seconds), without → an ordered
            transcript (retention per beat).
          </p>
        </div>
      )}

      {kind === "video" && (
        <details className="text-xs">
          <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider text-muted hover:text-ink-2">
            add a transcript ({asset.transcript_cues?.length ?? 0} cues)
          </summary>
          <div className="mt-1.5">
            {textarea(
              "Optional — one line per cue:\n0:03 Your tools were supposed to help\n0:11 One clean interface slides in",
              asset.transcript_cues?.map((c) => `${(c.t_ms / 1000).toFixed(1)} ${c.text}`).join("\n") ?? "",
              (v) => onChange({ ...asset, transcript_cues: parseCues(v).cues }),
              4
            )}
          </div>
        </details>
      )}

      {kind === "document" && (
        <div className="space-y-1.5">
          {textarea(
            "One slide or page per block, separated by a blank line.",
            (asset.pages ?? []).join("\n\n"),
            (v) =>
              onChange({
                ...asset,
                pages: v.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean),
              }),
            6
          )}
          <p className="font-mono text-[10px] text-muted">
            {(asset.pages ?? []).length} page{(asset.pages ?? []).length === 1 ? "" : "s"}
          </p>
        </div>
      )}

      {(kind === "text" || kind === "page") && (asset.text ?? "").length > 0 && (
        <details className="text-xs">
          <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider text-muted hover:text-ink-2">
            edit the copy ({(asset.text ?? "").length.toLocaleString()} characters)
          </summary>
          <div className="mt-1.5">
            {textarea("The copy the swarm will read.", asset.text ?? "", (v) =>
              onChange({ ...asset, text: v }), 8)}
          </div>
        </details>
      )}

      {(kind === "image" || kind === "video") && (
        <details className="text-xs">
          <summary className="cursor-pointer font-mono text-[10px] uppercase tracking-wider text-muted hover:text-ink-2">
            add a brief
          </summary>
          <div className="mt-1.5">
            {textarea(
              "What is this asset for, who is it aimed at?",
              asset.brief ?? "",
              (v) => onChange({ ...asset, brief: v }),
              2
            )}
          </div>
        </details>
      )}
    </div>
  );
}

/** Enough to run? Mirrors the server's refusals so the button explains itself
 *  instead of round-tripping to a 400. */
export function assetReady(kind: AssetKind, a: ContentAsset): string | null {
  if (kind === "image") return a.image_b64 ? null : "paste an image link, or choose a file";
  if (kind === "video") return a.frames?.length ? null : "paste a video link, or choose a file";
  if (kind === "audio")
    return a.transcript_cues?.length || (a.text ?? "").trim().length >= 20
      ? null
      : "paste a transcript";
  if (kind === "document")
    return (a.pages ?? []).length >= 2 ? null : "at least two pages or slides";
  return (a.text ?? "").trim().length >= 20 ? null : "paste at least 20 characters";
}

export type { ResolvedAsset };
