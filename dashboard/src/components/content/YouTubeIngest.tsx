"use client";

/**
 * A YouTube link becomes a study asset, in the browser.
 *
 * WHY THE CROPPING HAPPENS HERE. The server resolves what the video IS —
 * duration, chapters, transcript, and the geometry of YouTube's scrub-bar
 * filmstrip — and stops there, deliberately. Cutting tiles out of a spritesheet
 * means decoding a WebP that a third party supplied, and the API container has
 * no image decoder and no native dependencies by requirement. The browser has
 * both, inside a sandbox built for exactly this. So the server sends integers
 * and the browser does the pixels, which is the same boundary the uploaded-video
 * path has always respected.
 *
 * WHY THE SHEETS COME THROUGH THE RELAY. `i.ytimg.com` does send
 * `Access-Control-Allow-Origin: *`, so a direct canvas crop would work — but
 * the dashboard's CSP is `img-src 'self' data: blob:`, and reaching ytimg
 * directly means adding a third party to `img-src` AND `connect-src` on every
 * page of the app permanently, which hands any injected script an approved
 * destination to exfiltrate to. A sheet is ~25 kB. Relaying is free and the
 * policy stays shut.
 *
 * WHAT THIS REFUSES TO DO. It never presents a degraded ingest as a full one.
 * A video with no filmstrip and no captions can still be studied through its
 * chapter list, and that is a study of HOW THE VIDEO DESCRIBES ITSELF rather
 * than of the video — a different question, worth answering, and labelled as
 * such at every point it surfaces. The `rung` is carried all the way to the
 * findings for that reason.
 */

import { useEffect, useState } from "react";
import { type StageState } from "./IngestPipeline";
import type { AssetKind, ContentAsset, TranscriptCue } from "@/components/AssetInput";

export interface YouTubeChapter {
  start_s: number;
  end_s: number;
  title: string;
}

export interface YouTubeKeyframe {
  t_ms: number;
  sheet: number;
  sheet_url: string;
  x: number;
  y: number;
  w: number;
  h: number;
  label: string;
  source: "chapters" | "even";
}

export interface YouTubeManifest {
  video_id: string;
  title: string;
  author: string;
  duration_s: number;
  view_count: number | null;
  watch_url: string;
  thumbnail_url: string;
  thumbnail_candidates: string[];
  description: string;
  chapters: YouTubeChapter[];
  captions: {
    cues: TranscriptCue[];
    available: { language: string; name: string; auto: boolean }[];
  };
  keyframes: YouTubeKeyframe[];
  /** Unsigned interior frames, used when the filmstrip is out of reach. Whole
   *  images rather than tiles, so there is nothing to crop — just relay. */
  cdn_frames?: { name: string; fraction: number; url: string }[];
  storyboard: {
    width: number;
    height: number;
    interval_ms: number;
    frame_count: number;
    sheets: number;
  } | null;
  client: string;
  /** Why InnerTube refused, when it did — so the receipt can say "this server
   *  is rate-limited" rather than implying the video is private. */
  blocked_reason?: string;
  note: string;
  rung: "video" | "audio" | "text" | "cdn_frames" | "metadata";
  rung_basis: string;
  degraded: boolean;
}

export interface YouTubeIngestResult {
  kind: AssetKind;
  asset: Partial<ContentAsset>;
  manifest: YouTubeManifest;
}

/** Human runtime. `3:07`, `1:02:44` — never `187s`, which nobody thinks in. */
export function clock(totalSeconds: number): string {
  const s = Math.max(0, Math.round(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0
    ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}`
    : `${m}:${String(sec).padStart(2, "0")}`;
}

/** Relay one URL through the API and hand back the bytes as a blob. */
async function relay(url: string): Promise<Blob> {
  const res = await fetch("/api/cs/content/media", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `relay failed (${res.status})`);
  }
  return res.blob();
}

/**
 * Cut the planned tiles out of the sheets they live on.
 *
 * Sheets are fetched ONCE each and reused. Eight keyframes over a five-sheet
 * filmstrip touch three or four sheets; fetching per frame would pull the same
 * 25 kB image up to eight times and, worse, decode it eight times.
 */
/** A relayed image as base64, without the data-URL prefix. */
async function blobToBase64(blob: Blob): Promise<string> {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  // Chunked: spreading a large array into fromCharCode throws a RangeError.
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

async function cropKeyframes(
  keyframes: YouTubeKeyframe[],
  onProgress: (done: number, total: number, phase: "fetch" | "crop") => void
): Promise<{ t_ms: number; image_b64: string; media_type: string }[]> {
  const sheetUrls = Array.from(new Set(keyframes.map((f) => f.sheet_url)));
  const sheets = new Map<string, ImageBitmap>();

  for (const [i, url] of sheetUrls.entries()) {
    onProgress(i, sheetUrls.length, "fetch");
    const blob = await relay(url);
    // createImageBitmap decodes off the main thread, which matters: these are
    // 800×450 sheets and decoding five of them synchronously visibly stalls
    // the pipeline animation that is meant to be showing progress.
    sheets.set(url, await createImageBitmap(blob));
  }
  onProgress(sheetUrls.length, sheetUrls.length, "fetch");

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is unavailable in this browser.");

  const out: { t_ms: number; image_b64: string; media_type: string }[] = [];
  for (const [i, frame] of keyframes.entries()) {
    const sheet = sheets.get(frame.sheet_url);
    if (!sheet) continue;
    canvas.width = frame.w;
    canvas.height = frame.h;
    ctx.drawImage(sheet, frame.x, frame.y, frame.w, frame.h, 0, 0, frame.w, frame.h);
    // Quality 0.85 rather than the 0.7 the upload path uses. These tiles are
    // already only 160×90; a second lossy pass at 0.7 puts compression
    // artefacts into the very detail the vision model is being asked to read.
    out.push({
      t_ms: frame.t_ms,
      image_b64: canvas.toDataURL("image/jpeg", 0.85).split(",")[1],
      media_type: "image/jpeg",
    });
    onProgress(i + 1, keyframes.length, "crop");
  }
  for (const bitmap of sheets.values()) bitmap.close();
  return out;
}

/** The chapter list and description, as an ordered text asset.
 *
 *  Used only on the bottom rung, and shaped so the segmenter sees a running
 *  order rather than a paragraph. The heading names what this is, because the
 *  twins read the beats and should not be led to believe they watched a video.
 */
function chaptersAsText(manifest: YouTubeManifest): string {
  const head = `The creator's own description of a ${clock(manifest.duration_s)} video titled "${manifest.title}".`;
  const body = manifest.chapters
    .map((c) => `[${clock(c.start_s)}] ${c.title}`)
    .join("\n\n");
  const tail = manifest.description.slice(0, 1500).trim();
  return [head, body, tail].filter(Boolean).join("\n\n");
}

/**
 * A YouTube manifest becomes a study asset. No UI, no input, no URL box.
 *
 * Extracted from what used to be a self-contained `<YouTubeIngest>` panel,
 * because that panel was a second place to paste a link — and the whole point
 * of `UniversalInput` is that there is exactly one. The resolution step now
 * happens upstream (the server already told us this URL is a YouTube video and
 * handed back the manifest); everything left here is the part only a browser
 * can do, which is cutting tiles out of the filmstrip.
 *
 * `onStage` reports the fine-grained steps so the caller can fold them onto
 * whatever chain it is already showing, rather than this function owning a
 * pipeline of its own.
 */
export async function ingestYouTube(
  manifest: YouTubeManifest,
  onStage: (id: string, state: StageState, detail?: string) => void
): Promise<YouTubeIngestResult> {
  const cues = manifest.captions?.cues ?? [];
  onStage(
    "captions",
    cues.length ? "done" : "skipped",
    cues.length
      ? `${cues.length} cues`
      : manifest.captions?.available?.length
        ? "rate-limited by YouTube"
        : "none published"
  );

  // ── the CDN frames: whole images, no cropping ───────────────────────
  //
  // Reached when the player API refused us, which from a datacentre address is
  // most of the time. These are complete JPEGs rather than tiles in a sheet, so
  // the canvas work the storyboard path does is skipped entirely — relay each
  // one and hand it over. `t_ms: -1` marks the position as unknown; the server
  // reads that as a sequential axis rather than fabricating a runtime.
  if (manifest.rung === "cdn_frames" && (manifest.cdn_frames?.length ?? 0) >= 2) {
    const sources = manifest.cdn_frames!;
    onStage("filmstrip", "active", `0/${sources.length} frames`);
    const frames: { t_ms: number; image_b64: string; media_type: string }[] = [];
    for (const [i, source] of sources.entries()) {
      try {
        const blob = await relay(source.url);
        frames.push({
          t_ms: -1,
          image_b64: await blobToBase64(blob),
          media_type: blob.type && blob.type !== "application/octet-stream" ? blob.type : "image/jpeg",
        });
      } catch {
        // One missing frame is a thinner study, not a failed one.
      }
      onStage("filmstrip", "active", `${i + 1}/${sources.length} frames`);
    }
    if (frames.length < 2) throw new Error("Could not retrieve enough frames for this video.");
    onStage("filmstrip", "done", `${frames.length} frames`);
    onStage("crop", "done", "no cropping needed");
    return {
      kind: "video",
      asset: {
        kind: "video",
        frames,
        transcript_cues: cues,
        brief: `YouTube: "${manifest.title}"${manifest.author ? ` by ${manifest.author}` : ""}.`,
      },
      manifest,
    };
  }

  // ── the top rung: real keyframes ────────────────────────────────────
  if (manifest.rung === "video" && manifest.keyframes.length >= 2) {
    onStage("filmstrip", "active");
    const frames = await cropKeyframes(manifest.keyframes, (done, total, phase) => {
      if (phase === "fetch") {
        onStage("filmstrip", done >= total ? "done" : "active", `${done}/${total} sheets`);
        if (done >= total) onStage("crop", "active");
      } else {
        onStage("crop", done >= total ? "done" : "active", `${done}/${total} tiles`);
      }
    });
    if (frames.length < 2) throw new Error("The filmstrip decoded to fewer than two frames.");
    onStage("crop", "done", `${frames.length} tiles`);
    return {
      kind: "video",
      asset: {
        kind: "video",
        frames,
        transcript_cues: cues,
        brief: `YouTube: "${manifest.title}" by ${manifest.author} (${clock(manifest.duration_s)}).`,
      },
      manifest,
    };
  }

  // ── lower rungs ─────────────────────────────────────────────────────
  onStage("filmstrip", "skipped", manifest.storyboard ? "too few frames" : "not published");
  onStage("crop", "skipped");

  if (manifest.rung === "audio") {
    return {
      kind: "audio",
      asset: {
        kind: "audio",
        transcript_cues: cues,
        brief: `YouTube transcript: "${manifest.title}" by ${manifest.author} (${clock(manifest.duration_s)}).`,
      },
      manifest,
    };
  }

  return {
    kind: "text",
    asset: { kind: "text", text: chaptersAsText(manifest) },
    manifest,
  };
}

/**
 * What was actually retrieved, before any twins are spent.
 *
 * The same principle as the landing-page receipt: a study that silently ran
 * against the wrong thing is worse than one that refused, because it produces
 * numbers. Here the risk is subtler than a cookie wall — it is studying a
 * different video from the one intended, or studying a chapter list while
 * believing you studied footage.
 */
export function VideoReceipt({ manifest }: { manifest: YouTubeManifest }) {
  const thumbs = manifest.thumbnail_candidates?.length
    ? manifest.thumbnail_candidates
    : [manifest.thumbnail_url];
  // Pulled through the relay as bytes and shown from a blob URL.
  //
  // Not an `<img src>` pointing anywhere: the relay is a POST (it takes the
  // target URL in a JSON body, which is what keeps it from being a GET-shaped
  // open proxy), and `img-src` does not include ytimg for the reason set out at
  // the top of this file. `blob:` is already in the policy, so this needs no
  // widening at all. Candidates are tried in order because `maxresdefault`
  // 404s for anything never published above 720p.
  const [src, setSrc] = useState<string | null>(null);

  useEffect(() => {
    let revoked = false;
    let objectUrl: string | null = null;
    (async () => {
      for (const candidate of thumbs) {
        try {
          const blob = await relay(candidate);
          if (revoked) return;
          objectUrl = URL.createObjectURL(blob);
          setSrc(objectUrl);
          return;
        } catch {
          // Next candidate down the resolution ladder.
        }
      }
    })();
    return () => {
      revoked = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [thumbs]);

  return (
    <div className="rounded-lg border border-hairline/60 bg-black/20 p-2.5">
      <div className="flex gap-3">
        {/* Space is reserved whether or not the still has arrived, so the
            receipt does not jump sideways a beat after it renders. */}
        <div className="h-14 w-24 shrink-0 overflow-hidden rounded border border-hairline bg-surface-2">
          {src && (
            // eslint-disable-next-line @next/next/no-img-element -- a blob URL
            // for bytes already in memory; next/image would re-request it
            // through a loader that cannot see a blob.
            <img src={src} alt="" className="h-full w-full object-cover" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[12px] leading-snug text-ink">{manifest.title}</p>
          <p className="mt-0.5 flex flex-wrap gap-x-2 font-mono text-[10px] text-muted">
            {manifest.author && <span className="text-ink-2">{manifest.author}</span>}
            {/* Omitted rather than shown as 0:00. The public preview carries no
                runtime, and printing a zero duration states a fact about the
                video that is false — "0:00" reads as an empty clip, not as
                "we were not told". */}
            {manifest.duration_s > 0 && <span>{clock(manifest.duration_s)}</span>}
            {manifest.view_count != null && (
              <span>{manifest.view_count.toLocaleString()} views</span>
            )}
            <span className="text-accent-2">
              via {manifest.client === "oembed" ? "public preview" : manifest.client}
            </span>
          </p>
        </div>
      </div>

      {/* The rung, stated. On anything below the top one this is the single
          most important line on the screen: it is the difference between a
          study of the video and a study of its description. */}
      <p
        className={`mt-2 rounded border px-2 py-1 font-mono text-[9px] leading-relaxed ${
          manifest.degraded
            ? "border-[#f2ad1f]/40 bg-[#f2ad1f]/[0.07] text-[#f2ad1f]"
            : "border-good/30 bg-good/[0.06] text-good"
        }`}
      >
        <b className="uppercase tracking-wider">{manifest.rung} study</b> — {manifest.rung_basis}
      </p>
      <p className="mt-1 font-mono text-[9px] leading-relaxed text-muted">{manifest.note}</p>

      {manifest.chapters.length > 0 && (
        <ol className="mt-1.5 flex flex-wrap gap-1">
          {manifest.chapters.map((c, i) => (
            <li
              key={i}
              className="rounded border border-hairline px-1.5 py-0.5 font-mono text-[9px] text-ink-2"
            >
              <span className="text-muted">{clock(c.start_s)}</span> {c.title}
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
