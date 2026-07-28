"use client";

/**
 * Build a ContentAsset from whatever form the content actually exists in.
 *
 * Everything here happens in the BROWSER, including video keyframe extraction.
 * That is a deliberate boundary, not a shortcut: decoding caller-supplied
 * media server-side means running a decoder on untrusted input, which is a
 * large and historically hostile attack surface, and the API deliberately has
 * no `url` field because fetching a caller-supplied URL server-side is an SSRF
 * primitive. The browser already has a video decoder and already has
 * permission to read the user's own files.
 */

import { useRef, useState } from "react";

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
  { id: "image", label: "Image", hint: "a static ad, poster, banner, screenshot" },
  { id: "video", label: "Video", hint: "keyframes are extracted here in your browser" },
  { id: "audio", label: "Audio", hint: "a transcript, with timings if you have them" },
  { id: "page", label: "Landing page", hint: "paste the page HTML" },
  { id: "document", label: "Deck / document", hint: "one block per slide or page" },
];

/** Sampled evenly across the duration. Ten frames over a 30s spot is one every
 *  three seconds — finer than the beat structure the study can resolve, and
 *  each one costs a vision call, so more is spend without resolution. */
const FRAME_COUNT = 8;

async function fileToBase64(file: File): Promise<string> {
  const buf = await file.arrayBuffer();
  let binary = "";
  const bytes = new Uint8Array(buf);
  // Chunked: String.fromCharCode(...bytes) blows the argument limit and throws
  // a RangeError on anything above a few hundred kB.
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
}

async function extractKeyframes(
  file: File,
  onProgress: (done: number, total: number) => void
): Promise<{ t_ms: number; image_b64: string; media_type: string }[]> {
  const video = document.createElement("video");
  video.preload = "auto";
  video.muted = true;
  video.src = URL.createObjectURL(file);

  await new Promise<void>((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error("Could not decode that video in this browser."));
  });

  const duration = video.duration;
  if (!Number.isFinite(duration) || duration <= 0) {
    URL.revokeObjectURL(video.src);
    throw new Error("That video reports no duration, so frames cannot be sampled.");
  }

  const canvas = document.createElement("canvas");
  const scale = Math.min(1, 640 / (video.videoWidth || 640));
  canvas.width = Math.max(1, Math.round((video.videoWidth || 640) * scale));
  canvas.height = Math.max(1, Math.round((video.videoHeight || 360) * scale));
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas is unavailable in this browser.");

  const frames: { t_ms: number; image_b64: string; media_type: string }[] = [];
  for (let i = 0; i < FRAME_COUNT; i++) {
    // Sample at beat centres, not at 0 and the very end: the first frame of a
    // cut is often black and the last is often a logo card, and both would be
    // described as if they were the content.
    const t = (duration * (i + 0.5)) / FRAME_COUNT;
    await new Promise<void>((resolve) => {
      video.onseeked = () => resolve();
      video.currentTime = t;
    });
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.7);
    frames.push({
      t_ms: Math.round(t * 1000),
      image_b64: dataUrl.split(",")[1],
      media_type: "image/jpeg",
    });
    onProgress(i + 1, FRAME_COUNT);
  }
  URL.revokeObjectURL(video.src);
  return frames;
}

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

export function AssetInput({
  kind,
  asset,
  onChange,
  disabled,
}: {
  kind: AssetKind;
  asset: ContentAsset;
  onChange: (a: ContentAsset) => void;
  disabled?: boolean;
}) {
  const [note, setNote] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const textarea = (
    placeholder: string,
    value: string,
    set: (v: string) => void,
    rows = 6
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

  if (kind === "image" || kind === "video") {
    const isVideo = kind === "video";
    return (
      <div className="space-y-3">
        <input
          ref={fileRef}
          type="file"
          accept={isVideo ? "video/*" : "image/*"}
          disabled={disabled || working}
          className="block w-full text-xs text-muted file:mr-3 file:rounded-lg file:border file:border-hairline file:bg-surface-2 file:px-3 file:py-1.5 file:text-xs file:text-ink-2"
          onChange={async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            setWorking(true);
            setNote(null);
            try {
              if (isVideo) {
                const frames = await extractKeyframes(file, (d, t) =>
                  setNote(`extracting keyframe ${d} of ${t}…`)
                );
                onChange({ ...asset, kind, frames });
                setNote(`${frames.length} keyframes extracted in your browser`);
              } else {
                const b64 = await fileToBase64(file);
                onChange({ ...asset, kind, image_b64: b64, media_type: file.type || "image/png" });
                setNote(`${file.name} · ${(file.size / 1024).toFixed(0)} kB`);
              }
            } catch (err) {
              setNote(err instanceof Error ? err.message : "Could not read that file.");
            } finally {
              setWorking(false);
            }
          }}
        />
        {note && <p className="font-mono text-[11px] text-accent">{note}</p>}
        {isVideo &&
          textarea(
            "Optional transcript, one line per cue:\n0:03 Your tools were supposed to help\n0:11 One clean interface slides in",
            asset.transcript_cues?.map((c) => `${(c.t_ms / 1000).toFixed(1)} ${c.text}`).join("\n") ?? "",
            (v) => onChange({ ...asset, transcript_cues: parseCues(v).cues }),
            4
          )}
        {textarea(
          "Optional brief — what is this asset for, who is it aimed at?",
          asset.brief ?? "",
          (v) => onChange({ ...asset, brief: v }),
          2
        )}
      </div>
    );
  }

  if (kind === "audio") {
    return (
      <div className="space-y-2">
        {textarea(
          "Transcript. Prefix each line with a timestamp to get a timeline:\n0:00 Welcome back to the show\n0:14 Today we are talking about…",
          asset.transcript_cues?.length
            ? asset.transcript_cues.map((c) => `${(c.t_ms / 1000).toFixed(1)} ${c.text}`).join("\n")
            : asset.text ?? "",
          (v) => {
            const { cues, timed } = parseCues(v);
            // Untimed transcripts stay as plain text so the server treats them
            // as a sequence rather than fabricating a clock.
            onChange(timed ? { ...asset, transcript_cues: cues, text: undefined } : { ...asset, text: v, transcript_cues: [] });
          },
          8
        )}
        <p className="font-mono text-[10px] text-muted">
          with timestamps → a timeline (retention in seconds) · without → an
          ordered transcript (retention per beat)
        </p>
      </div>
    );
  }

  if (kind === "document") {
    return (
      <div className="space-y-2">
        {textarea(
          "One slide or page per block, separated by a blank line.",
          (asset.pages ?? []).join("\n\n"),
          (v) => onChange({ ...asset, pages: v.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean) }),
          8
        )}
        <p className="font-mono text-[10px] text-muted">
          {(asset.pages ?? []).length} page{(asset.pages ?? []).length === 1 ? "" : "s"} detected
        </p>
      </div>
    );
  }

  return textarea(
    kind === "page"
      ? "Paste the page HTML. Sections are read in DOM order — the order a scroller meets them."
      : "Paste a video script, storyboard, ad copy, or article.",
    asset.text ?? "",
    (v) => onChange({ ...asset, text: v }),
    kind === "page" ? 8 : 6
  );
}

/** Enough to run? Mirrors the server's refusals so the button explains itself
 *  instead of round-tripping to a 400. */
export function assetReady(kind: AssetKind, a: ContentAsset): string | null {
  if (kind === "image") return a.image_b64 ? null : "choose an image";
  if (kind === "video") return a.frames?.length ? null : "choose a video";
  if (kind === "audio")
    return a.transcript_cues?.length || (a.text ?? "").trim().length >= 20
      ? null
      : "paste a transcript";
  if (kind === "document")
    return (a.pages ?? []).length >= 2 ? null : "at least two pages or slides";
  return (a.text ?? "").trim().length >= 20 ? null : "paste at least 20 characters";
}
