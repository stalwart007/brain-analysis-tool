"use client";

/**
 * ONE BOX. Paste a link, paste text, or choose a file.
 *
 * WHAT THIS REPLACES, and why it was wrong. Every modality used to own its own
 * input: a YouTube box, a video-link box, a page-link box, a PDF-link box, an
 * audio-link box, each behind a content-type radio the researcher had to get
 * right FIRST. That made the picker a gate — you had to classify your own URL
 * before the product would look at it — and it made being wrong expensive,
 * because a link chosen under "Video" that turned out to be an article was
 * refused with a message about switching the content type rather than simply
 * being read.
 *
 * The classification was never the researcher's job. The server already
 * decides what a URL is from its content type, and does it more reliably than
 * a person eyeballing a link: `…/promo.mp4` served as HTML is a player page,
 * `…/x?id=9` served as image/png is an image, and no amount of looking at the
 * URL tells you either. So the picker stops being an input and becomes an
 * OUTCOME — it shows what was detected, and stays clickable only as an
 * override for the cases the server genuinely cannot know (pasted prose that
 * could be a script or an article).
 *
 * ONE CONSEQUENCE WORTH STATING. Because the kind is discovered rather than
 * declared, this component can hand back a kind the caller did not ask for,
 * and it always says so out loud rather than silently switching underneath. A
 * picker that moves on its own is confusing; a picker that moves and explains
 * itself is the product doing the work.
 */

import { useRef, useState } from "react";
import { motion } from "framer-motion";
import IngestPipeline, { type Stage, type StageState } from "./IngestPipeline";
import type { AssetKind, ContentAsset } from "@/components/AssetInput";
import type { YouTubeManifest } from "./YouTubeIngest";
import { ingestYouTube } from "./YouTubeIngest";

/** What the server said a link is, plus everything it learned on the way. */
export interface ResolvedAsset {
  kind: AssetKind;
  asset: Partial<ContentAsset>;
  /** Present only for YouTube, which resolves to a real timeline. */
  manifest?: YouTubeManifest;
  receipt: {
    finalUrl: string;
    hops: string[];
    bytes: number;
    contentType: string;
    note: string;
    /** `video` | `audio` | `text` | `metadata` | undefined for a direct read. */
    rung?: string;
    title?: string;
    sections?: string[];
    pageCount?: number;
  };
}

const URL_LIKE = /^https?:\/\/\S+$/i;

/** Sampled evenly across the duration — see the uploaded-video path. */
const FRAME_COUNT = 8;

async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
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
    // Beat centres, not boundaries: the first frame of a cut is often black
    // and the last is often a logo card.
    const t = (duration * (i + 0.5)) / FRAME_COUNT;
    await new Promise<void>((resolve) => {
      video.onseeked = () => resolve();
      video.currentTime = t;
    });
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    frames.push({
      t_ms: Math.round(t * 1000),
      image_b64: canvas.toDataURL("image/jpeg", 0.7).split(",")[1],
      media_type: "image/jpeg",
    });
    onProgress(i + 1, FRAME_COUNT);
  }
  URL.revokeObjectURL(video.src);
  return frames;
}

const LINK_STAGES: Stage[] = [
  { id: "resolve", label: "resolve link", state: "pending" },
  { id: "identify", label: "identify", state: "pending" },
  { id: "retrieve", label: "retrieve", state: "pending" },
  { id: "prepare", label: "prepare beats", state: "pending" },
];

const FILE_STAGES: Stage[] = [
  { id: "read", label: "read file", state: "pending" },
  { id: "prepare", label: "prepare beats", state: "pending" },
];

export default function UniversalInput({
  disabled,
  onResolved,
  onText,
}: {
  disabled?: boolean;
  onResolved: (r: ResolvedAsset) => void;
  /** Pasted prose, which needs no round trip — the caller keeps owning it. */
  onText: (text: string) => void;
}) {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);
  const runId = useRef(0);

  const mark = (id: string, state: StageState, detail?: string) =>
    setStages((prev) =>
      prev.map((s) => (s.id === id ? { ...s, state, detail: detail ?? s.detail } : s))
    );

  const looksLikeUrl = URL_LIKE.test(value.trim());

  async function submit() {
    const raw = value.trim();
    if (!raw) return;
    // Anything that is not a URL is content, not a pointer to content. No
    // round trip, no classification: the researcher pasted words.
    if (!URL_LIKE.test(raw)) {
      onText(raw);
      return;
    }
    const run = ++runId.current;
    setBusy(true);
    setError(null);
    setStages(LINK_STAGES.map((s) => ({ ...s })));

    try {
      mark("resolve", "active");
      const res = await fetch("/api/cs/content/fetch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: raw }),
      });
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error(data?.detail ?? `Could not read that link (${res.status})`);
      if (run !== runId.current) return;

      const receipt = {
        finalUrl: data.final_url ?? raw,
        hops: data.hops ?? [raw],
        bytes: data.bytes ?? 0,
        contentType: data.content_type ?? "",
        note: data.note ?? "",
        rung: data.rung ?? data.youtube?.rung,
        title: data.youtube?.title ?? data.meta?.title,
        sections: data.sections,
        pageCount: data.page_count,
      };
      mark("resolve", "done", new URL(receipt.finalUrl).hostname);

      // ── YouTube on the bottom rung: the server already sent the still ──
      //
      // When InnerTube refuses us — which, measured from the production host,
      // is the COMMON case — the server falls back to the public preview and
      // answers with a ready `image` asset plus the manifest. There is nothing
      // for the browser to crop, and handing this to `ingestYouTube` would run
      // it off the end of the ladder into `chaptersAsText` and submit a text
      // asset containing one sentence, discarding the thumbnail the server had
      // already fetched and encoded.
      if (data.youtube && data.rung === "metadata" && data.kind === "image") {
        mark("identify", "done", "youtube · preview only");
        mark("retrieve", "done", `${(receipt.bytes / 1024).toFixed(0)} kB`);
        mark("prepare", "done", "thumbnail");
        onResolved({
          kind: "image",
          asset: { ...data.asset, kind: "image" },
          manifest: data.youtube,
          receipt,
        });
        return;
      }

      // ── YouTube: a real timeline, cropped in this browser ───────────
      if (data.youtube) {
        mark("identify", "done", `youtube · ${data.youtube.rung}`);
        const result = await ingestYouTube(data.youtube, (id, state, detail) => {
          if (run !== runId.current) return;
          // The YouTube ingest reports its own finer-grained stages; they are
          // folded onto this chain's nodes rather than swapping the whole
          // pipeline out underneath the reader mid-run.
          if (id === "filmstrip") mark("retrieve", state, detail);
          if (id === "crop") mark("prepare", state, detail);
          if (id === "captions" && state === "skipped") mark("retrieve", "active", detail);
        });
        if (run !== runId.current) return;
        mark("retrieve", "done");
        mark("prepare", "done", `${result.asset.frames?.length ?? 0} beats`);
        onResolved({ ...result, receipt });
        return;
      }

      mark("identify", "done", data.kind + (receipt.rung ? ` · ${receipt.rung}` : ""));

      // ── a hosted video or audio file: relay, then decode here ───────
      if ((data.kind === "video" || data.kind === "audio") && data.media_relay) {
        if (data.kind === "audio") {
          // Reachable and confirmed, but a transcript is still the input —
          // nothing in this pipeline transcribes, and inventing beats from a
          // waveform would be fabricating the content rather than reading it.
          mark("retrieve", "done", "reachable");
          mark("prepare", "skipped", "transcript needed");
          onResolved({ kind: "audio", asset: { kind: "audio" }, receipt });
          return;
        }
        mark("retrieve", "active", "downloading");
        const relayed = await fetch("/api/cs/content/media", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ url: receipt.finalUrl }),
        });
        if (!relayed.ok) throw new Error(`Relay failed (${relayed.status})`);
        const blob = await relayed.blob();
        if (run !== runId.current) return;
        mark("retrieve", "done", `${(blob.size / 1_000_000).toFixed(1)} MB`);
        mark("prepare", "active");
        const file = new File([blob], "remote-video", { type: data.content_type || "video/mp4" });
        const frames = await extractKeyframes(file, (d, t) =>
          mark("prepare", "active", `${d}/${t} keyframes`)
        );
        if (run !== runId.current) return;
        mark("prepare", "done", `${frames.length} beats`);
        onResolved({ kind: "video", asset: { kind: "video", frames }, receipt });
        return;
      }

      // ── everything else arrives ready: image, PDF pages, page prose ──
      mark("retrieve", "done", receipt.bytes ? `${(receipt.bytes / 1024).toFixed(0)} kB` : "");
      const kind = data.kind as AssetKind;
      const detail =
        kind === "document"
          ? `${data.pages?.length ?? 0} pages`
          : kind === "page"
            ? `${data.sections?.length ?? 0} sections`
            : "ready";
      mark("prepare", "done", detail);
      onResolved({ kind, asset: { ...data.asset, kind }, receipt });
    } catch (e) {
      if (run !== runId.current) return;
      setError(e instanceof Error ? e.message : "Could not read that link");
      setStages((prev) => prev.map((s) => (s.state === "active" ? { ...s, state: "failed" } : s)));
    } finally {
      if (run === runId.current) setBusy(false);
    }
  }

  async function takeFile(file: File) {
    const run = ++runId.current;
    setBusy(true);
    setError(null);
    setStages(FILE_STAGES.map((s) => ({ ...s })));
    const receipt = {
      finalUrl: file.name,
      hops: [],
      bytes: file.size,
      contentType: file.type || "application/octet-stream",
      note: `${file.name} · ${(file.size / 1024).toFixed(0)} kB, read in your browser`,
    };
    try {
      mark("read", "active", file.type || "unknown type");
      // The file's own type decides, exactly as the content type decides for a
      // URL. The extension is not consulted anywhere.
      if (file.type.startsWith("video/")) {
        mark("read", "done");
        mark("prepare", "active");
        const frames = await extractKeyframes(file, (d, t) =>
          mark("prepare", "active", `${d}/${t} keyframes`)
        );
        if (run !== runId.current) return;
        mark("prepare", "done", `${frames.length} beats`);
        onResolved({ kind: "video", asset: { kind: "video", frames }, receipt });
        return;
      }
      if (file.type.startsWith("image/")) {
        const b64 = await fileToBase64(file);
        if (run !== runId.current) return;
        mark("read", "done");
        mark("prepare", "done", "1 image");
        onResolved({
          kind: "image",
          asset: { kind: "image", image_b64: b64, media_type: file.type },
          receipt,
        });
        return;
      }
      if (file.type.startsWith("text/") || file.type === "application/json") {
        const text = await file.text();
        if (run !== runId.current) return;
        mark("read", "done");
        mark("prepare", "done", `${text.length.toLocaleString()} chars`);
        onText(text);
        return;
      }
      throw new Error(
        `${file.type || "That file type"} cannot be read in the browser. ` +
          "Video, images and plain text work; for a PDF, paste its link instead."
      );
    } catch (e) {
      if (run !== runId.current) return;
      setError(e instanceof Error ? e.message : "Could not read that file");
      setStages((prev) => prev.map((s) => (s.state === "active" ? { ...s, state: "failed" } : s)));
    } finally {
      if (run === runId.current) setBusy(false);
    }
  }

  return (
    <div className="space-y-2.5">
      <div className="rounded-xl border border-hairline bg-surface-2 focus-within:border-accent/60">
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            // Enter submits a LINK, because a link is one line and nobody wants
            // a newline in it. Prose keeps its Enter key.
            if (e.key === "Enter" && !e.shiftKey && looksLikeUrl && !busy) {
              e.preventDefault();
              submit();
            }
          }}
          onPaste={(e) => {
            const pasted = e.clipboardData.getData("text").trim();
            // A pasted link into an empty box is unambiguous, and the extra
            // click to confirm it buys nothing.
            if (!value.trim() && URL_LIKE.test(pasted) && !busy) {
              e.preventDefault();
              setValue(pasted);
              setTimeout(() => submit(), 0);
            }
          }}
          rows={value.trim() && !looksLikeUrl ? 8 : 3}
          disabled={disabled || busy}
          placeholder={
            "Paste anything — a YouTube link, any video or podcast URL, a landing page, " +
            "a PDF link, an image link…\n\nOr paste the script, ad copy or article itself."
          }
          aria-label="Content, or a link to it"
          className="w-full resize-y bg-transparent p-3.5 font-mono text-xs leading-relaxed outline-none placeholder:text-muted disabled:opacity-50"
        />
        <div className="flex flex-wrap items-center gap-2 border-t border-hairline px-3 py-2">
          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={disabled || busy}
            className="rounded-lg border border-hairline px-2.5 py-1 font-mono text-[10px] text-ink-2 transition hover:border-accent/50 hover:text-ink disabled:opacity-40"
          >
            choose a file
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="video/*,image/*,text/*"
            className="hidden"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) takeFile(f);
              // Cleared so choosing the SAME file twice fires again — otherwise
              // a failed read cannot be retried without picking something else.
              e.target.value = "";
            }}
          />
          <span className="font-mono text-[9px] text-muted">
            {value.trim()
              ? looksLikeUrl
                ? "a link — we work out what it is"
                : `${value.trim().length.toLocaleString()} characters of copy`
              : "link, text, or a file"}
          </span>
          <motion.button
            type="button"
            whileTap={{ scale: 0.97 }}
            onClick={submit}
            disabled={disabled || busy || !value.trim()}
            className="ml-auto rounded-lg border border-accent/50 bg-accent/10 px-3 py-1.5 text-xs text-ink transition hover:bg-accent/20 disabled:opacity-40"
          >
            {busy ? "reading…" : looksLikeUrl ? "read link" : "use this copy"}
          </motion.button>
        </div>
      </div>

      {stages.length > 0 && <IngestPipeline stages={stages} />}
      {error && <p className="text-[11px] leading-relaxed text-critical">{error}</p>}
    </div>
  );
}
