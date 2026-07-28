"use client";

/**
 * Headless adaptive-bitrate player. hls.js is imported lazily only when a
 * stream loads (Safari plays HLS natively, so it's skipped there entirely).
 * Custom controls — no default browser chrome.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";

function formatTime(s: number): string {
  if (!isFinite(s)) return "–:––";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function Player({ src }: { src: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const hlsRef = useRef<{ destroy(): void } | null>(null);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !src) return;
    setError(null);

    let cancelled = false;
    (async () => {
      if (video.canPlayType("application/vnd.apple.mpegurl")) {
        video.src = src; // Safari: native HLS
        return;
      }
      const { default: Hls } = await import("hls.js");
      if (cancelled) return;
      if (!Hls.isSupported()) {
        setError("HLS is not supported in this browser.");
        return;
      }
      const hls = new Hls({ enableWorker: true, capLevelToPlayerSize: true });
      hlsRef.current = hls;
      hls.on(Hls.Events.ERROR, (_e: unknown, data: { fatal?: boolean; type?: string }) => {
        if (data?.fatal) setError(`Stream error (${data.type ?? "unknown"})`);
      });
      hls.loadSource(src);
      hls.attachMedia(video);
    })();

    return () => {
      cancelled = true;
      hlsRef.current?.destroy();
      hlsRef.current = null;
    };
  }, [src]);

  const toggle = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) void video.play();
    else video.pause();
  }, []);

  return (
    <div className="card group relative overflow-hidden">
      <video
        ref={videoRef}
        className="aspect-video w-full bg-black"
        playsInline
        onClick={toggle}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
        onDurationChange={(e) => setDuration(e.currentTarget.duration)}
      />

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 text-sm text-critical">
          {error}
        </div>
      )}

      {/* control bar — fades in on hover / while paused */}
      <AnimatePresence>
        <motion.div
          initial={false}
          animate={{ opacity: 1 }}
          className={`absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent px-4 pb-3 pt-10 transition-opacity ${
            playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
          }`}
        >
          <input
            type="range"
            min={0}
            max={duration || 0}
            step={0.1}
            value={time}
            aria-label="Seek"
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              if (videoRef.current) videoRef.current.currentTime = v;
            }}
            className="h-1 w-full cursor-pointer appearance-none rounded-full bg-white/20 accent-accent"
          />
          <div className="mt-2 flex items-center gap-4 text-sm text-ink-2">
            <button onClick={toggle} aria-label={playing ? "Pause" : "Play"} className="text-lg">
              {playing ? "❚❚" : "►"}
            </button>
            <span className="tabular-nums text-xs text-muted">
              {formatTime(time)} / {formatTime(duration)}
            </span>
            <button
              onClick={() => {
                const video = videoRef.current;
                if (!video) return;
                video.muted = !video.muted;
                setMuted(video.muted);
              }}
              aria-label={muted ? "Unmute" : "Mute"}
              className="ml-auto text-xs"
            >
              {muted ? "muted" : "sound"}
            </button>
            <button
              onClick={() => videoRef.current?.requestFullscreen()}
              aria-label="Fullscreen"
              className="text-xs"
            >
              ⛶
            </button>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
