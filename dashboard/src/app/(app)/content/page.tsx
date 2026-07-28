"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";
import Reveal, { useEntranceEnabled } from "@/components/Reveal";
import ContentImpactPanel from "@/components/ContentImpactPanel";
import { api, SessionRow } from "@/lib/api";

// Player (and hls.js behind it) stays out of the dashboard bundle entirely.
const Player = dynamic(() => import("@/components/Player"), { ssr: false });

const SAMPLE = "https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8";

export default function ContentLabPage() {
  const entrance = useEntranceEnabled();
  const [url, setUrl] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [personaCount, setPersonaCount] = useState(0);

  useEffect(() => {
    api
      .sessions()
      .then((s: SessionRow[]) => setPersonaCount(s.filter((x) => x.persona).length))
      .catch(() => {});
  }, []);

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Reveal>
        <div className="flex items-baseline gap-3">
          <span className="tag text-muted">Sector CNTL</span>
          <span className="h-px flex-1 bg-hairline" />
          <span className="panel-index">06</span>
        </div>
        <h1 className="display reg mt-2 text-[clamp(2.4rem,6vw,4.4rem)] text-bone">Content Lab</h1>
        <p className="mt-1 max-w-xl text-sm text-muted">
          Run content through the swarm&apos;s minds — or preview a stimulus
          stream before screening it.
        </p>

        <div className="mt-6">
          <ContentImpactPanel personaCount={personaCount} />
        </div>

        <h2 className="display mt-10 text-lg text-bone">Stream preview</h2>
        <p className="mt-1 max-w-xl text-sm text-muted">
          Paste an HLS manifest (.m3u8); playback adapts bitrate automatically.
        </p>

        <div className="mt-6 flex gap-2">
          <input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder={SAMPLE}
            className="flex-1 rounded-xl border border-hairline bg-surface-2 px-4 py-2.5 text-sm outline-none transition placeholder:text-muted focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
          />
          <motion.button
            whileTap={{ scale: 0.97 }}
            onClick={() => setActive(url.trim() || SAMPLE)}
            className="rounded-xl bg-accent px-5 py-2.5 font-display text-sm font-semibold text-white transition hover:brightness-110"
          >
            Load
          </motion.button>
        </div>

        {active && (
          <motion.div
            initial={entrance ? { opacity: 0, scale: 0.99 } : false}
            animate={{ opacity: 1, scale: 1 }}
            className="mt-6"
          >
            <Player src={active} />
          </motion.div>
        )}
      </Reveal>
    </div>
  );
}
