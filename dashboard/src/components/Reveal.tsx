"use client";

/**
 * Entrance animation that can never trap content invisible and never causes a
 * hydration mismatch: a pure-CSS keyframe (see .reveal in globals.css) with
 * `both` fill. Its timeline is time-based, not rAF-based, so it completes even
 * in hidden/background tabs — unlike a Framer mount animation, which parks at
 * opacity 0 until the tab is refocused. Server and client render identical
 * markup.
 */

import { CSSProperties, ReactNode, useState } from "react";

/**
 * True when JS-driven mount animations are safe (document visible at mount).
 * Use only for client-triggered motion (e.g. AnimatePresence result blocks
 * that mount after user interaction) — never for SSR'd markup, where the
 * server/client difference would be a hydration mismatch.
 */
export function useEntranceEnabled(): boolean {
  const [enabled] = useState(
    () => typeof document === "undefined" || document.visibilityState === "visible"
  );
  return enabled;
}

export default function Reveal({
  index = 0,
  className,
  children,
}: {
  index?: number;
  className?: string;
  children: ReactNode;
}) {
  return (
    <div
      className={`reveal ${className ?? ""}`}
      style={{ "--reveal-delay": `${index * 60}ms` } as CSSProperties}
    >
      {children}
    </div>
  );
}
