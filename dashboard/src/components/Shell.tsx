"use client";

import { ReactNode, useEffect, useRef, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import CortexRail from "./CortexRail";
import CortexNavigator from "./CortexNavigator";
import TissueField from "./TissueField";
import { Probe, ScrollWarp } from "./Craft";
import { REGION_RGB, RegionId, sectorFor } from "@/lib/cortex";
import { subscribeVitals, Vitals } from "@/lib/vitals";
import { JourneyProvider, useJourney } from "@/lib/journey";

/**
 * Live cortical trace — the console's vital sign.
 *
 * It no longer runs its own rAF clock. It subscribes to the shared
 * physiological clock, so the wiggle you see here is *the same signal*
 * driving every glow and pulse elsewhere on the page. Previously this canvas
 * animated forever, even in a hidden tab; the shared clock stops when the tab
 * is hidden or reduced motion is requested.
 */
function EEGStrip() {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const history = useRef<number[]>([]);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const W = 96;
    const H = 20;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr;
    cv.height = H * dpr;
    ctx.scale(dpr, dpr);

    // The trace is a scrolling *history* of the shared signal rather than a
    // function re-evaluated across x. That means the rightmost pixel is
    // genuinely "now" and matches the pulse of every dot on screen.
    const draw = (v: Vitals) => {
      const h = history.current;
      h.push(v.eeg);
      if (h.length > W) h.splice(0, h.length - W);

      ctx.clearRect(0, 0, W, H);
      ctx.strokeStyle = "rgba(223,217,217,0.75)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i < h.length; i++) {
        const x = W - h.length + i;
        const y = H / 2 - h[i] * (H / 2 - 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    return subscribeVitals(draw);
  }, []);

  return <canvas ref={ref} className="hidden h-5 w-24 opacity-70 md:block" aria-hidden />;
}

function Clock() {
  const [t, setT] = useState<string>("");
  useEffect(() => {
    const tick = () =>
      setT(
        new Date().toLocaleTimeString("en-GB", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      );
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, []);
  return <span className="font-mono text-xs tabular-nums text-ink-2">{t || "--:--:--"}</span>;
}

function ShellInner({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const sector = sectorFor(pathname);
  const [hovered, setHovered] = useState<RegionId | null>(null);
  const journey = useJourney();

  /* Two things can drive the camera. A scroll waypoint wins when the page
     declares one, because it is the more specific intent — the route says
     "you are in the prefrontal cortex", the waypoint says "you are at the
     pial surface of it, looking down". */
  const station = journey.active?.station ?? sector.station;
  const region = journey.active?.region ?? sector.region;
  const stationKey = journey.active?.id ?? sector.href;

  /* Keep the clock running for as long as the app shell is mounted. Nothing
     here reads the values in JS — the CSS variables it publishes are what the
     stylesheet consumes. */
  useEffect(() => subscribeVitals(), []);

  /* THE REGION CHANNEL. One assignment re-tints membranes, glows, mastheads,
     the depth meter and the tissue field, because every one of those rules is
     written against --region-rgb. Hover previews it, so pointing at a nav item
     warms the whole room before you commit. */
  useEffect(() => {
    const shown = hovered ?? region;
    const root = document.documentElement;
    root.style.setProperty("--region-rgb", REGION_RGB[shown]);
    root.style.setProperty("--depth", String(station.inside));
  }, [region, station.inside, hovered]);

  async function logout() {
    await fetch("/api/auth/logout", { method: "POST" });
    router.replace("/login");
  }

  return (
    <div className="relative min-h-dvh">
      {/* the brain lives behind everything and is the navigation */}
      <CortexNavigator
        station={station}
        region={region}
        stationKey={stationKey}
        hovered={hovered}
      />

      {/* histology sits between the cloud and the content, so the page has a
          middle distance instead of a backdrop */}
      <TissueField />

      {/* …and the reading channel sits above BOTH, dimming a soft column down
          the middle so type always wins against a luminous tractogram. Its
          strength scales with --depth, since that is where the problem is. */}
      <div className="reading-scrim" aria-hidden />

      <CortexRail
        active={sector}
        hovered={hovered}
        onHover={setHovered}
        onSignOut={logout}
        depthLabel={journey.active?.anatomy}
        depthRationale={journey.active?.rationale}
        inside={station.inside}
      />

      {/* vitals, pinned bottom-right so they never collide with the rail */}
      <div className="pointer-events-none fixed bottom-0 right-0 z-20 flex items-center gap-4 border-l border-t border-hairline bg-page/85 px-4 py-1.5 backdrop-blur-md">
        <span className="flex items-center gap-1.5">
          <span className="neon-dot pulse-dot inline-block h-1.5 w-1.5 rounded-full bg-bone" />
          <span className="hud-label text-ink-2">online</span>
        </span>
        <EEGStrip />
        <Clock />
      </div>

      {/* Content re-enters on every route with a CSS-driven transition: a
          rAF-based one parks at opacity 0 in hidden tabs, dimming the page. */}
      {/* scroll shear is published as a CSS variable; `warp` on <main> is the
          only element that consumes it, so the chrome stays rigid while the
          content sheet carries the inertia */}
      <ScrollWarp />
      <Probe />

      <main key={pathname} className="cortex-content warp relative z-10">
        <span className="tissue-flash" aria-hidden />
        {children}
      </main>
    </div>
  );
}

export default function Shell({ children }: { children: ReactNode }) {
  /* The provider wraps the shell, not just the page, because the shell's own
     chrome (camera, depth HUD, region tint) reads the active waypoint. */
  return (
    <JourneyProvider>
      <ShellInner>{children}</ShellInner>
    </JourneyProvider>
  );
}
