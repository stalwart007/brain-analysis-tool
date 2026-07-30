"use client";

/**
 * WHITE ROOM — the theory-of-mind sector.
 *
 * Every other station measures a reaction to something you made. This one
 * measures a reaction to each other: a cast of characters at a table, arguing a
 * motion, each modelling what the others believe while holding a position none
 * of them will say out loud.
 *
 * It lives in the temporoparietal junction because that is the structure that
 * represents another mind's beliefs as separate from your own. The route mapping
 * in lib/cortex.ts is functional, not decorative — descending into this sector
 * flies the camera laterally out through the temporal lobe, and the whole
 * interface re-tints to the temporal amber.
 */

import RoomPanel from "@/components/RoomPanel";
import Station, { StationHeader, StationSpec } from "@/components/Station";

const ROOM: StationSpec = {
  id: "room-table",
  kicker: "Sector ROOM",
  index: "01",
  anatomy: "temporoparietal junction",
  rationale:
    "theory of mind — the TPJ represents what someone ELSE believes as distinct from what you believe, which is the entire job of a room where everyone is reading everyone",
  region: "temporal",
  histology: "laminae",
  camera: { cam: [0.12, 0.18, 0.14], look: [1.22, -0.06, -0.34], inside: 1 },
};

export default function RoomPage() {
  return (
    <div className="mx-auto max-w-5xl px-6 pb-24">
      <Station spec={ROOM}>
        <div className="pt-16">
          <StationHeader
            spec={ROOM}
            title="Deliberate"
            lede="Describe the people in prose; they are cast, seated, and made to argue one motion over several rounds. Every statement is public and carries a position; every member also holds a private one the table never sees. The gap between those two numbers is the measurement — and a control arm, shown statements from a different room entirely, is what decides whether the gap means anything at all."
          />
        </div>
        <RoomPanel />
      </Station>
    </div>
  );
}
