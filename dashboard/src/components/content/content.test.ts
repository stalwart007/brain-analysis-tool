/**
 * The pure logic behind Content Lab.
 *
 * All of this decides what a study is RUN ON, which makes it the frontend code
 * whose bugs are least visible: a mis-parsed timestamp does not throw, it moves
 * every beat on the timeline; a wrong quadrant does not throw, it tells someone
 * their beat is boring when it is confusing, and those want opposite fixes.
 */

import { describe, expect, it } from "vitest";
import { parseCues } from "@/components/AssetInput";
import { clock } from "@/components/content/YouTubeIngest";
import { verdictFor } from "@/components/content/BeatDiagnostic";

describe("clock", () => {
  it("formats as minutes and seconds, never raw seconds", () => {
    expect(clock(0)).toBe("0:00");
    expect(clock(7)).toBe("0:07");
    expect(clock(187)).toBe("3:07");
    expect(clock(600)).toBe("10:00");
  });

  it("adds an hours field only when there is one", () => {
    expect(clock(3599)).toBe("59:59");
    expect(clock(3600)).toBe("1:00:00");
    expect(clock(3764)).toBe("1:02:44");
  });

  it("never renders a negative or fractional clock", () => {
    // A duration can arrive as 0 (oEmbed carries none) or as a float from a
    // fraction calculation; neither should produce "-1:-3" or "3:07.5".
    expect(clock(-5)).toBe("0:00");
    expect(clock(66.7)).toBe("1:07");
  });
});

describe("parseCues", () => {
  it("reads mm:ss stamps into milliseconds", () => {
    const { cues, timed } = parseCues("0:00 Welcome back\n1:30 The problem");
    expect(timed).toBe(true);
    expect(cues).toEqual([
      { t_ms: 0, text: "Welcome back" },
      { t_ms: 90_000, text: "The problem" },
    ]);
  });

  it("reads a bare seconds stamp", () => {
    const { cues } = parseCues("12.5 half a minute in");
    expect(cues).toEqual([{ t_ms: 12_500, text: "half a minute in" }]);
  });

  it("untimed lines produce a SEQUENCE, not invented timings", () => {
    // The load-bearing behaviour: with no stamps the server must see a
    // sequential axis, so retention is reported per beat rather than in
    // seconds that were never measured.
    const { cues, timed } = parseCues("First line\nSecond line");
    expect(timed).toBe(false);
    expect(cues).toEqual([]);
  });

  it("drops untimed lines once ANY line is timed", () => {
    // A half-stamped transcript is the dangerous case: keeping the unstamped
    // lines at t_ms = -1 would sort them all to the front of the timeline.
    const { cues, timed } = parseCues("0:00 Intro\nstray note\n0:30 Next");
    expect(timed).toBe(true);
    expect(cues.map((c) => c.t_ms)).toEqual([0, 30_000]);
    expect(cues.every((c) => c.t_ms >= 0)).toBe(true);
  });

  it("ignores blank lines and trims", () => {
    const { cues } = parseCues("\n\n  0:05   spaced out  \n\n");
    expect(cues).toEqual([{ t_ms: 5_000, text: "spaced out" }]);
  });

  it("a stamp with no words is not a cue", () => {
    expect(parseCues("0:00\n0:10").cues).toEqual([]);
  });
});

describe("verdictFor — why a beat lands or does not", () => {
  it("separates the two ways attention can be lost", () => {
    // The distinction the whole panel exists for. Both are low attention and
    // they want OPPOSITE fixes: boring needs more at stake, confusing needs
    // less to follow.
    expect(verdictFor(0.2, 0.2).key).toBe("boring");
    expect(verdictFor(0.2, 0.8).key).toBe("confusing");
  });

  it("separates effortless engagement from hard work", () => {
    expect(verdictFor(0.8, 0.2).key).toBe("working");
    expect(verdictFor(0.8, 0.8).key).toBe("demanding");
  });

  it("splits on the midpoint of the rated scale, not on this run's spread", () => {
    // Quadrants relative to the run's own distribution would ALWAYS find a
    // worst beat, including in content where every beat is fine.
    expect(verdictFor(0.5, 0.5).key).toBe("demanding");
    expect(verdictFor(0.5, 0.49).key).toBe("working");
    expect(verdictFor(0.49, 0.5).key).toBe("confusing");
    expect(verdictFor(0.49, 0.49).key).toBe("boring");
  });

  it("every verdict carries an instruction, not just a label", () => {
    for (const [a, e] of [[0.9, 0.1], [0.9, 0.9], [0.1, 0.1], [0.1, 0.9]] as const) {
      const v = verdictFor(a, e);
      expect(v.fix.length).toBeGreaterThan(20);
      expect(v.color).toMatch(/^#[0-9a-f]{6}$/i);
    }
  });
});
