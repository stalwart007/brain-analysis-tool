/**
 * @vitest-environment jsdom
 *
 * An absent interval is a FINDING, not a rendering gap.
 */

import { describe, expect, it } from "vitest";
import { render } from "@testing-library/react";
import Interval, { MIN_BOOTSTRAP_N } from "./Interval";

describe("Interval", () => {
  it("renders the interval when there is one", () => {
    const { container } = render(<Interval ci={[0.1234, 0.5678]} />);
    expect(container.textContent).toContain("[0.123, 0.568]");
  });

  it("says WHY there is none, rather than vanishing or saying n/a", () => {
    // Two of the three call sites used `{ci && …}` and rendered nothing, and
    // the third said "interval n/a". Both read as a missing field — something
    // to go and look for in the payload — when the truth is a fact about the
    // run. Opposite messages: one sends a reader to hunt a bug, the other
    // tells them to add twins.
    const { container } = render(<Interval ci={null} />);
    expect(container.textContent).toMatch(new RegExp(`under ${MIN_BOOTSTRAP_N} observations`));
    expect(container.textContent).not.toMatch(/n\/a/i);
    expect(container.querySelector("[title]")?.getAttribute("title")).toMatch(
      /at least 8 observations/i
    );
  });

  it("distinguishes 'no spread' from 'no interval'", () => {
    // A degenerate interval is a real measurement — every observation was
    // identical — and must not be reported as a tight one OR as a missing one.
    const { container } = render(<Interval ci={[0.5, 0.5]} />);
    expect(container.textContent).toContain("no observed spread");
    expect(container.textContent).not.toContain("under");
  });

  it("treats a malformed interval as absent rather than crashing", () => {
    const { container } = render(<Interval ci={[0.5] as unknown as [number, number]} />);
    expect(container.textContent).toContain("no interval");
  });

  it("lets a caller override the reason", () => {
    const { container } = render(<Interval ci={null} absentReason="Not identified here." />);
    expect(container.querySelector("[title]")?.getAttribute("title")).toBe("Not identified here.");
  });
});
