/**
 * @vitest-environment jsdom
 *
 * The cancellation contract, which is the one frontend behaviour that costs
 * money when it is wrong.
 *
 * Before this hook existed, ten panels called `streamRun` and none could stop
 * it: no abort signal, so navigating away or pressing Run twice left the fetch
 * open, the backend never saw a disconnect, and the twins ran to completion
 * billing OpenAI for a result nobody would see. Two runs at once also raced to
 * write the same state.
 */

import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { act, render } from "@testing-library/react";
import { useStreamRun } from "./useStreamRun";
import type { StreamEvent } from "./stream";

/** Records the signal it was handed and stays open until told otherwise. */
function openStream() {
  const encoder = new TextEncoder();
  const signals: AbortSignal[] = [];
  const push: ((s: string) => void)[] = [];
  const fetchImpl = vi.fn(async (_url: string, init?: RequestInit) => {
    if (init?.signal) signals.push(init.signal);
    return {
      ok: true,
      status: 200,
      body: new ReadableStream<Uint8Array>({
        start(c) {
          push.push((s) => c.enqueue(encoder.encode(s)));
        },
        pull() {
          return new Promise<void>((_, reject) => {
            init?.signal?.addEventListener("abort", () =>
              reject(new DOMException("Aborted", "AbortError"))
            );
          });
        },
      }),
    };
  });
  return { fetchImpl: fetchImpl as unknown as typeof fetch, signals, push };
}

function Harness({ onReady }: { onReady: (run: ReturnType<typeof useStreamRun>) => void }) {
  const run = useStreamRun();
  onReady(run);
  return null;
}

let captured: ReturnType<typeof useStreamRun>;

beforeEach(() => {
  captured = undefined as never;
});
afterEach(() => vi.unstubAllGlobals());

describe("useStreamRun", () => {
  it("passes an abort signal to every request", async () => {
    const { fetchImpl, signals } = openStream();
    vi.stubGlobal("fetch", fetchImpl);
    render(<Harness onReady={(r) => (captured = r)} />);
    act(() => void captured("/x", {}, () => {}));
    await act(async () => void (await new Promise((r) => setTimeout(r, 5))));
    expect(signals).toHaveLength(1);
    expect(signals[0].aborted).toBe(false);
  });

  it("aborts the in-flight run when the component unmounts", async () => {
    // THE money bug. The reader is gone and the twins keep going.
    const { fetchImpl, signals } = openStream();
    vi.stubGlobal("fetch", fetchImpl);
    const view = render(<Harness onReady={(r) => (captured = r)} />);
    act(() => void captured("/x", {}, () => {}));
    await act(async () => void (await new Promise((r) => setTimeout(r, 5))));
    expect(signals[0].aborted).toBe(false);
    view.unmount();
    expect(signals[0].aborted).toBe(true);
  });

  it("a second run supersedes the first rather than racing it", async () => {
    const { fetchImpl, signals } = openStream();
    vi.stubGlobal("fetch", fetchImpl);
    render(<Harness onReady={(r) => (captured = r)} />);
    act(() => void captured("/x", {}, () => {}));
    await act(async () => void (await new Promise((r) => setTimeout(r, 5))));
    act(() => void captured("/x", {}, () => {}));
    await act(async () => void (await new Promise((r) => setTimeout(r, 5))));
    expect(signals).toHaveLength(2);
    expect(signals[0].aborted).toBe(true);   // the first is cancelled…
    expect(signals[1].aborted).toBe(false);  // …and the second is live
  });

  it("drops frames from a superseded run so it cannot write over the current one", async () => {
    // Aborting is not instantaneous. A stream that has not yet noticed the
    // abort can still deliver buffered frames, and without the hook's
    // `aborted` guard those reach the OLD handler — so a slow first run lands
    // its `done` on top of a fast second one and the panel shows stale results.
    //
    // The mock deliberately IGNORES the signal, which is what makes this test
    // about the guard rather than about the transport: a stream that closed on
    // abort would pass whether or not the guard existed.
    const encoder = new TextEncoder();
    let emit: ((s: string) => void) | null = null;
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          start(c) {
            if (!emit) emit = (s) => c.enqueue(encoder.encode(s));
          },
          pull() {
            return new Promise<void>(() => {});
          },
        }),
      })) as unknown as typeof fetch
    );

    render(<Harness onReady={(r) => (captured = r)} />);
    const first: StreamEvent[] = [];
    act(() => void captured("/x", {}, (e) => first.push(e)));
    await act(async () => void (await new Promise((r) => setTimeout(r, 5))));

    act(() => void captured("/x", {}, () => {}));           // supersedes run 1
    await act(async () => void (await new Promise((r) => setTimeout(r, 5))));

    // Run 1's transport, still unaware, delivers a terminal frame.
    await act(async () => {
      emit?.('data: {"type":"done","result":{"stale":true}}\n\n');
      await new Promise((r) => setTimeout(r, 5));
    });

    expect(first.some((e) => e.type === "done")).toBe(false);
  });

  it("unmounting with no run in flight is harmless", () => {
    const { fetchImpl } = openStream();
    vi.stubGlobal("fetch", fetchImpl);
    const view = render(<Harness onReady={(r) => (captured = r)} />);
    expect(() => view.unmount()).not.toThrow();
  });
});
