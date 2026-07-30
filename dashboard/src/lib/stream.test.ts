/**
 * The SSE framer, which every study on the site reads its results through.
 *
 * This is the highest-consequence pure code in the frontend and it had no tests.
 * It parses bytes off a socket into events, and the failure modes are all silent:
 * a frame split across two chunks is not an error, it is a study that quietly
 * loses its `done`; a stream that closes early is not an error either, it is a
 * panel stuck on "running" with no message. Both have happened here before,
 * and both are recorded in the source as fixes with no test behind them.
 */

import { describe, expect, it, vi } from "vitest";
import { streamRun, type StreamEvent } from "./stream";

/** A fetch whose body yields exactly the chunks given, as bytes. */
function mockFetch(chunks: string[], init: Partial<Response> = {}) {
  const encoder = new TextEncoder();
  return vi.fn(async () => ({
    ok: true,
    status: 200,
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        for (const c of chunks) controller.enqueue(encoder.encode(c));
        controller.close();
      },
    }),
    ...init,
  })) as unknown as typeof fetch;
}

function frame(evt: Record<string, unknown>): string {
  return `data: ${JSON.stringify(evt)}\n\n`;
}

async function collect(fetchImpl: typeof fetch, signal?: AbortSignal) {
  vi.stubGlobal("fetch", fetchImpl);
  const seen: StreamEvent[] = [];
  await streamRun("/studies/content/stream", {}, (e) => seen.push(e), signal);
  vi.unstubAllGlobals();
  return seen;
}

describe("streamRun", () => {
  it("dispatches each frame in order", async () => {
    const seen = await collect(
      mockFetch([frame({ type: "start", total: 2 }), frame({ type: "agent" }), frame({ type: "done", result: {} })])
    );
    expect(seen.map((e) => e.type)).toEqual(["start", "agent", "done"]);
  });

  it("reassembles a frame split across chunk boundaries", async () => {
    // The failure this guards is invisible: TCP does not respect frame
    // boundaries, so a `done` arriving in two reads is a study that ends with
    // no result and no error rather than a parse error anyone would notice.
    const whole = frame({ type: "done", result: { twin_count: 7 } });
    const cut = Math.floor(whole.length / 2);
    const seen = await collect(mockFetch([whole.slice(0, cut), whole.slice(cut)]));
    expect(seen).toHaveLength(1);
    expect(seen[0].type).toBe("done");
    expect((seen[0].result as { twin_count: number }).twin_count).toBe(7);
  });

  it("reassembles many frames arriving in one chunk", async () => {
    const seen = await collect(
      mockFetch([frame({ type: "agent", i: 1 }) + frame({ type: "agent", i: 2 }) + frame({ type: "done" })])
    );
    expect(seen.map((e) => e.type)).toEqual(["agent", "agent", "done"]);
  });

  it("skips a malformed frame without losing the ones after it", async () => {
    const seen = await collect(
      mockFetch(["data: {not json\n\n", frame({ type: "done" })])
    );
    expect(seen.map((e) => e.type)).toEqual(["done"]);
  });

  it("ignores non-data lines such as comments and heartbeats", async () => {
    const seen = await collect(mockFetch([": keep-alive\n\n", frame({ type: "done" })]));
    expect(seen.map((e) => e.type)).toEqual(["done"]);
  });

  it("reports a stream that closes without a terminal frame", async () => {
    // THE regression the source documents: a backend that dies mid-run used to
    // resolve normally, so the panel's `finally { setBusy(false) }` closed the
    // stage on a run the user had just watched forty agents land in, with
    // nothing said.
    const seen = await collect(mockFetch([frame({ type: "start" }), frame({ type: "agent" })]));
    expect(seen.map((e) => e.type)).toEqual(["start", "agent", "error"]);
    expect(String(seen[2].detail)).toMatch(/ended before the run completed/i);
  });

  it("does not report an error when the run really did finish", async () => {
    const seen = await collect(mockFetch([frame({ type: "start" }), frame({ type: "done" })]));
    expect(seen.some((e) => e.type === "error")).toBe(false);
  });

  it("treats a server error frame as terminal", async () => {
    const seen = await collect(mockFetch([frame({ type: "error", detail: "no key" })]));
    expect(seen).toHaveLength(1);
    expect(seen[0].detail).toBe("no key");
  });

  it("surfaces the server's detail when the response is not ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 402,
        body: null,
        json: async () => ({ detail: "Budget exceeded." }),
      })) as unknown as typeof fetch
    );
    const seen: StreamEvent[] = [];
    await streamRun("/x", {}, (e) => seen.push(e));
    vi.unstubAllGlobals();
    expect(seen).toEqual([{ type: "error", detail: "Budget exceeded." }]);
  });

  it("falls back to the status code when the error body is unreadable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: false,
        status: 502,
        body: null,
        json: async () => {
          throw new Error("not json");
        },
      })) as unknown as typeof fetch
    );
    const seen: StreamEvent[] = [];
    await streamRun("/x", {}, (e) => seen.push(e));
    vi.unstubAllGlobals();
    expect(String(seen[0].detail)).toContain("502");
  });

  it("an abort is silent — it is a decision, not a failure", async () => {
    // Without this the deliberate cancellation in `useStreamRun` surfaced as
    // "the backend may have restarted", which is the same
    // misdiagnosis-in-a-message the early-close branch above exists to fix.
    //
    // The mock has to honour the signal the way a real `fetch` does: abort
    // rejects the in-flight `reader.read()` with an AbortError. A stream that
    // simply never resolves would hang instead of testing anything, which is
    // exactly what the first version of this test did.
    const controller = new AbortController();
    const encoder = new TextEncoder();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init?: RequestInit) => ({
        ok: true,
        status: 200,
        body: new ReadableStream<Uint8Array>({
          start(c) {
            c.enqueue(encoder.encode(frame({ type: "start" })));
          },
          pull() {
            return new Promise<void>((_, reject) => {
              init?.signal?.addEventListener("abort", () =>
                reject(new DOMException("Aborted", "AbortError"))
              );
            });
          },
        }),
      })) as unknown as typeof fetch
    );

    const seen: StreamEvent[] = [];
    const running = streamRun("/x", {}, (e) => seen.push(e), controller.signal);
    await new Promise((r) => setTimeout(r, 10));
    controller.abort();

    // Resolves rather than rejecting, and says nothing.
    await expect(running).resolves.toBeUndefined();
    vi.unstubAllGlobals();
    expect(seen.map((e) => e.type)).toEqual(["start"]);
    expect(seen.some((e) => e.type === "error")).toBe(false);
  });
});
