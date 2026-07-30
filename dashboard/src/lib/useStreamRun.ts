"use client";

/**
 * `streamRun`, bound to the lifetime of the component that calls it.
 *
 * WHAT THIS FIXES. Ten panels called `streamRun` and not one of them could
 * stop it. The `fetch` had no abort signal, so navigating away, closing the
 * tab, or pressing Run a second time left the connection open — the backend
 * never saw a disconnect, never cancelled its twin tasks, and billed OpenAI to
 * completion for a result nobody would ever see. Pressing Run twice was worse
 * than wasteful: both streams stayed live and raced to write the same state,
 * so a slow first run could overwrite a fast second one.
 *
 * A hook rather than a rule to remember. "Every panel must create an
 * AbortController, abort the previous run, and abort again on unmount" is
 * three things to get right in ten files, and the eleventh panel will be
 * written by copying whichever one was easiest to find. Here it is impossible
 * to get wrong: call it, and cancellation is already handled.
 */

import { useCallback, useEffect, useRef } from "react";
import { streamRun, type StreamEvent } from "./stream";

export function useStreamRun() {
  const controller = useRef<AbortController | null>(null);

  // Unmount is the case that costs money — the reader is gone and the twins
  // are still running.
  useEffect(
    () => () => {
      controller.current?.abort();
      controller.current = null;
    },
    []
  );

  return useCallback(
    (path: string, body: unknown, onEvent: (evt: StreamEvent) => void): Promise<void> => {
      // A second run supersedes the first. Aborting rather than racing is what
      // stops a slow earlier stream landing its `done` on top of a later one.
      controller.current?.abort();
      const next = new AbortController();
      controller.current = next;
      return streamRun(path, body, (evt) => {
        // Frames buffered before the abort took effect are dropped, so a
        // superseded run cannot write into the state of the current one.
        if (!next.signal.aborted) onEvent(evt);
      }, next.signal);
    },
    []
  );
}
