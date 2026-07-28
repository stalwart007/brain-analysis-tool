"use client";

import { FormEvent, Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { motion } from "framer-motion";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const res = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (res.ok) {
      router.replace(params.get("from") ?? "/");
      return;
    }
    const body = await res.json().catch(() => null);
    setError(body?.detail ?? "Sign-in failed");
    setBusy(false);
  }

  return (
    <div
      className="reveal card w-full max-w-sm p-8"
    >
      <div className="mb-8">
        <div className="mb-4 flex items-center gap-2 border-b border-hairline pb-3">
          <span className="pulse-dot inline-block h-2 w-2 rounded-full bg-bone" />
          <span className="display reg text-sm text-bone">CogniSwarm</span>
          <span className="hud-label ml-auto">cognitive instrument</span>
        </div>
        <h1 className="display text-[2.1rem] text-bone">Member sign-in</h1>
        <p className="mt-1 text-sm text-muted">
          Private research instance. Access is provisioned by your admin — there
          is no signup.
        </p>
      </div>

      <form onSubmit={submit} className="space-y-4">
        <label className="block">
          <span className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-muted">
            Email
          </span>
          <input
            type="email"
            required
            autoComplete="username"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full rounded-xl border border-hairline bg-surface-2 px-4 py-2.5 text-sm outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
          />
        </label>
        <label className="block">
          <span className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-muted">
            Password
          </span>
          <input
            type="password"
            required
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full rounded-xl border border-hairline bg-surface-2 px-4 py-2.5 text-sm outline-none transition focus:border-accent/60 focus:ring-2 focus:ring-accent/20"
          />
        </label>

        {error && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-sm text-critical"
            role="alert"
          >
            {error}
          </motion.p>
        )}

        <motion.button
          whileTap={{ scale: 0.98 }}
          disabled={busy}
          className="w-full rounded-xl bg-accent py-2.5 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Enter"}
        </motion.button>
      </form>
    </div>
  );
}

export default function LoginPage() {
  return (
    <main className="relative flex min-h-dvh items-center justify-center overflow-hidden p-6">
      {/* ambient gradient field */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(700px 500px at 22% 18%, rgba(223,217,217,0.05), transparent 62%)," +
            "radial-gradient(900px 700px at 78% 108%, rgba(223,217,217,0.035), transparent 68%)",
        }}
      />
      <Suspense>
        <LoginForm />
      </Suspense>
    </main>
  );
}
