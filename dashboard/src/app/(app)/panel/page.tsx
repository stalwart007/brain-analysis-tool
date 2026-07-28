"use client";

import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, PanelMember } from "@/lib/api";
import Reveal from "@/components/Reveal";

function tokenFrom(disclosureUrl: string): string {
  return disclosureUrl.split("/").pop() ?? "";
}

export default function PanelPage() {
  const [members, setMembers] = useState<PanelMember[]>([]);
  const [label, setLabel] = useState("");
  const [invite, setInvite] = useState<{ label: string; url: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    api.panelMembers().then(setMembers).catch(() => {});
  }, []);
  useEffect(refresh, [refresh]);

  async function createMember() {
    setError(null);
    try {
      const m = await api.createPanelMember(label.trim());
      // The disclosure page is served by the API server (its own public origin);
      // members visit it directly, never the closed-group dashboard.
      setInvite({ label: m.label, url: m.disclosure_url });
      setLabel("");
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create member");
    }
  }

  async function revoke(member: PanelMember) {
    if (!confirm(`Revoke ${member.label} and permanently delete their ${member.session_count} session(s)?`))
      return;
    try {
      await api.revokePanelMember(tokenFrom(member.disclosure_url));
      refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Revoke failed");
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-6 py-10">
      <Reveal>
        <div className="flex items-baseline gap-3">
          <span className="tag text-muted">Sector PANL</span>
          <span className="h-px flex-1 bg-hairline" />
          <span className="panel-index">05</span>
        </div>
        <h1 className="display reg mt-2 text-[clamp(2.4rem,6vw,4.4rem)] text-bone">Research Panel</h1>
        <p className="mt-1 max-w-2xl text-sm text-muted">
          The only defensible cross-site data model: a disclosed, opt-in,
          compensated panel. Each member gets a private disclosure link, consents
          explicitly, and can revoke at any time — revocation permanently erases
          all of their telemetry.
        </p>
      </Reveal>

      <Reveal index={1} className="mt-6 block">
        <div className="card p-6">
          <h2 className="mb-3 font-display text-lg font-medium tracking-tight">
            Invite a member
          </h2>
          <div className="flex gap-2">
            <input
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="Internal label (e.g. 'panelist 42, US cohort') — never PII"
              className="flex-1 rounded-xl border border-hairline bg-surface-2 px-4 py-2.5 text-sm outline-none placeholder:text-muted focus:border-accent/60"
            />
            <motion.button
              whileTap={{ scale: 0.97 }}
              disabled={!label.trim()}
              onClick={createMember}
              className="rounded-xl bg-accent px-5 py-2.5 font-display text-sm font-semibold text-white transition hover:brightness-110 disabled:opacity-40"
            >
              Generate invite
            </motion.button>
          </div>
          {error && <p className="mt-3 text-sm text-critical">{error}</p>}
          {invite && (
            <div className="mt-4 rounded-xl border border-accent/30 bg-accent/5 p-4">
              <p className="text-sm">
                Invite for <b>{invite.label}</b> — share this disclosure link
                (served by the API host, e.g. dev <code className="text-xs">:8000</code>):
              </p>
              <code className="mt-1 block break-all text-xs text-accent">{invite.url}</code>
              <p className="mt-1 text-xs text-muted">
                The member reads the full disclosure there and consents (or declines)
                themselves.
              </p>
            </div>
          )}
        </div>
      </Reveal>

      <Reveal index={2} className="mt-6 block">
        <div className="card p-6">
          <h2 className="mb-4 font-display text-lg font-medium tracking-tight">Members</h2>
          {members.length === 0 ? (
            <p className="text-sm text-muted">No panel members yet.</p>
          ) : (
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase tracking-wider text-muted">
                <tr>
                  <th className="pb-2 pr-2 font-medium">Label</th>
                  <th className="pb-2 pr-2 font-medium">Status</th>
                  <th className="pb-2 pr-2 font-medium tabular-nums">Sessions</th>
                  <th className="pb-2" />
                </tr>
              </thead>
              <tbody>
                {members.map((m) => {
                  const status = m.revoked_at
                    ? "revoked"
                    : m.consented_at
                      ? "active"
                      : "invited";
                  return (
                    <tr key={m.id} className="border-t border-hairline/60">
                      <td className="py-2 pr-2">{m.label}</td>
                      <td className="py-2 pr-2">
                        <span
                          className={`text-xs font-medium ${
                            status === "active"
                              ? "text-good"
                              : status === "revoked"
                                ? "text-critical"
                                : "text-muted"
                          }`}
                        >
                          {status}
                        </span>
                      </td>
                      <td className="py-2 pr-2 tabular-nums">{m.session_count}</td>
                      <td className="py-2 text-right">
                        {!m.revoked_at && (
                          <button
                            onClick={() => revoke(m)}
                            className="rounded-lg border border-hairline px-3 py-1 text-xs text-muted transition hover:border-critical/50 hover:text-critical"
                          >
                            Revoke &amp; erase
                          </button>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </Reveal>
    </div>
  );
}
