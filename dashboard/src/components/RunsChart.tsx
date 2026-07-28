"use client";

import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SwarmRunRow } from "@/lib/api";

// Series colors: validated dark categorical slots 1 & 2 (see dataviz reference).
const ENGAGEMENT = "#3987e5";
const INTENT = "#d95926";

export default function RunsChart({ runs }: { runs: SwarmRunRow[] }) {
  const data = [...runs]
    .reverse() // oldest → newest
    .map((r, i) => ({
      run: `#${i + 1}`,
      engagement: r.result.mean_engagement,
      intent: r.result.mean_intent,
      load: r.request.cognitive_load,
    }));

  if (data.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-muted">
        Run history appears here after your first swarm.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data} margin={{ top: 8, right: 12, left: -18, bottom: 0 }}>
        <defs>
          <linearGradient id="fill-eng" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={ENGAGEMENT} stopOpacity={0.28} />
            <stop offset="100%" stopColor={ENGAGEMENT} stopOpacity={0} />
          </linearGradient>
          <linearGradient id="fill-int" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={INTENT} stopOpacity={0.24} />
            <stop offset="100%" stopColor={INTENT} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid vertical={false} stroke="#2c2c2a" />
        <XAxis
          dataKey="run"
          tickLine={false}
          axisLine={{ stroke: "#383835" }}
          tick={{ fill: "#898781", fontSize: 12 }}
        />
        <YAxis
          domain={[0, 1]}
          tickLine={false}
          axisLine={false}
          tick={{ fill: "#898781", fontSize: 12 }}
        />
        <Tooltip
          cursor={{ stroke: "#383835" }}
          contentStyle={{
            background: "#1a1a19",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: 12,
            fontSize: 12,
            color: "#c3c2b7",
          }}
          formatter={(value: number, name: string, item) => [
            value.toFixed(2),
            name === "engagement" ? "Engagement" : "Intent",
          ]}
          labelFormatter={(label, payload) =>
            `Run ${label}${payload?.[0]?.payload?.load ? ` · load ${payload[0].payload.load}` : ""}`
          }
        />
        <Legend
          iconType="plainline"
          wrapperStyle={{ fontSize: 12, color: "#898781" }}
          formatter={(value) => (value === "engagement" ? "Engagement" : "Intent")}
        />
        <Area
          type="monotone"
          dataKey="engagement"
          stroke={ENGAGEMENT}
          strokeWidth={2}
          fill="url(#fill-eng)"
          dot={{ r: 3, fill: ENGAGEMENT, strokeWidth: 0 }}
          isAnimationActive={false}
        />
        <Area
          type="monotone"
          dataKey="intent"
          stroke={INTENT}
          strokeWidth={2}
          fill="url(#fill-int)"
          dot={{ r: 3, fill: INTENT, strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
