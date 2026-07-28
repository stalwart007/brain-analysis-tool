"use client";

/**
 * FUNCTIONAL BRAIN ATLAS — a stylized cortical profile whose seven regions
 * glow with model-derived activation scores (appraisal-theory constructs
 * projected onto named cognitive systems). Explicitly a visualization of
 * measured constructs — the disclaimer is part of the component.
 */

export interface SystemScores {
  salience: number;
  reward: number;
  threat: number;
  memory: number;
  social: number;
  semantic: number;
  executive: number;
}

const REGIONS: {
  key: keyof SystemScores;
  label: string;
  system: string;
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  rot: number;
  color: string;
  labelX: number;
  labelY: number;
}[] = [
  { key: "executive", label: "EXECUTIVE CONTROL", system: "frontoparietal", cx: 118, cy: 92, rx: 40, ry: 30, rot: -18, color: "#4fb6ff", labelX: 24, labelY: 40 },
  { key: "salience", label: "SALIENCE / ATTENTION", system: "cingulo-insular", cx: 176, cy: 128, rx: 34, ry: 24, rot: -8, color: "#f2ad1f", labelX: 150, labelY: 22 },
  { key: "reward", label: "REWARD / APPROACH", system: "striatal", cx: 150, cy: 160, rx: 26, ry: 17, rot: 0, color: "#16d016", labelX: 24, labelY: 258 },
  { key: "threat", label: "THREAT / AVOIDANCE", system: "amygdalar", cx: 186, cy: 178, rx: 20, ry: 13, rot: 0, color: "#f0605f", labelX: 132, labelY: 276 },
  { key: "memory", label: "MEMORY ENCODING", system: "hippocampal", cx: 224, cy: 170, rx: 26, ry: 15, rot: 8, color: "#b07ff0", labelX: 268, labelY: 258 },
  { key: "semantic", label: "SEMANTIC PROCESSING", system: "temporal", cx: 216, cy: 200, rx: 40, ry: 16, rot: 6, color: "#38c8c8", labelX: 300, labelY: 232 },
  { key: "social", label: "SOCIAL COGNITION", system: "temporoparietal", cx: 262, cy: 120, rx: 32, ry: 26, rot: 14, color: "#f078c8", labelX: 300, labelY: 40 },
];

export default function BrainMap({
  scores,
  caption,
}: {
  scores: SystemScores;
  caption?: string;
}) {
  return (
    <div>
      <svg viewBox="0 0 400 300" className="w-full">
        <defs>
          <filter id="bm-glow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="7" result="b" />
            <feMerge>
              <feMergeNode in="b" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* measurement grid backdrop */}
        {[60, 120, 180, 240].map((y) => (
          <line key={y} x1={12} y1={y} x2={388} y2={y} stroke="rgba(120,165,230,0.06)" />
        ))}

        {/* cortical outline, profile facing left */}
        <path
          d="M 64,158 C 58,108 96,58 158,46 C 226,34 300,52 330,100
             C 348,130 346,162 326,184 C 344,196 340,214 318,220
             C 322,238 298,248 278,240 C 274,254 244,258 230,246
             C 214,260 186,258 174,244 C 150,256 116,250 102,232
             C 78,222 60,196 64,158 Z"
          fill="rgba(20,30,48,0.55)"
          stroke="rgba(140,180,240,0.5)"
          strokeWidth="1.4"
        />
        {/* cerebellum + brainstem */}
        <path
          d="M 282,226 C 306,222 322,232 318,246 C 314,260 288,264 272,254 C 264,246 268,232 282,226 Z"
          fill="rgba(20,30,48,0.5)"
          stroke="rgba(140,180,240,0.4)"
          strokeWidth="1.2"
        />
        <path d="M 268,250 C 264,262 258,272 250,280" fill="none" stroke="rgba(140,180,240,0.4)" strokeWidth="3" strokeLinecap="round" />
        {/* a few gyri strokes for texture */}
        <path d="M 120,80 C 150,70 180,72 205,64 M 110,130 C 140,118 168,122 196,110 M 240,90 C 262,84 286,90 304,104 M 150,190 C 178,182 206,186 236,180"
          fill="none" stroke="rgba(140,180,240,0.16)" strokeWidth="1" />

        {/* regions */}
        {REGIONS.map((r) => {
          const v = Math.max(0, Math.min(1, scores[r.key] ?? 0));
          return (
            <g key={r.key} transform={`rotate(${r.rot} ${r.cx} ${r.cy})`}>
              <ellipse
                cx={r.cx} cy={r.cy} rx={r.rx} ry={r.ry}
                fill={r.color}
                opacity={0.10 + v * 0.55}
                filter={v > 0.35 ? "url(#bm-glow)" : undefined}
              />
              <ellipse
                cx={r.cx} cy={r.cy} rx={r.rx} ry={r.ry}
                fill="none" stroke={r.color} strokeWidth="0.8" opacity={0.35 + v * 0.4}
              />
            </g>
          );
        })}

        {/* leader lines + labels + values */}
        {REGIONS.map((r) => {
          const v = Math.max(0, Math.min(1, scores[r.key] ?? 0));
          const anchor = r.labelX > 200 ? "end" : "start";
          const lx = r.labelX > 200 ? r.labelX : r.labelX;
          return (
            <g key={`l-${r.key}`} fontFamily="ui-monospace, monospace">
              <line
                x1={r.cx} y1={r.cy} x2={lx + (anchor === "end" ? -2 : 2)} y2={r.labelY - 3}
                stroke={r.color} strokeWidth="0.6" opacity="0.45"
              />
              <text x={r.labelX} y={r.labelY - 6} fontSize="7.5" fill={r.color} textAnchor={anchor} letterSpacing="0.08em">
                {r.label}
              </text>
              <text x={r.labelX} y={r.labelY + 3} fontSize="7" fill="rgba(160,175,200,0.85)" textAnchor={anchor}>
                {r.system} · {(v * 100).toFixed(0)}%
              </text>
              {/* activation meter */}
              <rect x={anchor === "end" ? r.labelX - 44 : r.labelX} y={r.labelY + 7} width="44" height="2.4" rx="1.2" fill="rgba(255,255,255,0.08)" />
              <rect x={anchor === "end" ? r.labelX - 44 : r.labelX} y={r.labelY + 7} width={44 * v} height="2.4" rx="1.2" fill={r.color} opacity="0.85" />
            </g>
          );
        })}
      </svg>
      <p className="mt-1 text-[9px] leading-tight text-muted">
        {caption ??
          "Functional-system activation — appraisal constructs projected onto named cognitive systems. Model-derived scores, not neuroimaging."}
      </p>
    </div>
  );
}
