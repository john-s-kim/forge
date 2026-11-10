import { useMemo, useRef, useState } from "react";

interface Props {
  title: string;
  points: { step: number; value: number | null }[];
}

const W = 520;
const H = 200;
const M = { top: 12, right: 16, bottom: 26, left: 46 };

function ticks(lo: number, hi: number, n: number): number[] {
  if (hi <= lo) return [lo];
  const raw = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

const fmt = (v: number) =>
  Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2);

/** One metric, one axis. A single series, so the title names it and there is no legend. */
export function LineChart({ title, points }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  const g = useMemo(() => {
    const finite = points.filter((p): p is { step: number; value: number } => p.value != null);
    const steps = points.map((p) => p.step);
    const x0 = Math.min(...steps);
    const x1 = Math.max(...steps, x0 + 1);
    let y0 = Math.min(...finite.map((p) => p.value));
    let y1 = Math.max(...finite.map((p) => p.value));
    if (!finite.length) [y0, y1] = [0, 1];
    if (y0 === y1) [y0, y1] = [y0 - 0.5, y1 + 0.5];
    const pad = (y1 - y0) * 0.06;
    y0 -= pad;
    y1 += pad;
    const x = (s: number) => M.left + ((s - x0) / (x1 - x0)) * (W - M.left - M.right);
    const y = (v: number) => H - M.bottom - ((v - y0) / (y1 - y0)) * (H - M.top - M.bottom);
    // Break the line wherever the job logged a non-finite value.
    let path = "";
    let pen = false;
    for (const p of points) {
      if (p.value == null) {
        pen = false;
        continue;
      }
      path += `${pen ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.value).toFixed(1)}`;
      pen = true;
    }
    return { x, y, x0, x1, y0, y1, path, bad: points.filter((p) => p.value == null) };
  }, [points]);

  if (!points.length) return null;
  const at = hover == null ? null : points[hover];

  const onMove = (e: React.MouseEvent) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box) return;
    const px = ((e.clientX - box.left) / box.width) * W;
    let best = 0;
    points.forEach((p, i) => {
      if (Math.abs(g.x(p.step) - px) < Math.abs(g.x(points[best]!.step) - px)) best = i;
    });
    setHover(best);
  };

  return (
    <figure className="chart">
      <figcaption>
        <span>{title}</span>
        <span className="muted">
          {at ? `step ${at.step}: ${at.value == null ? "not finite" : fmt(at.value)}` : ""}
        </span>
      </figcaption>
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title} by training step`}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      >
        {ticks(g.y0, g.y1, 4).map((t) => (
          <g key={t}>
            <line className="grid" x1={M.left} x2={W - M.right} y1={g.y(t)} y2={g.y(t)} />
            <text className="tick" x={M.left - 6} y={g.y(t)} textAnchor="end" dominantBaseline="middle">
              {fmt(t)}
            </text>
          </g>
        ))}
        {ticks(g.x0, g.x1, 5).map((t) => (
          <text key={t} className="tick" x={g.x(t)} y={H - 8} textAnchor="middle">
            {t}
          </text>
        ))}
        <line className="axis" x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} />
        <path className="series" d={g.path} />
        {g.bad.map((p) => (
          <g key={p.step} className="bad">
            <line x1={g.x(p.step)} x2={g.x(p.step)} y1={M.top} y2={H - M.bottom} />
            <text x={g.x(p.step)} y={M.top + 2} textAnchor="middle" dominantBaseline="hanging">
              NaN
            </text>
          </g>
        ))}
        {at && (
          <g>
            <line className="crosshair" x1={g.x(at.step)} x2={g.x(at.step)} y1={M.top} y2={H - M.bottom} />
            {at.value != null && <circle className="dot" cx={g.x(at.step)} cy={g.y(at.value)} r={4} />}
          </g>
        )}
      </svg>
    </figure>
  );
}
