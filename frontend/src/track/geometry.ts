// Arc-length mapping from lap fraction to track X/Y: the same rule as the backend's
// presentation/track_geometry.py (fraction 0 = the start/finish line = first outline point).

export type Point = [number, number];

export class TrackPath {
  readonly xs: number[];
  readonly ys: number[];
  /** Arc length at each vertex; the last entry closes the loop back to the first point. */
  readonly cumulative: number[];

  constructor(points: Point[]) {
    if (points.length < 3) throw new Error("a track needs at least 3 points");
    this.xs = points.map((p) => p[0]);
    this.ys = points.map((p) => p[1]);
    this.cumulative = [0];
    const n = points.length;
    for (let i = 1; i <= n; i++) {
      const a = points[i - 1]!;
      const b = points[i % n]!;
      this.cumulative.push(this.cumulative[i - 1]! + Math.hypot(b[0] - a[0], b[1] - a[1]));
    }
  }

  get length(): number {
    return this.cumulative[this.cumulative.length - 1]!;
  }

  /** Segment index i (from vertex i-1 to vertex i mod n) containing `fraction`, and the target length. */
  private segment(fraction: number): [number, number] {
    const f = fraction - Math.floor(fraction);
    const target = f * this.length;
    let lo = 1;
    let hi = this.cumulative.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.cumulative[mid]! < target) lo = mid + 1;
      else hi = mid;
    }
    return [lo, target];
  }

  pointAt(fraction: number): Point {
    const [i, target] = this.segment(fraction);
    const n = this.xs.length;
    const start = this.cumulative[i - 1]!;
    const end = this.cumulative[i]!;
    const t = end === start ? 0 : (target - start) / (end - start);
    const x0 = this.xs[i - 1]!;
    const y0 = this.ys[i - 1]!;
    return [x0 + t * (this.xs[i % n]! - x0), y0 + t * (this.ys[i % n]! - y0)];
  }

  /** Direction of travel in radians, counter-clockwise from +X. */
  headingAt(fraction: number): number {
    const [i] = this.segment(fraction);
    const n = this.xs.length;
    return Math.atan2(this.ys[i % n]! - this.ys[i - 1]!, this.xs[i % n]! - this.xs[i - 1]!);
  }
}

/** Rotate a point by `deg` counter-clockwise (the track's display rotation). */
export function rotate([x, y]: Point, deg: number): Point {
  const a = (deg * Math.PI) / 180;
  const c = Math.cos(a);
  const s = Math.sin(a);
  return [x * c - y * s, x * s + y * c];
}

export interface View {
  scale: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
  rotationDeg: number;
}

/** A view that fits the rotated outline into width x height with `pad` pixels of margin. */
export function fitView(
  outline: Point[],
  rotationDeg: number,
  width: number,
  height: number,
  pad = 32,
): View {
  const pts = outline.map((p) => rotate(p, rotationDeg));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = Math.max(
    1e-6,
    Math.min((width - 2 * pad) / (maxX - minX || 1), (height - 2 * pad) / (maxY - minY || 1)),
  );
  return { scale, cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, width, height, rotationDeg };
}

/** Track metres -> canvas pixels (y up in the world, down on screen). */
export function toScreen(view: View, p: Point): Point {
  const [x, y] = rotate(p, view.rotationDeg);
  return [view.width / 2 + (x - view.cx) * view.scale, view.height / 2 - (y - view.cy) * view.scale];
}
