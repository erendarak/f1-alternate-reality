// 2D track view: real outline (FastF1-derived), cars as team-coloured markers with their codes
// (plan §14.4: simple markers first, no logos), safety car, retired cars, SC/VSC tint.
// Drawn at the display's frame rate from the latest two stream frames (track/interpolate.ts).

import { useEffect, useMemo, useRef } from "react";

import type { StreamHandle } from "../api/stream";
import type { Track } from "../api/types";
import { fitView, toScreen, TrackPath, type Point, type View } from "../track/geometry";
import { blendFactor, interpolateCars } from "../track/interpolate";

interface Props {
  track: Track;
  frames: StreamHandle["frames"];
  selected: string | null;
  onSelect: (driverId: string | null) => void;
}

const CAR_R = 6;
const PICK_PX = 14;

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function TrackCanvas({ track, frames, selected, onSelect }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const path = useMemo(() => new TrackPath(track.outline), [track]);
  const viewRef = useRef<View | null>(null);
  const shownRef = useRef<{ id: string; p: Point }[]>([]);
  const selectedRef = useRef(selected);
  selectedRef.current = selected;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;

    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      if (width < 50 || height < 50) return; // hidden pane: keep the last good view
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      viewRef.current = fitView(track.outline, track.rotation_deg, width, height);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    resize();

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const view = viewRef.current;
      const { frame, previous } = frames.current;
      if (!view) return;
      const colors = {
        bg: cssVar("--track-bg"),
        asphalt: cssVar("--asphalt"),
        edge: cssVar("--track-edge"),
        caution: cssVar("--caution"),
        text: cssVar("--text"),
        muted: cssVar("--muted"),
      };
      ctx.clearRect(0, 0, view.width, view.height);
      ctx.fillStyle = colors.bg;
      ctx.fillRect(0, 0, view.width, view.height);

      const phase = frame?.state.race_control.phase ?? "GREEN";
      const outline = track.outline.map((p) => toScreen(view, p));
      const trace = () => {
        ctx.beginPath();
        outline.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
        ctx.closePath();
      };
      ctx.lineJoin = "round";
      trace();
      ctx.strokeStyle = phase === "GREEN" ? colors.edge : colors.caution;
      ctx.lineWidth = 14;
      ctx.stroke();
      trace();
      ctx.strokeStyle = colors.asphalt;
      ctx.lineWidth = 10;
      ctx.stroke();

      // start/finish line, across the track
      const [sx, sy] = toScreen(view, track.start_finish);
      const heading = path.headingAt(0) + (view.rotationDeg * Math.PI) / 180;
      const nx = Math.sin(heading);
      const ny = Math.cos(heading);
      ctx.strokeStyle = colors.text;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(sx - nx * 9, sy - ny * 9);
      ctx.lineTo(sx + nx * 9, sy + ny * 9);
      ctx.stroke();

      ctx.font = "10px system-ui, sans-serif";
      ctx.fillStyle = colors.muted;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (const c of track.corners) {
        const [cx, cy] = toScreen(view, [c.x_m, c.y_m]);
        ctx.fillText(`${c.number}${c.letter ?? ""}`, cx, cy);
      }

      if (!frame) return;
      const now = performance.now();
      const state = frame.state;
      const shown = new Map(interpolateCars(previous, frame, now).map((s) => [s.driverId, s.fraction]));
      const cars = [...state.cars].reverse(); // leader drawn last, on top
      const placed: { id: string; p: Point }[] = [];
      for (const car of cars) {
        const p = toScreen(view, path.pointAt(shown.get(car.driver_id) ?? car.lap_fraction));
        placed.push({ id: car.driver_id, p });
        const isSel = car.driver_id === selectedRef.current;
        const out = car.status === "RETIRED";
        ctx.beginPath();
        ctx.arc(p[0], p[1], isSel ? CAR_R + 2 : CAR_R, 0, Math.PI * 2);
        ctx.fillStyle = out ? colors.muted : `#${car.team_color}`;
        ctx.fill();
        ctx.lineWidth = isSel ? 3 : 1.5;
        ctx.strokeStyle = isSel ? colors.text : colors.bg;
        ctx.stroke();
        if (out) {
          ctx.strokeStyle = "#e5484d";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(p[0] - 4, p[1] - 4);
          ctx.lineTo(p[0] + 4, p[1] + 4);
          ctx.moveTo(p[0] + 4, p[1] - 4);
          ctx.lineTo(p[0] - 4, p[1] + 4);
          ctx.stroke();
        }
        ctx.font = isSel ? "bold 12px system-ui, sans-serif" : "11px system-ui, sans-serif";
        ctx.fillStyle = out ? colors.muted : colors.text;
        ctx.textAlign = "left";
        ctx.fillText(car.code, p[0] + CAR_R + 3, p[1] - CAR_R - 2);
      }
      shownRef.current = placed;

      const sc = state.race_control.safety_car;
      if (sc) {
        let f = sc.lap_fraction;
        const before = previous?.state.race_control.safety_car;
        if (before && previous && state.status === "RUNNING") {
          let d = f - before.lap_fraction;
          if (d < -0.5) d += 1;
          if (d >= 0 && d < 0.5) f = before.lap_fraction + d * blendFactor(previous, frame, now);
        }
        const [x, y] = toScreen(view, path.pointAt(f));
        ctx.fillStyle = colors.caution;
        ctx.fillRect(x - 8, y - 6, 16, 12);
        ctx.fillStyle = "#111";
        ctx.font = "bold 9px system-ui, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("SC", x, y + 0.5);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, [track, path, frames]);

  const onClick = (ev: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = ev.currentTarget.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    let best: { id: string; d: number } | null = null;
    for (const { id, p } of shownRef.current) {
      const d = Math.hypot(p[0] - x, p[1] - y);
      if (d <= PICK_PX && (best === null || d < best.d)) best = { id, d };
    }
    onSelect(best?.id ?? null);
  };

  return (
    <div className="track-wrap">
      <canvas
        ref={canvasRef}
        className="track-canvas"
        onClick={onClick}
        aria-label="Track map with live car positions"
      />
      <div className="track-source" title={track.source}>
        Track: FastF1 outline (derived), corners numbered
      </div>
    </div>
  );
}
