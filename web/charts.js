// Small canvas chart library: line / point / band series and box plots.
// No dependencies, so the page works offline once loaded and is easy to test.

const css = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

function niceStep(span, target) {
  const raw = span / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

function linearTicks(min, max, target = 5) {
  if (!(max > min)) return [min];
  const step = niceStep(max - min, target);
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

function logTicks(min, max) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) {
    const v = Math.pow(10, e);
    if (v >= min * 0.999 && v <= max * 1.001) ticks.push(v);
  }
  return ticks.length >= 2 ? ticks : [min, max];
}

export function formatNumber(v) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
  return Number(v.toPrecision(3)).toString();
}

const finite = (v) => v !== null && v !== undefined && isFinite(v);

class Base {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.opts = opts;
    this.pad = { left: 58, right: 12, top: opts.title ? 28 : 12, bottom: opts.xlabel ? 40 : 26 };
    this._observer = new ResizeObserver(() => this.draw());
    this._observer.observe(canvas);
  }

  _prepare() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return null;
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = "12px system-ui, sans-serif";
    return { w, h, x0: this.pad.left, x1: w - this.pad.right, y0: this.pad.top, y1: h - this.pad.bottom };
  }

  _title(box) {
    const ctx = this.ctx;
    ctx.fillStyle = css("--text", "#222");
    if (this.opts.title) {
      ctx.font = "600 13px system-ui, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      ctx.fillText(this.opts.title, box.x0, 6);
      ctx.font = "12px system-ui, sans-serif";
    }
    if (this.opts.ylabel) {
      ctx.save();
      ctx.translate(12, (box.y0 + box.y1) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = css("--muted", "#666");
      ctx.fillText(this.opts.ylabel, 0, -2);
      ctx.restore();
    }
    if (this.opts.xlabel) {
      ctx.fillStyle = css("--muted", "#666");
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(this.opts.xlabel, (box.x0 + box.x1) / 2, box.h - 4);
    }
  }

  _yAxis(box, ticks, toY) {
    const ctx = this.ctx;
    ctx.strokeStyle = css("--grid", "#e5e5e5");
    ctx.fillStyle = css("--muted", "#666");
    ctx.lineWidth = 1;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of ticks) {
      const y = Math.round(toY(v)) + 0.5;
      if (y < box.y0 - 1 || y > box.y1 + 1) continue;
      ctx.beginPath();
      ctx.moveTo(box.x0, y);
      ctx.lineTo(box.x1, y);
      ctx.stroke();
      ctx.fillText(formatNumber(v), box.x0 - 6, y);
    }
  }

  _frame(box) {
    const ctx = this.ctx;
    ctx.strokeStyle = css("--axis", "#999");
    ctx.lineWidth = 1;
    ctx.strokeRect(box.x0 + 0.5, box.y0 + 0.5, box.x1 - box.x0, box.y1 - box.y0);
  }

  destroy() {
    this._observer.disconnect();
  }
}

export class LineChart extends Base {
  // series: {type: "line" | "points" | "band" | "hline", x, y | lo, hi | value, color, width, dash, alpha, size, label}
  constructor(canvas, opts = {}) {
    super(canvas, opts);
    this.series = [];
  }

  setSeries(series) {
    this.series = series;
    this.draw();
  }

  _ranges() {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    const log = !!this.opts.logY;
    const take = (v) => {
      if (!finite(v) || (log && v <= 0)) return;
      if (v < ymin) ymin = v;
      if (v > ymax) ymax = v;
    };
    for (const s of this.series) {
      if (s.type === "hline") { take(s.value); continue; }
      for (let i = 0; i < s.x.length; i++) {
        const x = s.x[i];
        if (!finite(x)) continue;
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (s.type === "band") { take(s.lo[i]); take(s.hi[i]); } else take(s.y[i]);
      }
    }
    if (this.opts.xRange) [xmin, xmax] = this.opts.xRange;
    if (!isFinite(xmin)) { xmin = 0; xmax = 1; }
    if (xmax <= xmin) xmax = xmin + 1;
    if (!isFinite(ymin)) { ymin = log ? 0.1 : 0; ymax = log ? 10 : 1; }
    if (log) {
      if (ymax <= ymin) { ymin /= 2; ymax *= 2; }
      const f = Math.pow(ymax / ymin, 0.05);
      ymin /= f; ymax *= f;
    } else {
      if (ymax <= ymin) { ymin -= 1; ymax += 1; }
      const m = (ymax - ymin) * 0.06;
      ymin -= m; ymax += m;
    }
    return { xmin, xmax, ymin, ymax };
  }

  draw() {
    const box = this._prepare();
    if (!box) return;
    const ctx = this.ctx;
    const r = this._ranges();
    const log = !!this.opts.logY;
    const toX = (x) => box.x0 + ((x - r.xmin) / (r.xmax - r.xmin)) * (box.x1 - box.x0);
    const toY = log
      ? (y) => box.y1 - ((Math.log10(y) - Math.log10(r.ymin)) / (Math.log10(r.ymax) - Math.log10(r.ymin))) * (box.y1 - box.y0)
      : (y) => box.y1 - ((y - r.ymin) / (r.ymax - r.ymin)) * (box.y1 - box.y0);
    const ok = (v) => finite(v) && (!log || v > 0);

    this._title(box);
    this._yAxis(box, log ? logTicks(r.ymin, r.ymax) : linearTicks(r.ymin, r.ymax, Math.max(2, (box.y1 - box.y0) / 40)), toY);
    // x ticks
    ctx.fillStyle = css("--muted", "#666");
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of linearTicks(r.xmin, r.xmax, Math.max(2, (box.x1 - box.x0) / 90))) {
      ctx.fillText(formatNumber(v), toX(v), box.y1 + 5);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(box.x0, box.y0, box.x1 - box.x0, box.y1 - box.y0);
    ctx.clip();
    for (const s of this.series) {
      ctx.globalAlpha = 1;
      if (s.type === "band") {
        ctx.fillStyle = s.color;
        ctx.globalAlpha = s.alpha ?? 0.18;
        let i = 0;
        while (i < s.x.length) {  // one polygon per run of valid points
          while (i < s.x.length && !(ok(s.lo[i]) && ok(s.hi[i]))) i++;
          const start = i;
          while (i < s.x.length && ok(s.lo[i]) && ok(s.hi[i])) i++;
          if (i - start < 2) continue;
          ctx.beginPath();
          for (let k = start; k < i; k++) ctx.lineTo(toX(s.x[k]), toY(s.hi[k]));
          for (let k = i - 1; k >= start; k--) ctx.lineTo(toX(s.x[k]), toY(s.lo[k]));
          ctx.closePath();
          ctx.fill();
        }
      } else if (s.type === "hline") {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width ?? 1;
        ctx.setLineDash(s.dash ?? [4, 4]);
        const y = toY(s.value);
        ctx.beginPath();
        ctx.moveTo(box.x0, y);
        ctx.lineTo(box.x1, y);
        ctx.stroke();
        ctx.setLineDash([]);
      } else if (s.type === "points") {
        ctx.fillStyle = s.color;
        const size = s.size ?? 2.5;
        for (let i = 0; i < s.x.length; i++) {
          if (!ok(s.y[i])) continue;
          ctx.beginPath();
          ctx.arc(toX(s.x[i]), toY(s.y[i]), size, 0, 2 * Math.PI);
          ctx.fill();
        }
      } else {
        ctx.strokeStyle = s.color;
        ctx.lineWidth = s.width ?? 1.5;
        ctx.setLineDash(s.dash ?? []);
        ctx.beginPath();
        let pen = false;
        for (let i = 0; i < s.x.length; i++) {
          if (!ok(s.y[i])) { pen = false; continue; }
          const X = toX(s.x[i]), Y = toY(s.y[i]);
          if (pen) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
          pen = true;
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
    ctx.restore();
    ctx.globalAlpha = 1;
    this._frame(box);
    this._legend(box);
  }

  _legend(box) {
    const items = this.series.filter((s) => s.label);
    if (!items.length) return;
    const ctx = this.ctx;
    ctx.font = "11px system-ui, sans-serif";
    const w = Math.max(...items.map((s) => ctx.measureText(s.label).width)) + 30;
    const h = items.length * 16 + 6;
    const x = box.x1 - w - 6, y = box.y0 + 6;
    ctx.fillStyle = css("--legend-bg", "rgba(255,255,255,0.85)");
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = css("--grid", "#ddd");
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    items.forEach((s, k) => {
      const cy = y + 11 + k * 16;
      ctx.globalAlpha = s.type === "band" ? 0.35 : 1;
      ctx.fillStyle = s.color;
      ctx.strokeStyle = s.color;
      if (s.type === "band") ctx.fillRect(x + 6, cy - 5, 16, 10);
      else if (s.type === "points") { ctx.beginPath(); ctx.arc(x + 14, cy, 3, 0, 2 * Math.PI); ctx.fill(); }
      else { ctx.lineWidth = 2; ctx.setLineDash(s.type === "hline" ? s.dash ?? [4, 4] : s.dash ?? []); ctx.beginPath(); ctx.moveTo(x + 6, cy); ctx.lineTo(x + 22, cy); ctx.stroke(); ctx.setLineDash([]); }
      ctx.globalAlpha = 1;
      ctx.fillStyle = css("--text", "#222");
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(s.label, x + 27, cy);
    });
    ctx.font = "12px system-ui, sans-serif";
  }
}

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export class BoxChart extends Base {
  // groups: [{label, color, values: number[]}]
  constructor(canvas, opts = {}) {
    super(canvas, opts);
    this.pad.bottom = 30;
    this.groups = [];
  }

  setGroups(groups) {
    this.groups = groups;
    this.draw();
  }

  draw() {
    const box = this._prepare();
    if (!box) return;
    const ctx = this.ctx;
    const stats = this.groups.map((g) => {
      const v = g.values.filter(finite).sort((a, b) => a - b);
      if (!v.length) return null;
      const q1 = quantile(v, 0.25), q3 = quantile(v, 0.75), iqr = q3 - q1;
      const inside = v.filter((x) => x >= q1 - 1.5 * iqr && x <= q3 + 1.5 * iqr);
      return {
        q1, q3, med: quantile(v, 0.5),
        lo: inside[0], hi: inside[inside.length - 1],
        out: v.filter((x) => x < q1 - 1.5 * iqr || x > q3 + 1.5 * iqr),
        min: v[0], max: v[v.length - 1],
      };
    });
    let ymin = Math.min(...stats.filter(Boolean).map((s) => s.min));
    let ymax = Math.max(...stats.filter(Boolean).map((s) => s.max));
    if (!isFinite(ymin)) { ymin = 0; ymax = 1; }
    if (ymax <= ymin) { ymin -= 0.5; ymax += 0.5; }
    const m = (ymax - ymin) * 0.08;
    ymin -= m; ymax += m;
    const toY = (y) => box.y1 - ((y - ymin) / (ymax - ymin)) * (box.y1 - box.y0);
    this._title(box);
    this._yAxis(box, linearTicks(ymin, ymax, Math.max(2, (box.y1 - box.y0) / 40)), toY);
    const slot = (box.x1 - box.x0) / Math.max(1, this.groups.length);
    this.groups.forEach((g, i) => {
      const cx = box.x0 + slot * (i + 0.5);
      ctx.fillStyle = css("--text", "#222");
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(g.label, cx, box.y1 + 6, slot - 4);
      const s = stats[i];
      if (!s) return;
      const half = Math.min(28, slot * 0.3);
      ctx.strokeStyle = css("--text", "#222");
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, toY(s.lo)); ctx.lineTo(cx, toY(s.q1));
      ctx.moveTo(cx, toY(s.q3)); ctx.lineTo(cx, toY(s.hi));
      ctx.moveTo(cx - half / 2, toY(s.lo)); ctx.lineTo(cx + half / 2, toY(s.lo));
      ctx.moveTo(cx - half / 2, toY(s.hi)); ctx.lineTo(cx + half / 2, toY(s.hi));
      ctx.stroke();
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = g.color;
      ctx.fillRect(cx - half, toY(s.q3), 2 * half, toY(s.q1) - toY(s.q3));
      ctx.globalAlpha = 1;
      ctx.strokeRect(cx - half, toY(s.q3), 2 * half, toY(s.q1) - toY(s.q3));
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx - half, toY(s.med)); ctx.lineTo(cx + half, toY(s.med));
      ctx.stroke();
      ctx.lineWidth = 1;
      for (const o of s.out) {
        ctx.beginPath();
        ctx.arc(cx, toY(o), 2.5, 0, 2 * Math.PI);
        ctx.stroke();
      }
    });
    this._frame(box);
  }
}
