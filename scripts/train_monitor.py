#!/usr/bin/env python3
"""实时监视 LeRobot 训练：训练/验证 loss 曲线 + best checkpoint 追踪。

LeRobot 只带 WandB 一种 logger（没有 tensorboard）。没有 wandb 账号时
``train.bash`` 会自动走 offline 模式，指标写进 ``<run-dir>/wandb/**/run-*.wandb``。
本脚本直接尾随那个文件（用的是 training_report.read_history，与事后报告同一套解析），
在本地起一个只读页面，所以不需要账号，指标也不出机器。

页面每 ``--interval`` 秒刷新一次：训练 L1 loss、验证 loss 双曲线，当前 best 验证点，
以及把所有 step checkpoint 列出来——点一下就把 best 那个链接成 ``checkpoints/best``
（LeRobot 自己只会维护 ``last`` 软链）。

用法：
    conda activate robotarm
    python scripts/train_monitor.py --run-dir artifacts/checkpoints/act_cup_global_20261009_v1
    # -> http://127.0.0.1:8420

    # 训练还没开始也能起，页面会显示"等待 run 出现"
    python scripts/train_monitor.py --run-dir <dir> --port 8420 --interval 2
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from robotarm.modeling.training_report import read_history

TRAIN_KEY = "train/l1_loss"
TOTAL_KEY = "train/loss"
EVAL_KEY = "eval/eval_loss"


def find_run_file(run_dir: Path) -> Path | None:
    """Newest WandB offline run file under ``run_dir`` (same rule as the report)."""
    files = sorted(
        run_dir.glob("wandb/**/run-*.wandb"), key=lambda p: p.stat().st_mtime
    )
    return files[-1] if files else None


def read_run_config(run_file: Path) -> dict:
    """The training config LeRobot logged into the run's ``run`` record.

    Carries ``steps``/``save_freq``/``eval_steps`` from the moment training
    starts, so the chart can show progress against the planned length before
    the first checkpoint lands.
    """
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(run_file))
    while (data := store.scan_data()) is not None:
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof("record_type") != "run":
            continue
        config = {}
        for item in record.run.config.update:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            try:
                config[key] = json.loads(item.value_json)
            except (TypeError, json.JSONDecodeError):
                continue
        return config
    return {}


def collect(run_dir: Path) -> dict:
    """Snapshot of everything the page needs.  Cheap enough to call per poll."""
    run_file = find_run_file(run_dir)
    train: list[list[float]] = []
    total: list[list[float]] = []
    evaluation: list[list[float]] = []
    if run_file is not None:
        try:
            rows = read_history(run_file)
        except Exception:
            # Training appends while we read, so a record can be caught
            # mid-write.  Serving the previous snapshot is correct: the next
            # poll picks it up.
            rows = None
        if rows is not None:
            for row in rows:
                step = row.get("_step")
                if step is None:
                    continue
                if TRAIN_KEY in row:
                    train.append([step, row[TRAIN_KEY]])
                if TOTAL_KEY in row:
                    total.append([step, row[TOTAL_KEY]])
                if EVAL_KEY in row:
                    evaluation.append([step, row[EVAL_KEY]])

    run_config: dict = {}
    if run_file is not None:
        try:
            run_config = read_run_config(run_file)
        except Exception:
            run_config = {}

    checkpoints = sorted(
        int(path.parent.name)
        for path in (run_dir / "checkpoints").glob("*/pretrained_model")
        if path.parent.name.isdigit()
    )
    best = None
    if evaluation:
        best_step, best_loss = min(evaluation, key=lambda point: point[1])
        nearest = (
            min(checkpoints, key=lambda step: abs(step - best_step))
            if checkpoints
            else None
        )
        best = {
            "step": best_step,
            "loss": best_loss,
            "nearest_checkpoint": nearest,
        }
    linked = None
    best_link = run_dir / "checkpoints" / "best"
    if best_link.is_symlink():
        linked = best_link.resolve().name

    return {
        "run_dir": str(run_dir),
        "run_file": str(run_file) if run_file else None,
        "train": train,
        "total": total,
        "eval": evaluation,
        "checkpoints": checkpoints,
        "best": best,
        "best_link": linked,
        "steps": run_config.get("steps"),
        "save_freq": run_config.get("save_freq"),
        "eval_steps": run_config.get("eval_steps"),
        "now": time.time(),
    }


def link_best(run_dir: Path, step: int) -> str:
    """Point ``checkpoints/best`` at a step dir, mirroring LeRobot's ``last``."""
    target = run_dir / "checkpoints" / f"{step:06d}"
    if not (target / "pretrained_model").is_dir():
        raise FileNotFoundError(f"no checkpoint at {target}")
    link = run_dir / "checkpoints" / "best"
    if link.is_symlink() or link.exists():
        link.unlink()
    # Relative, like LeRobot's own `last`, so the run dir stays movable.
    link.symlink_to(target.name)
    return target.name


class State:
    """Cached snapshot, refreshed at most once per ``interval`` seconds."""

    def __init__(self, run_dir: Path, interval: float) -> None:
        self.run_dir = run_dir
        self.interval = interval
        self._lock = threading.Lock()
        self._snapshot: dict | None = None
        self._at = 0.0

    def get(self) -> dict:
        with self._lock:
            if self._snapshot is None or time.monotonic() - self._at >= self.interval:
                self._snapshot = collect(self.run_dir)
                self._at = time.monotonic()
            return self._snapshot

    def refresh(self) -> dict:
        with self._lock:
            self._snapshot = collect(self.run_dir)
            self._at = time.monotonic()
            return self._snapshot


def build_handler(state: State):
    page = PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, payload: dict) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path.startswith("/api/metrics"):
                self._json(200, state.get())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if self.path != "/api/link-best":
                self._json(404, {"error": "not found"})
                return
            snapshot = state.refresh()
            best = snapshot.get("best")
            if not best or best.get("nearest_checkpoint") is None:
                self._json(409, {"error": "no eval point or no checkpoint yet"})
                return
            try:
                linked = link_best(state.run_dir, best["nearest_checkpoint"])
            except (OSError, FileNotFoundError) as exc:
                self._json(500, {"error": str(exc)})
                return
            state.refresh()
            self._json(200, {"linked": linked})

        def log_message(self, *args) -> None:
            pass  # keep the terminal readable during a long run

    return Handler


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>训练监视</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface-1: #fcfcfb;
    --text-primary: #0b0b0b; --text-secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6; --series-2: #eb6834;
    --ok: #006300; --bad: #d03b3b; --tip-bg: #fff;
  }
  @media (prefers-color-scheme: dark) {
    :root:where(:not([data-theme="light"])) {
      color-scheme: dark;
      --page: #0d0d0d; --surface-1: #1a1a19;
      --text-primary: #fff; --text-secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
      --series-1: #3987e5; --series-2: #d95926;
      --ok: #0ca30c; --bad: #d03b3b; --tip-bg: #242422;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 22px 16px 48px; background: var(--page);
    color: var(--text-primary);
    font: 15px/1.55 system-ui, -apple-system, "Segoe UI", "Noto Sans SC", sans-serif;
  }
  .wrap { max-width: 900px; margin: 0 auto; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  h1 { font-size: 19px; margin: 0; letter-spacing: -0.01em; }
  .status { font-size: 12.5px; color: var(--text-secondary); }
  .status .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                 background: var(--muted); margin-right: 6px; vertical-align: middle; }
  .status.live .dot { background: var(--ok); }
  .path { font-size: 12px; color: var(--muted); margin: 6px 0 16px; word-break: break-all; }

  .stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-bottom: 16px; }
  .tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 11px 13px; }
  .tile .k { font-size: 11.5px; color: var(--muted); margin-bottom: 5px; }
  .tile .v { font-size: 20px; font-weight: 650; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
  .tile .v.s1 { color: var(--series-1); }
  .tile .v.s2 { color: var(--series-2); }
  .tile .s { font-size: 11.5px; color: var(--text-secondary); margin-top: 3px; font-variant-numeric: tabular-nums; }

  figure { margin: 0 0 16px; background: var(--surface-1); border: 1px solid var(--border);
           border-radius: 10px; padding: 12px 10px 4px; }
  .legend { display: flex; gap: 16px; font-size: 12.5px; color: var(--text-secondary); padding: 0 6px 6px; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .legend i { width: 16px; height: 2px; border-radius: 1px; display: inline-block; }
  svg { width: 100%; height: auto; display: block; }
  .grid { stroke: var(--grid); stroke-width: 1; }
  .axis { stroke: var(--baseline); stroke-width: 1; }
  .tick { fill: var(--muted); font-size: 10.5px; }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
  .l1 { stroke: var(--series-1); } .l2 { stroke: var(--series-2); }
  .dot { stroke: var(--surface-1); stroke-width: 2; }
  .d1 { fill: var(--series-1); } .d2 { fill: var(--series-2); }
  .endlabel { fill: var(--text-secondary); font-size: 11px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .anno { fill: var(--muted); font-size: 10.5px; }
  .cross { stroke: var(--muted); stroke-width: 1; }

  .tip { position: fixed; pointer-events: none; opacity: 0; background: var(--tip-bg);
         border: 1px solid var(--border); border-radius: 8px; padding: 7px 10px;
         font-size: 12.5px; box-shadow: 0 6px 20px rgba(0,0,0,0.13); z-index: 10; }
  .tip.on { opacity: 1; }
  .tip .h { font-size: 11px; color: var(--muted); margin-bottom: 4px; font-variant-numeric: tabular-nums; }
  .tip .r { display: grid; grid-template-columns: 14px 1fr auto; gap: 6px; align-items: baseline; }
  .tip .r i { width: 14px; height: 2px; border-radius: 1px; align-self: center; }
  .tip .n { color: var(--text-secondary); }
  .tip .v { font-weight: 650; color: var(--text-primary); font-variant-numeric: tabular-nums; }

  .cps { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
  .cps h2 { font-size: 13px; margin: 0 0 8px; font-weight: 600; }
  .cpwrap { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 10px; }
  .cp { font-size: 12px; font-variant-numeric: tabular-nums; padding: 3px 9px;
        border-radius: 999px; border: 1px solid var(--border); color: var(--text-secondary); }
  .cp.isbest { border-color: var(--series-2); color: var(--text-primary); font-weight: 650; }
  .cp.linked { background: color-mix(in srgb, var(--series-2) 12%, transparent); }
  button { font: inherit; font-size: 12px; padding: 5px 11px; border-radius: 7px;
           border: 1px solid var(--border); background: var(--page); color: var(--text-secondary); cursor: pointer; }
  button:hover { color: var(--text-primary); }
  button:disabled { opacity: 0.45; cursor: default; }
  .empty { color: var(--muted); font-size: 12.5px; padding: 6px 0; }
  @media (max-width: 700px) { .stats { grid-template-columns: repeat(2, minmax(0,1fr)); } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>训练监视</h1>
    <div class="status" id="status"><span class="dot"></span><span id="statustext">连接中…</span></div>
  </header>
  <div class="path" id="path"></div>

  <section class="stats">
    <div class="tile"><div class="k">训练 L1（最新）</div><div class="v s1" id="m-train">—</div><div class="s" id="s-train"></div></div>
    <div class="tile"><div class="k">验证 loss（最新）</div><div class="v s2" id="m-eval">—</div><div class="s" id="s-eval"></div></div>
    <div class="tile"><div class="k">验证最低</div><div class="v s2" id="m-best">—</div><div class="s" id="s-best"></div></div>
    <div class="tile"><div class="k">训练 / 验证 差距</div><div class="v" id="m-gap">—</div><div class="s">等比例差距</div></div>
  </section>

  <figure>
    <div class="legend">
      <span><i style="background:var(--series-1)"></i>训练 L1 loss</span>
      <span><i style="background:var(--series-2)"></i>验证 loss</span>
    </div>
    <svg id="chart" viewBox="0 0 860 380" role="img" aria-label="训练与验证 loss 曲线"></svg>
  </figure>

  <div class="cps">
    <h2>Checkpoints</h2>
    <div id="cplist"><div class="empty">等待 checkpoint…</div></div>
  </div>
</div>
<div class="tip" id="tip"></div>

<script>
const W = 860, H = 380, M = { t: 16, r: 92, b: 40, l: 58 };
const PW = W - M.l - M.r, PH = H - M.t - M.b;
const NS = "http://www.w3.org/2000/svg";
const $ = id => document.getElementById(id);
const el = (n, a) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };
const fmt = v => v == null ? "—" : (v >= 0.1 ? v.toFixed(4) : v.toFixed(5));
const fmtStep = s => s.toLocaleString("en-US");
let SNAP = null, GEOM = null;

function niceTicks(lo, hi) {
  const out = [];
  const start = Math.floor(Math.log10(lo)), end = Math.ceil(Math.log10(hi));
  for (let e = start; e <= end; e++)
    for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
  if (out.length < 3) out.push(lo, hi);
  return out;
}

function draw(snap) {
  const svg = $("chart");
  svg.replaceChildren();
  const train = snap.train || [], ev = snap.eval || [];
  const all = train.concat(ev);
  if (!all.length) {
    const t = el("text", { x: W / 2, y: H / 2, class: "tick", "text-anchor": "middle" });
    t.textContent = "等待第一批指标（run 文件出现后约 100 步）…";
    svg.appendChild(t); GEOM = null; return;
  }
  const values = all.map(p => p[1]).filter(v => v > 0);
  let lo = Math.min(...values), hi = Math.max(...values);
  lo = Math.pow(10, Math.floor(Math.log10(lo)) * 1 - 0.05);
  hi = Math.pow(10, Math.ceil(Math.log10(hi)) + 0.05);
  // Span the chart to the planned run length so early progress is visible
  // against the target, not stretched across a two-point axis.
  const xmax = Math.max(snap.steps || 0, all[all.length - 1][0]) || 1;
  const yPix = v => M.t + PH - (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) * PH;
  const xPix = s => M.l + (s / xmax) * PW;
  GEOM = { yPix, xPix, xmax };

  for (const v of niceTicks(lo, hi)) {
    const y = yPix(v);
    svg.appendChild(el("line", { x1: M.l, x2: M.l + PW, y1: y, y2: y, class: "grid" }));
    const t = el("text", { x: M.l - 9, y: y + 4, class: "tick", "text-anchor": "end" });
    t.textContent = v >= 0.01 ? String(+v.toPrecision(2)) : v.toExponential(0);
    svg.appendChild(t);
  }
  svg.appendChild(el("line", { x1: M.l, x2: M.l + PW, y1: M.t + PH, y2: M.t + PH, class: "axis" }));
  const steps = 5;
  for (let i = 0; i <= steps; i++) {
    const s = xmax * i / steps, x = xPix(s);
    svg.appendChild(el("line", { x1: x, x2: x, y1: M.t + PH, y2: M.t + PH + 5, class: "axis" }));
    const t = el("text", { x, y: M.t + PH + 19, class: "tick", "text-anchor": "middle" });
    t.textContent = s >= 1000 ? (s / 1000).toFixed(1).replace(/\.0$/, "") + "k" : String(Math.round(s));
    svg.appendChild(t);
  }

  const series = [[train, "l1", "d1"], [ev, "l2", "d2"]];
  for (const [pts, cls, dcls] of series) {
    if (!pts.length) continue;
    svg.appendChild(el("path", { class: "line " + cls, fill: "none",
      d: pts.map((p, i) => `${i ? "L" : "M"}${xPix(p[0]).toFixed(2)},${yPix(p[1]).toFixed(2)}`).join("") }));
    const last = pts[pts.length - 1];
    svg.appendChild(el("circle", { cx: xPix(last[0]), cy: yPix(last[1]), r: 4, class: "dot " + dcls }));
    const lb = el("text", { x: xPix(last[0]) + 9, y: yPix(last[1]) + 4, class: "endlabel" });
    lb.textContent = (cls === "l1" ? "训练 " : "验证 ") + fmt(last[1]);
    svg.appendChild(lb);
  }

  const best = snap.best;
  if (best && ev.length) {
    const cx = xPix(best.step), cy = yPix(best.loss);
    svg.appendChild(el("circle", { cx, cy, r: 5, class: "dot d2" }));
    const t = el("text", { x: cx, y: cy - 12, class: "anno", "text-anchor": "middle" });
    t.textContent = `best ${fmt(best.loss)} @ ${fmtStep(best.step)}`;
    svg.appendChild(t);
  }

  const cross = el("line", { y1: M.t, y2: M.t + PH, class: "cross", opacity: 0 });
  svg.appendChild(cross);
  const hd = ["d1", "d2"].map(c => { const d = el("circle", { r: 4, class: "dot " + c, opacity: 0 }); svg.appendChild(d); return d; });
  const hit = el("rect", { x: M.l, y: M.t, width: PW, height: PH, fill: "transparent" });
  svg.appendChild(hit);

  const near = (pts, s) => pts.reduce((a, b) => Math.abs(b[0] - s) < Math.abs(a[0] - s) ? b : a);
  hit.addEventListener("pointermove", evt => {
    const box = svg.getBoundingClientRect();
    const vx = (evt.clientX - box.left) / (box.width / W);
    const step = Math.max(0, Math.min(xmax, (vx - M.l) / PW * xmax));
    cross.setAttribute("x1", xPix(step)); cross.setAttribute("x2", xPix(step)); cross.setAttribute("opacity", 1);
    const rows = [];
    if (train.length) { const p = near(train, step); hd[0].setAttribute("cx", xPix(p[0])); hd[0].setAttribute("cy", yPix(p[1])); hd[0].setAttribute("opacity", 1); rows.push(["d1", "训练 L1", fmt(p[1]), p[0]]); }
    if (ev.length) { const p = near(ev, step); hd[1].setAttribute("cx", xPix(p[0])); hd[1].setAttribute("cy", yPix(p[1])); hd[1].setAttribute("opacity", 1); rows.push(["d2", "验证", fmt(p[1]), p[0]]); }
    const tip = $("tip");
    tip.replaceChildren();
    const h = document.createElement("div"); h.className = "h"; h.textContent = "step " + fmtStep(Math.round(step)); tip.appendChild(h);
    for (const [cls, name, val, at] of rows) {
      const r = document.createElement("div"); r.className = "r";
      const i = document.createElement("i"); i.style.background = cls === "d1" ? "var(--series-1)" : "var(--series-2)";
      const n = document.createElement("span"); n.className = "n"; n.textContent = name + " (step " + fmtStep(at) + ")";
      const v = document.createElement("span"); v.className = "v"; v.textContent = val;
      r.append(i, n, v); tip.appendChild(r);
    }
    tip.classList.add("on");
    let left = box.left + xPix(step) * (box.width / W) + 12;
    if (left + tip.offsetWidth > window.innerWidth - 8) left = box.left + xPix(step) * (box.width / W) - tip.offsetWidth - 12;
    tip.style.left = Math.max(8, left) + "px";
    tip.style.top = (box.top + 40) + "px";
  });
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("opacity", 0); hd.forEach(d => d.setAttribute("opacity", 0));
    $("tip").classList.remove("on");
  });
}

function render(snap) {
  SNAP = snap;
  $("path").textContent = snap.run_dir + (snap.run_file ? "  ·  " + snap.run_file.split("/").slice(-2).join("/") : "  ·  尚无 wandb run 文件");
  const live = snap.train.length > 0;
  $("status").className = "status" + (live ? " live" : "");
  const plan = [];
  if (snap.steps) plan.push(`共 ${fmtStep(snap.steps)} 步`);
  if (snap.save_freq) plan.push(`save 每 ${fmtStep(snap.save_freq)}`);
  if (snap.eval_steps) plan.push(`eval 每 ${fmtStep(snap.eval_steps)}`);
  $("statustext").textContent = live
    ? `${snap.train.length} 训练点 / ${snap.eval.length} 验证点` + (plan.length ? "  ·  " + plan.join(" · ") : "")
    : "等待 run 出现…";

  const tr = snap.train.length ? snap.train[snap.train.length - 1] : null;
  const ev = snap.eval.length ? snap.eval[snap.eval.length - 1] : null;
  $("m-train").textContent = tr ? fmt(tr[1]) : "—";
  $("s-train").textContent = tr ? "step " + fmtStep(tr[0]) : "";
  $("m-eval").textContent = ev ? fmt(ev[1]) : "—";
  $("s-eval").textContent = ev ? "step " + fmtStep(ev[0]) : "";
  const b = snap.best;
  $("m-best").textContent = b ? fmt(b.loss) : "—";
  $("s-best").textContent = b ? "step " + fmtStep(b.step) : "";
  if (tr && ev) {
    const ratio = ev[1] / tr[1];
    $("m-gap").textContent = ratio.toFixed(2) + "×";
    $("m-gap").style.color = ratio > 2 ? "var(--bad)" : "var(--ok)";
  } else { $("m-gap").textContent = "—"; $("m-gap").style.color = ""; }

  const list = $("cplist");
  list.replaceChildren();
  if (!snap.checkpoints.length) {
    const d = document.createElement("div"); d.className = "empty"; d.textContent = "等待 checkpoint（save_freq 步后出现）…";
    list.appendChild(d);
  } else {
    const wrap = document.createElement("div"); wrap.className = "cpwrap";
    for (const step of snap.checkpoints) {
      const chip = document.createElement("span");
      const isBest = b && b.nearest_checkpoint === step;
      const isLinked = snap.best_link === String(step).padStart(6, "0");
      chip.className = "cp" + (isBest ? " isbest" : "") + (isLinked ? " linked" : "");
      chip.textContent = fmtStep(step) + (isBest ? "  ← best" : "");
      chip.title = isLinked ? "checkpoints/best 当前指向这里" : "";
      wrap.appendChild(chip);
    }
    list.appendChild(wrap);
    const btn = document.createElement("button");
    btn.textContent = "把 best 链接为 checkpoints/best";
    btn.disabled = !b || b.nearest_checkpoint == null;
    btn.onclick = async () => {
      btn.disabled = true;
      try {
        const r = await fetch("/api/link-best", { method: "POST" });
        const j = await r.json();
        if (!r.ok) throw new Error(j.error || "failed");
        btn.textContent = "已链接 → " + j.linked;
      } catch (e) { btn.textContent = "失败：" + e.message; }
      setTimeout(tick, 300);
    };
    list.appendChild(btn);
  }
  draw(snap);
}

async function tick() {
  try {
    const r = await fetch("/api/metrics");
    render(await r.json());
  } catch (e) {
    $("status").className = "status";
    $("statustext").textContent = "服务已断开";
  }
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


def print_status(run_dir: Path) -> int:
    """One-shot summary of a run, for when a browser is inconvenient."""
    run_dir = run_dir.expanduser().resolve()
    snap = collect(run_dir)
    print(f"run 目录   : {snap['run_dir']}")
    if not run_dir.is_dir():
        print("             (还不存在——训练尚未启动，或已被清理)")
        return 0
    if snap["run_file"] is None:
        print("wandb      : (还没有 run 文件)")
    else:
        print(f"wandb      : {snap['run_file']}")
    plan = [
        f"共 {snap['steps']:,} 步" if snap["steps"] else None,
        f"save 每 {snap['save_freq']:,}" if snap["save_freq"] else None,
        f"eval 每 {snap['eval_steps']:,}" if snap["eval_steps"] else None,
    ]
    print("配置       : " + (" · ".join(p for p in plan if p) or "(未读到)"))
    if snap["train"]:
        step, value = snap["train"][-1]
        print(f"训练 L1    : {value:.5f}  @ step {step:,.0f}")
    else:
        print("训练 L1    : (还没有数据)")
    if snap["eval"]:
        step, value = snap["eval"][-1]
        print(f"最新验证   : {value:.5f}  @ step {step:,.0f}")
    best = snap["best"]
    if best:
        print(
            f"验证最低   : {best['loss']:.5f}  @ step {best['step']:,.0f}"
            f"  -> 最近 checkpoint {best['nearest_checkpoint']:,}"
        )
    if snap["train"] and snap["eval"]:
        print(f"训练/验证  : {snap['eval'][-1][1] / snap['train'][-1][1]:.2f}×")
    print(
        "checkpoints: "
        + (", ".join(f"{s:,}" for s in snap["checkpoints"]) or "(还没有)")
    )
    if snap["best_link"]:
        print(f"best 链接  : -> {snap['best_link']}")
    newest = run_dir / "checkpoints" / "last" / "pretrained_model"
    if newest.is_dir():
        print(f"最新模型   : {newest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="重新读取 run 文件的最小间隔（秒）",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="只打印一次当前 run 的摘要然后退出，不起服务",
    )
    args = parser.parse_args(argv)

    if args.status:
        return print_status(args.run_dir)

    run_dir = args.run_dir.expanduser().resolve()
    # Deliberately never create --run-dir.  LeRobot refuses to train into an
    # existing output directory (TrainPipelineConfig.validate raises
    # FileExistsError when resume is False), so a monitor that mkdir'd it would
    # block the very run it exists to watch.  Every lookup in collect() already
    # tolerates a missing directory, so starting before training is fine.
    state = State(run_dir, max(0.5, args.interval))
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(f"监视 {run_dir}", flush=True)
    if not run_dir.is_dir():
        print(
            "  目录尚不存在——等训练自己创建，页面会显示“等待 run 出现”",
            flush=True,
        )
    print(f"打开 http://{args.host}:{args.port}  (Ctrl+C 退出)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
