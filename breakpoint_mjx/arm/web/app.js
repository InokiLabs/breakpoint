// BreakPoint AR4 Stress Lab: the browser side. Talks to server.py's JSON API, or reads
// window.BP_STATIC when opened as an exported report.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const STATIC = window.BP_STATIC || null;
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const JC = ["#4c8dff", "#2fbf9b", "#f5a524", "#b48cff", "#ff5f56", "#5fd0e6"];
const DIM = {
  payload_kg: ["Payload", "kg"], cog_x_mm: ["CoG offset x", "mm"], cog_y_mm: ["CoG offset y", "mm"],
  cog_z_mm: ["CoG offset z", "mm"], speed_scale: ["Speed", "×"], accel_scale: ["Acceleration", "×"],
  decel_scale: ["Deceleration", "×"], torque_scale: ["Motor torque", "×"],
  supply_scale: ["Supply voltage", "×"], friction_scale: ["Friction", "×"],
  backlash_scale: ["Backlash", "×"], stiffness_scale: ["Gearbox stiffness", "×"],
  shift_x_mm: ["Fixture shift x", "mm"], shift_y_mm: ["Fixture shift y", "mm"],
  shift_z_mm: ["Fixture shift z", "mm"], reach_scale: ["Reach", "×"],
  world_model_scale: ["World-model correction", "×"], ff_scale: ["Payload as configured", "×"],
  gain_scale: ["Servo gain", "×"],
};
const dimLabel = (n) => (DIM[n] ? `${DIM[n][0]} (${DIM[n][1]})` : n);
const fmtDim = (n, v) => {
  const u = DIM[n] ? DIM[n][1] : "";
  const x = Math.abs(v) >= 10 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
  return u === "×" ? `${DIM[n][0]} ×${x}` : `${DIM[n] ? DIM[n][0] : n} ${x} ${u}`;
};
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const S = {
  state: null, programId: null, preview: null, replay: null, frame: 0, playing: false,
  t0: 0, speed: 1, mag: 50, tele: "err", useWm: true, search: null, searchId: null,
  selFamily: null, budget: "quick",
};

// ------------------------------------------------------------------------------ api
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

async function runJob(body, progEl, onDone) {
  const bar = $(".bar", progEl), label = $("span", progEl);
  progEl.hidden = false;
  bar.style.width = "2%";
  label.textContent = "Queued…";
  const { id } = await api("/api/jobs", body);
  for (;;) {
    await new Promise((r) => setTimeout(r, 700));
    const j = await api(`/api/jobs/${id}`);
    const frac = j.total ? Math.min(1, j.done / j.total) : 0;
    bar.style.width = `${Math.max(2, frac * 100)}%`;
    const el = j.started ? ((Date.now() / 1000) - j.started).toFixed(0) : 0;
    label.textContent = j.status === "running" ? `${j.note} · ${j.done}/${j.total} · ${el}s` : j.status;
    if (j.status === "done") {
      progEl.hidden = true;
      const st = await api(`/api/study/${j.result_id}`);
      await onDone(st);
      refreshStudies();
      return;
    }
    if (j.status === "error") {
      label.textContent = `Failed: ${j.error}`;
      bar.style.width = "100%";
      bar.style.background = "color-mix(in srgb, var(--fail) 35%, transparent)";
      throw new Error(j.error);
    }
  }
}

// --------------------------------------------------------------------------- viewer
const V = {};
function initViewer() {
  const el = $("#viewport");
  V.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  V.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio));
  el.appendChild(V.renderer.domElement);
  V.scene = new THREE.Scene();
  V.camera = new THREE.PerspectiveCamera(38, 1, 0.01, 20);
  V.camera.up.set(0, 0, 1);
  V.camera.position.set(0.95, -1.05, 0.72);
  V.controls = new OrbitControls(V.camera, V.renderer.domElement);
  V.controls.target.set(0.02, -0.22, 0.24);
  V.controls.enableDamping = true;
  V.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242a, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(1.2, -1.5, 2.2);
  V.scene.add(key);
  const rim = new THREE.DirectionalLight(0x9cc0ff, 0.6);
  rim.position.set(-1.5, 1.2, 0.8);
  V.scene.add(rim);
  const grid = new THREE.GridHelper(1.6, 32, 0x2c323a, 0x1d2228);
  grid.rotation.x = Math.PI / 2;
  V.scene.add(grid);
  const table = new THREE.Mesh(new THREE.CircleGeometry(0.8, 64),
    new THREE.MeshStandardMaterial({ color: 0x15191e, roughness: 1, transparent: true, opacity: 0.6 }));
  table.position.z = -0.0005;
  V.scene.add(table);
  V.actual = makeRobot(false);
  V.ghost = makeRobot(true);
  V.scene.add(V.actual.root, V.ghost.root);
  V.refPath = new THREE.Line(new THREE.BufferGeometry(),
    new THREE.LineDashedMaterial({ color: 0x7d8fb3, dashSize: 0.008, gapSize: 0.006 }));
  V.actPath = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xff5f56 }));
  V.marks = new THREE.Group();
  V.scene.add(V.refPath, V.actPath, V.marks);
  new ResizeObserver(resize).observe(el);
  resize();
  requestAnimationFrame(tick);
}

function resize() {
  const el = $("#viewport");
  const w = el.clientWidth, h = el.clientHeight;
  V.renderer.setSize(w, h);
  V.camera.aspect = w / Math.max(1, h);
  V.camera.updateProjectionMatrix();
}

function makeRobot(ghost) {
  const root = new THREE.Group();
  const base = new THREE.Group();
  root.add(base);
  const links = [];
  for (let i = 0; i < 6; i++) { const g = new THREE.Group(); links.push(g); root.add(g); }
  const mat = ghost
    ? new THREE.MeshBasicMaterial({ color: 0x7d8fb3, transparent: true, opacity: 0.14, depthWrite: false })
    : null;
  // fallback shapes until meshes arrive: a capsule from each body to the next
  const fallback = [];
  const payload = new THREE.Mesh(new THREE.SphereGeometry(0.022, 20, 14), ghost ? mat
    : new THREE.MeshStandardMaterial({ color: 0xf5a524, roughness: 0.5 }));
  links[5].add(payload);   // MuJoCo fuses the rigid payload into link_6
  return { root, base, links, ghost, mat, fallback, payload };
}

function setFallback(robot, frame) {
  // simple sticks between body origins, used only if meshes failed to load
  if (V.meshesOk) return;
  const P = [[0, 0, 0], ...frame];
  while (robot.fallback.length < 6) {
    const m = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1, 14),
      robot.ghost ? robot.mat : new THREE.MeshStandardMaterial({ color: 0xc8ccd2, metalness: 0.3, roughness: 0.5 }));
    robot.fallback.push(m);
    robot.root.add(m);
  }
  for (let i = 0; i < 6; i++) {
    const a = new THREE.Vector3(...P[i]), b = new THREE.Vector3(...P[i + 1]);
    const m = robot.fallback[i];
    const len = a.distanceTo(b);
    m.visible = len > 1e-4;
    m.position.copy(a).add(b).multiplyScalar(0.5);
    m.scale.set(1, Math.max(len, 1e-4), 1);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  }
}

async function loadMeshes(dir) {
  if (V.meshDir === dir) return;
  V.meshDir = dir;
  V.meshesOk = false;
  for (const r of [V.actual, V.ghost]) {
    for (const g of [r.base, ...r.links]) [...g.children].filter((c) => c !== r.payload).forEach((c) => g.remove(c));
    r.fallback.forEach((m) => (m.visible = true));
  }
  try {
    let manifest, getBuf;
    if (STATIC) {
      manifest = STATIC.meshes.manifest;
      getBuf = async (f) => Uint8Array.from(atob(STATIC.meshes.files[f]), (c) => c.charCodeAt(0)).buffer;
    } else {
      manifest = await (await fetch(`/static/meshes/${dir}/manifest.json`)).json();
      getBuf = async (f) => (await fetch(`/static/meshes/${dir}/${f}`)).arrayBuffer();
    }
    for (const p of manifest.parts) {
      const buf = await getBuf(p.file);
      const dv = new DataView(buf);
      const nv = dv.getUint32(0, true), nf = dv.getUint32(4, true);
      const pos = new Float32Array(buf, 8, nv * 3);
      const idx = new Uint32Array(buf, 8 + nv * 12, nf * 3);
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      geo.setIndex(new THREE.BufferAttribute(idx, 1));
      geo.computeVertexNormals();
      for (const robot of [V.actual, V.ghost]) {
        const m = new THREE.Mesh(geo, robot.ghost ? robot.mat : new THREE.MeshStandardMaterial({
          color: p.color, metalness: ["#dcdcdc", "#c9cdd2"].includes(p.color) ? 0.45 : 0.15, roughness: 0.5, flatShading: false,
        }));
        m.position.set(...p.xyz);
        m.rotation.set(p.rpy[0], p.rpy[1], p.rpy[2], "ZYX");
        (p.link < 0 ? robot.base : robot.links[p.link]).add(m);
      }
    }
    V.meshesOk = true;
    for (const r of [V.actual, V.ghost]) r.fallback.forEach((m) => (m.visible = false));
    if (S.replay) showFrame(S.frame);
  } catch (e) {
    console.warn("meshes unavailable, using simple shapes", e);
  }
}

function poseRobot(robot, xpos, xquat) {
  for (let i = 0; i < Math.min(6, xpos.length); i++) {
    const g = robot.links[i];
    g.position.set(...xpos[i]);
    const q = xquat[i];
    g.quaternion.set(q[1], q[2], q[3], q[0]);
  }
  setFallback(robot, xpos.slice(0, 6));
}

function setPaths() {
  const R = S.replay && S.replay.replay;
  if (!R) return;
  V.refPath.geometry.dispose();
  V.refPath.geometry = new THREE.BufferGeometry().setFromPoints(R.tcp_ref.map((p) => new THREE.Vector3(...p)));
  V.refPath.computeLineDistances();
  // cap the magnification so a large deviation is not blown up off screen (<= 4 cm drawn)
  R._maxdev = Math.max(1e-9, ...R.tcp.map((a, k) => Math.hypot(...a.map((v, j) => v - R.tcp_ref[k][j]))));
  V.marks.clear();
  for (const p of (S.replay.taught_points || []).filter((x) => x.cycle === 1)) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.006, 12, 8),
      new THREE.MeshBasicMaterial({ color: 0x9cc0ff }));
    m.position.set(...p.ref_mm.map((v) => v / 1000));
    V.marks.add(m);
  }
}

function showFrame(i) {
  const R = S.replay && S.replay.replay;
  if (!R) return;
  i = Math.max(0, Math.min(R.t.length - 1, i));
  S.frame = i;
  poseRobot(V.actual, R.xpos[i], R.xquat[i]);
  poseRobot(V.ghost, R.xpos_ref[i], R.xquat_ref[i]);
  const pts = [];
  const mag = Math.max(1, Math.min(S.mag, 0.04 / (R._maxdev || 1e-9)));
  for (let k = 0; k <= i; k++) {
    const a = R.tcp[k], r = R.tcp_ref[k];
    pts.push(new THREE.Vector3(...r.map((v, j) => v + mag * (a[j] - v))));
  }
  V.actPath.geometry.dispose();
  V.actPath.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  $("#scrub").value = Math.round((i / Math.max(1, R.t.length - 1)) * 1000);
  $("#clock").textContent = `${R.t[i].toFixed(2)} s`;
  // flash slip events near this frame
  const t = R.t[i];
  const ev = (R.slip_events || []).filter((e) => e.t <= t && e.t > t - 0.6);
  const toast = $("#toast");
  if (ev.length) {
    const e = ev[ev.length - 1];
    toast.textContent = `J${e.joint} lost ${e.steps} steps at ${e.t.toFixed(2)} s`;
    toast.hidden = false;
  } else toast.hidden = true;
  drawCursor();
}

function tick(now) {
  requestAnimationFrame(tick);
  const R = S.replay && S.replay.replay;
  if (S.playing && R) {
    const t = R.t[0] + ((now - S.t0) / 1000) * S.speed;
    let i = S.frame;
    while (i < R.t.length - 1 && R.t[i + 1] <= t) i++;
    if (i >= R.t.length - 1) { S.playing = false; $("#play").textContent = "▶"; }
    if (i !== S.frame) showFrame(i);
  }
  V.controls.update();
  V.renderer.render(V.scene, V.camera);
}

function play() {
  const R = S.replay && S.replay.replay;
  if (!R) return;
  if (S.playing) { S.playing = false; $("#play").textContent = "▶"; return; }
  if (S.frame >= R.t.length - 1) S.frame = 0;
  S.t0 = performance.now() - ((R.t[S.frame] - R.t[0]) / S.speed) * 1000;
  S.playing = true;
  $("#play").textContent = "❚❚";
}

function drawMarkers() {
  const c = $("#markers"), R = S.replay && S.replay.replay;
  const w = (c.width = c.clientWidth * 2);
  c.height = 20;
  const g = c.getContext("2d");
  g.clearRect(0, 0, w, 20);
  if (!R) return;
  const T = R.t[R.t.length - 1] || 1;
  g.fillStyle = "#9cc0ff";
  for (const m of R.measure || []) g.fillRect((m.t / T) * w - 1, 0, 2, 8);
  g.fillStyle = "#ff5f56";
  for (const e of R.slip_events || []) g.fillRect((e.t / T) * w - 1.5, 10, 3, 10);
  const ab = S.replay.metrics && S.replay.metrics.aborted_at_s;
  if (ab != null) { g.fillStyle = "#f5a524"; g.fillRect((ab / T) * w - 2, 0, 4, 20); }
}

async function loadReplay(theta, title, sub) {
  $("#viewer-busy").hidden = false;
  try {
    let r;
    if (STATIC) {
      const key = S.selFamily || "__nominal__";
      r = STATIC.replays[key] || STATIC.replays.__nominal__;
    } else {
      r = await api("/api/replay", { program: S.programId, theta, use_wm: S.useWm });
    }
    S.replay = r;
    S.frame = 0;
    $("#replay-title").textContent = title;
    $("#replay-sub").textContent = sub || "";
    setPaths();
    showFrame(0);
    drawMarkers();
    renderVerdict();
    renderTele();
  } finally {
    $("#viewer-busy").hidden = true;
  }
}

// ------------------------------------------------------------------------ charts
function lineChart(el, { series, xlab = "", ylab = "", hlines = [], cursor = null, ymin = null, ymax = null, legend = true }) {
  const W = el.clientWidth || 600, H = el.clientHeight || 180;
  const m = { l: 46, r: 10, t: 8, b: 22 };
  const xs = series.flatMap((s) => s.x), ys = series.flatMap((s) => s.y).filter((v) => v != null && isFinite(v));
  if (!xs.length) { el.innerHTML = ""; return; }
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = ymin ?? Math.min(...ys, ...hlines.map((h) => h.y)), y1 = ymax ?? Math.max(...ys, ...hlines.map((h) => h.y));
  if (y1 - y0 < 1e-9) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.06; y0 -= ymin == null ? pad : 0; y1 += ymax == null ? pad : 0;
  if (x1 - x0 < 1e-9) x1 = x0 + 1;
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const ticks = (a, b, n) => { const st = niceStep((b - a) / n); const out = []; for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(+v.toFixed(10)); return out; };
  let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img">`;
  for (const v of ticks(y0, y1, 4)) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(v)}" y2="${Y(v)}" stroke="#20252b"/><text x="${m.l - 6}" y="${Y(v) + 3}" text-anchor="end">${fmtTick(v)}</text>`;
  for (const v of ticks(x0, x1, 6)) s += `<text x="${X(v)}" y="${H - 6}" text-anchor="middle">${fmtTick(v)}</text>`;
  for (const h of hlines) s += `<line x1="${m.l}" x2="${W - m.r}" y1="${Y(h.y)}" y2="${Y(h.y)}" stroke="${h.color || "#ff5f56"}" stroke-dasharray="4 3"/><text x="${W - m.r - 2}" y="${Y(h.y) - 3}" text-anchor="end" fill="${h.color || "#ff5f56"}">${esc(h.label || "")}</text>`;
  for (const se of series) {
    let d = "";
    se.x.forEach((xv, i) => { const yv = se.y[i]; if (yv == null || !isFinite(yv)) return; d += `${d ? "L" : "M"}${X(xv).toFixed(1)},${Y(yv).toFixed(1)}`; });
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="${se.width || 1.4}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} opacity="${se.opacity ?? 1}"/>`;
    if (se.dots) se.x.forEach((xv, i) => { s += `<circle cx="${X(xv)}" cy="${Y(se.y[i])}" r="2.5" fill="${se.color}"/>`; });
  }
  if (cursor != null) s += `<line class="cursor" x1="${X(cursor)}" x2="${X(cursor)}" y1="${m.t}" y2="${H - m.b}" stroke="#e7e9ec" opacity=".35"/>`;
  s += `<text x="${m.l}" y="${m.t + 9}" fill="#8d949e">${esc(ylab)}</text>`;
  if (legend) {
    let lx = m.l + 6 + ylab.length * 6.2 + 12;
    for (const se of series.filter((q) => q.label)) { s += `<rect x="${lx}" y="${m.t + 3}" width="8" height="3" fill="${se.color}"/><text x="${lx + 11}" y="${m.t + 9}">${esc(se.label)}</text>`; lx += 16 + se.label.length * 6.2; }
  }
  s += "</svg>";
  el.innerHTML = s;
  el._x = X;
}
const niceStep = (r) => { const p = Math.pow(10, Math.floor(Math.log10(r || 1))); const f = r / p; return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p; };
const fmtTick = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? +v.toFixed(2) : +v.toPrecision(2));

function renderTele() {
  const R = S.replay && S.replay.replay, el = $("#tele-chart");
  if (!R) { el.innerHTML = ""; return; }
  const t = R.t;
  const per = (fn) => [0, 1, 2, 3, 4, 5].map((j) => ({ x: t, y: t.map((_, i) => fn(i, j)), color: JC[j], label: `J${j + 1}` }));
  let cfg;
  if (S.tele === "err") cfg = { series: per((i, j) => R.q[i][j] - R.q_ref[i][j]), ylab: "link − commanded (deg)" };
  else if (S.tele === "util") cfg = { series: per((i, j) => R.util[i][j]), ylab: "share of pull-out torque", hlines: [{ y: 1, label: "stall" }], ymin: 0 };
  else if (S.tele === "load") cfg = { series: per((i, j) => R.aux[i][j]), ylab: R.aux_label || "", hlines: (R.aux_lines || []).map((y, i) => ({ y, label: i ? "" : "slips past ±90°" })) };
  else {
    const d = t.map((_, i) => 1000 * Math.hypot(...R.tcp[i].map((v, k) => v - R.tcp_ref[i][k])));
    cfg = { series: [{ x: t, y: d, color: "#ff5f56", label: "actual vs commanded" }], ylab: "TCP deviation (mm)", ymin: 0 };
  }
  lineChart(el, { ...cfg, cursor: t[S.frame] });
}

function drawCursor() {
  const el = $("#tele-chart"), R = S.replay && S.replay.replay;
  const ln = el.querySelector(".cursor");
  if (ln && el._x && R) { const x = el._x(R.t[S.frame]); ln.setAttribute("x1", x); ln.setAttribute("x2", x); }
}

function renderVerdict() {
  const r = S.replay, v = $("#verdict"), ev = $("#events");
  if (!r || !r.valid) { v.textContent = r ? `Invalid: ${r.invalid_reason}` : ""; ev.innerHTML = ""; return; }
  const m = r.metrics;
  v.className = `verdict ${r.failed ? "fail" : "pass"}`;
  v.textContent = r.failed ? `FAIL · ${S.state.modes[r.dominant] || r.dominant}` : `PASS · margin ${(1 - r.failure_score).toFixed(2)}`;
  const items = r.events.length ? r.events.map((e) => `<li>${esc(e)}</li>`) : [];
  items.push(`<li class="ok">Off taught point ≤ ${m.max_tcp_error_mm.toFixed(3)} mm · encoder-blind ${m.max_hidden_error_mm.toFixed(3)} mm · drift ${m.repeat_drift_mm.toFixed(3)} mm</li>`);
  items.push(`<li class="ok">Peak torque used ${(m.peak_torque_util * 100).toFixed(0)}% · lost steps ${m.lost_steps} · ${m.duration_s.toFixed(1)} s program</li>`);
  ev.innerHTML = items.join("");
}

// ------------------------------------------------------------------------ program
async function selectProgram(id) {
  S.programId = id;
  if (!STATIC) fillEnvSelects();
  const p = STATIC ? STATIC.preview : await api(`/api/program/${encodeURIComponent(id)}`);
  S.preview = p;
  const pr = p.program;
  $("#program-desc").textContent = pr.description || "";
  $("#program-facts").innerHTML = [
    ["Duration / cycle", `${(p.duration_s / Math.max(1, pr.cycles)).toFixed(1)} s`],
    ["Cycles", pr.cycles], ["Payload", `${pr.payload_kg} kg`],
    ["Tool CoG", `${pr.payload_cog_mm.join(", ")} mm`], ["Tolerance", `${pr.tolerance_mm} mm`],
    ["Controller", pr.control], ["Taught points", new Set(p.measure.map((m) => m.label)).size],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("");
  $("#program-issues").innerHTML = p.issues.map((i) => `<div class="issue">${esc(i.detail)}</div>`).join("");
  $("#program-moves").innerHTML = pr.moves.map((m) => {
    if (m.type === "joint") return `<li><div><span class="kind">Joint</span> ${m.speed ?? 100}% / ${m.accel ?? 100}%<span class="mono">${m.deg.map((d) => (+d).toFixed(1)).join("  ")}°</span></div></li>`;
    if (m.type === "linear") return `<li><div><span class="kind">Linear</span> ${m.speed_mm_s ?? 50} mm/s<span class="mono">${m.xyz_mm.join("  ")} mm</span></div></li>`;
    return `<li><div><span class="kind dwell">Dwell</span> ${m.s}s${m.measure ? ` · measure <b>${esc(m.measure)}</b>` : ""}</div></li>`;
  }).join("");
  $("#program-yaml").textContent = p.yaml || "";
  for (const r of [V.actual, V.ghost]) r.payload.position.set(...pr.payload_cog_mm.map((v) => v / 1000));
  const rb = p.robot;
  $("#chip-robot").textContent = rb.name;
  $("#chip-control").textContent = { ros2: "ROS 2 driver · velocity mode (MV)", ros2_pid: "ROS 2 · velocity PID", ros2_position: "ROS 2 · position mode (MT)", open_loop: "Ideal step generator", servo: `Servo + harmonic drive · ${rb.controller.rate_hz} Hz` }[pr.control] || pr.control;
  $$("#tele-tabs button")[3].textContent = rb.drive === "servo" ? "Gearbox wind-up" : "Load angle";
  S.robot = rb;
  const reach = rb.reach_mm / 1000;
  if (V.reach !== reach) {
    V.reach = reach;
    V.camera.position.set(1.5 * reach, -1.65 * reach, 1.15 * reach);
    V.controls.target.set(rb.drive === "servo" ? 0.25 * reach : 0.03, rb.drive === "servo" ? 0.1 * reach : -0.35 * reach, 0.35 * reach);
  }
  loadMeshes(p.mesh_dir);
  S.selFamily = null;
  await loadReplay(null, "Nominal run", `${pr.name}: the program at nominal conditions, ${S.useWm && S.state.active_world_model ? "calibrated" : "raw"} simulator`);
}

// ------------------------------------------------------------------------ stress
function renderSearch(st) {
  S.search = st.result;
  S.searchId = st.id;
  const r = st.result;
  $("#search-summary").innerHTML = `<div class="stats">
    <div><b>${r.n_worlds}</b><span>conditions run</span></div>
    <div><b>${r.n_failed}</b><span>failed</span></div>
    <div><b>${r.families.length}</b><span>failure families</span></div>
    <div><b>${r.seconds}s</b><span>search time</span></div></div>
    ${r.world_model ? `<div class="note">Run on the calibrated simulator (world model).</div>` : `<div class="note">Run on the raw simulator.</div>`}`;
  const fams = r.families;
  $("#families").innerHTML = fams.length ? fams.map((f, i) => famCard(f, i)).join("")
    : `<div class="empty">No failure found under this budget. That is a search result, not a proof: try Standard or Deep.</div>`;
  $$("#families .fam").forEach((el) => {
    const f = fams[+el.dataset.i];
    $("[data-act=replay]", el).onclick = () => replayFamily(f);
    const iso = $("[data-act=cause]", el);
    if (iso) iso.onclick = () => isolate(f, el);
  });
  const sel = $("#rep-cond");
  sel.innerHTML = `<option value="">Nominal</option>` + fams.map((f, i) => `<option value="${i}">${esc(f.name)}</option>`).join("");
}

function famCard(f, i) {
  const devs = f.deviations.slice(0, 6).map((d) => `<span class="dev ${d.outside_plausible ? "out" : ""}" title="${d.outside_plausible ? "outside the plausible band" : "within the plausible band"}">${esc(fmtDim(d.name, d.value))}</span>`).join("");
  return `<div class="fam ${S.selFamily === f.key ? "sel" : ""}" data-i="${i}">
    <div class="fam-head"><span class="fam-name">${esc(f.name)}</span><span class="fam-count">${f.count} of ${S.search.n_worlds}</span></div>
    <div class="plaus"><span>plausibility</span><div class="track"><div class="fill" style="width:${(f.plausibility * 100).toFixed(0)}%"></div></div><span class="mono">${f.plausibility.toFixed(2)}</span></div>
    <div class="devs">${devs}</div>
    <div class="ev">${esc(f.events[0] || "")}</div>
    <div class="fam-actions"><button class="btn small" data-act="replay">Replay</button>${STATIC ? "" : `<button class="btn small ghost" data-act="cause">Isolate cause</button>`}</div>
    <div class="cause-slot">${f.minimal_cause ? causeHtml(f.minimal_cause) : ""}</div></div>`;
}

function causeHtml(mc) {
  if (!mc.reproduced) return `<div class="cause muted">Did not reproduce on re-run.</div>`;
  const rows = mc.thresholds.map((t) => `<div class="thr"><span>${esc(DIM[t.name] ? DIM[t.name][0] : t.name)} ${t.direction === "above" ? "≥" : "≤"} <b>${fmtNum(t.threshold)}</b> ${DIM[t.name] ? DIM[t.name][1] : ""}</span><span class="${t.within_plausible ? "in" : "muted"}">${t.within_plausible ? "inside plausible band" : "outside plausible band"}</span></div>`).join("");
  return `<div class="cause"><div class="muted small">Minimal cause: fails only with these, everything else back to nominal${mc.thresholds.length > 1 ? " (thresholds with the others held)" : ""}</div>${rows}</div>`;
}
const fmtNum = (v) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

async function replayFamily(f) {
  S.selFamily = f.key;
  $$("#families .fam").forEach((el) => el.classList.toggle("sel", S.search.families[+el.dataset.i].key === f.key));
  const devs = f.deviations.slice(0, 4).map((d) => fmtDim(d.name, d.value)).join(" · ");
  await loadReplay(f.theta, f.name, `Most plausible failing condition: ${devs}`);
  play();
}

async function isolate(f, el) {
  const slot = $(".cause-slot", el);
  slot.innerHTML = `<div class="cause"><div class="progress"><div class="bar"></div><span></span></div></div>`;
  try {
    await runJob({ kind: "mincause", program: S.programId, theta: f.theta, use_wm: S.useWm, study: S.searchId, family: f.key },
      $(".progress", slot), async (st) => { f.minimal_cause = st.result.cause; slot.innerHTML = causeHtml(st.result.cause); });
  } catch (e) { slot.innerHTML = `<div class="cause muted">${esc(e.message)}</div>`; }
}

// ---------------------------------------------------------------------- envelope
function renderEnvelope(st) {
  const r = st.result, el = $("#envelope");
  const W = 396, H = 330, m = { l: 52, r: 8, t: 10, b: 40 };
  const nx = r.xs.length, ny = r.ys.length;
  const cw = (W - m.l - m.r) / nx, ch = (H - m.t - m.b) / ny;
  const [x0, x1] = [r.xs[0], r.xs[nx - 1]], [y0, y1] = [r.ys[0], r.ys[ny - 1]];
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r - cw) + cw / 2;
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b - ch) - ch / 2;
  let s = `<svg viewBox="0 0 ${W} ${H}">`;
  r.cells.forEach((c, k) => {
    const ix = k % nx, iy = Math.floor(k / nx);
    const col = !c.valid ? "#333" : c.failed ? `rgba(255,95,86,${0.35 + 0.5 * Math.min(1, (c.score - 1) / 2)})` : mix(c.score);
    s += `<rect class="cell" data-k="${k}" x="${m.l + ix * cw + 1}" y="${H - m.b - (iy + 1) * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="3" fill="${col}"><title>${dimLabel(r.x)} ${fmtNum(c.x)}, ${dimLabel(r.y)} ${fmtNum(c.y)}: ${c.failed ? "FAIL (" + (S.state.modes[c.dominant] || c.dominant) + ")" : "pass"}, score ${c.score}</title></rect>`;
  });
  const px = r.plausible[r.x], py = r.plausible[r.y];
  const bx0 = Math.max(X(px[0]) - cw / 2, m.l), bx1 = Math.min(X(px[1]) + cw / 2, W - m.r);
  const by0 = Math.min(Y(py[0]) + ch / 2, H - m.b), by1 = Math.max(Y(py[1]) - ch / 2, m.t);
  s += `<rect x="${bx0}" y="${by1}" width="${bx1 - bx0}" height="${by0 - by1}" fill="none" stroke="#e7e9ec" stroke-dasharray="5 4" opacity=".6"/>`;
  s += `<circle cx="${X(r.base[r.x])}" cy="${Y(r.base[r.y])}" r="4" fill="#e7e9ec"/>`;
  r.xs.forEach((v, i) => { if (i % Math.ceil(nx / 5) === 0 || i === nx - 1) s += `<text x="${m.l + i * cw + cw / 2}" y="${H - m.b + 13}" text-anchor="middle" class="axis">${fmtNum(v)}</text>`; });
  r.ys.forEach((v, i) => { if (i % Math.ceil(ny / 5) === 0 || i === ny - 1) s += `<text x="${m.l - 6}" y="${H - m.b - i * ch - ch / 2 + 3}" text-anchor="end">${fmtNum(v)}</text>`; });
  s += `<text x="${(W + m.l) / 2}" y="${H - 6}" text-anchor="middle">${esc(dimLabel(r.x))}</text>`;
  s += `<text transform="translate(12 ${(H - m.b) / 2}) rotate(-90)" text-anchor="middle">${esc(dimLabel(r.y))}</text></svg>`;
  const fails = r.cells.filter((c) => c.failed).length;
  // largest passing value along x at nominal y, the number an integrator asks for
  el.innerHTML = `<div class="heat">${s}</div><div class="note">${fails} of ${r.cells.length} cells fail. Click a cell to replay it. Colour runs from teal (comfortable) to amber (near the limit); red fails.</div>`;
  if (!STATIC) $$("rect.cell", el).forEach((rc) => (rc.onclick = () => {
    const c = r.cells[+rc.dataset.k];
    S.selFamily = null;
    loadReplay({ ...r.base, [r.x]: c.x, [r.y]: c.y }, `${fmtDim(r.x, c.x)} · ${fmtDim(r.y, c.y)}`, c.failed ? `Fails: ${S.state.modes[c.dominant] || c.dominant}` : "Passes").then(play);
  }));
}
function mix(score) {
  const a = [47, 191, 155], b = [245, 165, 36], f = Math.max(0, Math.min(1, score));
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * f));
  return `rgba(${c[0]},${c[1]},${c[2]},${0.25 + 0.45 * f})`;
}

// --------------------------------------------------------------------- repeatability
function renderRepeat(st) {
  const r = st.result, el = $("#repeat");
  const labels = Object.keys(r.per_point);
  el.innerHTML = `<div class="stats"><div><b>${r.cycles}</b><span>cycles</span></div><div><b>${r.metrics.repeat_drift_mm.toFixed(3)}</b><span>max drift (mm)</span></div><div><b>${r.metrics.lost_steps}</b><span>lost steps</span></div><div><b class="${r.failed ? "" : ""}">${r.failed ? "FAIL" : "PASS"}</b><span>${r.failed ? esc(S.state.modes[r.dominant] || "") : "verdict"}</span></div></div>
    <div class="mini-chart" id="rep-chart"></div>
    ${r.events.map((e) => `<div class="note warn">${esc(e)}</div>`).join("")}
    <div class="note">Error at each taught point against the as-taught (nominal, cycle 1) position. Growth with cycle number is accumulated error: lost steps the velocity-mode driver never makes up.</div>`;
  const cols = ["#4c8dff", "#f5a524", "#2fbf9b", "#b48cff"];
  lineChart($("#rep-chart"), {
    series: labels.map((l, i) => ({ x: r.per_point[l].map((p) => p.cycle), y: r.per_point[l].map((p) => p.err_mm), color: cols[i % 4], label: l, dots: true })),
    ylab: "off taught point (mm)", ymin: 0,
    hlines: [{ y: S.preview.program.tolerance_mm, label: "tolerance", color: "#ff5f56" }],
  });
}

// ------------------------------------------------------------------------ sim vs real
function renderCalibration(rep, name) {
  const el = $("#calibration");
  if (!rep) { el.innerHTML = ""; return; }
  const b = rep.before, a = rep.after;
  const src = rep.log && rep.log.meta && rep.log.meta.source;
  const imp = (x, y) => (y > 0 ? `${(x / y).toFixed(1)}× better` : "");
  let h = `${src && String(src).startsWith("SYNTHETIC") ? `<div class="note warn">This log is SYNTHETIC: a simulated stand-in with hidden drive-train values and an unmodelled 17 ms latency, so the fit can be checked against a known answer. Replace it with your robot's log.</div>` : ""}
  <div class="big"><div><span>Joint feedback, raw sim (RMS)</span><b>${b.joint_rms_deg.toFixed(3)}°</b></div><div class="after"><span>Calibrated sim (RMS) · ${imp(b.joint_rms_deg, a.joint_rms_deg)}</span><b>${a.joint_rms_deg.toFixed(3)}°</b></div>
  ${b.tcp_rms_mm != null ? `<div><span>TCP at measured points, raw (RMS)</span><b>${b.tcp_rms_mm.toFixed(3)} mm</b></div><div class="after"><span>Calibrated · ${imp(b.tcp_rms_mm, a.tcp_rms_mm)}</span><b>${a.tcp_rms_mm.toFixed(3)} mm</b></div>` : ""}</div>
  <table class="t"><tr><th>Parameter</th><th>Default</th><th>Fitted</th><th>From this log</th></tr>
  ${Object.entries(rep.params).map(([k, v]) => { const id = rep.identifiability[k] || {}; const cls = id.status === "identified" ? "ok" : id.status === "weak" ? "weak" : "no"; return `<tr><td>${esc(DIM[k] ? DIM[k][0] : k)}</td><td class="n">${(rep.defaults[k] ?? "").toString()}</td><td class="n">${(+v).toFixed(3)}</td><td><span class="badge ${cls}" title="${esc(id.why || "")}">${esc(id.status || "")}</span></td></tr>`; }).join("")}</table>
  ${Object.values(rep.identifiability).some((x) => x.why) ? `<div class="note warn">${esc(Object.values(rep.identifiability).find((x) => x.why).why)}</div>` : ""}
  <div class="row"><div class="seg" id="cal-joint">${[1, 2, 3, 4, 5, 6].map((j) => `<button data-j="${j - 1}" class="${j === 2 ? "on" : ""}">J${j}</button>`).join("")}</div><span class="muted small">real vs simulated encoder − command</span></div>
  <div class="mini-chart" id="cal-chart"></div>
  <table class="t"><tr><th>Joint RMS (deg)</th>${[1, 2, 3, 4, 5, 6].map((j) => `<th>J${j}</th>`).join("")}</tr>
  <tr><td>raw</td>${b.joint_rms_by_joint_deg.map((v) => `<td class="n">${v.toFixed(3)}</td>`).join("")}</tr>
  <tr><td>calibrated</td>${a.joint_rms_by_joint_deg.map((v) => `<td class="n">${v.toFixed(3)}</td>`).join("")}</tr></table>
  <div class="note">World model <b>${esc(name)}</b> is ${S.state.active_world_model === name ? "active: stress tests run on the calibrated simulator" : "saved"}. Calibrated physics re-centres the drive-train axes of the stress space; the residual is applied as a correction on the commanded motion (<code>world_model_scale</code>).</div>`;
  el.innerHTML = h;
  const draw = (j) => {
    const sr = rep.series, t = sr.t;
    const e = (arr) => t.map((_, i) => arr[i][j] - sr.cmd[i][j]);
    lineChart($("#cal-chart"), { series: [
      { x: t, y: e(sr.real), color: "#e7e9ec", label: "real", width: 1.6 },
      { x: t, y: e(sr.sim_before), color: "#ff5f56", label: "raw sim", dash: "4 3" },
      { x: t, y: e(sr.sim_after), color: "#2fbf9b", label: "calibrated" }], ylab: `J${j + 1} feedback − command (deg)` });
  };
  $$("#cal-joint button").forEach((bt) => (bt.onclick = () => { $$("#cal-joint button").forEach((x) => x.classList.toggle("on", x === bt)); draw(+bt.dataset.j); }));
  draw(1);
}

function renderLogs() {
  const logs = S.state.logs || [];
  $("#log-select").innerHTML = logs.length ? logs.map((l) => `<option value="${esc(l.id)}">${esc(l.id)}${l.duration_s ? ` · ${l.duration_s}s${l.tcp_samples ? ` · ${l.tcp_samples} TCP pts` : ""}` : ""}</option>`).join("") : `<option value="">No logs yet</option>`;
}

function renderSimChip() {
  const c = $("#chip-sim"), wm = S.state.active_world_model;
  if (wm && S.useWm) { c.textContent = `Simulator: calibrated (${wm})`; c.classList.add("cal"); }
  else { c.textContent = wm ? "Simulator: raw (calibration off)" : "Simulator: raw (not calibrated)"; c.classList.remove("cal"); }
}

// ------------------------------------------------------------------------ report
async function refreshStudies() {
  if (STATIC) {
    const st = STATIC.study;
    $("#studies").innerHTML = `<div class="note">Exported ${esc(STATIC.report.generated)} · schema <code>${STATIC.report.schema}</code></div>` +
      STATIC.report.limits.map((l) => `<div class="note">${esc(l)}</div>`).join("");
    return;
  }
  S.state = await api("/api/state");
  const list = S.state.studies;
  $("#studies").innerHTML = list.length ? list.map((s) => `<div class="study"><div><span class="k">${esc(s.kind)}</span> <span class="muted">${esc(s.program || "")} · ${esc(s.created.replace("T", " "))}</span></div><div><a class="btn small ghost" href="/api/report/${s.id}.html">HTML</a> <a class="btn small ghost" href="/api/report/${s.id}.json">JSON</a></div></div>`).join("") : `<div class="empty">Run a test to create a study.</div>`;
  renderLogs();
  renderSimChip();
}

// ------------------------------------------------------------------------ wiring
function wire() {
  $$(".tabs button").forEach((b) => (b.onclick = () => {
    $$(".tabs button").forEach((x) => x.classList.toggle("on", x === b));
    $$(".tab").forEach((t) => t.classList.toggle("on", t.dataset.tab === b.dataset.tab));
  }));
  $$("#tele-tabs button").forEach((b) => (b.onclick = () => {
    $$("#tele-tabs button").forEach((x) => x.classList.toggle("on", x === b));
    S.tele = b.dataset.k; renderTele();
  }));
  $$("#budget button").forEach((b) => (b.onclick = () => {
    $$("#budget button").forEach((x) => x.classList.toggle("on", x === b)); S.budget = b.dataset.b;
  }));
  $("#play").onclick = play;
  $("#scrub").oninput = (e) => { const R = S.replay && S.replay.replay; if (!R) return; S.playing = false; $("#play").textContent = "▶"; showFrame(Math.round((+e.target.value / 1000) * (R.t.length - 1))); };
  $("#speed").onchange = (e) => { S.speed = +e.target.value; if (S.playing) { S.playing = false; play(); } };
  $("#magnify").onchange = (e) => { S.mag = +e.target.value; showFrame(S.frame); };
  window.addEventListener("resize", () => { renderTele(); drawMarkers(); });
  if (STATIC) return;

  $("#program-select").onchange = (e) => selectProgram(e.target.value);
  $("#upload-program").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try { const { id } = await api("/api/program", { yaml: await f.text() }); await boot(id); } catch (err) { alert(`Program rejected: ${err.message}`); }
  };
  $("#chip-sim").onclick = async () => {
    if (!S.state.active_world_model) { $$(".tabs button").find((b) => b.dataset.tab === "real").click(); return; }
    S.useWm = !S.useWm; renderSimChip(); await selectProgram(S.programId);
  };
  $("#run-search").onclick = async (e) => {
    e.target.disabled = true;
    try { await runJob({ kind: "search", program: S.programId, budget: S.budget, use_wm: S.useWm }, $("#search-progress"), async (st) => { renderSearch(st); if (st.result.families[0]) replayFamily(st.result.families[0]); }); }
    catch (err) { console.error(err); } finally { e.target.disabled = false; }
  };
  $("#run-envelope").onclick = async (e) => {
    e.target.disabled = true;
    try { await runJob({ kind: "envelope", program: S.programId, x: $("#env-x").value, y: $("#env-y").value, n: +$("#env-n").value, use_wm: S.useWm }, $("#env-progress"), async (st) => renderEnvelope(st)); }
    catch (err) { console.error(err); } finally { e.target.disabled = false; }
  };
  $("#run-repeat").onclick = async (e) => {
    e.target.disabled = true;
    const ci = $("#rep-cond").value;
    const theta = ci !== "" && S.search ? S.search.families[+ci].theta : null;
    try { await runJob({ kind: "endurance", program: S.programId, cycles: +$("#rep-cycles").value, theta, use_wm: S.useWm }, $("#rep-progress"), async (st) => renderRepeat(st)); }
    catch (err) { console.error(err); } finally { e.target.disabled = false; }
  };
  $("#upload-log").onchange = async (e) => {
    const f = e.target.files[0]; if (!f) return;
    try { await api("/api/logs", { csv: await f.text(), name: f.name }); await refreshStudies(); $("#log-select").value = f.name.replace(/\.csv$/, ""); } catch (err) { alert(`Log rejected: ${err.message}`); }
  };
  $("#synth-log").onclick = async (e) => {
    e.target.disabled = true;
    try { const { id } = await api("/api/logs/synthetic", { program: S.programId }); await refreshStudies(); $("#log-select").value = id; } finally { e.target.disabled = false; }
  };
  $("#run-fit").onclick = async (e) => {
    const log = $("#log-select").value; if (!log) return;
    e.target.disabled = true;
    try {
      await runJob({ kind: "calibrate", log, fit_payload: $("#fit-payload").checked }, $("#fit-progress"), async (st) => {
        S.state = await api("/api/state"); S.useWm = true; renderSimChip();
        renderCalibration(st.result.report, st.result.world_model);
        await selectProgram(S.programId);
      });
    } catch (err) { console.error(err); } finally { e.target.disabled = false; }
  };
}

function fillEnvSelects() {
  const dims = ((S.state.spaces || {})[S.programId] || Object.keys(DIM)).filter((n) => DIM[n]);
  const opts = (sel) => dims.map((n) => `<option value="${n}" ${n === sel ? "selected" : ""}>${esc(dimLabel(n))}</option>`).join("");
  $("#env-x").innerHTML = opts("payload_kg");
  $("#env-y").innerHTML = opts("accel_scale");
}

async function boot(pick) {
  if (STATIC) {
    S.state = STATIC.state;
    $("#static-badge").hidden = false;
    if (STATIC.title) $(".brand .product").textContent = STATIC.title;
    $$("#run-search, #run-envelope, #run-repeat, #run-fit, #upload-program-btn, #synth-log, #budget, label:has(#upload-log), label:has(#fit-payload), .form").forEach((x) => (x.style.display = "none"));
    $("#program-select").innerHTML = `<option>${esc(STATIC.preview.program.name)}</option>`;
    S.useWm = !!STATIC.study.world_model;
    renderSimChip();
    await selectProgram(STATIC.preview.id);
    const sts = STATIC.studies || [STATIC.study];
    for (const st of sts) {
      if (st.kind === "search") renderSearch(st);
      if (st.kind === "envelope") renderEnvelope(st);
      if (st.kind === "endurance") renderRepeat(st);
      if (st.kind === "calibration") renderCalibration(st.result.report, st.result.world_model);
    }
    const first = { search: "stress", envelope: "envelope", endurance: "repeat", calibration: "real" }[sts[0].kind];
    if (first) $$(".tabs button").find((b) => b.dataset.tab === first).click();
    refreshStudies();
    return;
  }
  S.state = await api("/api/state");
  $("#workers").textContent = `${S.state.workers} sim workers`;
  const groups = {};
  for (const p of S.state.programs) (groups[p.robot_name || "Other"] ||= []).push(p);
  $("#program-select").innerHTML = Object.entries(groups).map(([g, ps]) => `<optgroup label="${esc(g)}">${ps.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}${p.builtin ? "" : " (uploaded)"}</option>`).join("")}</optgroup>`).join("");
  const id = pick || S.state.programs[0].id;
  $("#program-select").value = id;

  renderLogs();
  renderSimChip();
  refreshStudies();
  if (S.state.active_world_model) {
    api(`/api/world_model/${encodeURIComponent(S.state.active_world_model)}`).then((w) => renderCalibration(w.report, S.state.active_world_model)).catch(() => {});
  }
  await selectProgram(id);
}

initViewer();
wire();
boot().catch((e) => { $("#replay-sub").textContent = `Could not start: ${e.message}`; console.error(e); });
