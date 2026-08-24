'use strict';
// 상태 서버: 의존성 없음. node server.js 로 실행.
// - GET  /overlay.html   OBS 브라우저 소스용 오버레이
// - GET  /control.html   오퍼레이터 컨트롤 패널
// - GET  /api/state      현재 상태 스냅샷
// - GET  /api/events     SSE 스트림 (state / cmd)
// - POST /api/state      부분 병합 패치 → 저장 + 브로드캐스트
// - POST /api/cmd        일회성 명령 (저장 안 함)

const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = __dirname;
const STATE_FILE = path.join(ROOT, 'state.json');
const PORT = Number(process.env.PORT) || 8787;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
  '.gif': 'image/gif', '.webp': 'image/webp', '.svg': 'image/svg+xml',
  '.mp4': 'video/mp4', '.webm': 'video/webm', '.mkv': 'video/x-matroska',
  '.woff2': 'font/woff2', '.woff': 'font/woff', '.ttf': 'font/ttf'
};

// state.json 이 없거나 깨져 있으면 이 값으로 새로 만듭니다.
const DEFAULT_STATE = {
  view: 'select',
  round: { kicker: 'ARENA BATTLE', title: 'ROUND 1', sub: 'EVOLVE' },
  teams: {
    left: { name: '왼쪽 팀', logo: '', score: 0, players: [] },
    right: { name: '오른쪽 팀', logo: '', score: 0, players: [] }
  },
  songs: {
    left: [
      { title: '', jacket: '', diff: 'MXM', level: '', banned: false },
      { title: '', jacket: '', diff: 'MXM', level: '', banned: false },
      { title: '', jacket: '', diff: 'MXM', level: '', banned: false }
    ],
    right: [
      { title: '', jacket: '', diff: 'MXM', level: '', banned: false },
      { title: '', jacket: '', diff: 'MXM', level: '', banned: false },
      { title: '', jacket: '', diff: 'MXM', level: '', banned: false }
    ]
  },
  match: {
    label: '', meter: 0, meterVisible: true,
    autoMeter: false, autoRank: false,
    combinedVideo: '', maxDiff: 150000, labelStyle: 'short',
    slots: [
      { rank: '1st', team: 'left', player: '', icon: '', video: '', offsetMs: 0 },
      { rank: '2nd', team: 'left', player: '', icon: '', video: '', offsetMs: 0 },
      { rank: '3rd', team: 'right', player: '', icon: '', video: '', offsetMs: 0 },
      { rank: '4th', team: 'right', player: '', icon: '', video: '', offsetMs: 0 }
    ]
  }
};

let state;
try {
  state = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
  // 예전 state.json 에 새 기능의 필드가 없을 수 있으므로 기본값을 밑에 깔아줍니다.
  // (사용자가 입력한 값은 그대로 두고, 빠진 키만 채웁니다)
  state = fillDefaults(state, DEFAULT_STATE);
} catch (e) {
  state = DEFAULT_STATE;
  fs.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
  console.log(`state.json 을 새로 만들었습니다 (${e.code === 'ENOENT' ? '파일 없음' : '읽기 실패'})`);
}

function fillDefaults(cur, def) {
  if (cur === null || cur === undefined) return def;
  if (Array.isArray(def) || typeof def !== 'object') return cur;
  if (typeof cur !== 'object' || Array.isArray(cur)) return cur;
  const out = { ...cur };
  for (const k of Object.keys(def)) out[k] = fillDefaults(cur[k], def[k]);
  return out;
}
let clients = [];
let diag = { timelines: {}, at: 0 };   // 오버레이가 보고하는 타임라인 로드 상태
let saveTimer = null;

/* ── 프리셋 저장소 ────────────────────────────────────
   presets/<이름>.json. 이름에 경로 문자가 섞이면 서버 밖 파일을 덮어쓸 수
   있으므로, 허용 문자만 남기고 검증에 실패하면 아예 거부합니다. */
const PRESET_DIR = path.join(ROOT, 'presets');

function presetPath(name) {
  if (typeof name !== 'string') return null;
  const clean = name.trim();
  if (!clean || clean.length > 60) return null;
  // 경로 구분자·상위 이동·제어문자 차단
  if (/[\\/:*?"<>|]/.test(clean) || clean.includes('..') || /[\x00-\x1f]/.test(clean)) return null;
  return path.join(PRESET_DIR, clean + '.json');
}

function listPresets() {
  try {
    return fs.readdirSync(PRESET_DIR)
      .filter((f) => f.endsWith('.json'))
      .map((f) => f.slice(0, -5))
      .sort();
  } catch (_) {
    return [];
  }
}

function saveSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2), () => {});
  }, 300);
}

function send(event, data) {
  const payload = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  clients = clients.filter((res) => {
    try { res.write(payload); return true; } catch (_) { return false; }
  });
}

// 객체는 재귀 병합, 배열은 통째로 교체 (곡 목록/슬롯은 항상 전체를 보냄)
// 단, 원래 배열이던 자리에 배열이 아닌 값이 오면 무시합니다 — 잘못된 패치 하나로
// 오버레이의 forEach 가 터져 방송 화면이 멈추는 것을 막습니다.
function merge(target, patch) {
  for (const k of Object.keys(patch)) {
    const v = patch[k];
    if (Array.isArray(target[k]) && v !== undefined && v !== null && !Array.isArray(v)) {
      console.warn(`[state] ${k}: 배열 자리에 ${typeof v} 가 와서 무시했습니다`);
      continue;
    }
    const isPlainObj = v && typeof v === 'object' && !Array.isArray(v);
    const targetIsPlainObj = target[k] && typeof target[k] === 'object' && !Array.isArray(target[k]);
    if (isPlainObj && targetIsPlainObj) merge(target[k], v);
    else target[k] = v;
  }
  return target;
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let buf = '';
    req.on('data', (c) => {
      buf += c;
      if (buf.length > 4e6) { req.destroy(); reject(new Error('body too large')); }
    });
    req.on('end', () => {
      try { resolve(buf ? JSON.parse(buf) : {}); } catch (e) { reject(e); }
    });
  });
}

function serveStatic(req, res, pathname) {
  const rel = pathname === '/' ? '/control.html' : decodeURIComponent(pathname);
  const base = path.basename(rel);
  // 폴더 구조가 평평해져도 찾도록 후보를 넓게 잡음
  const candidates = [
    path.join(ROOT, 'public', rel), path.join(ROOT, rel),
    path.join(ROOT, 'public', base), path.join(ROOT, base)
  ];
  const file = candidates.find((p) => p.startsWith(ROOT) && fs.existsSync(p) && fs.statSync(p).isFile());
  if (!file) {
    console.warn(`404  ${pathname}`);
    res.writeHead(404); return res.end('Not found');
  }

  const type = MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
  const stat = fs.statSync(file);

  // 영상은 Range 요청 지원 (seek 필수)
  const range = req.headers.range;
  if (range && /^bytes=/.test(range)) {
    const [startRaw, endRaw] = range.replace('bytes=', '').split('-');
    const start = parseInt(startRaw, 10) || 0;
    const end = endRaw ? parseInt(endRaw, 10) : stat.size - 1;
    res.writeHead(206, {
      'Content-Type': type,
      'Content-Range': `bytes ${start}-${end}/${stat.size}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': end - start + 1
    });
    return fs.createReadStream(file, { start, end }).pipe(res);
  }

  res.writeHead(200, { 'Content-Type': type, 'Content-Length': stat.size, 'Accept-Ranges': 'bytes', 'Cache-Control': 'no-cache' });
  fs.createReadStream(file).pipe(res);
}

const server = http.createServer(async (req, res) => {
  const { pathname } = new URL(req.url, `http://${req.headers.host}`);
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') { res.writeHead(204); return res.end(); }

  if (pathname === '/api/events') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream; charset=utf-8',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive'
    });
    res.write(`event: state\ndata: ${JSON.stringify(state)}\n\n`);
    clients.push(res);
    const ping = setInterval(() => { try { res.write(': ping\n\n'); } catch (_) {} }, 15000);
    req.on('close', () => { clearInterval(ping); clients = clients.filter((c) => c !== res); });
    return;
  }

  if (pathname === '/api/state' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': MIME['.json'] });
    return res.end(JSON.stringify(state));
  }

  if (pathname === '/api/state' && req.method === 'POST') {
    try {
      const patch = await readBody(req);
      merge(state, patch);
      saveSoon();
      send('state', state);
      res.writeHead(200, { 'Content-Type': MIME['.json'] });
      return res.end(JSON.stringify({ ok: true }));
    } catch (e) {
      res.writeHead(400); return res.end(String(e.message));
    }
  }

  /* ── 프리셋 (경기별 데이터를 미리 만들어두고 당일엔 불러오기만) ──
     presets/ 폴더에 파일 하나씩. 라이브 상태(state.json)와 완전히 분리되어
     있어서, 프리셋을 저장해도 방송 화면은 꿈쩍하지 않습니다. */
  if (pathname === '/api/presets' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': MIME['.json'] });
    return res.end(JSON.stringify({ presets: listPresets() }));
  }

  if (pathname.startsWith('/api/presets/') && req.method === 'GET') {
    const name = decodeURIComponent(pathname.slice('/api/presets/'.length));
    const file = presetPath(name);
    if (!file) { res.writeHead(400); return res.end('잘못된 이름'); }
    try {
      const data = fs.readFileSync(file, 'utf-8');
      res.writeHead(200, { 'Content-Type': MIME['.json'] });
      return res.end(data);
    } catch (e) {
      res.writeHead(404); return res.end('프리셋을 찾을 수 없습니다');
    }
  }

  if (pathname.startsWith('/api/presets/') && req.method === 'PUT') {
    const name = decodeURIComponent(pathname.slice('/api/presets/'.length));
    const file = presetPath(name);
    if (!file) { res.writeHead(400); return res.end('잘못된 이름'); }
    try {
      const body = await readBody(req);
      fs.mkdirSync(PRESET_DIR, { recursive: true });
      // 임시 파일에 먼저 쓰고 교체 — 저장 중 정전/강제종료에도 기존 파일이 안 깨집니다
      const tmp = file + '.tmp';
      fs.writeFileSync(tmp, JSON.stringify(body, null, 2));
      fs.renameSync(tmp, file);
      res.writeHead(200, { 'Content-Type': MIME['.json'] });
      return res.end(JSON.stringify({ ok: true, presets: listPresets() }));
    } catch (e) {
      res.writeHead(400); return res.end(String(e.message));
    }
  }

  if (pathname.startsWith('/api/presets/') && req.method === 'DELETE') {
    const name = decodeURIComponent(pathname.slice('/api/presets/'.length));
    const file = presetPath(name);
    if (!file) { res.writeHead(400); return res.end('잘못된 이름'); }
    try { fs.unlinkSync(file); } catch (_) {}
    res.writeHead(200, { 'Content-Type': MIME['.json'] });
    return res.end(JSON.stringify({ ok: true, presets: listPresets() }));
  }

  /* ── 진단 ── 오버레이가 타임라인 로드 결과를 보고하면 컨트롤이 읽어갑니다.
     상태(state)에 섞지 않는 이유: 방송 데이터가 아니고, SSE 로 돌려보내면
     오버레이가 다시 렌더되는 순환이 생기기 때문입니다. */
  if (pathname === '/api/diag' && req.method === 'POST') {
    try {
      diag = await readBody(req);
      diag.at = Date.now();
      res.writeHead(200, { 'Content-Type': MIME['.json'] });
      return res.end(JSON.stringify({ ok: true }));
    } catch (e) {
      res.writeHead(400); return res.end(String(e.message));
    }
  }

  if (pathname === '/api/diag' && req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': MIME['.json'] });
    return res.end(JSON.stringify(diag));
  }

  if (pathname === '/api/cmd' && req.method === 'POST') {
    try {
      const cmd = await readBody(req);
      send('cmd', cmd);
      res.writeHead(200, { 'Content-Type': MIME['.json'] });
      return res.end(JSON.stringify({ ok: true }));
    } catch (e) {
      res.writeHead(400); return res.end(String(e.message));
    }
  }

  serveStatic(req, res, pathname);
});

for (const p of ['public/overlay.html', 'public/control.html', 'state.json']) {
  if (!fs.existsSync(path.join(ROOT, p))) console.warn(`[누락] ${p}`);
}

server.listen(PORT, () => {
  console.log(`오버레이  http://localhost:${PORT}/overlay.html`);
  console.log(`컨트롤    http://localhost:${PORT}/control.html`);
});
