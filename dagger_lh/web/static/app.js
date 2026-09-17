/* Participant client for the DAgger learned-helplessness study.
 *
 * Thin client: the simulator runs on the server. This file reports key *state*
 * (not events) for continuous control, renders JPEG frames onto a canvas, and
 * renders whatever screen the server says to show. The server drives the study
 * flow, so the client holds no condition information and cannot leak it.
 */
'use strict';

const KEYMAP = {
  KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd', KeyR: 'r', KeyF: 'f',
  KeyQ: 'q', KeyE: 'e', KeyX: 'x', KeyP: 'p', KeyH: 'h',
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  Space: 'space', Tab: 'tab', Enter: 'enter', Backspace: 'backspace',
  ShiftLeft: 'lshift', ShiftRight: 'rshift',
};
// Keys the browser would otherwise act on (scroll, focus change, back-navigation).
const SWALLOW = new Set(['Space','Tab','ArrowUp','ArrowDown','ArrowLeft',
                         'ArrowRight','Backspace','Enter']);
const INPUT_HZ = 30;

const state = {
  sid: null, ws: null, held: new Set(), pending: [],
  screen: null, teleopActive: false, latency: null,
  lastSent: 0, seq: 0, surveyStart: 0, reconnects: 0, pingTimer: null,
  pendingReturn: false, screenSeq: null,
};

const root = document.getElementById('root');
const esc = s => String(s ?? '').replace(/[&<>"]/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

/* ------------------------------------------------------------------ enrolment */
async function boot() {
  const qs = new URLSearchParams(location.search);
  const body = {};
  for (const k of ['PROLIFIC_PID','STUDY_ID','SESSION_ID','source','arm','token',
                   'external_id','pid','participant_id','ResponseID']) {
    if (qs.get(k)) body[k] = qs.get(k);
  }

  // Coming back from an external questionnaire. The worker is still parked on
  // its redirect screen, so answer that screen instead of rendering it again --
  // otherwise the participant is bounced straight back to Qualtrics.
  if (qs.get('returned') === '1') {
    state.pendingReturn = true;
    try {
      const clean = new URL(location.href);
      clean.searchParams.delete('returned');
      history.replaceState({}, '', clean);   // so a refresh doesn't re-trigger
    } catch (e) {}
  }

  // Resume across an accidental refresh so a reload does not restart the study.
  const saved = qs.get('sid') || sessionStorage.getItem('dagger_sid');
  if (saved) {
    state.sid = saved;
    try { sessionStorage.setItem('dagger_sid', saved); } catch (e) {}
    connect();
    return;
  }

  let r;
  try { r = await fetch('/api/enroll', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body) });
  } catch (e) { return fatal('Could not reach the study server.'); }

  if (r.status === 503) {
    return card('Study is busy', `<p>All experiment slots are in use right now.
      Please wait a couple of minutes and reload this page.</p>
      <div class="row"><button onclick="location.reload()">Try again</button></div>`);
  }
  if (!r.ok) return fatal('The study could not be started (' + r.status + ').');
  const j = await r.json();
  state.sid = j.sid;
  try { sessionStorage.setItem('dagger_sid', j.sid); } catch (e) {}
  connect();
}

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/${state.sid}`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    state.reconnects = 0;
    send({t: 'hello', viewport: [innerWidth, innerHeight], ua: navigator.userAgent});
    // A heartbeat is mandatory, not just nice: proxies (Cloudflare included)
    // close a WebSocket that goes idle in both directions, and the survey and
    // debrief screens send no frames for minutes at a time.
    // Cleared and restarted per connection so reconnects don't stack up timers.
    if (state.pingTimer) clearInterval(state.pingTimer);
    state.pingTimer = setInterval(
      () => send({t: 'ping', ts: performance.now()}), 3000);
  };
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') return onJSON(JSON.parse(ev.data));
    onBinary(ev.data);
  };
  ws.onclose = ev => {
    if (state.pingTimer) { clearInterval(state.pingTimer); state.pingTimer = null; }
    if (ev.code === 4404) {
      try { sessionStorage.removeItem('dagger_sid'); } catch (e) {}
      return card('Session not found', `<p>This session has expired.</p>
        <div class="row"><button onclick="location.reload()">Start over</button></div>`);
    }
    if (state.reconnects < 12) {
      state.reconnects += 1;
      banner(`Connection lost — reconnecting (${state.reconnects})…`);
      setTimeout(connect, 1200);
    } else {
      fatal('Connection lost. Your progress up to this point was saved.');
    }
  };
}

function send(o) {
  if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(o));
}

/* ------------------------------------------------------------------ inbound */
let sideExpected = false;

function onBinary(buf) {
  const dv = new DataView(buf);
  if (sideExpected) {           // the wrist view follows its main frame
    sideExpected = false;
    return paint('side', buf, 0);
  }
  const n = dv.getUint32(0);
  let header;
  try { header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n))); }
  catch (e) { return; }
  sideExpected = !!header.side;
  paint('view', buf, 4 + n);
  if (header.hud) drawHud(header.hud);
}

function paint(id, buf, offset) {
  const el = document.getElementById(id);
  if (!el) return;
  const blob = new Blob([new Uint8Array(buf, offset)], {type: 'image/jpeg'});
  createImageBitmap(blob).then(bmp => {
    const ctx = el.getContext('2d');
    if (el.width !== bmp.width) { el.width = bmp.width; el.height = bmp.height; }
    ctx.drawImage(bmp, 0, 0);
    bmp.close();
    if (id === 'side') el.style.display = 'block';
  }).catch(() => {});
}

function onJSON(m) {
  if (m.t === 'pong') {
    state.latency = Math.round(performance.now() - m.ts);
    document.getElementById('latency').textContent = state.latency + ' ms';
    return;
  }
  if (m.t === 'error') return fatal(m.message || 'Technical problem.');
  if (m.t !== 'screen') return;
  // Echoed back with the answer so the server can discard stale or duplicate
  // responses (a reconnect replays the current screen).
  state.screenSeq = m.seq ?? null;
  state.screen = m.kind;
  state.teleopActive = (m.kind === 'teleop');
  if (m.kind === 'redirect' && state.pendingReturn) {
    state.pendingReturn = false;
    respond('redirect', {returned: true});
    return;
  }
  const r = ({
    consent: renderConsent, instructions: renderInstructions,
    message: renderMessage, teleop: renderTeleop, survey: renderSurvey,
    debrief: renderDebrief, done: renderDone, screenout: renderScreenout,
    redirect: renderRedirect,
  })[m.kind];
  if (r) r(m.payload || {});
}

/* ------------------------------------------------------------------ screens */
function card(title, html) {
  root.innerHTML = `<div class="card"><h1>${esc(title)}</h1>${html}</div>`;
}
function fatal(msg) {
  card('Something went wrong', `<p class="warn">${esc(msg)}</p>
    <p class="muted">Please contact the researchers, quoting this page.</p>`);
}
function banner(msg) {
  let b = document.getElementById('banner');
  if (!b) {
    b = document.createElement('div');
    b.id = 'banner'; b.className = 'warn';
    root.prepend(b);
  }
  b.textContent = msg;
  setTimeout(() => b && b.remove(), 6000);
}

function renderConsent(p) {
  card('Consent to take part', `
    <div class="prose">${esc(p.text)}</div>
    <label class="check"><input type="checkbox" id="c1">
      <span>I have read the information above, I am 18 or older, and I agree to
      take part.</span></label>
    <label class="check"><input type="checkbox" id="c2">
      <span>I understand my session is recorded under the code
      <code>${esc(p.participant)}</code>, and that I may withdraw at any time.</span></label>
    <div class="row">
      <button id="go" disabled>Begin</button>
      <button class="ghost" id="no">I do not wish to take part</button>
    </div>`);
  const go = document.getElementById('go');
  const upd = () => go.disabled = !(document.getElementById('c1').checked &&
                                    document.getElementById('c2').checked);
  document.getElementById('c1').onchange = upd;
  document.getElementById('c2').onchange = upd;
  go.onclick = () => respond('consent', {consent: true, ts: Date.now()});
  document.getElementById('no').onclick = () => respond('consent', {consent: false});
}

function keymapHtml(keymap) {
  return (keymap || []).map(([k, d]) =>
    `<kbd>${esc(k)}</kbd><span>${esc(d)}</span>`).join('');
}

function renderInstructions(p) {
  const body = p.practice ? `
      <p>You will control a simulated robot arm with your keyboard to complete the
      task <b>${esc(p.task)}</b>: pick up the red cube and lift it off the table.</p>
      <p>First you will <b>practise with full control</b> so you can get used to the
      keys. Take your time.</p>`
    : `<p>Now the robot will attempt the task <b>on its own</b>.</p>
      <p>Watch it. When it goes wrong, ${p.hold_to_intervene
        ? 'hold <kbd>TAB</kbd> to take control' : 'press <kbd>TAB</kbd> to take control'}
      and show it what to do${p.hold_to_intervene ? '' : ', then press <kbd>TAB</kbd> again to hand control back'}.</p>
      <p>After each attempt the robot updates from your corrections. There are
      <b>${esc(p.rounds)}</b> attempts, with a few short questions in between.</p>`;
  card(p.practice ? 'How to control the robot' : 'Teaching the robot', `
    ${body}
    <h2 style="margin-top:22px">Controls</h2>
    <div class="keys" style="font-size:13px">${keymapHtml(p.keymap)}</div>
    <p class="muted" style="margin-top:16px">Keep this browser tab focused —
    key presses are only registered while the tab is active.</p>
    <div class="row"><button id="go">${p.practice ? 'Start practice' : "Start"}</button></div>`);
  document.getElementById('go').onclick = () => respond('instructions', {action: 'ok'});
}

function renderMessage(p) {
  card(p.title, `<div class="prose">${esc(p.body || '')}</div>
    <div class="row"><button id="go">${esc(p.cta || 'Continue')}</button></div>`);
  document.getElementById('go').onclick = () => respond('message', {action: 'ok'});
  // Enter should advance a modal, but must not leak into the simulator.
  const onKey = e => {
    if (e.code === 'Enter' && state.screen === 'message') {
      e.preventDefault();
      document.removeEventListener('keydown', onKey);
      respond('message', {action: 'ok'});
    }
  };
  document.addEventListener('keydown', onKey);
}

function renderTeleop(p) {
  if (document.getElementById('view')) return;   // already mounted; keep canvas
  root.innerHTML = `
    <div id="stage">
      <div class="viewwrap">
        <canvas id="view"></canvas>
        <canvas id="side"></canvas>
        <div id="controlBadge">…</div>
      </div>
      <div class="hud">
        <dl id="hudlist"></dl>
        <div class="feedback" id="hudfeedback" style="display:none"></div>
        <h3>Controls</h3>
        <div class="keys">${keymapHtml(p.keymap)}</div>
      </div>
    </div>`;
}

function drawHud(h) {
  const badge = document.getElementById('controlBadge');
  if (!badge) return;
  const you = String(h.controller || '').toUpperCase().startsWith('YOU');
  badge.textContent = you ? 'YOU ARE IN CONTROL' : 'ROBOT IS IN CONTROL';
  badge.style.color = you ? 'var(--human)' : 'var(--robot)';
  const keys = ['round','episode','step','time','gripper','phase'];
  document.getElementById('hudlist').innerHTML = keys
    .filter(k => h[k] !== undefined && h[k] !== '')
    .map(k => `<dt>${esc(k)}</dt><dd>${esc(h[k])}</dd>`).join('');
  const fb = document.getElementById('hudfeedback');
  const txt = [h.feedback, h.status].filter(Boolean).join('\n');
  fb.style.display = txt ? 'block' : 'none';
  fb.innerHTML = txt ? `<span class="${h.status ? 'bignote' : ''}">${esc(txt)}</span>` : '';
}

function renderSurvey(p) {
  state.surveyStart = performance.now();
  const items = (p.items || []).map((it, i) => {
    const opts = Array.from({length: it.scale || 7}, (_, k) => {
      const v = k + 1, id = `q${i}_${v}`;
      return `<input type="radio" name="q${i}" id="${id}" value="${v}">
              <label for="${id}">${v}</label>`;
    }).join('');
    return `<div class="item" data-key="${esc(it.key)}" data-idx="${i}">
      <div class="prompt">${esc(it.prompt)}</div>
      <div class="scale"><span class="end">${esc(it.low)}</span>
        <span class="opts">${opts}</span>
        <span class="end">${esc(it.high)}</span></div></div>`;
  }).join('');
  root.innerHTML = `<div class="card"><h1>${esc(p.header)}</h1>
    <p class="muted">There are no right or wrong answers.</p>
    ${items}
    <div class="row"><button id="go" disabled>Continue</button>
      <span class="muted" id="left"></span></div></div>`;

  const n = (p.items || []).length;
  const go = document.getElementById('go');
  const upd = () => {
    const done = root.querySelectorAll('.item input:checked').length;
    go.disabled = done < n;
    document.getElementById('left').textContent =
      done < n ? `${n - done} remaining` : '';
  };
  root.querySelectorAll('input[type=radio]').forEach(r => r.onchange = upd);
  upd();
  go.onclick = () => {
    const data = {elapsed: (performance.now() - state.surveyStart) / 1000};
    root.querySelectorAll('.item').forEach(el => {
      const sel = el.querySelector('input:checked');
      if (sel) data[el.dataset.key] = Number(sel.value);
    });
    respond('survey', data);
  };
}

function renderDebrief(p) {
  card(p.title, `<div class="prose">${esc(p.body)}</div>
    <h2 style="margin-top:22px">Anything you'd like to tell us?</h2>
    <textarea id="comments" placeholder="Optional"></textarea>
    <label class="check"><input type="checkbox" id="wd">
      <span>Please <b>delete my data</b>. I do not want it used in the research.</span></label>
    <div class="row"><button id="go">I understand — finish</button></div>`);
  document.getElementById('go').onclick = () => respond('debrief', {
    acknowledged: true,
    withdraw: document.getElementById('wd').checked,
    comments: document.getElementById('comments').value.slice(0, 2000),
  });
}

function renderDone(p) {
  try { sessionStorage.removeItem('dagger_sid'); } catch (e) {}
  card(p.title || 'Thank you', `
    <div class="prose">${esc(p.body || '')}</div>
    ${p.code ? `<p style="margin-top:18px">Your completion code:</p>
      <p style="font-size:26px;font-weight:700;letter-spacing:.06em">
      <code>${esc(p.code)}</code></p>` : ''}
    ${p.redirect ? `<div class="row">
      <button onclick="location.href='${esc(p.redirect)}'">Return to Prolific</button>
      </div>` : ''}
    <p class="muted" style="margin-top:18px">You may now close this tab.</p>`);
  respond('done', {ack: true});
}

function renderRedirect(p) {
  // Qualtrics serves X-Frame-Options: SAMEORIGIN, so its surveys cannot be
  // shown in an iframe; the whole tab has to go. The session id stays in
  // sessionStorage so the return trip resumes rather than restarting.
  card(p.title || 'Next: a short questionnaire', `
    <div class="prose">${esc(p.body || '')}</div>
    <div class="row"><button id="go">${esc(p.cta || 'Continue')}</button></div>
    <p class="muted" style="margin-top:14px">Keep this tab open — do not close
    it. You will come back here automatically.</p>`);
  document.getElementById('go').onclick = () => { location.href = p.url; };
}

function renderScreenout(p) {
  try { sessionStorage.removeItem('dagger_sid'); } catch (e) {}
  card(p.title || 'Thank you', `<div class="prose">${esc(p.body || '')}</div>
    ${p.code ? `<p style="margin-top:16px">Code: <code>${esc(p.code)}</code></p>` : ''}`);
  respond('screenout', {ack: true});
}

function respond(screen, data) {
  send({t: 'response', screen, data, seq: state.screenSeq});
}

/* ------------------------------------------------------------------ input */
document.addEventListener('keydown', e => {
  if (!state.teleopActive) return;
  const k = KEYMAP[e.code];
  if (!k) return;
  if (SWALLOW.has(e.code)) e.preventDefault();
  if (!state.held.has(k)) state.pending.push(k);   // edge press, sent once
  state.held.add(k);
  flush(true);
});
document.addEventListener('keyup', e => {
  const k = KEYMAP[e.code];
  if (!k) return;
  if (state.teleopActive && SWALLOW.has(e.code)) e.preventDefault();
  state.held.delete(k);
  flush(true);
});

// A blurred window keeps stale keys "held" forever, so clear them and tell the
// server — otherwise a tab switch looks like a participant holding a direction.
function clearKeys() {
  if (state.held.size) { state.held.clear(); flush(true); }
}
addEventListener('blur', () => { clearKeys(); send({t: 'focus', focused: false}); });
addEventListener('focus', () => send({t: 'focus', focused: true}));
document.addEventListener('visibilitychange', () => {
  const vis = document.visibilityState === 'visible';
  if (!vis) clearKeys();
  send({t: 'focus', focused: vis});
});

function flush(force) {
  const now = performance.now();
  if (!force && now - state.lastSent < 1000 / INPUT_HZ) return;
  state.lastSent = now;
  send({t: 'input', keys: [...state.held], press: state.pending, seq: ++state.seq});
  state.pending = [];
}
// Steady heartbeat so held keys keep arriving even with no new browser events.
setInterval(() => { if (state.teleopActive) flush(false); }, 1000 / INPUT_HZ);

boot();
