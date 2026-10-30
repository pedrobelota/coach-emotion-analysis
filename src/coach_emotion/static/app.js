const input = document.querySelector('#audio');
const consent = document.querySelector('#consent');
const analyze = document.querySelector('#analyze');
const example = document.querySelector('#example');
const status = document.querySelector('#status');
const mode = document.querySelector('#mode');
const tbody = document.querySelector('#rows');
const timeline = document.querySelector('#timeline');

function render(data) {
  tbody.replaceChildren(); timeline.replaceChildren();
  for (const [start, end, code, label] of data.rows) {
    const row = document.createElement('tr');
    for (const value of [`${start.toFixed(1)} s`, `${end.toFixed(1)} s`, code, label]) {
      const cell = document.createElement('td'); cell.textContent = value; row.append(cell);
    }
    tbody.append(row);
    const segment = document.createElement('div');
    segment.className = `interval code-${code}`;
    segment.textContent = `${start}–${end}s · ${label}`;
    timeline.append(segment);
  }
  mode.textContent = data.synthetic ? 'Synthetic · no inference' : 'Model output';
  status.textContent = data.message;
}
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Unable to process this request.');
  render(data);
}
async function perform(action) {
  analyze.disabled = example.disabled = true;
  try { await action(); }
  catch (error) { status.textContent = error.message; mode.textContent = 'Request not completed'; }
  finally { analyze.disabled = example.disabled = false; }
}
example.addEventListener('click', () => perform(() => request('/api/example')));
analyze.addEventListener('click', () => perform(async () => {
  const file = input.files[0];
  if (!file) throw new Error('Choose a PCM WAV recording first.');
  if (!consent.checked) throw new Error('Confirm that you are authorized to process this recording.');
  if (file.size > 8 * 1024 * 1024) throw new Error('The recording exceeds 8 MiB.');
  if (!file.name.toLowerCase().endsWith('.wav')) throw new Error('Choose a PCM WAV file.');
  mode.textContent = 'Processing'; status.textContent = 'Processing locally on this server. First use downloads models; please allow several minutes.';
  await request('/api/analyze?consent=true', {method:'POST', headers:{'Content-Type':'audio/wav'}, body:file});
}));
