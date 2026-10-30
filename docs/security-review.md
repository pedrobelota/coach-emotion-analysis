# Security and quality review — 2026-10-30

## Scope and privacy findings

The review covered the public main history beginning at release commit `7bea555`,
publishable source files, evaluator edge cases, runtime dependencies and the new
demo HTTP boundary. Gitleaks 8.30.1 found **no matched secrets** in the five-commit
published history. File/history inspection found no research recordings, private
annotation tables, transcripts, model weights or personal absolute paths. Commit
identities use the GitHub noreply address. These bounded checks do not prove that
every possible sensitive value or vulnerability is absent.

The local working directory also contains ignored research outputs and isolated
environments. They are not publication candidates. `scripts/prepare_space.py`
exports an explicit source allowlist; it must be used instead of uploading the
working directory wholesale. Do not publish raw scanner logs or experiment outputs.

## Confirmed issues and fixes

| Finding | Consequence | Release 0.2 correction |
| --- | --- | --- |
| Any positive overlap counted as prediction coverage | A prediction lasting 0.001 s could score a fully labeled 5 s interval | Require complete coverage of each labeled interval; reject gaps |
| Duplicate or overlapping ground truth was accepted | Labels could be counted twice | Reject overlap, duplication and out-of-order intervals |
| Blank interval cells became the string `nan` | Valid compact annotation rows failed to load | Reconstruct blank intervals from the five-second schema |
| Legacy four-column labels could enter the final-code evaluator | Different coding conventions could be mixed | Require compact three-column final-code annotations; keep the legacy evaluator separate |
| Only aggregate scores were returned | Poor minority-class performance was easy to miss | Add neutral baselines, per-class precision/recall/F1 and a confusion matrix |
| Old inference pins had known dependency advisories | Vulnerable library code was included in installations | Upgrade and validate the current CPU runtime |
| File decoding relied on a removed PyAV keyword | New PyAV could trigger a slow ASR fallback | Pass normalized audio samples to faster-whisper |
| CTranslate2 4.6 used removed `pkg_resources` | Inference failed after updating setuptools | Update CTranslate2 to 4.8.4 and faster-whisper to 1.2.1 |
| Source distribution omitted supporting material | An sdist lacked tests, configs and documentation | Add an explicit manifest; package browser assets in the wheel |

The old torch/Transformers direct pins produced 23 advisory records (19 unique IDs)
across two packages. This is a dependency finding, not evidence that the project
was exploited or that private data leaked. Some affected features are not used here.
The final installed Windows CPU environment was enumerated and checked with
pip-audit 2.10.1: **98 third-party packages, zero known advisories returned**.
The local project itself is excluded from the advisory lookup. The CPU local-version
suffix is normalized for the PyPI advisory query. This result is time-bounded;
CI also audits freshly resolved Linux dependencies.

## Demo boundary

The demo accepts only short PCM WAV recordings, requires an explicit authorization
checkbox, limits both declared and streamed upload size, and permits one active
inference plus three queued requests. It runs inference in an isolated child process
with a five-minute timeout, terminates the owned process tree on timeout, returns
only temporal labels, and removes temporary input/output folders after each request.
Model files remain in the local cache; user recordings do not go into that cache.

Client error messages do not include child-process transcripts or tracebacks.
The app uses same-origin scripts, a restrictive content security policy, no-store
responses and no inference access logs. Hosting infrastructure has separate logs
and policies; this is not a production authentication or abuse-prevention system.

The GitHub workflows validate core/API tests on Linux and Windows, model-head
compatibility without downloading weights, wheel/sdist packaging, the Docker Space
build, container HTTP routes, secret detection and dependency advisories. Consult
the checks on the exact commit for their outcome. Real model inference is validated
separately on local authorized audio; no private sample is committed.

Local validation: **83 unit tests and 13 integration tests passed**, including
HTTP rejection/cleanup checks, synthetic FFmpeg input and the random SER head.
The wheel and sdist built successfully, with all three browser assets in the wheel.
The 22-file Space export passed a separate Gitleaks directory scan. Hash comparison
confirmed that all 127 inventoried original research files remained unchanged.

A 30-second authorized local recording completed with the updated CPU runtime in
407.32 seconds (including cold imports/loading): 14 windows, six final intervals,
no SER-window failures, and faster-whisper execution without Transformers fallback.
All six codes were neutral without intervention. The real HTTP inference endpoint
also processed ten seconds of generated silence in 33.5 seconds and returned two
intervals. These checks establish execution, not classification quality or a latency
benchmark. Timings are not comparable across different inputs or cache states.

A subsequent HTTP request with the authorized 30-second recording completed in
27.0 seconds with six final intervals using Whisper tiny and warmed local caches.
Automated browser interaction was unavailable in the local environment; the HTTP
boundary and real model execution were checked directly, not claimed as visual QA.

## What still matters for an AI Engineer portfolio

The historical collapsed macro F1 remains **0.172**. The all-neutral baseline's
84.11% accuracy exceeds the pipeline's 73.83% on the same small private slice.
Software fixes and a working demo do not establish improved scientific accuracy.
The next useful work is a held-out recording split, more minority examples,
annotation-agreement analysis and ablations of acoustic/text/rule contributions.

The system integrates pretrained ASR and SER; it does not yet implement an LLM
application or train a new model. An evidence-grounded LLM report with timestamp
citations and explicit evaluation would be a separate, meaningful GenAI extension.
Public source visibility does not grant an open-source license; ownership and
institutional reuse rights remain to be confirmed before assigning one.
