# Release validation

Validated locally on 2026-10-28. Software checks establish that the package executes;
they do not establish emotion-classification accuracy.

## Environment

| Item | Local value |
| --- | --- |
| OS | Windows 10 Pro, build 19045 |
| Python | 3.13.7 |
| CPU | Intel Core i3-9100F, 3.60 GHz |
| RAM | 16 GB |
| GPU | NVIDIA GeForce GTX 1050 Ti, 4 GB VRAM |
| FFmpeg | 8.0 |
| PyTorch | 2.7.1+cu118 |
| Transformers | 4.57.0 |
| faster-whisper / CTranslate2 | 1.2.0 / 4.6.0 |
| librosa | 0.11.0 |

Two environments were used. Core/development dependencies were installed into a new
isolated virtual environment. The neural test used a separate virtual environment
with access to already-installed inference libraries and existing local model caches.
It is not evidence of a clean, from-scratch installation of the full neural stack.

The inference environment used NumPy 2.3.3, pandas 2.3.2, SciPy 1.16.2 and
scikit-learn 1.7.2. The fresh core environment resolved NumPy 2.5.4, pandas 2.3.3,
SciPy 1.18.1 and scikit-learn 1.9.1. Tested direct inference versions are recorded
in `constraints-inference.txt`; this is not a complete transitive lock file.

## Automated checks

| Check | Result |
| --- | --- |
| Default unit suite, without neural dependencies | 59 passed |
| FFmpeg integration with synthetic stereo input | Passed: cropped mono 16 kHz output |
| Legacy SER adapter, tiny random model, offline | Passed: expected checkpoint-head parameter names |
| Ruff lint and format | Passed |
| Isolated sdist and wheel build | Passed |
| Wheel installation and CLI | Passed |
| Synthetic CLI evaluation | 4 intervals, 50% exact / 75% collapsed accuracy, as constructed |
| Core dependency consistency (`pip check`) | Passed |

The 59-test suite completed in approximately four seconds after dependencies were
cached. Both integration tests passed together in 15.80 seconds. The initial cold
environment was slower. The compatibility integration checks parameter names; actual
pretrained checkpoint loading is a separate smoke-test concern.

CI runs the lightweight suite, lint, formatting, packaging, synthetic CLI example and
publication audit on Linux (Python 3.11/3.13) and Windows (Python 3.13). The media test
skips if FFmpeg is unavailable. CI does not download neural checkpoints. See the
[workflow runs](https://github.com/pedrobelota/coach-emotion-analysis/actions/workflows/ci.yml)
for the status of each commit.

## Local end-to-end smoke test

The release smoke test used a 30-second local research excerpt, with network model
access disabled (`--offline`). The recording, transcript, predictions and console
diagnostics remain in ignored local output directories and are not published.

The automatic-device run completed successfully in **753.64 seconds (12.56 minutes)**,
including dependency/model loading, media conversion and inference. Whisper attempted
CUDA and fell back to CPU int8; SER loaded its complete checkpoint on the GPU in
float16. The run processed **14 windows with zero SER failures** and produced six
five-second intervals. It wrote `predictions.tsv`, `details.tsv`, `windows.json`,
`console.log` and an environment/timing `summary.json`, all outside version control.

A second run with **`--device cpu --offline` completed in 82.78 seconds**, using
Whisper CPU int8 and SER CPU float32. It also produced 14 windows, zero SER failures
and six final intervals. The files were already cached by the first run. These two
wall times are not a controlled GPU-versus-CPU comparison and must not be used to
claim a speedup. They verify both the automatic-fallback path and fully CPU execution.

This run is a functional check, not a latency benchmark or an accuracy evaluation.
Dependency imports and checkpoint reads were slow on the local machine; an earlier
attempt was interrupted during loading. The VAD retained only about 0.83 seconds of
the 30-second excerpt, and all six final predictions were neutral without intervention.
That further limits what can be inferred about transcription or classification quality
from this one smoke test. No ground truth was needed to execute analysis.

## Corrections validated during migration

- Imports no longer create logs, reconfigure root logging or load neural runtimes.
- JSON/CLI settings control model, backend, device, timing and batching.
- Output/input collisions and output overwrites are rejected.
- Invalid window, interval, crop and emotion-code values fail explicitly.
- Evaluation rejects missing predictions and overlapping final prediction rows.
- Fractional human emotion codes cannot be silently truncated into valid labels.
- Failed SER windows abort instead of generating fabricated neutral observations.
- Nonaligned audio tails receive an additional window.
- Cropped raw-window evaluation restores the annotation time origin.
- Models are loaded sequentially; the caller releases its Whisper reference before SER.

The original research folder is retained in place. The public project uses its own Git
root and includes only curated code, invented examples and aggregate metrics. Candidate
files and all historical Git blobs are inspected for excluded artifacts, personal
paths, unexpectedly large files and common secret signatures before publication.
This is a bounded hygiene audit, not a formal privacy or security certification.

Historical accuracy figures remain documented separately in [results](results.md).

The subsequent release 0.2 audit and dependency/demo work are documented separately
in [security and quality review](security-review.md). The timings and dependency
versions above remain historical records of the initial migration.
