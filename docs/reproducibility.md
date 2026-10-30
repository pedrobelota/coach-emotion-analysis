# Reproducibility

## Software versus research reproduction

The package, CLI, tests and synthetic examples can be run from a clean clone. The
private pilot metrics cannot be independently reproduced from this repository because
recordings and human labels are intentionally withheld. Historical model aliases,
hardware and native libraries can also affect exact numerical outputs.

Use Python 3.11–3.13. `pyproject.toml` separates lightweight evaluation/development
dependencies from the inference extra. Transformers 5.22.0 and faster-whisper 1.2.1
are pinned to the verified interfaces; the SER revision is pinned in `runtime.py`.
Core ranges allow supported Python versions. They are not a universal lock file.

`constraints-inference.txt` records the tested direct inference versions without
including machine paths or unrelated global packages:

```bash
python -m pip install -e ".[inference]" -c constraints-inference.txt
```

Select the PyTorch CPU/CUDA index separately as described in the README. This is a
direct-dependency reference, not a cryptographic lock of all transitive packages.

For a new experiment, record the commit, configuration, resolved model snapshots,
package versions, FFmpeg version, hardware, native libraries and data identifiers in
private experiment records. The smoke runner produces a sanitized environment/timing
summary while keeping transcripts and console diagnostics in ignored local outputs.

## CPU

Install a compatible CPU PyTorch build before the inference extra and pass `--device
cpu`. faster-whisper uses int8 by default. `configs/cpu-demo.json` selects Whisper
tiny to reduce computation; its outputs are not equivalent to the research model.
The acoustic model remains large even with a small Whisper configuration.

## GPU

PyTorch and CTranslate2 may require different CUDA runtime libraries. A CUDA-enabled
PyTorch installation alone does not ensure faster-whisper CUDA support. Automatic
fallback preserves an executable path when supported failures occur. The selected
device and actual fallback should be reported with every benchmark.

## Offline tests

The default tests need no model files and import no inference runtimes. Optional
integration tests use synthetic media and a tiny randomly initialized Wav2Vec2
architecture. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to prohibit Hub
requests. A real `scripts/smoke_test.py --offline` run additionally requires complete
local caches for the selected Whisper and SER revisions.

## Known boundaries

GPU/CPU outputs can differ. Model aliases can resolve to newer snapshots. Fine-grained
timing in the Transformers fallback is approximate. Softmax confidence is not calibrated.
This project includes a bounded demo API and a Docker Space recipe, but no production
service or newly trained weights. Hosting requires a separate deployment; see
[Hugging Face](huggingface.md). The current CPU environment and regression results
are documented in the [security and quality review](security-review.md).
