# Evidence and authorship

This PIBIC project combines an existing Whisper ASR checkpoint, an existing
Wav2Vec2 emotion checkpoint and domain-specific fusion rules. Its contribution is
the integration, temporal processing, evaluation and software delivery, not a new
foundation model. AI assistants supported migration, review, tests and documentation;
the author remains responsible for explaining and validating the work.

| Claim | Inspectable evidence | Boundary |
| --- | --- | --- |
| Audio-to-table execution | CLI, pipeline, local smoke report in docs/security-review.md | Functional execution does not measure emotion accuracy |
| Research evaluation | Evaluator, regression tests, per-class metrics and neutral baselines | Private pilot data cannot be independently reproduced from the public repo |
| Reproducible engineering | CI on Linux/Windows, wheel/sdist and Docker build | Unit tests use synthetic data or stubs; native checkpoints are tested separately |
| Privacy-aware publication | Data card, explicit Space export, secret scans | Scanners are bounded checks, not a privacy certification |
| Deployable demonstration | HTTP API, UI assets and Docker Space recipe | No public Hugging Face deployment is claimed |

Historical macro F1 remains 0.172; collapsed accuracy 73.83% is below the 84.11%
all-neutral baseline on the same slice. Do not describe this as a validated emotion
detector or transfer its scores to arbitrary audio. The strongest next research
evidence is independent recordings, annotation agreement and component ablations.

The code does not implement RAG, an agent system or LLM report generation. Those
are potential extensions, not current accomplishments. The companion Android
project demonstrates direct LLM integration. Neither project guarantees a hiring
outcome; claims should be supported with code and evidence in a technical interview.

Review scope: source/documentation consistency, public checks and publication hygiene.
The original research files and contributors' attribution remain preserved. No new
project-wide license has been assigned without ownership confirmation.
