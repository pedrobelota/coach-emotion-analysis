# Methodology and implementation

## Data flow

FFmpeg extracts the selected media range and normalizes it to one channel at 16 kHz.
Whisper transcribes the clip in Portuguese. faster-whisper requests word timestamps;
the Transformers fallback may provide segment timestamps. When exact word timing is
unavailable, legacy proportional text subdivision is only an approximation.

The pipeline examines four-second windows with a two-second hop. A tail-aligned
window is added if the last regular window would leave audio uncovered. Clips shorter
than the window are rejected. This tail behavior is a release correction and can
change predictions near a nonaligned ending compared with the original prototype.

The legacy SER adapter preserves `classifier.dense` and `classifier.output` weights.
Checkpoint loading rejects missing, unexpected and mismatched weights. Each window
receives the top acoustic label and its softmax score. The score is uncalibrated.
Whisper is released before SER loading. CPU fallback is available, but hardware and
native library compatibility still determine which backend can run.

The domain vocabulary scores the aligned transcription. The five-second generator
combines textual, contextual and acoustic evidence with explicit thresholds and
class-dependent weights. Strong text evidence can override acoustic labels; otherwise
neutral remains a frequent fallback. These are heuristics retained from the research
prototype, not newly learned parameters or a calibrated posterior.

## Two evaluation levels

`coach-emotion evaluate` measures the **final** 0–6 table. Overlapping final rows and
invalid/nonintegral codes are errors. Every labeled human interval must overlap at
least one prediction; otherwise evaluation fails instead of silently selecting only
covered intervals. If multiple nonoverlapping prediction rows intersect an annotation,
each receives one vote; votes are not weighted by overlap duration. Matching
five-second boundaries are preferred. The full duration of each labeled interval must
be covered, including its start/end and internal gaps. An overlap of a few milliseconds
cannot count as coverage for a five-second annotation. Duplicate, overlapping and
out-of-order ground-truth intervals are rejected.

The public final-code evaluator requires the compact three-column annotation schema;
legacy four-column labels must use the raw-window evaluator. Its numeric codes and
text labels can follow a different historical mapping and must not be conflated.
Reports include per-class precision/recall/F1, a confusion matrix (rows: truth;
columns: prediction), its explicit class order, and always-neutral baselines.

`coach_emotion.evaluation` and `coach_emotion.evaluate_audio` retain the original
raw-window evaluator for compatibility. Its report explicitly counts missing
predictions and excluded labels. Its scores must not be conflated with the final
dataset metrics. The audio wrapper restores source-time offsets after FFmpeg crops.

Final macro F1 averages the union of classes observed in labels/predictions after
collapsing codes 0 and 1. It is not averaged over absent classes in a small sample;
reports with different class sets are not directly comparable.

## Package boundaries

| Module | Responsibility |
| --- | --- |
| `cli` | Validated public commands and output collision checks |
| `runtime` | Immutable per-call JSON/CLI settings through a context variable |
| `analyze` | Media preparation and final dataset export |
| `pipeline` | Lazy model imports, transcription, fallback and window inference |
| `emotion_recognition` | Legacy checkpoint architecture and loading checks |
| `symbolic_rules`, `dataset` | Domain evidence and fusion |
| `evaluate`, `evaluation`, `evaluate_audio` | Final and legacy evaluation contracts |
| `benchmark` | Retained local Whisper timing utility |

The inference module remains relatively large to limit risky rewrites during the
migration. It is designed for sequential local CLI use, not concurrent production
serving. The CUDA-failure flag persists within a process. A service would require
explicit model lifecycle management and concurrency testing.

## Research improvements still needed

Use more independent recordings, speaker/recording-level held-out splits, class
balancing and stronger annotation analysis. Compare neutral baselines, text-only,
audio-only and combined systems. Keep tuning separate from final evaluation. The
archived metrics alone cannot certify that historical tuning used fully independent
data; this repository does not make that stronger claim.
