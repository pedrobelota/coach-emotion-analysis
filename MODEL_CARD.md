# System / model card

## Identity

Coach Emotion Analysis 0.2.0 is a research inference system assembled from pretrained
models and hand-authored domain rules. It is not a newly trained model. The public
release distributes code only.

## Components

| Stage | Source | Role |
| --- | --- | --- |
| ASR | openai/whisper-large-v3-turbo | Portuguese transcription and timing |
| ASR runtime | faster-whisper / CTranslate2, or Transformers fallback | Local inference |
| SER | ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition | Acoustic class scores |
| Fusion | Portuguese football vocabulary and explicit weights | Temporal research labels |

The SER revision is pinned to `b520c9c46a719e36e1b9a91cad2cb5d0668757d8`, the checkpoint
used during the verified local runs. The adapter checks for missing, unexpected and
mismatched checkpoint weights; loading an untrained replacement head is an error.
Whisper model aliases are not immutable revisions; record the resolved snapshot when
conducting a new experiment. Upstream architecture and model terms remain applicable.

The browser demo substitutes Whisper tiny for the research ASR model and runs on
CPU. No evaluation result from the historical pilot is attributed to that configuration.

## Language and domain

Whisper is configured for Portuguese. The SER upstream card describes fine-tuning
on RAVDESS acted English speech. Portuguese spontaneous speech on a football sideline
is a substantial domain shift. Multilingual pretraining does not eliminate that shift.
Source: [upstream SER model card](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition).

## Intended use

Inspection of a research pipeline, reproducible software experiments, and supervised
error analysis on appropriately authorized recordings. Outputs are hypotheses for
researchers to review against human annotation, not evidence of a person's feelings.

## Outputs and failure behavior

Seven final codes distinguish neutral without/with intervention, joy, surprise,
anger, sadness and fear. Acoustic disgust is mapped into the anger-related evidence;
calm and neutral are collapsed. The fusion system can override acoustic predictions.
Reported confidence is a raw model score; it is not a calibrated posterior for the
final fused code. Failed inference exits nonzero. Failed SER windows no longer become
synthetic neutral observations.

## Evaluation

The archived pilot evaluated 107 labeled intervals: 51.40% exact accuracy, 73.83%
collapsed accuracy and 0.172 macro F1. These values belong to the historical
experiment and are not a fresh evaluation of every migrated behavior. See
[results](docs/results.md). The release smoke test demonstrates executable inference
only. No performance claim is made for unseen speakers or recordings.

## Risks and limitations

Severe imbalance, tiny class supports, correlated adjacent intervals, ASR mistakes,
heuristic temporal interpolation, noisy audio, overlapping voices and language shift
limit reliability. There is no diarization, demographic fairness validation or
validated mental-state inference. Human labels are themselves uncertain observations.
Do not use the outputs for personnel decisions, discipline, medical diagnosis or
other consequential automated decisions about people.

## Training, privacy and licensing

This migration performs no training or fine-tuning. No private recording or annotation
is distributed. Inference runs locally after model download; debug output and raw JSON
can contain transcript text and should be protected. No general reuse license is
granted for this research code pending ownership confirmation. Upstream model licenses
and dataset conditions must be reviewed independently.
