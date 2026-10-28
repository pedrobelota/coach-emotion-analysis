# Synthetic examples

These four intervals are invented schema examples, not observations from a person or
model inference. They intentionally contain errors: exact accuracy is 2/4, and
collapsed-neutral accuracy is 3/4. Never present these numbers as research results.

The compact ground truth has three **headerless**, tab-separated columns:
`interval`, `sentiment group`, `emotion code`. Times use `minutes:seconds`.
Prediction files have a header and explicit second offsets.

Run `coach-emotion evaluate --predictions examples/synthetic_predictions.csv
--ground-truth examples/synthetic_ground_truth.tsv` on one line.

No speech recording or transcribed private utterance is included. You can supply
your own local recording, or generate silence with `scripts/generate_demo_audio.py`
to check decoding. Silence cannot measure transcription accuracy or emotional quality.
