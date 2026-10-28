# Data card

## Distributed data

The repository contains only invented tabular examples and code to generate a silent
WAV. They contain no real speakers, transcriptions or research annotations and have
no value for estimating model quality. Example labels intentionally contain prediction
errors to demonstrate the evaluator.

## Private pilot corpus (not distributed)

The archived evaluation describes 550 seconds, split into 110 intervals of five
seconds. There are 107 labeled intervals and three null labels. Label counts are
62 neutral-without-intervention, 28 neutral-with-intervention, 1 joy, 2 surprise,
7 anger, 5 sadness and 2 fear. These aggregate counts are published solely to explain
the reported metrics and imbalance.

Research documentation describes independent human annotation followed by majority
vote. This release does not redistribute the raw forms, identities, recordings or
individual annotations, and does not independently certify consent or agreement
statistics. Demographic coverage and a validated representative sampling frame are
not established by this repository.

## Schema and time origin

The final prediction table has explicit `inicio_s`, `fim_s` and `codigo_emocao`
columns. The historical compact ground-truth format uses three headerless fields:
interval text, coarse sentiment, and emotion code. A legacy four-column annotated
format is also parsed internally. See [examples](examples/README.md).

Analysis timestamps begin at zero for the selected clip, including after cropping.
Annotation timestamps must match that origin for final-table evaluation. Null labels
are excluded; missing predictions for labeled intervals are an error in the public CLI.

## Access, protection and retention

Users must supply recordings they are entitled to process. Research audio, video,
private annotations, transcript JSON, local logs and model caches are excluded from
version control. There is no public download link for the private pilot corpus.
Do not upload individual recordings in issues or pull requests. Keep any institutional
retention and participant-consent arrangements outside this public repository.

## Validity

Intervals from one recording are correlated. Neutral classes dominate, and minority
supports are too small for reliable comparisons. Future splits should be made by
recording/speaker before tuning; adjacent intervals must not be randomly scattered
across train and test. Public examples cannot reproduce the private pilot metrics.
