# Historical research results

The numbers below were transcribed from the archived final evaluation summary supplied
with the original research project. They describe one 550-second pilot slice and have
not been recomputed as a new full benchmark during the software migration.

| Measure | Value |
| --- | ---: |
| Labeled intervals | 107 |
| Exact correct codes | 55 |
| Exact accuracy | 51.401869% |
| Wilson 95% interval | 42.048589%–60.657979% |
| Collapsed-neutral correct classes | 79 |
| Collapsed-neutral accuracy | 73.831776% |
| Collapsed macro precision | 0.161111 |
| Collapsed macro recall | 0.227778 |
| Collapsed macro F1 | 0.172222 |

Three additional intervals have null human labels and were excluded. The historical
record reports no missing overlapping prediction among these 107 labeled intervals.

| Code | Meaning | Human support | Predicted support |
| --- | --- | ---: | ---: |
| 0 | Neutral without intervention | 62 | 87 |
| 1 | Neutral with intervention | 28 | 3 |
| 2 | Joy | 1 | 3 |
| 3 | Surprise | 2 | 0 |
| 4 | Anger | 7 | 2 |
| 5 | Sadness | 5 | 2 |
| 6 | Fear | 2 | 10 |

## Baselines and interpretation

Always predicting code 0 yields 62/107 = **57.94% exact accuracy**, higher than this
pipeline's 51.40%. After collapsing neutral codes, always predicting neutral yields
90/107 = **84.11% accuracy**, higher than 73.83%. This shows why execution success
and aggregated accuracy do not establish model usefulness.

The reported macro F1 is low, and the model overpredicts fear in this slice. Several
classes have only one or two examples. More data, separate tuning and evaluation,
and explicit baselines are necessary before claiming generalization or improvement.
The Wilson interval treats observations as binomial trials and does not account for
temporal correlation; interpret it as descriptive only.

## Release testing is separate

Unit tests check contracts and regression fixes; synthetic examples check the evaluator;
real model smoke tests check execution. None supplies an independent emotion-quality
benchmark. The release corrects failures, tail coverage and timestamp handling, so
these archived metrics are historical evidence rather than a guarantee of unchanged
output on every input. See [validation](validation.md).
