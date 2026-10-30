---
title: Coach Emotion Analysis
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Research demo for temporal speech emotion analysis
---

This Space integrates pretrained Whisper and Wav2Vec2 models with Portuguese
football-domain rules. It is a **research demo**, not an independently trained model.
The UI uses Whisper tiny on CPU; historical pilot metrics used a different setup.

Only upload authorized, non-sensitive PCM WAV clips (4–30 seconds, up to 8 MiB).
The app emits temporal hypotheses, not measurements of a person's internal state.
It must not be used for consequential decisions about people. Uploads are processed
on the Space server; read the retention notice in the app before submitting audio.

The synthetic walkthrough contains invented rows and performs no inference.
No private research recording, label, transcript, checkpoint or API credential is
included in the Space source. Pretrained models are downloaded from their upstream
repositories at first inference.

See the [GitHub repository](https://github.com/pedrobelota/coach-emotion-analysis)
for architecture, tests, model/data cards and the archived macro F1 of 0.172.
No project-wide reuse license has been assigned pending ownership confirmation.
The model providers' terms remain separate.
