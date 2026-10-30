# Audio demo and Hugging Face Space

Publish this as an **application Space**, not as a newly trained model. It combines
upstream Whisper and SER checkpoints; there is no original fine-tuned weight artifact
to upload to the model Hub. Keep upstream attribution and research limitations visible.

## Local demo

Install FFmpeg and Python 3.11–3.13 in an isolated environment, then:

```bash
python -m pip install --upgrade pip==26.2.2 setuptools==84.2.1
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[inference,demo]"
coach-emotion-demo
```

Visit http://127.0.0.1:7860. The synthetic button shows invented schema examples and
does not invoke models. Real analysis accepts PCM WAV, 4–30 seconds, mono/stereo,
8–48 kHz and at most 8 MiB. It uses Whisper tiny and the same pinned SER checkpoint.
Cold inference downloads model files and may exceed the five-minute request timeout;
retry after the model cache has populated. The large acoustic model still needs
substantial RAM and disk even though Whisper is small.

The API takes a raw WAV body at `POST /api/analyze?consent=true`, with
`Content-Type: audio/wav`. It returns JSON rows and a research-status message.
Uploads and generated output files are temporary; provider infrastructure has its
own policies. Do not use identifiable or sensitive research recordings on a public
Space. The source default binds localhost; the container listens on port 7860.

## Prepare only the publishable source

From the repository root:

```bash
python scripts/prepare_space.py --output outputs/hf-space
```

Choose a new directory if it exists. The script copies only package source, static
assets, `pyproject.toml`, the Space README and Dockerfile. It does not copy recordings,
annotations, outputs, logs, Git history, environments, caches or credentials. It
neither authenticates nor publishes. Never upload the entire working directory.

Optional local container check, with Docker Engine running:

```bash
docker build -t coach-emotion-demo outputs/hf-space
docker run --rm -p 127.0.0.1:7860:7860 coach-emotion-demo
```

The CI container job validates the build and HTTP health/asset routes without
downloading checkpoints. Full model inference is a separate local smoke test.

## Publish after account setup

1. Confirm code ownership and institutional publication rights, and review upstream
   model/data licenses. No project-wide reuse license has been assigned automatically.
2. Check current [Spaces eligibility and pricing](https://huggingface.co/docs/hub/spaces-overview).
   At preparation time (October 2026), creating a compute Space such as Docker requires
   an eligible paid plan. Do not assume CPU hosting creation is free. Static Spaces
   cannot run this Python inference server. Eligible ZeroGPU options use a different
   Gradio/GPU design; this Docker recipe is not a ZeroGPU deployment.
3. Create a Docker Space in your account with suitable hardware and review any costs
   yourself. Follow the [official Docker Space guide](https://huggingface.co/docs/hub/spaces-sdks-docker).
4. Authenticate locally using `hf auth login`. Keep the token out of chat, Git and
   configuration files; use a token scoped to that destination repository.
5. Upload only the prepared folder, replacing `YOUR_ACCOUNT/YOUR_SPACE` below:

```bash
hf upload YOUR_ACCOUNT/YOUR_SPACE outputs/hf-space . --repo-type space
```

6. Wait for a successful build, test the synthetic example, then test a recording
   you own and may share. Confirm the returned rows, limits and error messages.
   Only then add the verified Space link to the GitHub README and portfolio.

No account login, subscription or Hugging Face publication is performed by the
export script. A model-Hub release is appropriate later only if there is an original
trained artifact with compatible rights, a model card and independently evaluated results.
