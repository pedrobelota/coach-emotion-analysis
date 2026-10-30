FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1
RUN useradd --create-home --uid 1000 --home-dir /opt/runner runner
WORKDIR /opt/app
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.2 setuptools==84.2.1
RUN python -m pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY --chown=runner:runner . /opt/app
RUN python -m pip install --no-cache-dir ".[inference,demo]" && python -m pip check
ENV HF_HOME=/opt/runner/.cache/huggingface HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
USER runner
EXPOSE 7860
CMD ["python", "-m", "coach_emotion.server", "--host", "0.0.0.0", "--port", "7860"]
