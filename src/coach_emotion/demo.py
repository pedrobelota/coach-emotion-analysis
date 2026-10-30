"""Bounded local/Spaces demo. No model imports or server launch at import time."""

import csv
import logging
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 30
HEADERS = ["Start (s)", "End (s)", "Code", "Research label"]
SYNTHETIC_ROWS = [
    [0, 5, 0, "neutro sem intervencao"],
    [5, 10, 2, "alegria"],
    [10, 15, 4, "raiva"],
    [15, 20, 1, "neutro com intervencao"],
]
logger = logging.getLogger(__name__)


def _run_command(command, *, timeout, env):
    """Terminate the owned process tree on timeout, including Windows venv launchers."""
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=os.name != "nt",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    ) as child:
        try:
            stdout, stderr = child.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(child.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            child.kill()
            child.communicate()
            raise
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)


def validate_upload(filename: str | None, consent: bool) -> Path:
    if consent is not True:
        raise ValueError("Confirm that you are authorized to process this recording.")
    if not filename:
        raise ValueError("Choose a WAV file first.")
    path = Path(filename)
    if path.suffix.lower() != ".wav" or not path.is_file():
        raise ValueError("Use an uncompressed PCM WAV file.")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("The WAV must be no larger than 8 MiB.")
    try:
        with wave.open(str(path), "rb") as audio:
            rate = audio.getframerate()
            if not 8000 <= rate <= 48000 or audio.getnchannels() not in (1, 2):
                raise ValueError("Use mono or stereo audio sampled between 8 and 48 kHz.")
            if not 4 <= audio.getnframes() / rate <= MAX_SECONDS:
                raise ValueError("Use a recording between 4 and 30 seconds long.")
    except (wave.Error, EOFError) as exc:
        raise ValueError("Use an uncompressed PCM WAV file.") from exc
    return path


def analyze_upload(filename: str | None, consent: bool):
    """One isolated CLI process per request; outputs and diagnostics are temporary."""
    source = validate_upload(filename, consent)
    started = time.perf_counter()
    with TemporaryDirectory(prefix="coach_demo_") as folder:
        output = Path(folder) / "predictions.tsv"
        command = [
            sys.executable,
            "-m",
            "coach_emotion",
            "analyze",
            "--input",
            str(source),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--model",
            "tiny",
            "--backend",
            "faster-whisper",
        ]
        try:
            result = _run_command(
                command,
                timeout=300,
                env=dict(os.environ, PYTHONUTF8="1", HF_HUB_DISABLE_TELEMETRY="1"),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "Analysis timed out. Try a shorter clip after the models finish downloading."
            ) from exc
        if result.returncode:
            # Do not expose paths, transcript fragments or tracebacks to the client.
            logger.warning("Demo inference failed with exit code %s", result.returncode)
            raise RuntimeError("Analysis failed. Check the WAV format and model availability.")
        with output.open(encoding="utf-8-sig", newline="") as handle:
            rows = [
                [
                    float(row["inicio_s"]),
                    float(row["fim_s"]),
                    int(row["codigo_emocao"]),
                    row["rotulo_modelo"],
                ]
                for row in csv.DictReader(handle, delimiter="\t")
            ]
    return (
        rows,
        f"Processed in {time.perf_counter() - started:.1f} s. Whisper tiny + acoustic SER, CPU. These are research hypotheses.",
    )
