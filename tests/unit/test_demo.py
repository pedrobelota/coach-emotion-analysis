import csv
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from coach_emotion.demo import _run_command, analyze_upload, validate_upload


def create_wav(path, seconds=4):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * 16000 * seconds)
    return path


def test_consent_precedes_file_access():
    with pytest.raises(ValueError, match="authorized"):
        validate_upload("nonexistent.wav", False)


@pytest.mark.parametrize("seconds", [1, 31])
def test_demo_rejects_out_of_bounds_duration(tmp_path, seconds):
    path = create_wav(tmp_path / "input.wav", seconds)
    with pytest.raises(ValueError, match="4 and 30"):
        validate_upload(str(path), True)


def test_demo_rejects_oversized_input_before_decoding(tmp_path):
    path = tmp_path / "input.wav"
    with path.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="8 MiB"):
        validate_upload(str(path), True)


def test_demo_rejects_malformed_wav(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a recording")
    with pytest.raises(ValueError, match="PCM WAV"):
        validate_upload(str(path), True)


def test_demo_returns_only_predictions_and_removes_outputs(tmp_path):
    path = create_wav(tmp_path / "input.wav")
    produced = []

    def infer(command, **kwargs):
        assert "--windows" not in command
        assert command[command.index("--device") + 1] == "cpu"
        assert command[command.index("--model") + 1] == "tiny"
        assert kwargs["timeout"] == 300
        output = Path(command[command.index("--output") + 1])
        produced.append(output)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["inicio_s", "fim_s", "codigo_emocao", "rotulo_modelo"])
            writer.writerow([0, 4, 0, "neutro sem intervencao"])
        return subprocess.CompletedProcess(command, 0, b"private transcript", b"")

    with patch("coach_emotion.demo._run_command", side_effect=infer):
        rows, status = analyze_upload(str(path), True)
    assert rows == [[0.0, 4.0, 0, "neutro sem intervencao"]]
    assert "private transcript" not in status
    assert not produced[0].parent.exists()


def test_demo_failure_does_not_expose_child_diagnostics(tmp_path):
    path = create_wav(tmp_path / "input.wav")
    failed = subprocess.CompletedProcess([], 1, b"sensitive", b"sensitive")
    with patch("coach_emotion.demo._run_command", return_value=failed):
        with pytest.raises(RuntimeError) as exc:
            analyze_upload(str(path), True)
    assert "sensitive" not in str(exc.value)


def test_owned_process_is_terminated_on_timeout():
    import os

    with pytest.raises(subprocess.TimeoutExpired):
        _run_command(
            [sys.executable, "-c", "import time; time.sleep(60)"], timeout=1, env=dict(os.environ)
        )
