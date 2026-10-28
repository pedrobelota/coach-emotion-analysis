"""Real FFmpeg conversion on generated signal, with no human data or models."""

import shutil
import wave

import pytest

from coach_emotion.analyze import obter_duracao_wav, preparar_midia_para_dataset

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_ffmpeg_normalizes_and_crops_synthetic_audio(tmp_path):
    source = tmp_path / "input.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setparams((2, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0" * 8000 * 2 * 2 * 7)
    target = tmp_path / "output.wav"
    preparar_midia_para_dataset(source, target, inicio_s=1.0, fim_s=6.0)
    assert obter_duracao_wav(target) == 5.0
    with wave.open(str(target), "rb") as audio:
        assert audio.getnchannels() == 1
        assert audio.getframerate() == 16000
