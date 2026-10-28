"""Regression tests for migration defects and public CLI contracts."""

import json
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

from coach_emotion import pipeline
from coach_emotion.cli import main
from coach_emotion.dataset import ConfigDatasetModelo
from coach_emotion.evaluate import _ler_dataset_final, avaliar_dataset_final
from coach_emotion.runtime import RuntimeConfig, configuration, settings


def test_imports_do_not_load_models_write_logs_or_change_root_logging(tmp_path):
    script = """
import logging, pathlib, sys, os
before = (list(logging.getLogger().handlers), dict(os.environ))
import coach_emotion.config, coach_emotion.pipeline, coach_emotion.cli
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'faster_whisper' not in sys.modules
assert list(pathlib.Path('.').iterdir()) == []
assert before == (list(logging.getLogger().handlers), dict(os.environ))
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


def test_context_restored_after_failure():
    previous = settings()
    with pytest.raises(RuntimeError), configuration(RuntimeConfig(device="cpu")):
        assert settings().device == "cpu"
        raise RuntimeError("test")
    assert settings() is previous


@pytest.mark.parametrize(
    "values",
    [{"backend": "other"}, {"cpu_threads": -1}, {"batch_size_cpu": 0}, {"use_vad": "true"}],
)
def test_invalid_runtime_options(values):
    with pytest.raises(ValueError):
        RuntimeConfig(**values)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_interval_duration(value):
    with pytest.raises(ValueError):
        ConfigDatasetModelo(duracao_intervalo_s=value)


def test_configuration_file_rejects_typos(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"devcie":"cpu"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        RuntimeConfig.from_file(path)


def test_cli_never_overwrites_input(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(b"private input")
    with pytest.raises(SystemExit) as exc:
        main(["analyze", "--input", str(path), "--output", str(path)])
    assert exc.value.code == 1
    assert path.read_bytes() == b"private input"


def test_cli_analyze_forwards_settings_without_ground_truth(tmp_path):
    with patch("coach_emotion.analyze.main_cli") as run:

        def verify(args):
            assert settings().device == "cpu"
            assert settings().stt_model == "openai/whisper-small"
            assert "--entrada" in args
            return 0

        run.side_effect = verify
        assert (
            main(
                [
                    "analyze",
                    "--input",
                    "input.wav",
                    "--output",
                    str(tmp_path / "out.tsv"),
                    "--device",
                    "cpu",
                    "--model",
                    "small",
                ]
            )
            == 0
        )


def test_evaluation_rejects_missing_prediction_instead_of_inflating_accuracy(tmp_path):
    truth = tmp_path / "truth.tsv"
    truth.write_text("0:00 - 0:05:\t0\t0\n0:05 - 0:10:\t2\t4\n", encoding="utf-8")
    pred = tmp_path / "pred.tsv"
    pred.write_text("inicio_s\tfim_s\tcodigo_emocao\n0\t5\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No prediction"):
        avaliar_dataset_final(truth, pred)


@pytest.mark.parametrize(
    "row", ["0\t5\t1.5", "0\tinf\t0", "0\t5\t9", "-1\t5\t0", "5\t0\t0", "nan\t5\t0"]
)
def test_invalid_prediction_rows(tmp_path, row):
    pred = tmp_path / "pred.tsv"
    pred.write_text("inicio_s\tfim_s\tcodigo_emocao\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        _ler_dataset_final(pred)


def test_overlapping_final_predictions_rejected(tmp_path):
    pred = tmp_path / "pred.csv"
    pred.write_text("inicio_s,fim_s,codigo_emocao\n0,5,0\n4,9,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        _ler_dataset_final(pred)


@pytest.mark.parametrize("code", ["1.5", "-1", "9", "inf"])
def test_invalid_ground_truth_codes_are_not_truncated_or_silently_dropped(tmp_path, code):
    truth = tmp_path / "truth.tsv"
    truth.write_text(f"0:00 - 0:05:\t0\t{code}\n", encoding="utf-8")
    pred = tmp_path / "pred.tsv"
    pred.write_text("inicio_s\tfim_s\tcodigo_emocao\n0\t5\t0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="[Gg]round-truth emotion code"):
        avaliar_dataset_final(truth, pred)


def test_cli_evaluation_with_synthetic_labels(tmp_path):
    truth = tmp_path / "truth.tsv"
    truth.write_text("0:00 - 0:05:\t0\t0\n0:05 - 0:10:\t2\t4\n", encoding="utf-8")
    pred = tmp_path / "pred.csv"
    pred.write_text("inicio_s,fim_s,codigo_emocao\n0,5,0\n5,10,4\n", encoding="utf-8")
    output = tmp_path / "nested/metrics.json"
    assert (
        main(
            [
                "evaluate",
                "--ground-truth",
                str(truth),
                "--predictions",
                str(pred),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["acuracia_exata_0a6"] == 1.0


def test_complete_tail_is_analyzed_and_failure_is_not_neutral(tmp_path):
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fixture")
    runtime = pipeline.STTRuntime(
        "faster-whisper", Mock(), "test", "cpu", "int8", 30, 1, 1, 1, True, False
    )
    ser = Mock(return_value=[{"label": "neutral", "score": 0.8}])
    transcript = pipeline.TranscricaoResultado("", [])
    with (
        patch.object(
            pipeline,
            "carregar_audio_mono",
            return_value=(np.zeros(7 * 16000, dtype=np.float32), 16000),
        ),
        patch.object(
            pipeline, "transcrever_audio_com_fallback", return_value=(transcript, runtime)
        ),
        patch.object(pipeline, "carregar_ser", return_value=ser),
        patch.object(pipeline, "liberar_memoria_modelos"),
    ):
        result = pipeline.main(str(audio))
        assert [(r["inicio_s"], r["fim_s"]) for r in result] == [(0.0, 4.0), (2.0, 6.0), (3.0, 7.0)]
        ser.side_effect = RuntimeError("inference failed")
        with pytest.raises(RuntimeError, match="no emotion will be fabricated"):
            pipeline.main(str(audio))


def test_audio_evaluation_preserves_crop_offset(tmp_path):
    from coach_emotion.evaluate_audio import main_cli

    truth = tmp_path / "truth.tsv"
    truth.write_text("0:10 - 0:15:\t0\t0\n", encoding="utf-8")
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"fixture")
    predictions = [
        {"inicio_s": 0.0, "fim_s": 5.0, "emocao": "neu", "escore": 0.0, "confianca_emo": 1.0}
    ]
    with (
        patch("coach_emotion.evaluate_audio.preparar_midia_para_avaliacao", return_value=audio),
        patch("coach_emotion.evaluate_audio.executar_pipeline", return_value=predictions),
        patch("coach_emotion.evaluation.AvaliadorAcuracia.avaliar") as evaluate,
        patch("coach_emotion.evaluation.AvaliadorAcuracia.formatar_relatorio", return_value="ok"),
    ):
        assert main_cli(["--audio", str(audio), "--csv-ground-truth", str(truth)]) == 0
        assert evaluate.call_args.args[0][0]["inicio_s"] == 10.0
