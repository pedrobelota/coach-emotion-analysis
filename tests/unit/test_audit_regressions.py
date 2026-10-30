from pathlib import Path

import pytest

from coach_emotion.evaluate import avaliar_dataset_final
from coach_emotion.evaluation import AvaliadorAcuracia, ConfigAvaliacao


def write_tables(folder: Path, truth: str, predictions: str):
    gt, pred = folder / "truth.tsv", folder / "pred.csv"
    gt.write_text(truth, encoding="utf-8")
    pred.write_text("inicio_s,fim_s,codigo_emocao\n" + predictions, encoding="utf-8")
    return gt, pred


@pytest.mark.parametrize("rows", ["0,0.001,0\n", "1,5,0\n", "0,4,0\n", "0,2,0\n3,5,0\n"])
def test_partial_coverage_cannot_produce_perfect_accuracy(tmp_path, rows):
    gt, pred = write_tables(tmp_path, "0:00 - 0:05:\t0\t0\n", rows)
    with pytest.raises(ValueError, match="Incomplete prediction coverage"):
        avaliar_dataset_final(gt, pred)


def test_contiguous_predictions_covering_annotation_are_accepted(tmp_path):
    gt, pred = write_tables(tmp_path, "0:00 - 0:05:\t0\t0\n", "0,2,0\n2,5,0\n")
    assert avaliar_dataset_final(gt, pred).acuracia_exata_0a6 == 1.0


@pytest.mark.parametrize(
    "truth",
    [
        "0:00 - 0:05:\t0\t0\n0:00 - 0:05:\t0\t0\n",
        "0:05 - 0:10:\t0\t0\n0:00 - 0:05:\t0\t0\n",
        "0:00 - 0:05:\t0\t0\n0:04 - 0:09:\t0\t0\n",
    ],
)
def test_duplicate_overlapping_or_unordered_annotations_fail(tmp_path, truth):
    gt, _ = write_tables(tmp_path, truth, "0,10,0\n")
    with pytest.raises(ValueError, match="chronological"):
        AvaliadorAcuracia(gt).carregar_ground_truth()


def test_empty_interval_is_reconstructed_in_compact_schema(tmp_path):
    gt, pred = write_tables(tmp_path, "\t0\t0\n", "0,5,0\n")
    assert avaliar_dataset_final(gt, pred).total_intervalos == 1


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), 0, -1])
def test_annotation_duration_must_be_finite_and_positive(duration):
    with pytest.raises(ValueError):
        ConfigAvaliacao(duracao_intervalo_padrao_s=duration)


def test_legacy_label_codes_cannot_be_interpreted_as_final_codes(tmp_path):
    gt, pred = write_tables(tmp_path, "0:00 - 0:05:,2,Tristeza,5\n", "0,5,2\n")
    with pytest.raises(ValueError, match="compact three-column"):
        avaliar_dataset_final(gt, pred)


def test_report_exposes_baselines_confusion_matrix_and_minority_failure(tmp_path):
    gt, pred = write_tables(tmp_path, "0:00 - 0:05:\t0\t0\n0:05 - 0:10:\t2\t4\n", "0,5,0\n5,10,0\n")
    report = avaliar_dataset_final(gt, pred)
    assert report.baseline_codigo_0 == report.baseline_neutro_colapsado == 0.5
    assert report.classes_colapsadas == ["ang", "neu"]
    assert report.matriz_confusao_colapsada == [[0, 1], [0, 1]]
    assert report.metricas_por_classe_colapsada["ang"]["recall"] == 0
    assert report.metricas_por_classe_colapsada["ang"]["support"] == 1
