from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from coach_emotion.dataset import EMOCAO_FINAL_POR_CODIGO, ROTULOS_MODELO_0_A_6
from coach_emotion.evaluation import AvaliadorAcuracia, ConfigAvaliacao

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class PredicaoFinal:
    inicio_s: float
    fim_s: float
    codigo_emocao: int


@dataclass(frozen=True)
class ResultadoAvaliacaoFinal:
    total_intervalos: int
    acertos_exatos_0a6: int
    acuracia_exata_0a6: float
    intervalo_confianca_95: tuple[float, float]
    acertos_emocao_colapsada: int
    acuracia_emocao_colapsada: float
    macro_precision_colapsada: float
    macro_recall_colapsada: float
    macro_f1_colapsada: float
    suporte_por_codigo_real: Dict[int, int]
    suporte_por_codigo_predito: Dict[int, int]
    observacao_validade: str


def _ha_sobreposicao(inicio_a: float, fim_a: float, inicio_b: float, fim_b: float) -> bool:
    return min(fim_a, fim_b) - max(inicio_a, inicio_b) > 0.0


def _ler_dataset_final(caminho: str | Path) -> List[PredicaoFinal]:
    arquivo = Path(caminho)
    separador = "\t" if arquivo.suffix.lower() == ".tsv" else ","
    dataframe = pd.read_csv(arquivo, sep=separador, encoding="utf-8-sig")
    obrigatorias = {"inicio_s", "fim_s", "codigo_emocao"}
    ausentes = obrigatorias.difference(dataframe.columns)
    if ausentes:
        raise ValueError(f"Dataset final sem colunas obrigatorias: {sorted(ausentes)}")

    predicoes: List[PredicaoFinal] = []
    for indice, linha in dataframe.iterrows():
        valor = float(linha["codigo_emocao"])
        if not isfinite(valor) or not valor.is_integer():
            raise ValueError(f"codigo_emocao must be an integer at row {indice}")
        codigo = int(valor)
        if codigo not in ROTULOS_MODELO_0_A_6:
            raise ValueError(f"codigo_emocao invalido na linha {indice}: {codigo}")
        inicio_s = float(linha["inicio_s"])
        fim_s = float(linha["fim_s"])
        if not isfinite(inicio_s) or not isfinite(fim_s) or inicio_s < 0 or fim_s <= inicio_s:
            raise ValueError(f"Intervalo invalido na linha {indice}")
        predicoes.append(PredicaoFinal(inicio_s, fim_s, codigo))
    predicoes.sort(key=lambda item: (item.inicio_s, item.fim_s))
    for anterior, atual in zip(predicoes, predicoes[1:]):
        if atual.inicio_s < anterior.fim_s:
            raise ValueError("Final prediction intervals must not overlap")
    return predicoes


def _voto_majoritario(codigos: Iterable[int]) -> int:
    contagem = Counter(codigos)
    return max(contagem, key=lambda codigo: (contagem[codigo], -codigo))


def _wilson_95(acertos: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proporcao = acertos / total
    denominador = 1 + (Z_95**2 / total)
    centro = (proporcao + Z_95**2 / (2 * total)) / denominador
    margem = (
        Z_95
        * sqrt((proporcao * (1 - proporcao) / total) + (Z_95**2 / (4 * total**2)))
        / denominador
    )
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def avaliar_dataset_final(
    caminho_ground_truth: str | Path,
    caminho_dataset_final: str | Path,
) -> ResultadoAvaliacaoFinal:
    """Compara o dataset neurossimbólico final com os códigos humanos de 0 a 6."""
    avaliador = AvaliadorAcuracia(
        caminho_ground_truth,
        config=ConfigAvaliacao(politica_nulos="ignore"),
    )
    ground_truth = avaliador.carregar_ground_truth()
    predicoes = _ler_dataset_final(caminho_dataset_final)

    reais: List[int] = []
    preditos: List[int] = []
    for intervalo in ground_truth:
        if intervalo.rotulo_sistema is None:
            continue
        candidatas = [
            predicao.codigo_emocao
            for predicao in predicoes
            if _ha_sobreposicao(
                intervalo.inicio_s,
                intervalo.fim_s,
                predicao.inicio_s,
                predicao.fim_s,
            )
        ]
        if not candidatas:
            raise ValueError(
                f"No prediction overlaps labeled interval {intervalo.inicio_s}-{intervalo.fim_s}; "
                "evaluation refuses to silently exclude missing predictions"
            )
        reais.append(intervalo.codigo_emocao)
        preditos.append(_voto_majoritario(candidatas))

    if not reais:
        raise ValueError("Nao ha intervalos rotulados com predicao sobreposta")

    acertos_exatos = sum(real == predito for real, predito in zip(reais, preditos))
    reais_colapsados = [EMOCAO_FINAL_POR_CODIGO[real] for real in reais]
    preditos_colapsados = [EMOCAO_FINAL_POR_CODIGO[predito] for predito in preditos]
    acertos_colapsados = sum(
        real == predito for real, predito in zip(reais_colapsados, preditos_colapsados)
    )
    classes = sorted(set(reais_colapsados) | set(preditos_colapsados))
    precision, recall, f1, _ = precision_recall_fscore_support(
        reais_colapsados,
        preditos_colapsados,
        labels=classes,
        average="macro",
        zero_division=0,
    )
    return ResultadoAvaliacaoFinal(
        total_intervalos=len(reais),
        acertos_exatos_0a6=acertos_exatos,
        acuracia_exata_0a6=acertos_exatos / len(reais),
        intervalo_confianca_95=_wilson_95(acertos_exatos, len(reais)),
        acertos_emocao_colapsada=acertos_colapsados,
        acuracia_emocao_colapsada=acertos_colapsados / len(reais),
        macro_precision_colapsada=float(precision),
        macro_recall_colapsada=float(recall),
        macro_f1_colapsada=float(f1),
        suporte_por_codigo_real=dict(sorted(Counter(reais).items())),
        suporte_por_codigo_predito=dict(sorted(Counter(preditos).items())),
        observacao_validade=(
            "As metricas descrevem este unico recorte anotado. Elas nao devem ser "
            "interpretadas como estimativa de generalizacao sem conjunto de teste independente."
        ),
    )


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Avalia o dataset neurossimbolico final contra o ground truth humano."
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--dataset-modelo", required=True)
    parser.add_argument("--salvar-json", default=None)
    args = parser.parse_args(argv)
    resultado = avaliar_dataset_final(args.ground_truth, args.dataset_modelo)
    dados: Mapping[str, Any] = asdict(resultado)
    print(json.dumps(dados, ensure_ascii=False, indent=2))
    if args.salvar_json:
        Path(args.salvar_json).write_text(
            json.dumps(dados, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
