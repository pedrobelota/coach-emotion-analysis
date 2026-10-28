from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import pandas as pd

if TYPE_CHECKING:
    from coach_emotion.evaluation import AvaliadorAcuracia, ResultadoAvaliacao


EXTENSOES_AUDIO = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
EXTENSOES_VIDEO = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpeg", ".mpg")
EXTENSOES_MIDIA = EXTENSOES_AUDIO + EXTENSOES_VIDEO


def descobrir_ground_truth_padrao() -> Optional[Path]:
    caminho_env = os.environ.get("PIBIC_GROUND_TRUTH_CSV")
    if caminho_env:
        caminho = Path(caminho_env).expanduser()
        return caminho if caminho.exists() else None

    return None


def descobrir_midia_padrao(base_dir: Path) -> Optional[Path]:
    candidatos: list[Path] = []
    for extensao in EXTENSOES_MIDIA:
        candidatos.extend(sorted(base_dir.glob(f"*{extensao}")))
    return candidatos[0] if candidatos else None


def executar_pipeline(
    caminho_audio: str,
    jan_ms: Optional[int] = None,
    hop_ms: Optional[int] = None,
) -> Sequence[Mapping[str, Any]] | None:
    from coach_emotion.pipeline import main

    return main(caminho_audio=caminho_audio, jan_ms=jan_ms, hop_ms=hop_ms)


def exportar_detalhes_csv(
    resultado: "ResultadoAvaliacao",
    caminho_saida: str | Path,
) -> Path:
    caminho = Path(caminho_saida)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    registros = []

    for intervalo in resultado.intervalos:
        erro_escore: Optional[float] = None
        erro_escore_abs: Optional[float] = None
        if intervalo.metrica_humana is not None and intervalo.escore_predito is not None:
            erro_escore = float(intervalo.escore_predito - intervalo.metrica_humana)
            erro_escore_abs = abs(erro_escore)

        acerto_classe: Optional[bool] = None
        if intervalo.rotulo_real is not None and intervalo.rotulo_predito is not None:
            acerto_classe = intervalo.rotulo_real == intervalo.rotulo_predito

        registros.append(
            {
                "indice_csv": intervalo.indice_csv,
                "inicio_s": intervalo.inicio_s,
                "fim_s": intervalo.fim_s,
                "duracao_s": round(intervalo.fim_s - intervalo.inicio_s, 3),
                "rotulo_humano": intervalo.rotulo_humano,
                "rotulo_real": intervalo.rotulo_real,
                "rotulo_predito": intervalo.rotulo_predito,
                "acerto_classe": acerto_classe,
                "metrica_humana": intervalo.metrica_humana,
                "escore_predito": intervalo.escore_predito,
                "erro_escore": erro_escore,
                "erro_escore_abs": erro_escore_abs,
                "janelas_agregadas": intervalo.janelas_agregadas,
                "incluido_nas_metricas": intervalo.incluido_nas_metricas,
                "motivo_exclusao": intervalo.motivo_exclusao,
            }
        )

    dataframe = pd.DataFrame(registros)
    dataframe.to_csv(caminho, index=False, encoding="utf-8-sig")
    return caminho


def preparar_midia_para_avaliacao(
    caminho_midia: Path,
    avaliador: "AvaliadorAcuracia",
    caminho_saida: Path,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg nao esta disponivel no PATH para preparar o audio da avaliacao.")

    inicio_s, fim_s = avaliador.obter_faixa_temporal_csv()
    duracao_s = fim_s - inicio_s
    if duracao_s <= 0:
        raise ValueError("Faixa temporal do ground truth invalida para recorte.")

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    comando = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{inicio_s:.3f}",
        "-t",
        f"{duracao_s:.3f}",
        "-i",
        str(caminho_midia),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(caminho_saida),
    ]

    try:
        subprocess.run(
            comando,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg nao foi encontrado para preparar a avaliacao.") from exc
    except subprocess.CalledProcessError as exc:
        detalhe = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Falha ao preparar o audio com ffmpeg. Detalhes: {detalhe}") from exc

    return caminho_saida


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Executa o pipeline de audio e valida os resultados contra o ground truth.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help=(
            "Caminho do arquivo de audio ou video. "
            "Se omitido, tenta detectar automaticamente na pasta atual."
        ),
    )
    parser.add_argument(
        "--csv-ground-truth",
        type=str,
        default=None,
        help=("Caminho do CSV/TSV de validacao humana. Se omitido, usa PIBIC_GROUND_TRUTH_CSV."),
    )
    parser.add_argument(
        "--jan-ms",
        type=int,
        default=None,
        help="Duracao da janela em milissegundos. Se omitido, usa o valor do pipeline.",
    )
    parser.add_argument(
        "--hop-ms",
        type=int,
        default=None,
        help="Hop size em milissegundos. Se omitido, usa o valor do pipeline.",
    )
    parser.add_argument(
        "--politica-nulos",
        choices=("ignore", "map_to_neu"),
        default="ignore",
        help="Como tratar intervalos com rotulo humano nulo.",
    )
    parser.add_argument(
        "--salvar-detalhes-csv",
        type=str,
        default=None,
        help="Se informado, salva um CSV com o alinhamento intervalo a intervalo.",
    )
    parser.add_argument(
        "--sem-preparo-automatico",
        action="store_true",
        help=(
            "Desativa a preparacao automatica do audio com base na faixa temporal do ground truth."
        ),
    )
    parser.add_argument(
        "--salvar-audio-preparado",
        type=str,
        default=None,
        help=("Se informado, salva o audio recortado e normalizado usado na avaliacao."),
    )
    return parser


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = criar_parser()
    args = parser.parse_args(argv)

    base_dir = Path.cwd()

    caminho_midia = (
        Path(args.audio).expanduser() if args.audio else descobrir_midia_padrao(base_dir)
    )
    if caminho_midia is None:
        parser.error(
            "Nenhum arquivo de audio/video encontrado. "
            "Informe --audio ou coloque um arquivo suportado na pasta atual."
        )
    if not caminho_midia.exists():
        parser.error(f"Arquivo de entrada nao encontrado: {caminho_midia}")

    caminho_ground_truth = (
        Path(args.csv_ground_truth).expanduser()
        if args.csv_ground_truth
        else descobrir_ground_truth_padrao()
    )
    if caminho_ground_truth is None:
        parser.error(
            "Ground truth nao encontrado. "
            "Informe --csv-ground-truth ou defina PIBIC_GROUND_TRUTH_CSV."
        )
    if not caminho_ground_truth.exists():
        parser.error(f"Ground truth nao encontrado: {caminho_ground_truth}")

    from coach_emotion.evaluation import AvaliadorAcuracia, ConfigAvaliacao

    avaliador = AvaliadorAcuracia(
        caminho_ground_truth,
        config=ConfigAvaliacao(politica_nulos=args.politica_nulos),
    )

    caminho_pipeline = caminho_midia
    audio_preparado_salvo: Optional[Path] = None

    with TemporaryDirectory(prefix="pibic_avaliacao_") as pasta_temporaria:
        if not args.sem_preparo_automatico:
            if args.salvar_audio_preparado:
                destino_preparado = Path(args.salvar_audio_preparado).expanduser()
            else:
                destino_preparado = (
                    Path(pasta_temporaria) / f"{caminho_midia.stem}_preparado_gt.wav"
                )

            caminho_pipeline = preparar_midia_para_avaliacao(
                caminho_midia,
                avaliador,
                destino_preparado,
            )
            if args.salvar_audio_preparado:
                audio_preparado_salvo = caminho_pipeline

        resultados_pipeline = executar_pipeline(
            caminho_audio=str(caminho_pipeline),
            jan_ms=args.jan_ms,
            hop_ms=args.hop_ms,
        )
        if not resultados_pipeline:
            print("Pipeline nao retornou resultados validos para avaliacao.")
            return 1

        if not args.sem_preparo_automatico:
            # FFmpeg resets the cropped media clock to zero; labels retain source time.
            offset, _ = avaliador.obter_faixa_temporal_csv()
            resultados_pipeline = [
                {**item, "inicio_s": item["inicio_s"] + offset, "fim_s": item["fim_s"] + offset}
                for item in resultados_pipeline
            ]
        resultado = avaliador.avaliar(resultados_pipeline)

        print(f"Entrada original: {caminho_midia}")
        print(f"Ground truth: {caminho_ground_truth}")
        if not args.sem_preparo_automatico:
            faixa_ini, faixa_fim = avaliador.obter_faixa_temporal_csv()
            print(f"Faixa usada na avaliacao: {faixa_ini:.2f}s - {faixa_fim:.2f}s")
            if audio_preparado_salvo is not None:
                print(f"Audio preparado salvo em: {audio_preparado_salvo}")
            else:
                print("Audio preparado automaticamente em arquivo temporario.")
        else:
            print("Preparo automatico desativado: usando arquivo de entrada completo.")
        print("")
        print(avaliador.formatar_relatorio(resultado))

        if args.salvar_detalhes_csv:
            caminho_saida = exportar_detalhes_csv(resultado, args.salvar_detalhes_csv)
            print("")
            print(f"Detalhes exportados para: {caminho_saida}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
