from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import wave
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Sequence

from coach_emotion.dataset import ConfigDatasetModelo, GeradorDatasetModelo

EXTENSOES_MIDIA = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".mpeg",
    ".mpg",
)


def executar_pipeline(
    caminho_audio: str,
    jan_ms: Optional[int] = None,
    hop_ms: Optional[int] = None,
):
    from coach_emotion.pipeline import main

    return main(caminho_audio=caminho_audio, jan_ms=jan_ms, hop_ms=hop_ms)


def preparar_midia_para_dataset(
    caminho_entrada: Path,
    caminho_saida: Path,
    inicio_s: Optional[float] = None,
    fim_s: Optional[float] = None,
) -> Path:
    if caminho_entrada.resolve() == caminho_saida.resolve():
        raise ValueError("Prepared audio cannot overwrite the input")
    if inicio_s is not None and (not math.isfinite(inicio_s) or inicio_s < 0):
        raise ValueError("Start must be finite and nonnegative")
    if fim_s is not None and (not math.isfinite(fim_s) or fim_s <= (inicio_s or 0)):
        raise ValueError("End must be finite and greater than start")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg nao esta disponivel no PATH.")

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    comando = ["ffmpeg", "-y"]

    if inicio_s is not None:
        comando.extend(["-ss", f"{inicio_s:.3f}"])

    comando.extend(["-i", str(caminho_entrada)])

    if fim_s is not None:
        if inicio_s is not None:
            duracao_s = fim_s - inicio_s
            if duracao_s <= 0:
                raise ValueError("fim_s deve ser maior que inicio_s")
            comando.extend(["-t", f"{duracao_s:.3f}"])
        else:
            comando.extend(["-to", f"{fim_s:.3f}"])

    comando.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(caminho_saida),
        ]
    )

    try:
        subprocess.run(
            comando,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detalhe = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"Falha ao preparar a midia com ffmpeg. Detalhes: {detalhe}") from exc

    return caminho_saida


def obter_duracao_wav(caminho_wav: Path) -> float:
    with closing(wave.open(str(caminho_wav), "rb")) as arquivo:
        frames = arquivo.getnframes()
        frame_rate = arquivo.getframerate()
        if frame_rate <= 0:
            raise ValueError("Arquivo WAV com frame rate invalido.")
        return frames / float(frame_rate)


def parsear_pesos_fusao(valor: str) -> Dict[int, float]:
    """Converte ``2:1.35,3:1.35`` em pesos configuraveis por classe."""
    pesos: Dict[int, float] = {}
    for item in valor.split(","):
        codigo_texto, separador, peso_texto = item.strip().partition(":")
        if not separador:
            raise argparse.ArgumentTypeError("Use o formato codigo:peso, separado por virgulas.")
        try:
            codigo = int(codigo_texto.strip())
            peso = float(peso_texto.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Peso invalido: {item!r}. Use, por exemplo, 2:1.35."
            ) from exc
        if codigo not in {2, 3, 4, 5, 6} or not math.isfinite(peso) or peso <= 0:
            raise argparse.ArgumentTypeError(
                "Os codigos devem estar entre 2 e 6 e os pesos devem ser positivos."
            )
        pesos[codigo] = peso
    if not pesos:
        raise argparse.ArgumentTypeError("Informe ao menos um peso de emocao.")
    return pesos


def salvar_janelas_pipeline(
    resultados_pipeline: Sequence[Dict[str, Any]],
    caminho_saida: Path,
) -> Path:
    """Salva as evidencias brutas para auditoria e recalibracao reproduzivel."""
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    caminho_saida.write_text(
        json.dumps(list(resultados_pipeline), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return caminho_saida


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um dataset de saida do modelo no formato de recortes temporais com avaliacao 0-6."
        )
    )
    parser.add_argument(
        "--entrada",
        type=str,
        required=True,
        help="Caminho do arquivo de audio ou video.",
    )
    parser.add_argument(
        "--saida",
        type=str,
        required=True,
        help="Caminho do CSV/TSV de saida do dataset do modelo.",
    )
    parser.add_argument(
        "--jan-ms",
        type=int,
        default=None,
        help="Duracao da janela do pipeline em milissegundos.",
    )
    parser.add_argument(
        "--hop-ms",
        type=int,
        default=None,
        help="Hop size do pipeline em milissegundos.",
    )
    parser.add_argument(
        "--intervalo-s",
        type=float,
        default=5.0,
        help="Duracao dos recortes do dataset final em segundos.",
    )
    parser.add_argument(
        "--inicio-s",
        type=float,
        default=None,
        help="Opcional: inicio do recorte da midia em segundos.",
    )
    parser.add_argument(
        "--fim-s",
        type=float,
        default=None,
        help="Opcional: fim do recorte da midia em segundos.",
    )
    parser.add_argument(
        "--salvar-audio-preparado",
        type=str,
        default=None,
        help="Se informado, salva o WAV preparado usado pelo pipeline.",
    )
    parser.add_argument(
        "--confianca-min-emocao",
        type=float,
        default=0.18,
        help=("Limiar conservador: abaixo disso a emocao do recorte vira neutra."),
    )
    parser.add_argument(
        "--escore-neutro-max-abs",
        type=float,
        default=2.0,
        help=(
            "Se a emocao nao for neutra, mas o escore medio tiver modulo abaixo "
            "desse valor em contexto neutro, o recorte vira neutro. Padrao: 2.0."
        ),
    )
    parser.add_argument(
        "--pesos-fusao-nao-neutra",
        type=parsear_pesos_fusao,
        default=None,
        help=(
            "Opcional: pesos por classe no formato 2:1.35,3:1.35,4:1.30,"
            "5:1.35,6:1.20. Pesos nao informados mantem o padrao calibrado."
        ),
    )
    parser.add_argument(
        "--saida-detalhes",
        type=str,
        default=None,
        help=("Opcional: exporta um segundo CSV/TSV com colunas internas de depuracao do modelo."),
    )
    parser.add_argument(
        "--saida-janelas",
        type=str,
        default=None,
        help=("Opcional: salva as janelas brutas do pipeline em JSON para auditoria."),
    )
    return parser


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = criar_parser()
    args = parser.parse_args(argv)

    caminho_entrada = Path(args.entrada).expanduser()
    if not caminho_entrada.exists():
        parser.error(f"Arquivo de entrada nao encontrado: {caminho_entrada}")
    if caminho_entrada.suffix.lower() not in EXTENSOES_MIDIA:
        parser.error(
            f"Extensao de entrada nao suportada. Use uma destas: {sorted(EXTENSOES_MIDIA)}"
        )

    caminho_saida = Path(args.saida).expanduser()
    # Validate before loading large models or starting FFmpeg.
    ConfigDatasetModelo(duracao_intervalo_s=args.intervalo_s)

    with TemporaryDirectory(prefix="pibic_dataset_modelo_") as pasta_temporaria:
        if args.salvar_audio_preparado:
            caminho_audio = Path(args.salvar_audio_preparado).expanduser()
        else:
            caminho_audio = Path(pasta_temporaria) / f"{caminho_entrada.stem}_dataset.wav"

        preparar_midia_para_dataset(
            caminho_entrada=caminho_entrada,
            caminho_saida=caminho_audio,
            inicio_s=args.inicio_s,
            fim_s=args.fim_s,
        )

        duracao_total_s = obter_duracao_wav(caminho_audio)
        resultados_pipeline = executar_pipeline(
            caminho_audio=str(caminho_audio),
            jan_ms=args.jan_ms,
            hop_ms=args.hop_ms,
        )
        if not resultados_pipeline:
            print("Pipeline nao retornou resultados validos para gerar o dataset.")
            return 1

        caminho_saida_janelas: Optional[Path] = None
        if args.saida_janelas:
            caminho_saida_janelas = salvar_janelas_pipeline(
                resultados_pipeline,
                Path(args.saida_janelas).expanduser(),
            )

        gerador = GeradorDatasetModelo(
            config=ConfigDatasetModelo(
                duracao_intervalo_s=args.intervalo_s,
                confianca_minima_emocao=args.confianca_min_emocao,
                escore_neutro_max_abs=args.escore_neutro_max_abs,
                pesos_fusao_nao_neutra=(
                    args.pesos_fusao_nao_neutra
                    if args.pesos_fusao_nao_neutra is not None
                    else ConfigDatasetModelo().pesos_fusao_nao_neutra
                ),
            )
        )
        linhas = gerador.gerar(
            resultados_pipeline=resultados_pipeline,
            duracao_total_s=duracao_total_s,
        )
        gerador.exportar(linhas, caminho_saida, incluir_detalhes=False)

        caminho_saida_detalhes: Optional[Path] = None
        if args.saida_detalhes:
            caminho_saida_detalhes = Path(args.saida_detalhes).expanduser()
            gerador.exportar(
                linhas,
                caminho_saida_detalhes,
                incluir_detalhes=True,
            )

        print(f"Entrada analisada: {caminho_entrada}")
        if args.salvar_audio_preparado:
            print(f"Audio preparado: {caminho_audio}")
        else:
            print("Audio preparado: arquivo temporario normalizado em 16 kHz")
        print(f"Duracao considerada: {duracao_total_s:.2f}s")
        print(f"Intervalo do dataset: {args.intervalo_s:.2f}s")
        print(f"Confianca minima de emocao: {args.confianca_min_emocao:.2f}")
        print(f"Escore neutro max abs: {args.escore_neutro_max_abs:.2f}")
        print(f"Pesos de fusao nao neutra: {gerador.config.pesos_fusao_nao_neutra}")
        print(f"Total de linhas geradas: {len(linhas)}")
        print(f"Dataset salvo em: {caminho_saida}")
        if caminho_saida_detalhes is not None:
            print(f"Detalhes internos salvos em: {caminho_saida_detalhes}")
        if caminho_saida_janelas is not None:
            print(f"Janelas brutas salvas em: {caminho_saida_janelas}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
