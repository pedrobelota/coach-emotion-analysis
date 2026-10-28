from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

AUDIO_EXTENSOES = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
PERFIS_DISPONIVEIS = ("rapido", "turbo", "balanceado", "preciso", "pt_otimizado")


@dataclass(frozen=True)
class ResultadoBenchmarkWhisper:
    perfil: str
    backend_solicitado: str
    backend_real: str
    modelo: str
    load_s: float
    transcribe_s: float
    total_s: float
    total_chars: int
    total_chunks: int
    chars_por_segundo: float
    idioma_detectado: Optional[str]
    sucesso: bool
    erro: str = ""


def descobrir_audio_padrao(base_dir: Path) -> Optional[Path]:
    candidatos: List[Path] = []
    for extensao in AUDIO_EXTENSOES:
        candidatos.extend(sorted(base_dir.glob(f"*{extensao}")))
    return candidatos[0] if candidatos else None


def obter_whisper_configs():
    from coach_emotion.config import WHISPER_CONFIGS

    return WHISPER_CONFIGS


def executar_benchmark(
    caminho_audio: Path,
    perfil: str,
    backend: str,
) -> ResultadoBenchmarkWhisper:
    whisper_configs = obter_whisper_configs()
    if perfil not in whisper_configs:
        raise ValueError(f"Perfil Whisper invalido: {perfil}")

    from coach_emotion.pipeline import carregar_stt, transcrever_com_stt

    modelo = whisper_configs[perfil].id

    inicio_carga = time.perf_counter()
    stt = carregar_stt(model_id=modelo, backend_preferido=backend)
    load_s = time.perf_counter() - inicio_carga

    inicio_transcricao = time.perf_counter()
    transcricao = transcrever_com_stt(stt, caminho_audio)
    transcribe_s = time.perf_counter() - inicio_transcricao

    total_chars = len(transcricao.texto_completo)
    total_chunks = len(transcricao.chunks)
    chars_por_segundo = total_chars / transcribe_s if transcribe_s > 0 else 0.0

    return ResultadoBenchmarkWhisper(
        perfil=perfil,
        backend_solicitado=backend,
        backend_real=stt.backend,
        modelo=modelo,
        load_s=load_s,
        transcribe_s=transcribe_s,
        total_s=load_s + transcribe_s,
        total_chars=total_chars,
        total_chunks=total_chunks,
        chars_por_segundo=chars_por_segundo,
        idioma_detectado=transcricao.idioma_detectado,
        sucesso=True,
    )


def exportar_resultados_csv(
    resultados: Sequence[ResultadoBenchmarkWhisper],
    caminho_saida: str | Path,
) -> Path:
    caminho = Path(caminho_saida)
    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=list(asdict(resultados[0]).keys()))
        writer.writeheader()
        for resultado in resultados:
            writer.writerow(asdict(resultado))
    return caminho


def formatar_resultados(resultados: Sequence[ResultadoBenchmarkWhisper]) -> str:
    linhas = [
        "perfil        backend          load_s  transcribe_s  total_s  chars/s   chunks  status",
    ]
    for resultado in resultados:
        status = "ok" if resultado.sucesso else f"erro: {resultado.erro}"
        linhas.append(
            f"{resultado.perfil:<12}"
            f"{resultado.backend_real:<17}"
            f"{resultado.load_s:>7.2f}  "
            f"{resultado.transcribe_s:>12.2f}  "
            f"{resultado.total_s:>7.2f}  "
            f"{resultado.chars_por_segundo:>7.1f}  "
            f"{resultado.total_chunks:>6d}  "
            f"{status}"
        )
    return "\n".join(linhas)


def criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara perfis e backends do Whisper no audio informado.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Caminho do audio a ser transcrito. Se omitido, tenta detectar na pasta atual.",
    )
    parser.add_argument(
        "--perfis",
        nargs="+",
        default=["turbo"],
        choices=sorted(PERFIS_DISPONIVEIS),
        help="Perfis Whisper a comparar.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["auto"],
        choices=["auto", "faster-whisper", "transformers"],
        help="Backends de STT a comparar.",
    )
    parser.add_argument(
        "--salvar-csv",
        type=str,
        default=None,
        help="Se informado, exporta o benchmark para CSV.",
    )
    return parser


def main_cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = criar_parser()
    args = parser.parse_args(argv)

    caminho_audio = (
        Path(args.audio).expanduser() if args.audio else descobrir_audio_padrao(Path.cwd())
    )
    if caminho_audio is None:
        parser.error(
            "Nenhum audio encontrado. Informe --audio ou coloque um arquivo suportado na pasta atual."
        )
    from coach_emotion.pipeline import validar_arquivo_audio

    if not validar_arquivo_audio(caminho_audio):
        parser.error(f"Arquivo de audio invalido: {caminho_audio}")

    whisper_configs = obter_whisper_configs()
    resultados: List[ResultadoBenchmarkWhisper] = []
    for perfil in args.perfis:
        for backend in args.backends:
            try:
                resultados.append(executar_benchmark(caminho_audio, perfil, backend))
            except Exception as e:
                resultados.append(
                    ResultadoBenchmarkWhisper(
                        perfil=perfil,
                        backend_solicitado=backend,
                        backend_real=backend,
                        modelo=whisper_configs[perfil].id,
                        load_s=0.0,
                        transcribe_s=0.0,
                        total_s=0.0,
                        total_chars=0,
                        total_chunks=0,
                        chars_por_segundo=0.0,
                        idioma_detectado=None,
                        sucesso=False,
                        erro=str(e),
                    )
                )

    print(f"Audio benchmark: {caminho_audio}")
    print("")
    print(formatar_resultados(resultados))

    if args.salvar_csv and resultados:
        caminho_saida = exportar_resultados_csv(resultados, args.salvar_csv)
        print("")
        print(f"CSV salvo em: {caminho_saida}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main_cli())
