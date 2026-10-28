"""Transcription, acoustic emotion inference and temporal football rules."""

import gc
import logging
import mimetypes
import re
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from coach_emotion.config import (
    HOP_MS,
    JANELA_MS,
    MAPA_ROTULOS,
    MODULADORES_DE_EMOCAO,
    TAXA_DE_AMOSTRAGEM,
)
from coach_emotion.runtime import settings
from coach_emotion.symbolic_rules import VOCABULARIO_NORMALIZADO, normalizar_texto

LIBROSA_AVAILABLE = find_spec("librosa") is not None
SCIPY_AVAILABLE = find_spec("scipy") is not None
TORCH_AVAILABLE = find_spec("torch") is not None
CTRANSLATE2_AVAILABLE = find_spec("ctranslate2") is not None
TRANSFORMERS_AVAILABLE = find_spec("transformers") is not None
FASTER_WHISPER_AVAILABLE = find_spec("faster_whisper") is not None
BATCHED_FASTER_WHISPER_AVAILABLE = FASTER_WHISPER_AVAILABLE
SER_COMPAT_AVAILABLE = TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE
librosa = wavfile = resample = torch = ctranslate2 = pipeline = None
WhisperModel = BatchedInferencePipeline = None
_BACKENDS_LOADED = False


def _ensure_backends():
    """Import model libraries only when inference is actually requested."""
    global librosa, wavfile, resample, torch, ctranslate2, pipeline
    global WhisperModel, BatchedInferencePipeline, _BACKENDS_LOADED
    if _BACKENDS_LOADED:
        return
    if LIBROSA_AVAILABLE:
        import librosa as audio_lib

        librosa = audio_lib
    if SCIPY_AVAILABLE:
        from scipy.io import wavfile as wav_lib
        from scipy.signal import resample as resample_fn

        wavfile, resample = wav_lib, resample_fn
    if TORCH_AVAILABLE:
        import torch as torch_lib

        torch = torch_lib
    if CTRANSLATE2_AVAILABLE:
        import ctranslate2 as ct_lib

        ctranslate2 = ct_lib
    if TRANSFORMERS_AVAILABLE:
        from transformers import pipeline as hf_pipeline

        pipeline = hf_pipeline
    if FASTER_WHISPER_AVAILABLE:
        from faster_whisper import BatchedInferencePipeline as fw_batch
        from faster_whisper import WhisperModel as fw_model

        WhisperModel, BatchedInferencePipeline = fw_model, fw_batch
    _BACKENDS_LOADED = True


def carregar_ser_compativel(**kwargs):
    from coach_emotion.emotion_recognition import carregar_ser_compativel as loader

    return loader(revision=settings().ser_revision, **kwargs)


# === CONFIGURAÇÃO DE LOGGER ===
logger = logging.getLogger(__name__)

MODELOS_COMPATIVEIS_FASTER_WHISPER: Dict[str, str] = {
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-medium": "medium",
    "openai/whisper-tiny": "tiny",
    "openai/whisper-base": "base",
    "openai/whisper-small": "small",
}
FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO = False


def erro_indica_falha_cuda_faster_whisper(erro: Exception) -> bool:
    mensagem = str(erro).casefold()
    return "cublas64_12.dll" in mensagem or "cannot be loaded" in mensagem


# ========================================
# UTILITÁRIOS DE VALIDAÇÃO
# ========================================
def validar_arquivo_audio(caminho: Path) -> bool:
    """
    Valida se o arquivo existe e é um formato de áudio suportado.

    Returns:
        bool: True se válido, False caso contrário
    """
    if not caminho.exists():
        logger.error(f"Arquivo não encontrado: {caminho.resolve()}")
        return False

    # Verifica extensão
    ext = caminho.suffix.lower()
    extensoes_validas = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    if ext not in extensoes_validas:
        logger.error(f"Extensão não suportada: {ext}. Use: {extensoes_validas}")
        return False

    # Verifica MIME type (proteção extra)
    mime, _ = mimetypes.guess_type(str(caminho))
    if mime and not mime.startswith("audio/"):
        logger.error(f"Arquivo não é áudio válido. MIME type: {mime}")
        return False

    return True


def validar_duracao_audio(y: np.ndarray, sr: int, duracao_min_s: float) -> bool:
    """
    Verifica se o áudio tem duração mínima necessária.

    Args:
        y: Array de áudio
        sr: Taxa de amostragem
        duracao_min_s: Duração mínima em segundos

    Returns:
        bool: True se duração suficiente
    """
    if y.size == 0:
        logger.error("Áudio vazio (0 amostras)")
        return False

    duracao_s = y.size / sr
    if duracao_s < duracao_min_s:
        logger.error(
            f"Áudio muito curto: {duracao_s:.2f}s. Mínimo necessário: {duracao_min_s:.2f}s"
        )
        return False

    logger.info(f"Duração do áudio: {duracao_s:.2f}s ({y.size} amostras @ {sr}Hz)")
    return True


# ========================================
# CARREGAMENTO DE ÁUDIO
# ========================================
def carregar_audio_mono(caminho: Path, sr_alvo: int) -> Tuple[np.ndarray, int]:
    """
    Carrega áudio como mono (1D) float32 na taxa de amostragem desejada.

    CORREÇÃO: Usa scipy.signal.resample para reamostragem sem depender de librosa.

    Args:
        caminho: Caminho do arquivo de áudio
        sr_alvo: Taxa de amostragem alvo

    Returns:
        Tuple[np.ndarray, int]: (áudio_mono, taxa_de_amostragem)
    """
    _ensure_backends()
    # Estratégia 1: Librosa (mais robusto)
    if LIBROSA_AVAILABLE:
        logger.debug(f"Carregando com librosa: {caminho}")
        y, sr = librosa.load(str(caminho), sr=sr_alvo, mono=True)
        y = y.astype(np.float32, copy=False)
        return y, sr

    # Estratégia 2: Scipy (fallback)
    if not SCIPY_AVAILABLE:
        raise RuntimeError(
            "Nenhum backend de áudio disponível. "
            "Instale 'librosa' ou 'scipy': pip install librosa scipy"
        )

    logger.debug(f"Carregando com scipy.io.wavfile: {caminho}")
    sr, data = wavfile.read(str(caminho))

    # Converte para float32 normalizado [-1, 1]
    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32:
        y = data.astype(np.float32, copy=False)
    else:
        logger.warning(f"Tipo de dado incomum: {data.dtype}. Convertendo para float32.")
        y = data.astype(np.float32)

    # Força mono (média dos canais se estéreo)
    if y.ndim > 1:
        logger.debug(f"Convertendo de {y.ndim}D para mono")
        y = y.mean(axis=1).astype(np.float32)

    # Reamostra se necessário (CORREÇÃO: usa scipy.signal.resample)
    if sr != sr_alvo:
        logger.info(f"Reamostrando: {sr}Hz → {sr_alvo}Hz")
        num_samples_alvo = int(len(y) * sr_alvo / sr)
        y = resample(y, num_samples_alvo).astype(np.float32)
        sr = sr_alvo

    return y, sr


# ========================================
# CARREGAMENTO DE MODELOS
# ========================================
@dataclass(frozen=True)
class STTRuntime:
    backend: str
    model: Any
    model_id: str
    device: str
    compute_type: Optional[str]
    chunk_length_s: int
    batch_size: int
    num_beams: int
    best_of: int
    use_vad: bool
    condition_on_previous_text: bool
    supports_batch_size: bool = False


@dataclass(frozen=True)
class TranscricaoResultado:
    texto_completo: str
    chunks: List[Dict[str, Any]]
    idioma_detectado: Optional[str] = None


def detectar_dispositivo_torch() -> Tuple[int, Optional[object]]:
    """
    Detecta melhor dispositivo (GPU/CPU) e dtype para os modelos.

    Returns:
        Tuple[int, dtype]: (device_id, torch_dtype)
            device_id: -1 para CPU, 0+ para GPU
            torch_dtype: torch.bfloat16, torch.float16 ou None
    """
    _ensure_backends()
    if settings().device == "cpu":
        return -1, None
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        logger.info("Usando CPU (GPU não disponível)")
        return -1, None

    device = 0
    logger.info(f"GPU detectada: {torch.cuda.get_device_name(0)}")

    # Tenta usar bfloat16 se GPU suportar (Ampere+)
    try:
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 8:  # Ampere (RTX 30xx+) ou superior
            logger.info("Usando torch.bfloat16 (GPU Ampere+)")
            return device, torch.bfloat16
        else:
            logger.info("Usando torch.float16 (GPU pré-Ampere)")
            return device, torch.float16
    except Exception as e:
        logger.warning(f"Erro ao detectar capability da GPU: {e}. Usando float16.")
        return device, torch.float16


def detectar_dispositivo_faster_whisper() -> Tuple[str, int]:
    """Detecta dispositivo para faster-whisper sem depender de torch."""
    _ensure_backends()
    if settings().device == "cpu":
        return "cpu", 0
    if CTRANSLATE2_AVAILABLE:
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                logger.info("GPU detectada para faster-whisper via ctranslate2")
                return "cuda", 0
        except Exception as e:
            logger.warning(f"Falha ao detectar GPU via ctranslate2: {e}. Usando CPU.")

    logger.info("Usando CPU no faster-whisper")
    return "cpu", 0


def obter_compute_types_faster_whisper(device: str) -> List[str]:
    """Escolhe compute types mais realistas para o hardware disponivel."""
    if device == "cpu":
        return [settings().compute_type_cpu, "int8", "default"]

    if torch is not None and torch.cuda.is_available():
        try:
            capability = torch.cuda.get_device_capability(0)
            if capability[0] < 7:
                return ["default"]
        except Exception:
            logger.debug(
                "Nao foi possivel inspecionar a capability CUDA para faster-whisper.",
                exc_info=True,
            )

    return ["int8_float16", settings().compute_type_gpu, "default"]


def selecionar_backend_stt(preferencia: Optional[str] = None) -> str:
    """Resolve o backend de STT a partir da preferencia configurada."""
    preferencia = preferencia or settings().backend
    preferencia_normalizada = preferencia.strip().lower()
    opcoes_validas = {"auto", "faster-whisper", "transformers"}
    if preferencia_normalizada not in opcoes_validas:
        raise ValueError(f"Backend STT invalido: {preferencia}. Use: {sorted(opcoes_validas)}")

    if preferencia_normalizada == "auto":
        if FASTER_WHISPER_AVAILABLE:
            return "faster-whisper"
        return "transformers"

    if preferencia_normalizada == "faster-whisper" and not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(
            "Backend 'faster-whisper' selecionado, mas a biblioteca nao esta instalada."
        )

    if preferencia_normalizada == "transformers" and not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Backend 'transformers' selecionado, mas a biblioteca nao esta instalada."
        )

    return preferencia_normalizada


def resolver_modelo_faster_whisper(model_id: str) -> Optional[str]:
    """Mapeia IDs do Hugging Face para nomes compativeis com faster-whisper."""
    return MODELOS_COMPATIVEIS_FASTER_WHISPER.get(model_id)


def carregar_stt_faster_whisper(
    model_id: str,
    device_forcado: Optional[str] = None,
) -> STTRuntime:
    """Carrega Whisper otimizado com faster-whisper."""
    _ensure_backends()
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper nao esta disponivel. Instale com: pip install faster-whisper"
        )

    model_id_faster = resolver_modelo_faster_whisper(model_id)
    if model_id_faster is None:
        raise RuntimeError(
            f"Modelo nao compativel com faster-whisper sem conversao CTranslate2: {model_id}"
        )

    if device_forcado is not None:
        device = device_forcado
        device_index = 0
    elif FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO:
        device = "cpu"
        device_index = 0
        logger.info("CUDA do faster-whisper indisponivel nesta sessao. Usando fallback em CPU.")
    else:
        device, device_index = detectar_dispositivo_faster_whisper()
    compute_types = obter_compute_types_faster_whisper(device)

    ultimo_erro: Optional[Exception] = None
    for compute_type in dict.fromkeys(compute_types):
        try:
            logger.info(
                "Carregando Whisper com faster-whisper: "
                f"modelo={model_id_faster}, origem={model_id}, "
                f"device={device}, compute_type={compute_type}"
            )
            model = WhisperModel(
                model_id_faster,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                cpu_threads=settings().cpu_threads,
                num_workers=settings().num_workers,
            )
            runtime_model: Any = model
            supports_batch_size = False
            if BATCHED_FASTER_WHISPER_AVAILABLE and BatchedInferencePipeline is not None:
                runtime_model = BatchedInferencePipeline(model)
                supports_batch_size = True
                logger.info("BatchedInferencePipeline ativado para faster-whisper")

            logger.info("✓ Whisper carregado com faster-whisper")
            return STTRuntime(
                backend="faster-whisper",
                model=runtime_model,
                model_id=model_id,
                device=device,
                compute_type=compute_type,
                chunk_length_s=settings().chunk_length_s,
                batch_size=settings().batch_size_gpu
                if device == "cuda"
                else settings().batch_size_cpu,
                num_beams=settings().num_beams,
                best_of=settings().best_of,
                use_vad=settings().use_vad,
                condition_on_previous_text=settings().condition_on_previous_text,
                supports_batch_size=supports_batch_size,
            )
        except Exception as e:
            ultimo_erro = e
            logger.warning(f"Falha ao carregar faster-whisper (compute_type={compute_type}): {e}")

    raise RuntimeError("Falha ao carregar Whisper com faster-whisper") from ultimo_erro


def carregar_stt_transformers(model_id: str) -> STTRuntime:
    """Carrega Whisper via pipeline do transformers."""
    _ensure_backends()
    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError(
            "Backend transformers indisponivel. Instale com: pip install transformers"
        )
    if not TORCH_AVAILABLE:
        raise RuntimeError("Backend transformers requer torch. Instale com: pip install torch")

    device_id, dtype = detectar_dispositivo_torch()
    stt_kwargs: Dict[str, Any] = {"device": device_id}
    if dtype is not None:
        stt_kwargs["dtype"] = dtype

    logger.info(
        "Carregando Whisper com transformers: "
        f"modelo={model_id}, device={'cuda' if device_id != -1 else 'cpu'}"
    )
    model = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        **stt_kwargs,
    )
    logger.info("✓ Whisper carregado com transformers")

    return STTRuntime(
        backend="transformers",
        model=model,
        model_id=model_id,
        device="cuda" if device_id != -1 else "cpu",
        compute_type=str(dtype) if dtype is not None else None,
        chunk_length_s=settings().chunk_length_s,
        batch_size=settings().batch_size_gpu if device_id != -1 else settings().batch_size_cpu,
        num_beams=settings().num_beams,
        best_of=settings().best_of,
        use_vad=False,
        condition_on_previous_text=True,
        supports_batch_size=False,
    )


def carregar_stt(
    model_id: Optional[str] = None,
    backend_preferido: Optional[str] = None,
) -> STTRuntime:
    """Carrega STT com o backend mais rapido disponivel."""
    backend_preferido = backend_preferido or settings().backend
    model_id = model_id or settings().stt_model
    backend = selecionar_backend_stt(backend_preferido)

    if backend == "faster-whisper":
        try:
            return carregar_stt_faster_whisper(model_id)
        except Exception as e:
            if backend_preferido.strip().lower() == "auto":
                logger.warning(
                    f"Faster-whisper falhou no modo automatico. Voltando para transformers: {e}"
                )
                return carregar_stt_transformers(model_id)
            raise
    if backend == "transformers":
        return carregar_stt_transformers(model_id)

    raise RuntimeError(f"Backend STT nao suportado: {backend}")


def liberar_memoria_modelos(*modelos: Any) -> None:
    """Libera referencias de modelos e limpa cache de GPU quando possivel."""
    for modelo in modelos:
        del modelo

    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            logger.debug("Nao foi possivel limpar o cache CUDA.", exc_info=True)


def carregar_ser() -> Optional[object]:
    """Carrega o modelo de reconhecimento de emocao."""
    _ensure_backends()
    if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE or not SER_COMPAT_AVAILABLE:
        logger.error("SER requer transformers e torch. Instale com: pip install transformers torch")
        return None

    device, dtype = detectar_dispositivo_torch()

    tentativas: List[Tuple[int, Optional[object], str]] = [(device, dtype, "preferencial")]
    if device != -1:
        tentativas.append((-1, None, "fallback_cpu"))

    ultimo_erro: Optional[Exception] = None
    for device_id, dtype_tentativa, descricao in tentativas:
        try:
            logger.info(
                "Carregando SER: "
                f"{settings().ser_model} (modo={descricao}, device={'cuda' if device_id != -1 else 'cpu'})"
            )
            if carregar_ser_compativel is None:
                raise RuntimeError("Carregador SER compativel nao esta disponivel")
            ser = carregar_ser_compativel(
                identificador_modelo=settings().ser_model,
                device_id=device_id,
                dtype=dtype_tentativa,
            )
            logger.info("✓ SER carregado com todos os pesos do checkpoint")
            return ser
        except Exception as e:
            ultimo_erro = e
            logger.warning(
                "Falha ao carregar SER "
                f"(modo={descricao}, device={'cuda' if device_id != -1 else 'cpu'}): {e}"
            )

    logger.error(f"Falha ao carregar SER: {ultimo_erro}", exc_info=True)
    return None


def converter_segmentos_faster_whisper_em_chunks(
    segmentos: Iterable[Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Converte segmentos do faster-whisper para o formato de chunks do pipeline."""
    textos: List[str] = []
    chunks: List[Dict[str, Any]] = []

    for segmento in segmentos:
        texto = str(getattr(segmento, "text", "")).strip()
        inicio = getattr(segmento, "start", None)
        fim = getattr(segmento, "end", None)
        palavras: List[Dict[str, Any]] = []
        for palavra in getattr(segmento, "words", None) or []:
            texto_palavra = str(getattr(palavra, "word", ""))
            inicio_palavra = getattr(palavra, "start", None)
            fim_palavra = getattr(palavra, "end", None)
            if inicio_palavra is None or fim_palavra is None:
                continue
            palavras.append(
                {
                    "word": texto_palavra,
                    "start": float(inicio_palavra),
                    "end": float(fim_palavra),
                    "probability": float(getattr(palavra, "probability", 0.0)),
                }
            )

        if texto:
            textos.append(texto)

        chunks.append(
            {
                "text": texto,
                "timestamp": (
                    float(inicio) if inicio is not None else None,
                    float(fim) if fim is not None else None,
                ),
                "words": palavras,
            }
        )

    return " ".join(textos).strip(), chunks


def transcrever_com_stt(stt: STTRuntime, caminho: Path) -> TranscricaoResultado:
    """Executa a transcricao usando o backend de STT carregado."""
    if stt.backend == "faster-whisper":
        vad_parameters = None
        if stt.use_vad:
            vad_parameters = {"min_silence_duration_ms": settings().vad_min_silence_ms}

        kwargs_transcricao: Dict[str, Any] = {
            "language": settings().language,
            "task": "transcribe",
            "beam_size": stt.num_beams,
            "best_of": stt.best_of,
            "condition_on_previous_text": stt.condition_on_previous_text,
            "vad_filter": stt.use_vad,
            "vad_parameters": vad_parameters,
            "word_timestamps": True,
            "without_timestamps": False,
            "chunk_length": stt.chunk_length_s,
            "temperature": 0.0,
        }
        if stt.supports_batch_size:
            kwargs_transcricao["batch_size"] = stt.batch_size

        segmentos, info = stt.model.transcribe(
            str(caminho),
            **kwargs_transcricao,
        )
        texto_completo, chunks = converter_segmentos_faster_whisper_em_chunks(segmentos)
        return TranscricaoResultado(
            texto_completo=texto_completo,
            chunks=chunks,
            idioma_detectado=getattr(info, "language", None),
        )

    asr_output = stt.model(
        str(caminho),
        chunk_length_s=stt.chunk_length_s,
        batch_size=stt.batch_size,
        generate_kwargs={
            "language": settings().language,
            "task": "transcribe",
            "num_beams": stt.num_beams,
        },
        return_timestamps=True,
    )

    if isinstance(asr_output, dict):
        return TranscricaoResultado(
            texto_completo=str(asr_output.get("text", "")),
            chunks=list(asr_output.get("chunks", [])),
            idioma_detectado=settings().language,
        )

    return TranscricaoResultado(
        texto_completo=str(asr_output),
        chunks=[],
        idioma_detectado=settings().language,
    )


def transcrever_audio_com_fallback(caminho: Path) -> Tuple[TranscricaoResultado, STTRuntime]:
    """Transcreve audio com fallback automatico entre backends de STT."""
    global FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO

    tentativas_backend = [settings().backend]
    if settings().backend in {"auto", "faster-whisper"}:
        tentativas_backend.append("transformers")

    erros: List[str] = []
    for backend_preferido in dict.fromkeys(tentativas_backend):
        stt: Optional[STTRuntime] = None
        try:
            stt = carregar_stt(backend_preferido=backend_preferido)
            logger.info(
                "Transcrevendo com Whisper "
                f"(backend={stt.backend}, modelo={stt.model_id}, device={stt.device})..."
            )
            transcricao = transcrever_com_stt(stt, caminho)
            return transcricao, stt
        except Exception as e:
            erros.append(f"{backend_preferido}: {e}")
            if (
                stt is not None
                and stt.backend == "faster-whisper"
                and erro_indica_falha_cuda_faster_whisper(e)
            ):
                FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO = True
                logger.warning(
                    "Falha recorrente de CUDA detectada no faster-whisper. "
                    "A sessao passara a usar fallback em CPU quando necessario."
                )
                try:
                    if stt is not None:
                        liberar_memoria_modelos(stt.model)
                    logger.info(
                        "Tentando fallback do faster-whisper em CPU "
                        "para preservar timestamps mais estaveis."
                    )
                    stt_cpu = carregar_stt_faster_whisper(
                        model_id=stt.model_id,
                        device_forcado="cpu",
                    )
                    transcricao_cpu = transcrever_com_stt(stt_cpu, caminho)
                    return transcricao_cpu, stt_cpu
                except Exception as erro_cpu:
                    erros.append(f"faster-whisper-cpu: {erro_cpu}")
                    logger.warning(f"Fallback do faster-whisper em CPU falhou: {erro_cpu}")
            logger.warning(f"Falha na transcricao com backend STT {backend_preferido}: {e}")
            if stt is not None:
                liberar_memoria_modelos(stt.model)

    raise RuntimeError(
        "Falha ao transcrever audio em todos os backends tentados: " + " | ".join(erros)
    )


# ========================================
# ANÁLISE DE CONTEXTO
# ========================================
def pontuar_contexto(texto: str) -> Tuple[str, float]:
    """
    Analisa o contexto do texto usando vocabulário de futebol.

    CORREÇÕES APLICADAS:
    - Usa "termos" (consistente com dicionario.py)
    - Regex correto: \\b em vez de \\\\b
    - Usa VOCABULARIO_NORMALIZADO (pré-processado)

    Args:
        texto: Texto transcrito para análise

    Returns:
        Tuple[str, float]: (categoria_detectada, pontuacao_base)
    """
    txt_norm = normalizar_texto(texto)
    melhor_categoria = "neutro"
    melhor_pontuacao = 0.0

    for categoria, cfg in VOCABULARIO_NORMALIZADO.items():
        pontuacao_base = float(cfg["pontuacao_base"])

        for termo_norm in cfg["termos"]:
            # CORREÇÃO CRÍTICA: \b em vez de \\b
            # \b = word boundary (limite de palavra)
            padrao = rf"\b{re.escape(termo_norm)}\b"

            if re.search(padrao, txt_norm):
                logger.debug(f"Match '{termo_norm}' → {categoria} ({pontuacao_base}pts)")

                if pontuacao_base > melhor_pontuacao:
                    melhor_categoria = categoria
                    melhor_pontuacao = pontuacao_base

    return melhor_categoria, melhor_pontuacao


def _segmentar_texto_temporalmente(texto: str) -> List[str]:
    """Divide chunks muito longos em subfrases para reduzir vazamento temporal."""
    texto_bruto = str(texto).strip()
    if not texto_bruto:
        return []

    for padrao in (r"(?<=[\.\!\?])\s+", r"\s{2,}", r",\s+"):
        partes = [parte.strip() for parte in re.split(padrao, texto_bruto) if parte.strip()]
        if len(partes) > 1:
            return partes

    palavras = texto_bruto.split()
    if len(palavras) <= 8:
        return [texto_bruto]

    tamanho_bloco = 8
    return [
        " ".join(palavras[indice : indice + tamanho_bloco]).strip()
        for indice in range(0, len(palavras), tamanho_bloco)
        if " ".join(palavras[indice : indice + tamanho_bloco]).strip()
    ]


def _subdividir_chunk_temporalmente(
    texto: str,
    ts_ini: float,
    ts_fim: float,
) -> List[Tuple[float, float, str]]:
    texto_limpo = str(texto).strip()
    if not texto_limpo or ts_fim <= ts_ini:
        return []

    duracao_chunk = ts_fim - ts_ini
    partes = _segmentar_texto_temporalmente(texto_limpo)
    if duracao_chunk <= 8.0 or len(partes) <= 1:
        return [(ts_ini, ts_fim, texto_limpo)]

    duracao_parte = duracao_chunk / len(partes)
    return [
        (
            ts_ini + indice * duracao_parte,
            ts_ini + (indice + 1) * duracao_parte,
            parte,
        )
        for indice, parte in enumerate(partes)
    ]


def _extrair_texto_por_intervalo_legado(chunks: List[Dict], inicio_s: float, fim_s: float) -> str:
    """
    Extrai texto dos chunks do Whisper que se sobrepõem ao intervalo especificado.

    OTIMIZAÇÃO: Usa timestamps do Whisper para precisão temporal.

    Args:
        chunks: Lista de chunks com timestamps do Whisper
        inicio_s: Início do intervalo em segundos
        fim_s: Fim do intervalo em segundos

    Returns:
        str: Texto concatenado do intervalo
    """
    texto_intervalo = []

    for chunk in chunks:
        if "timestamp" not in chunk:
            continue

        ts = chunk["timestamp"]
        if ts is None or len(ts) != 2:
            continue

        ts_ini, ts_fim = ts

        # CORREÇÃO: Verifica se timestamps são None
        if ts_ini is None or ts_fim is None:
            continue

        # Verifica sobreposição: chunk toca o intervalo?
        if ts_ini <= fim_s and ts_fim >= inicio_s:
            texto_intervalo.append(chunk["text"])

    return " ".join(texto_intervalo).strip()


def _limpar_texto_extraido_por_palavras(partes: List[str]) -> str:
    texto = "".join(partes).strip()
    if not texto:
        return ""
    texto = re.sub(r"\s+", " ", texto)
    texto = re.sub(r"\s+([,.;:!?])", r"\1", texto)
    return texto.strip()


def _extrair_texto_por_palavras(
    chunks: List[Dict[str, Any]],
    inicio_s: float,
    fim_s: float,
) -> Optional[str]:
    partes: List[str] = []
    vistos: set[Tuple[int, int, str]] = set()
    encontrou_palavras = False

    for chunk in chunks:
        for palavra in chunk.get("words") or []:
            encontrou_palavras = True
            inicio_palavra = palavra.get("start")
            fim_palavra = palavra.get("end")
            texto_palavra = str(palavra.get("word", ""))
            if inicio_palavra is None or fim_palavra is None or not texto_palavra:
                continue
            if float(inicio_palavra) < fim_s and float(fim_palavra) > inicio_s:
                chave = (
                    int(round(float(inicio_palavra) * 100)),
                    int(round(float(fim_palavra) * 100)),
                    normalizar_texto(texto_palavra),
                )
                if chave in vistos:
                    continue
                vistos.add(chave)
                partes.append(texto_palavra)

    if not encontrou_palavras:
        return None
    return _limpar_texto_extraido_por_palavras(partes)


def extrair_texto_por_intervalo(chunks: List[Dict], inicio_s: float, fim_s: float) -> str:
    """
    Versao refinada da extracao temporal.

    Quando o Whisper devolve chunks muito longos, reparte o texto em subfrases
    e as distribui ao longo do intervalo do chunk para reduzir vazamento de
    contexto entre janelas vizinhas.
    """
    texto_intervalo = []
    textos_vistos = set()

    texto_por_palavras = _extrair_texto_por_palavras(chunks, inicio_s, fim_s)
    if texto_por_palavras is not None:
        return texto_por_palavras

    for chunk in chunks:
        if "timestamp" not in chunk:
            continue

        ts = chunk["timestamp"]
        if ts is None or len(ts) != 2:
            continue

        ts_ini, ts_fim = ts
        if ts_ini is None or ts_fim is None:
            continue

        for sub_ini, sub_fim, sub_texto in _subdividir_chunk_temporalmente(
            chunk.get("text", ""),
            float(ts_ini),
            float(ts_fim),
        ):
            if sub_ini < fim_s and sub_fim > inicio_s:
                chave = normalizar_texto(sub_texto)
                if chave and chave not in textos_vistos:
                    textos_vistos.add(chave)
                    texto_intervalo.append(sub_texto)

    return " ".join(texto_intervalo).strip()


def _chunks_possuem_timestamps_validos(chunks: List[Dict[str, Any]]) -> bool:
    for chunk in chunks:
        ts = chunk.get("timestamp")
        if ts is None or len(ts) != 2:
            continue
        ts_ini, ts_fim = ts
        if ts_ini is not None and ts_fim is not None:
            return True
    return False


def resolver_texto_janela(
    chunks: List[Dict[str, Any]],
    texto_completo: str,
    inicio_s: float,
    fim_s: float,
) -> str:
    if _chunks_possuem_timestamps_validos(chunks):
        return extrair_texto_por_intervalo(chunks, inicio_s, fim_s)
    return texto_completo


# ========================================
# PIPELINE PRINCIPAL
# ========================================
def main(
    caminho_audio: str = "teste_curto.wav", jan_ms: int = None, hop_ms: int = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Pipeline principal de análise.

    Args:
        caminho_audio: Caminho do arquivo de áudio
        jan_ms: Duração da janela em milissegundos (padrão: configs.JANELA_MS)
        hop_ms: Hop size em milissegundos (padrão: configs.HOP_MS)
    """
    # Usa configurações padrão se não especificado
    if jan_ms is None:
        jan_ms = JANELA_MS
    if hop_ms is None:
        hop_ms = HOP_MS

    if jan_ms <= 0 or hop_ms <= 0 or hop_ms > jan_ms:
        raise ValueError("Window and hop must be positive; hop cannot exceed window")

    logger.info("=" * 60)
    logger.info("INICIANDO ANÁLISE DE SENTIMENTO POR ÁUDIO")
    logger.info("=" * 60)

    # === VALIDAÇÃO DE ARQUIVO ===
    caminho = Path(caminho_audio)
    if not validar_arquivo_audio(caminho):
        return

    # === CARREGAMENTO DE ÁUDIO ===
    try:
        logger.info(f"Carregando áudio: {caminho.name}")
        y, sr = carregar_audio_mono(caminho, TAXA_DE_AMOSTRAGEM)
    except Exception as e:
        logger.error(f"ERRO ao carregar áudio: {e}", exc_info=True)
        return

    # === VALIDAÇÃO DE DURAÇÃO ===
    duracao_min_s = jan_ms / 1000.0
    if not validar_duracao_audio(y, sr, duracao_min_s):
        return

    # === TRANSCRIÇÃO COM WHISPER ===
    stt: Optional[STTRuntime] = None
    try:
        transcricao, stt = transcrever_audio_com_fallback(caminho)
        texto_completo = transcricao.texto_completo
        chunks = transcricao.chunks

        logger.info(f"\n{'=' * 60}")
        logger.info("TRANSCRICAO COMPLETA")
        logger.info(f"{'=' * 60}")
        logger.debug(texto_completo)
        logger.info(f"{'=' * 60}\n")

        if chunks:
            logger.info(f"Chunks com timestamps: {len(chunks)}")
        if transcricao.idioma_detectado:
            logger.info(f"Idioma da transcricao: {transcricao.idioma_detectado}")

    except Exception as e:
        logger.error(f"ERRO na transcrição: {e}", exc_info=True)
        return
    finally:
        if stt is not None:
            stt = None
            liberar_memoria_modelos()

    # === CARREGAMENTO DO SER ===
    ser = carregar_ser()
    if ser is None:
        logger.error("Falha ao carregar SER. Abortando.")
        return

    # === CONFIGURAÇÃO DE JANELAS ===
    jan_amostras = int((jan_ms / 1000.0) * sr)
    hop_amostras = int((hop_ms / 1000.0) * sr)

    logger.info("Parâmetros de janelamento:")
    logger.info(f"  Janela: {jan_ms}ms ({jan_amostras} amostras)")
    logger.info(f"  Hop: {hop_ms}ms ({hop_amostras} amostras)")
    logger.info(f"  Overlap: {100 * (1 - hop_amostras / jan_amostras):.1f}%")

    # === ANÁLISE POR JANELAS ===
    resultados = []
    janelas_processadas = 0
    janelas_falhadas = 0

    logger.info("\nProcessando janelas de áudio...")

    starts = list(range(0, len(y) - jan_amostras + 1, hop_amostras))
    if starts and starts[-1] + jan_amostras < len(y):
        starts.append(len(y) - jan_amostras)
    for inicio in starts:
        fim = inicio + jan_amostras
        recorte = y[inicio:fim]

        if recorte.size == 0:
            continue

        janelas_processadas += 1
        inicio_s = inicio / sr
        fim_s = fim / sr

        # === ANÁLISE DE EMOÇÃO (SER) ===
        try:
            recorte = recorte.astype(np.float32, copy=False)
            ser_output = ser({"array": recorte, "sampling_rate": sr})

            # Pega a emoção com maior confiança
            raw_label = str(ser_output[0]["label"]).lower().strip()
            confianca = float(ser_output[0].get("score", 0.0))

        except Exception as e:
            logger.warning(f"Falha SER [janela {inicio_s:.2f}-{fim_s:.2f}s]: {e}")
            ser = None
            liberar_memoria_modelos()
            raise RuntimeError(
                f"SER failed at {inicio_s:.2f}-{fim_s:.2f}s; no emotion will be fabricated"
            ) from e

        # Mapeia rótulo SER → abreviação
        emocao_abrev = MAPA_ROTULOS.get(raw_label, "neu")
        peso_emocao = float(MODULADORES_DE_EMOCAO.get(emocao_abrev, 0.0))

        # === ANÁLISE DE CONTEXTO ===
        # Usa texto completo apenas quando nao ha timestamps validos.
        # Se o intervalo estiver silencioso, ele deve permanecer vazio.
        texto_janela = resolver_texto_janela(
            chunks=chunks,
            texto_completo=texto_completo,
            inicio_s=inicio_s,
            fim_s=fim_s,
        )

        cat_contexto, peso_contexto = pontuar_contexto(texto_janela)

        # === ESCORE FINAL ===
        escore_final = peso_emocao + peso_contexto

        resultado = {
            "inicio_s": round(inicio_s, 2),
            "fim_s": round(fim_s, 2),
            "texto": texto_janela,
            "texto_preview": (
                texto_janela[:50] + "..." if len(texto_janela) > 50 else texto_janela
            ),
            "emocao_raw": raw_label,
            "emocao": emocao_abrev,
            "confianca_emo": round(confianca, 3),
            "peso_emo": peso_emocao,
            "categoria_ctx": cat_contexto,
            "peso_ctx": peso_contexto,
            "escore": round(escore_final, 3),
        }

        resultados.append(resultado)

        # Log detalhado da primeira janela (debug)
        if janelas_processadas == 1:
            logger.debug(f"Primeira janela (exemplo): {resultado}")

    # === RESULTADOS FINAIS ===
    logger.info(f"\n{'=' * 60}")
    logger.info("RESULTADOS DA ANÁLISE")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total de janelas processadas: {janelas_processadas}")
    logger.info(f"Janelas com falha no SER: {janelas_falhadas}")

    if not resultados:
        logger.warning("Nenhuma janela válida foi processada!")
        return

    # Mostra primeiras 10 janelas
    logger.info("\nPrimeiras 10 janelas:")
    for i, r in enumerate(resultados[:10], 1):
        logger.info(
            f"  [{i}] {r['inicio_s']}-{r['fim_s']}s | "
            f"Emo: {r['emocao']}({r['confianca_emo']}) | "
            f"Ctx: {r['categoria_ctx']} | "
            f"Escore: {r['escore']}"
        )

    if len(resultados) > 10:
        logger.info(f"  ... (+{len(resultados) - 10} janelas)")

    # === ESTATÍSTICAS GLOBAIS ===
    escores = [r["escore"] for r in resultados]
    emocoes = [r["emocao"] for r in resultados]
    categorias = [r["categoria_ctx"] for r in resultados]

    # Estatísticas de escore
    escore_medio = float(np.mean(escores))
    escore_min = float(np.min(escores))
    escore_max = float(np.max(escores))
    escore_std = float(np.std(escores))

    # Distribuição de emoções
    from collections import Counter

    dist_emocoes = Counter(emocoes)
    dist_categorias = Counter(categorias)

    logger.info(f"\n{'=' * 60}")
    logger.info("ESTATÍSTICAS GLOBAIS")
    logger.info(f"{'=' * 60}")
    logger.info(f"Escore médio: {escore_medio:.3f}")
    logger.info(f"Escore mín/máx: {escore_min:.3f} / {escore_max:.3f}")
    logger.info(f"Desvio padrão: {escore_std:.3f}")

    logger.info("\nDistribuição de Emoções:")
    for emo, count in dist_emocoes.most_common():
        perc = 100 * count / len(emocoes)
        logger.info(f"  {emo}: {count} ({perc:.1f}%)")

    logger.info("\nTop 5 Categorias de Contexto:")
    for cat, count in dist_categorias.most_common(5):
        perc = 100 * count / len(categorias)
        logger.info(f"  {cat}: {count} ({perc:.1f}%)")

    logger.info("Raw scores are heuristic signals; see the calibrated output and limitations.")

    # Alerta de janelas falhadas
    if janelas_falhadas > 0:
        taxa_falha = 100 * janelas_falhadas / janelas_processadas
        logger.warning(
            f"\nAVISO: {janelas_falhadas} janelas falharam no SER ({taxa_falha:.1f}%). "
            f"Considere verificar qualidade do audio."
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("ANÁLISE CONCLUÍDA")
    logger.info(f"{'=' * 60}\n")

    ser = None
    liberar_memoria_modelos()

    return resultados
