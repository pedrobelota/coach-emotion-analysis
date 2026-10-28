# -*- coding: utf-8 -*-
"""
Configurações do projeto de Análise de Sentimento por Áudio (técnico de futebol).

Melhorias aplicadas:
- Modelo SER otimizado para multilíngue (XLSR)
- Opções de Whisper configuráveis por desempenho
- Logger estruturado
- Validação de configurações
"""

import logging
from dataclasses import dataclass


# ------------------------------
# Modelos (Hugging Face Hub)
# ------------------------------
@dataclass
class ModelConfig:
    """Configuração de modelo com metadados"""

    id: str
    descricao: str
    vram_gb: float
    idiomas: list[str]


# Opções de Whisper por caso de uso
WHISPER_CONFIGS = {
    "rapido": ModelConfig(
        id="openai/whisper-medium",
        descricao="Boa velocidade com precisao equilibrada",
        vram_gb=1.5,
        idiomas=["pt", "en", "es"],
    ),
    "turbo": ModelConfig(
        id="openai/whisper-large-v3-turbo",
        descricao="Melhor relacao velocidade/qualidade para GPU",
        vram_gb=6.0,
        idiomas=["pt", "en", "es"],
    ),
    "balanceado": ModelConfig(
        id="openai/whisper-medium",
        descricao="Balanceado precisão/velocidade",
        vram_gb=1.5,
        idiomas=["pt", "en", "es"],
    ),
    "preciso": ModelConfig(
        id="openai/whisper-large-v3",
        descricao="Máxima precisão (requer GPU)",
        vram_gb=3.0,
        idiomas=["pt", "en", "es"],
    ),
    "pt_otimizado": ModelConfig(
        id="primeline/whisper-large-v3-portuguese",
        descricao="Otimizado para português brasileiro",
        vram_gb=3.0,
        idiomas=["pt"],
    ),
}

# Selecione o perfil desejado (ajuste aqui)
PERFIL_WHISPER = "turbo"  # Opções: rapido, turbo, balanceado, preciso, pt_otimizado

MODELOS = {
    "stt": WHISPER_CONFIGS[PERFIL_WHISPER].id,
    # SER fine-tuned on English acted speech (RAVDESS); Portuguese transfer is unvalidated.
    "ser": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
}

# Idioma do Whisper (código ISO 639-1)
STT_LANGUAGE = "pt"

# Backend e otimizacoes de STT
STT_BACKEND = "auto"  # auto, faster-whisper, transformers
STT_CHUNK_LENGTH_S = 30
STT_BATCH_SIZE_GPU = 8
STT_BATCH_SIZE_CPU = 1
STT_NUM_BEAMS = 1
STT_BEST_OF = 1
STT_USE_VAD = True
STT_VAD_MIN_SILENCE_MS = 500
STT_CONDITION_ON_PREVIOUS_TEXT = False
STT_COMPUTE_TYPE_GPU = "float16"
STT_COMPUTE_TYPE_CPU = "int8"
STT_NUM_WORKERS = 1
STT_CPU_THREADS = 0

# Taxa de amostragem padrão (16kHz é padrão para a maioria dos modelos)
TAXA_DE_AMOSTRAGEM = 16000

# Parâmetros de janelamento
JANELA_MS = 4000  # 4 segundos por janela
HOP_MS = 2000  # 2 segundos de overlap (50%)

# ------------------------------
# Mapa de rótulos do SER -> abreviações
# ------------------------------
# Harmoniza diferentes saídas de modelos SER
MAPA_ROTULOS = {
    "happy": "hap",
    "happiness": "hap",
    "joy": "hap",
    "neutral": "neu",
    "calm": "neu",
    "angry": "ang",
    "anger": "ang",
    "sad": "sad",
    "sadness": "sad",
    "disgust": "dis",
    "fear": "fea",
    "fearful": "fea",
    # Surpresa passa a ser preservada como classe propria para avaliacao.
    "surprise": "sur",
    "surprised": "sur",
}

# ------------------------------
# Moduladores de emoção
# ------------------------------
# Pesos aplicados por emoção detectada
# Ajuste conforme o comportamento desejado no contexto do futebol
MODULADORES_DE_EMOCAO = {
    "hap": 0.8,  # Felicidade/excitação → positivo
    "neu": 0.0,  # Neutro → sem modulação
    "ang": 1.0,  # Raiva/irritação → alta intensidade
    "sad": -0.6,  # Tristeza → negativo moderado
    "dis": -0.8,  # Nojo/repulsa → forte negativo
    "fea": 0.6,  # Medo/apreensão → tensão/pressão
    "sur": 0.0,  # Surpresa → classe distinta, sem enviesar o escore final
}


# ------------------------------
# Validação de configurações
# ------------------------------
def validar_configs() -> bool:
    """Valida se as configurações são consistentes"""
    logger = logging.getLogger(__name__)

    # Verifica se todas as emoções mapeadas têm modulador
    emocoes_sem_modulador = set(MAPA_ROTULOS.values()) - set(MODULADORES_DE_EMOCAO.keys())
    if emocoes_sem_modulador:
        logger.warning(f"Emoções sem modulador definido: {emocoes_sem_modulador}")

    # Verifica se TAXA_DE_AMOSTRAGEM é válida
    if TAXA_DE_AMOSTRAGEM not in [8000, 16000, 22050, 44100, 48000]:
        logger.warning(f"Taxa de amostragem incomum: {TAXA_DE_AMOSTRAGEM} Hz")

    # Verifica se janela e hop fazem sentido
    if HOP_MS > JANELA_MS:
        logger.error("HOP_MS não pode ser maior que JANELA_MS")
        return False

    if STT_BACKEND not in {"auto", "faster-whisper", "transformers"}:
        logger.error(f"Backend STT inválido: {STT_BACKEND}")
        return False

    if STT_CHUNK_LENGTH_S <= 0:
        logger.error("STT_CHUNK_LENGTH_S deve ser maior que zero")
        return False

    if STT_NUM_BEAMS <= 0 or STT_BEST_OF <= 0:
        logger.error("STT_NUM_BEAMS e STT_BEST_OF devem ser maiores que zero")
        return False

    logger.info(
        "Configuração validada: "
        f"Whisper={PERFIL_WHISPER}, backend={STT_BACKEND}, SER={MODELOS['ser']}"
    )
    return True
