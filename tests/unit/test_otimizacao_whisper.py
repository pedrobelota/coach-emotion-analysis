import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from coach_emotion import pipeline
from coach_emotion.pipeline import (
    STTRuntime,
    TranscricaoResultado,
    carregar_stt,
    converter_segmentos_faster_whisper_em_chunks,
    erro_indica_falha_cuda_faster_whisper,
    extrair_texto_por_intervalo,
    resolver_modelo_faster_whisper,
    resolver_texto_janela,
    selecionar_backend_stt,
    transcrever_audio_com_fallback,
    transcrever_com_stt,
)
from coach_emotion.runtime import RuntimeConfig, configuration


class TesteOtimizacaoWhisper(unittest.TestCase):
    def test_backend_auto_prefere_faster_whisper(self) -> None:
        with patch("coach_emotion.pipeline.FASTER_WHISPER_AVAILABLE", True):
            with patch("coach_emotion.pipeline.FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO", False):
                self.assertEqual(selecionar_backend_stt("auto"), "faster-whisper")

    def test_backend_auto_vai_para_transformers_quando_cuda_esta_indisponivel(self) -> None:
        with patch("coach_emotion.pipeline.FASTER_WHISPER_AVAILABLE", True):
            with patch("coach_emotion.pipeline.FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO", True):
                self.assertEqual(selecionar_backend_stt("auto"), "faster-whisper")

    def test_detecta_mensagem_de_falha_cuda_do_faster_whisper(self) -> None:
        self.assertTrue(
            erro_indica_falha_cuda_faster_whisper(
                RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
            )
        )
        self.assertFalse(erro_indica_falha_cuda_faster_whisper(RuntimeError("falha generica")))

    def test_falha_cuda_marca_sessao_para_evitar_novas_tentativas(self) -> None:
        runtime_fast = STTRuntime(
            backend="faster-whisper",
            model=Mock(),
            model_id="openai/whisper-large-v3-turbo",
            device="cuda",
            compute_type="default",
            chunk_length_s=30,
            batch_size=8,
            num_beams=1,
            best_of=1,
            use_vad=True,
            condition_on_previous_text=False,
            supports_batch_size=True,
        )
        runtime_cpu = STTRuntime(
            backend="faster-whisper",
            model=Mock(),
            model_id="openai/whisper-large-v3-turbo",
            device="cpu",
            compute_type="int8",
            chunk_length_s=30,
            batch_size=1,
            num_beams=1,
            best_of=1,
            use_vad=True,
            condition_on_previous_text=False,
        )
        trans_ok = TranscricaoResultado(
            texto_completo="fala valida",
            chunks=[{"text": "fala valida", "timestamp": (0.0, 1.0)}],
            idioma_detectado="pt",
        )

        with configuration(RuntimeConfig(backend="auto")):
            with patch("coach_emotion.pipeline.FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO", False):
                with patch("coach_emotion.pipeline.carregar_stt", return_value=runtime_fast):
                    with patch(
                        "coach_emotion.pipeline.carregar_stt_faster_whisper",
                        return_value=runtime_cpu,
                    ):
                        with patch(
                            "coach_emotion.pipeline.transcrever_com_stt",
                            side_effect=[
                                RuntimeError(
                                    "Library cublas64_12.dll is not found or cannot be loaded"
                                ),
                                trans_ok,
                            ],
                        ):
                            with patch("coach_emotion.pipeline.liberar_memoria_modelos"):
                                _, runtime = transcrever_audio_com_fallback(Path("audio.wav"))

                    self.assertEqual(runtime.backend, "faster-whisper")
                    self.assertEqual(runtime.device, "cpu")
                    self.assertTrue(pipeline.FASTER_WHISPER_CUDA_INDISPONIVEL_NA_SESSAO)
                    with patch("coach_emotion.pipeline.FASTER_WHISPER_AVAILABLE", True):
                        self.assertEqual(selecionar_backend_stt("auto"), "faster-whisper")

    def test_resolve_modelo_compativel_faster_whisper(self) -> None:
        self.assertEqual(
            resolver_modelo_faster_whisper("openai/whisper-large-v3-turbo"),
            "large-v3-turbo",
        )
        self.assertIsNone(resolver_modelo_faster_whisper("primeline/whisper-large-v3-portuguese"))

    def test_auto_faz_fallback_para_transformers_quando_faster_whisper_falha(self) -> None:
        runtime_transformers = STTRuntime(
            backend="transformers",
            model=Mock(),
            model_id="openai/whisper-medium",
            device="cuda",
            compute_type="torch.float16",
            chunk_length_s=30,
            batch_size=8,
            num_beams=1,
            best_of=1,
            use_vad=False,
            condition_on_previous_text=True,
        )

        with patch(
            "coach_emotion.pipeline.carregar_stt_faster_whisper", side_effect=RuntimeError("falhou")
        ):
            with patch(
                "coach_emotion.pipeline.carregar_stt_transformers",
                return_value=runtime_transformers,
            ):
                runtime = carregar_stt(
                    model_id="openai/whisper-large-v3-turbo",
                    backend_preferido="auto",
                )

        self.assertEqual(runtime.backend, "transformers")

    def test_transcrever_audio_faz_fallback_quando_primeira_inferencia_falha(self) -> None:
        runtime_fast = STTRuntime(
            backend="faster-whisper",
            model=Mock(),
            model_id="openai/whisper-large-v3-turbo",
            device="cuda",
            compute_type="default",
            chunk_length_s=30,
            batch_size=8,
            num_beams=1,
            best_of=1,
            use_vad=True,
            condition_on_previous_text=False,
            supports_batch_size=True,
        )
        runtime_cpu = STTRuntime(
            backend="faster-whisper",
            model=Mock(),
            model_id="openai/whisper-large-v3-turbo",
            device="cpu",
            compute_type="int8",
            chunk_length_s=30,
            batch_size=1,
            num_beams=1,
            best_of=1,
            use_vad=True,
            condition_on_previous_text=False,
        )
        trans_ok = TranscricaoResultado(
            texto_completo="fala valida",
            chunks=[{"text": "fala valida", "timestamp": (0.0, 1.0)}],
            idioma_detectado="pt",
        )

        with configuration(RuntimeConfig(backend="auto")):
            with patch("coach_emotion.pipeline.carregar_stt", return_value=runtime_fast):
                with patch(
                    "coach_emotion.pipeline.carregar_stt_faster_whisper",
                    return_value=runtime_cpu,
                ):
                    with patch(
                        "coach_emotion.pipeline.transcrever_com_stt",
                        side_effect=[
                            RuntimeError(
                                "Library cublas64_12.dll is not found or cannot be loaded"
                            ),
                            trans_ok,
                        ],
                    ):
                        with patch("coach_emotion.pipeline.liberar_memoria_modelos") as liberar:
                            transcricao, runtime = transcrever_audio_com_fallback(Path("audio.wav"))

        self.assertEqual(runtime.backend, "faster-whisper")
        self.assertEqual(runtime.device, "cpu")
        self.assertEqual(transcricao.texto_completo, "fala valida")
        liberar.assert_called()

    def test_converte_segmentos_faster_whisper_para_chunks(self) -> None:
        segmentos = [
            SimpleNamespace(start=0.0, end=1.2, text=" Ola "),
            SimpleNamespace(start=1.2, end=2.7, text=" mundo "),
        ]

        texto, chunks = converter_segmentos_faster_whisper_em_chunks(segmentos)

        self.assertEqual(texto, "Ola mundo")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["timestamp"], (0.0, 1.2))
        self.assertEqual(chunks[1]["text"], "mundo")
        self.assertEqual(chunks[0]["words"], [])

    def test_extrai_texto_temporalizado_de_chunk_longo(self) -> None:
        chunks = [
            {
                "text": "Boa! Bateu a cabeca? Quer trocar? Vamos!",
                "timestamp": (0.0, 20.0),
            }
        ]

        self.assertEqual(extrair_texto_por_intervalo(chunks, 0.0, 5.0), "Boa!")
        self.assertEqual(
            extrair_texto_por_intervalo(chunks, 5.0, 10.0),
            "Bateu a cabeca?",
        )
        self.assertEqual(
            extrair_texto_por_intervalo(chunks, 10.0, 15.0),
            "Quer trocar?",
        )
        self.assertEqual(extrair_texto_por_intervalo(chunks, 15.0, 20.0), "Vamos!")

    def test_extrai_texto_prioriza_timestamps_por_palavra(self) -> None:
        chunks = [
            {
                "text": "Boa! Bateu a cabeca? Quer trocar?",
                "timestamp": (0.0, 20.0),
                "words": [
                    {"word": " Boa!", "start": 0.3, "end": 0.8},
                    {"word": " Bateu", "start": 7.1, "end": 7.5},
                    {"word": " a", "start": 7.5, "end": 7.7},
                    {"word": " cabeca?", "start": 7.7, "end": 8.4},
                    {"word": " Quer", "start": 15.2, "end": 15.6},
                    {"word": " trocar?", "start": 15.6, "end": 16.3},
                ],
            }
        ]

        self.assertEqual(extrair_texto_por_intervalo(chunks, 0.0, 5.0), "Boa!")
        self.assertEqual(
            extrair_texto_por_intervalo(chunks, 5.0, 10.0),
            "Bateu a cabeca?",
        )
        self.assertEqual(extrair_texto_por_intervalo(chunks, 10.0, 14.0), "")
        self.assertEqual(
            extrair_texto_por_intervalo(chunks, 14.0, 18.0),
            "Quer trocar?",
        )

    def test_resolver_texto_janela_nao_vaza_texto_completo_em_silencio(self) -> None:
        chunks = [
            {
                "text": "Boa! Bateu a cabeca?",
                "timestamp": (0.0, 20.0),
                "words": [
                    {"word": " Boa!", "start": 0.3, "end": 0.8},
                    {"word": " Bateu", "start": 7.1, "end": 7.5},
                    {"word": " a", "start": 7.5, "end": 7.7},
                    {"word": " cabeca?", "start": 7.7, "end": 8.4},
                ],
            }
        ]

        self.assertEqual(
            resolver_texto_janela(chunks, "Boa! Bateu a cabeca?", 10.0, 14.0),
            "",
        )

    def test_transcricao_faster_whisper_repassa_parametros_otimizados(self) -> None:
        segmentos = [SimpleNamespace(start=0.0, end=1.0, text="teste")]
        info = SimpleNamespace(language="pt")
        modelo = Mock()
        modelo.transcribe.return_value = (segmentos, info)

        runtime = STTRuntime(
            backend="faster-whisper",
            model=modelo,
            model_id="openai/whisper-large-v3-turbo",
            device="cuda",
            compute_type="float16",
            chunk_length_s=30,
            batch_size=8,
            num_beams=1,
            best_of=1,
            use_vad=True,
            condition_on_previous_text=False,
            supports_batch_size=True,
        )

        with patch("coach_emotion.pipeline.carregar_audio_mono", return_value=([0.0], 16000)):
            resultado = transcrever_com_stt(runtime, Path("audio.wav"))
        self.assertEqual(modelo.transcribe.call_args.args[0], [0.0])

        kwargs = modelo.transcribe.call_args.kwargs
        self.assertEqual(kwargs["beam_size"], 1)
        self.assertEqual(kwargs["best_of"], 1)
        self.assertEqual(kwargs["chunk_length"], 30)
        self.assertEqual(kwargs["batch_size"], 8)
        self.assertTrue(kwargs["vad_filter"])
        self.assertFalse(kwargs["condition_on_previous_text"])
        self.assertTrue(kwargs["word_timestamps"])
        self.assertEqual(resultado.texto_completo, "teste")
        self.assertEqual(resultado.idioma_detectado, "pt")

    def test_transcricao_transformers_repassa_chunk_batch_e_num_beams(self) -> None:
        modelo = Mock(
            return_value={
                "text": "texto",
                "chunks": [{"text": "texto", "timestamp": (0.0, 1.0)}],
            }
        )
        runtime = STTRuntime(
            backend="transformers",
            model=modelo,
            model_id="openai/whisper-medium",
            device="cpu",
            compute_type=None,
            chunk_length_s=30,
            batch_size=1,
            num_beams=1,
            best_of=1,
            use_vad=False,
            condition_on_previous_text=True,
        )

        resultado = transcrever_com_stt(runtime, Path("audio.wav"))

        kwargs = modelo.call_args.kwargs
        self.assertEqual(kwargs["chunk_length_s"], 30)
        self.assertEqual(kwargs["batch_size"], 1)
        self.assertEqual(kwargs["generate_kwargs"]["num_beams"], 1)
        self.assertEqual(resultado.texto_completo, "texto")
        self.assertEqual(len(resultado.chunks), 1)


if __name__ == "__main__":
    unittest.main()
