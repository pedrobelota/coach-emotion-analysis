from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from coach_emotion.evaluate_audio import main_cli, preparar_midia_para_avaliacao


class TesteExecucaoAvaliacao(unittest.TestCase):
    def test_preparar_midia_para_avaliacao_usa_faixa_do_ground_truth(self) -> None:
        with TemporaryDirectory() as pasta_temporaria:
            base = Path(pasta_temporaria)
            entrada = base / "treinadora.mp4"
            entrada.write_bytes(b"mp4")
            saida = base / "preparado.wav"
            avaliador = Mock()
            avaliador.obter_faixa_temporal_csv.return_value = (5.0, 20.0)

            with patch("coach_emotion.evaluate_audio.shutil.which", return_value="ffmpeg"):
                with patch("coach_emotion.evaluate_audio.subprocess.run") as run:
                    caminho = preparar_midia_para_avaliacao(entrada, avaliador, saida)

            self.assertEqual(caminho, saida)
            comando = run.call_args.args[0]
            self.assertIn("ffmpeg", comando[0])
            self.assertEqual(comando[3], "5.000")
            self.assertEqual(comando[5], "15.000")
            self.assertEqual(comando[-1], str(saida))

    def test_main_cli_executa_pipeline_stubado_e_exporta_csv(self) -> None:
        with TemporaryDirectory() as pasta_temporaria:
            base = Path(pasta_temporaria)
            caminho_audio = base / "trecho.wav"
            caminho_audio.write_bytes(b"RIFF")

            caminho_csv = base / "gt.csv"
            caminho_csv.write_text(
                "\n".join(
                    [
                        "0:00 - 0:05:,2,Tristeza,5",
                        "0:05 - 0:10:,3,Raiva,6",
                        "0:10 - 0:15:,0,,1",
                    ]
                ),
                encoding="utf-8",
            )

            caminho_saida = base / "detalhes.csv"
            predicoes_stubadas = [
                {
                    "inicio_s": 0.0,
                    "fim_s": 4.0,
                    "emocao": "sad",
                    "escore": 4.0,
                    "confianca_emo": 0.90,
                },
                {
                    "inicio_s": 2.0,
                    "fim_s": 6.0,
                    "emocao": "sad",
                    "escore": 6.0,
                    "confianca_emo": 0.70,
                },
                {
                    "inicio_s": 6.0,
                    "fim_s": 10.0,
                    "emocao": "ang",
                    "escore": 5.0,
                    "confianca_emo": 0.80,
                },
                {
                    "inicio_s": 8.0,
                    "fim_s": 10.0,
                    "emocao": "ang",
                    "escore": 7.0,
                    "confianca_emo": 0.95,
                },
                {
                    "inicio_s": 10.0,
                    "fim_s": 15.0,
                    "emocao": "neutral",
                    "escore": 1.0,
                    "confianca_emo": 0.60,
                },
            ]

            saida_capturada = io.StringIO()
            with patch(
                "coach_emotion.evaluate_audio.preparar_midia_para_avaliacao",
                return_value=caminho_audio,
            ):
                with patch(
                    "coach_emotion.evaluate_audio.executar_pipeline",
                    return_value=predicoes_stubadas,
                ):
                    with redirect_stdout(saida_capturada):
                        codigo_saida = main_cli(
                            [
                                "--audio",
                                str(caminho_audio),
                                "--csv-ground-truth",
                                str(caminho_csv),
                                "--politica-nulos",
                                "map_to_neu",
                                "--salvar-detalhes-csv",
                                str(caminho_saida),
                            ]
                        )

            saida = saida_capturada.getvalue()
            self.assertEqual(codigo_saida, 0)
            self.assertIn("AVALIACAO DE ACURACIA", saida)
            self.assertIn("Macro metrics", saida)
            self.assertTrue(caminho_saida.exists())

            conteudo_csv = caminho_saida.read_text(encoding="utf-8-sig")
            self.assertIn("rotulo_predito", conteudo_csv)
            self.assertIn("duracao_s", conteudo_csv)
            self.assertIn("erro_escore_abs", conteudo_csv)
            self.assertIn("sad", conteudo_csv)
            self.assertIn("neu", conteudo_csv)


if __name__ == "__main__":
    unittest.main()
