from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coach_emotion.evaluation import AvaliadorAcuracia, ConfigAvaliacao


class TesteAvaliador(unittest.TestCase):
    def test_agregacao_e_metricas_com_nulos_ignorados(self) -> None:
        with TemporaryDirectory() as pasta_temporaria:
            caminho_csv = Path(pasta_temporaria) / "gt.csv"
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

            resultados_pipeline = [
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

            avaliador = AvaliadorAcuracia(
                caminho_csv,
                config=ConfigAvaliacao(politica_nulos="ignore"),
            )
            resultado = avaliador.avaliar(resultados_pipeline)

            self.assertEqual(resultado.total_intervalos_metricas, 2)
            self.assertEqual(resultado.intervalos_ignorados_nulos, 1)
            self.assertEqual(resultado.intervalos_sem_predicao, 0)
            self.assertEqual(resultado.rotulos_reais, ["sad", "ang"])
            self.assertEqual(resultado.rotulos_preditos, ["sad", "ang"])
            self.assertAlmostEqual(resultado.macro_f1, 1.0)
            self.assertIsNotNone(resultado.mae_escore)
            self.assertIsNotNone(resultado.rmse_escore)
            self.assertAlmostEqual(resultado.mae_escore, 0.0)
            self.assertAlmostEqual(resultado.rmse_escore, 0.0)

    def test_nulos_podem_ser_mapeados_para_neutro(self) -> None:
        with TemporaryDirectory() as pasta_temporaria:
            caminho_csv = Path(pasta_temporaria) / "gt.csv"
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

            resultados_pipeline = [
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

            avaliador = AvaliadorAcuracia(
                caminho_csv,
                config=ConfigAvaliacao(politica_nulos="map_to_neu"),
            )
            resultado = avaliador.avaliar(resultados_pipeline)

            self.assertEqual(resultado.total_intervalos_metricas, 3)
            self.assertEqual(resultado.intervalos_ignorados_nulos, 0)
            self.assertEqual(resultado.rotulos_reais, ["sad", "ang", "neu"])
            self.assertEqual(resultado.rotulos_preditos, ["sad", "ang", "neu"])
            self.assertIn("neu", resultado.metricas_por_classe)
            self.assertEqual(resultado.metricas_por_classe["neu"].support, 1)

    def test_csv_compacto_reconstroi_intervalos_e_mapeia_codigos_humanos(self) -> None:
        with TemporaryDirectory() as pasta_temporaria:
            caminho_csv = Path(pasta_temporaria) / "gt_compacto.csv"
            caminho_csv.write_text(
                "\n".join(
                    [
                        "  ,0,0",
                        "0:05 - 0:10:,0,1",
                        "0;15,2,4",
                        "0;20,,",
                    ]
                ),
                encoding="utf-8",
            )

            resultados_pipeline = [
                {
                    "inicio_s": 0.0,
                    "fim_s": 5.0,
                    "emocao": "neutral",
                    "escore": 0.0,
                    "confianca_emo": 0.4,
                },
                {
                    "inicio_s": 5.0,
                    "fim_s": 10.0,
                    "emocao": "neutral",
                    "escore": 0.0,
                    "confianca_emo": 0.4,
                },
                {
                    "inicio_s": 10.0,
                    "fim_s": 15.0,
                    "emocao": "ang",
                    "escore": 2.0,
                    "confianca_emo": 0.9,
                },
                {
                    "inicio_s": 15.0,
                    "fim_s": 20.0,
                    "emocao": "ang",
                    "escore": 5.0,
                    "confianca_emo": 0.9,
                },
            ]

            avaliador = AvaliadorAcuracia(
                caminho_csv,
                config=ConfigAvaliacao(politica_nulos="ignore"),
            )
            intervalos = avaliador.carregar_ground_truth()
            resultado = avaliador.avaliar(resultados_pipeline)

            self.assertEqual(
                [(item.inicio_s, item.fim_s) for item in intervalos],
                [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0)],
            )
            self.assertEqual(intervalos[0].rotulo_humano, "neutro sem intervencao")
            self.assertEqual(intervalos[0].rotulo_sistema, "neu")
            self.assertEqual(intervalos[1].rotulo_humano, "neutro com intervencao")
            self.assertEqual(intervalos[1].rotulo_sistema, "neu")
            self.assertEqual(intervalos[2].rotulo_humano, "raiva")
            self.assertEqual(intervalos[2].rotulo_sistema, "ang")
            self.assertIsNone(intervalos[3].rotulo_humano)
            self.assertEqual(resultado.intervalos_rotulados_originais, 3)
            self.assertEqual(resultado.total_intervalos_metricas, 3)
            self.assertEqual(resultado.intervalos_ignorados_nulos, 1)
            self.assertEqual(resultado.intervalos_sem_metrica_humana, 1)
            self.assertEqual(resultado.rotulos_reais, ["neu", "neu", "ang"])
            self.assertEqual(resultado.rotulos_preditos, ["neu", "neu", "ang"])
            self.assertAlmostEqual(resultado.mae_escore or 0.0, 0.0)
            self.assertAlmostEqual(resultado.rmse_escore or 0.0, 0.0)

    def test_tsv_compacto_tambem_e_suportado(self) -> None:
        with TemporaryDirectory() as pasta_temporaria:
            caminho_tsv = Path(pasta_temporaria) / "gt_compacto.tsv"
            caminho_tsv.write_text(
                "\n".join(
                    [
                        "  \t0\t0",
                        "0:05 - 0:10:\t1\t2",
                        "0;15\t2\t3",
                        "0;20\t2\t4",
                    ]
                ),
                encoding="utf-8",
            )

            avaliador = AvaliadorAcuracia(caminho_tsv)
            intervalos = avaliador.carregar_ground_truth()

            self.assertEqual(len(intervalos), 4)
            self.assertEqual(intervalos[0].rotulo_sistema, "neu")
            self.assertEqual(intervalos[1].rotulo_sistema, "hap")
            self.assertEqual(intervalos[2].rotulo_sistema, "sur")
            self.assertEqual(intervalos[3].rotulo_sistema, "ang")


if __name__ == "__main__":
    unittest.main()
