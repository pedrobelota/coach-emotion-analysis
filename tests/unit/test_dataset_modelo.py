from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coach_emotion.dataset import ConfigDatasetModelo, GeradorDatasetModelo


class TesteDatasetModelo(unittest.TestCase):
    def test_gera_codigos_0_a_6_por_intervalo(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 2.4,
                "texto": "",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 0.0,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 2.5,
                "fim_s": 4.9,
                "texto": "",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 0.0,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 5.0,
                "fim_s": 7.4,
                "texto": "Muito bem, boa!",
                "emocao": "neu",
                "categoria_ctx": "REFORCO_POSITIVO",
                "escore": 3.0,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 7.5,
                "fim_s": 9.9,
                "texto": "Muito bem, boa!",
                "emocao": "neu",
                "categoria_ctx": "REFORCO_POSITIVO",
                "escore": 3.0,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 10.0,
                "fim_s": 12.4,
                "texto": "Eita, caramba!",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 12.5,
                "fim_s": 14.9,
                "texto": "Eita, caramba!",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 15.0,
                "fim_s": 17.4,
                "texto": "Nao pode, acorda!",
                "emocao": "neu",
                "categoria_ctx": "CRITICA_CONSTRUTIVA",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 17.5,
                "fim_s": 19.9,
                "texto": "Nao pode, acorda!",
                "emocao": "neu",
                "categoria_ctx": "CRITICA_CONSTRUTIVA",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 20.0,
                "fim_s": 22.4,
                "texto": "Poxa, errou, era pra fazer.",
                "emocao": "neu",
                "categoria_ctx": "FRUSTRACAO_E_LAMENTO",
                "escore": -1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 22.5,
                "fim_s": 24.9,
                "texto": "Poxa, errou, era pra fazer.",
                "emocao": "neu",
                "categoria_ctx": "FRUSTRACAO_E_LAMENTO",
                "escore": -1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 25.0,
                "fim_s": 27.4,
                "texto": "Ta doendo? Bateu a cabeca? Quer trocar?",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 27.5,
                "fim_s": 29.9,
                "texto": "Ta doendo? Bateu a cabeca? Quer trocar?",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
        ]

        gerador = GeradorDatasetModelo(config=ConfigDatasetModelo(duracao_intervalo_s=5.0))
        linhas = gerador.gerar(resultados_pipeline, duracao_total_s=30.0)

        self.assertEqual(
            [linha.intervalo for linha in linhas[:6]],
            [
                "0:00 - 0:05:",
                "0:05 - 0:10:",
                "0:10 - 0:15:",
                "0:15 - 0:20:",
                "0:20 - 0:25:",
                "0:25 - 0:30:",
            ],
        )
        self.assertEqual([linha.codigo_emocao for linha in linhas[:6]], [0, 2, 3, 4, 5, 6])
        self.assertEqual([linha.avaliacao_modelo for linha in linhas[:6]], [0, 1, 1, 2, 2, 2])
        self.assertEqual(linhas[1].rotulo_modelo, "alegria")
        self.assertEqual(linhas[5].rotulo_modelo, "medo")

    def test_neutro_com_intervencao_vira_codigo_1(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "Fecha a linha e abre na esquerda",
                "emocao": "neu",
                "categoria_ctx": "INSTRUCAO_TATICA_GERAL",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 6.0,
                "texto": "Fecha a linha e abre na esquerda",
                "emocao": "neu",
                "categoria_ctx": "INSTRUCAO_TATICA_GERAL",
                "escore": 1.2,
                "confianca_emo": 0.2,
            },
        ]

        linhas = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0].codigo_emocao, 1)
        self.assertEqual(linhas[0].rotulo_modelo, "neutro com intervencao")

    def test_texto_de_preocupacao_sobrepoe_audio_fraco(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "Ele bateu a cabeca, ta doendo?",
                "emocao": "sad",
                "categoria_ctx": "neutro",
                "escore": -0.2,
                "confianca_emo": 0.13,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 5.0,
                "texto": "Quer trocar? Ta bem?",
                "emocao": "sad",
                "categoria_ctx": "neutro",
                "escore": -0.1,
                "confianca_emo": 0.14,
            },
        ]

        linhas = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0].emocao_agregada_bruta, "sad")
        self.assertEqual(linhas[0].emocao_agregada, "fea")
        self.assertEqual(linhas[0].codigo_emocao, 6)
        self.assertEqual(linhas[0].rotulo_modelo, "medo")

    def test_audio_confiante_recupera_tristeza_sem_texto_forte(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "",
                "emocao": "sad",
                "categoria_ctx": "neutro",
                "escore": -2.2,
                "confianca_emo": 0.35,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 5.0,
                "texto": "",
                "emocao": "sad",
                "categoria_ctx": "neutro",
                "escore": -2.0,
                "confianca_emo": 0.32,
            },
        ]

        linhas = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0].emocao_agregada_bruta, "sad")
        self.assertEqual(linhas[0].emocao_agregada, "sad")
        self.assertEqual(linhas[0].codigo_emocao, 5)

    def test_sinais_leves_nao_forcam_preocupacao(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "Calma, bora trocar e volta pro jogo.",
                "emocao": "neu",
                "categoria_ctx": "INSTRUCAO_TATICA_GERAL",
                "escore": 0.4,
                "confianca_emo": 0.12,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 5.0,
                "texto": "Calma, bora trocar e volta pro jogo.",
                "emocao": "neu",
                "categoria_ctx": "INSTRUCAO_TATICA_GERAL",
                "escore": 0.4,
                "confianca_emo": 0.12,
            },
        ]

        linhas = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0].codigo_emocao, 1)
        self.assertEqual(linhas[0].rotulo_modelo, "neutro com intervencao")

    def test_audio_negativo_com_baixa_confianca_permanece_neutro_sem_intervencao(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "",
                "emocao": "fea",
                "categoria_ctx": "neutro",
                "escore": 0.4,
                "confianca_emo": 0.12,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 5.0,
                "texto": "",
                "emocao": "fea",
                "categoria_ctx": "neutro",
                "escore": 0.4,
                "confianca_emo": 0.12,
            },
        ]

        linhas = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(len(linhas), 1)
        self.assertEqual(linhas[0].codigo_emocao, 0)
        self.assertEqual(linhas[0].rotulo_modelo, "neutro sem intervencao")

    def test_audio_negativo_sem_texto_nao_cria_intervencao(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "",
                "emocao": "sad",
                "categoria_ctx": "neutro",
                "escore": -0.6,
                "confianca_emo": 0.9,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 5.0,
                "texto": "",
                "emocao": "sad",
                "categoria_ctx": "neutro",
                "escore": -0.6,
                "confianca_emo": 0.9,
            },
        ]

        linhas = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(linhas[0].codigo_emocao, 0)
        self.assertEqual(linhas[0].motivo_calibracao, "neutro_final")

    def test_pesos_de_fusao_valorizam_evidencia_nao_neutra_existente(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "Nossa",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 0.0,
                "confianca_emo": 0.1,
            },
            {
                "inicio_s": 2.0,
                "fim_s": 5.0,
                "texto": "Nossa",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 0.0,
                "confianca_emo": 0.1,
            },
        ]

        calibrado = GeradorDatasetModelo().gerar(resultados_pipeline, duracao_total_s=5.0)
        sem_compensacao = GeradorDatasetModelo(
            ConfigDatasetModelo(pesos_fusao_nao_neutra={2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0})
        ).gerar(resultados_pipeline, duracao_total_s=5.0)

        self.assertEqual(calibrado[0].codigo_emocao, 3)
        self.assertEqual(sem_compensacao[0].codigo_emocao, 0)

    def test_pesos_de_fusao_invalidos_sao_rejeitados(self) -> None:
        with self.assertRaises(ValueError):
            ConfigDatasetModelo(pesos_fusao_nao_neutra={2: 0.0})

    def test_exporta_csv_principal_com_colunas_de_sentimento(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 0.0,
                "confianca_emo": 0.8,
            },
        ]
        gerador = GeradorDatasetModelo()
        linhas = gerador.gerar(resultados_pipeline, duracao_total_s=5.0)

        with TemporaryDirectory() as pasta_temporaria:
            caminho_saida = Path(pasta_temporaria) / "coach_emotion.dataset.csv"
            gerador.exportar(linhas, caminho_saida)
            conteudo = caminho_saida.read_text(encoding="utf-8-sig")

        self.assertIn("intervalo,avaliacao_modelo,codigo_emocao", conteudo)
        self.assertIn("rotulo_modelo,inicio_s,fim_s", conteudo)
        self.assertNotIn("emocao_agregada_bruta", conteudo)
        self.assertNotIn("motivo_calibracao", conteudo)
        self.assertIn("0:00 - 0:05:,0,0,neutro sem intervencao,0.0,5.0", conteudo)

    def test_exporta_csv_detalhado_quando_solicitado(self) -> None:
        resultados_pipeline = [
            {
                "inicio_s": 0.0,
                "fim_s": 4.0,
                "texto": "",
                "emocao": "neu",
                "categoria_ctx": "neutro",
                "escore": 0.0,
                "confianca_emo": 0.8,
            },
        ]
        gerador = GeradorDatasetModelo()
        linhas = gerador.gerar(resultados_pipeline, duracao_total_s=5.0)

        with TemporaryDirectory() as pasta_temporaria:
            caminho_saida = Path(pasta_temporaria) / "dataset_modelo_detalhado.csv"
            gerador.exportar(linhas, caminho_saida, incluir_detalhes=True)
            conteudo = caminho_saida.read_text(encoding="utf-8-sig")

        self.assertIn("emocao_agregada_bruta", conteudo)
        self.assertIn("motivo_calibracao", conteudo)
        self.assertIn("0:00 - 0:05:,0,0,neutro sem intervencao,0.0,5.0,neu,neu", conteudo)


if __name__ == "__main__":
    unittest.main()
