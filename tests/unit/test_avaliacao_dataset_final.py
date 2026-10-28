from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coach_emotion.evaluate import avaliar_dataset_final


class TesteAvaliacaoDatasetFinal(unittest.TestCase):
    def test_avalia_codigo_exato_e_emocao_colapsada(self) -> None:
        with TemporaryDirectory() as pasta:
            base = Path(pasta)
            ground_truth = base / "ground_truth.tsv"
            ground_truth.write_text(
                "  \t0\t0\n0:05 - 0:10:\t0\t1\n0:10 - 0:15:\t2\t4\n",
                encoding="utf-8",
            )
            dataset = base / "dataset.tsv"
            dataset.write_text(
                "inicio_s\tfim_s\tcodigo_emocao\n0\t5\t1\n5\t10\t0\n10\t15\t4\n",
                encoding="utf-8",
            )
            resultado = avaliar_dataset_final(ground_truth, dataset)

        self.assertEqual(resultado.total_intervalos, 3)
        self.assertEqual(resultado.acertos_exatos_0a6, 1)
        self.assertAlmostEqual(resultado.acuracia_exata_0a6, 1 / 3)
        self.assertEqual(resultado.acertos_emocao_colapsada, 3)
        self.assertAlmostEqual(resultado.acuracia_emocao_colapsada, 1.0)


if __name__ == "__main__":
    unittest.main()
