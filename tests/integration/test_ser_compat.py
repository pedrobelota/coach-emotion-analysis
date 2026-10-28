from __future__ import annotations

import unittest

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Wav2Vec2Config

from coach_emotion.emotion_recognition import Wav2Vec2ParaClassificacaoLegada


class TesteSERCompativel(unittest.TestCase):
    def test_cabeca_legada_preserva_nomes_dos_pesos(self) -> None:
        config = Wav2Vec2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8, 8, 8, 8, 8, 8),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            num_labels=8,
        )
        modelo = Wav2Vec2ParaClassificacaoLegada(config)
        chaves = set(modelo.state_dict())

        self.assertIn("classifier.dense.weight", chaves)
        self.assertIn("classifier.dense.bias", chaves)
        self.assertIn("classifier.output.weight", chaves)
        self.assertIn("classifier.output.bias", chaves)
        self.assertNotIn("projector.weight", chaves)


if __name__ == "__main__":
    unittest.main()
