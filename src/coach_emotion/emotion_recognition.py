"""Compatibilidade para checkpoints SER treinados com Transformers antigos.

O checkpoint configurado pelo projeto armazena a cabeca como
``classifier.dense`` e ``classifier.output``. Versoes recentes do
Transformers esperam ``projector`` e ``classifier``, o que recriava a cabeca
aleatoriamente. Este modulo restaura a arquitetura original do checkpoint.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class CabecaClassificacaoLegada(nn.Module):
    """Replica os nomes e as operacoes da cabeca salva no checkpoint."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dropout = nn.Dropout(config.final_dropout)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(features)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.output(hidden_states)


class Wav2Vec2ParaClassificacaoLegada(Wav2Vec2PreTrainedModel):
    """Wav2Vec2 com a cabeca de classificacao usada pelo checkpoint SER."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = CabecaClassificacaoLegada(config)
        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **_: Any,
    ) -> SequenceClassifierOutput | Tuple[torch.Tensor, ...]:
        usar_retorno_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=usar_retorno_dict,
        )
        hidden_states = outputs[0]

        if attention_mask is None:
            representacao = hidden_states.mean(dim=1)
        else:
            mascara = self._get_feature_vector_attention_mask(
                hidden_states.shape[1],
                attention_mask,
            )
            mascara_expandida = mascara.unsqueeze(-1).to(hidden_states.dtype)
            representacao = (hidden_states * mascara_expandida).sum(dim=1)
            representacao = representacao / mascara_expandida.sum(dim=1).clamp_min(1.0)

        logits = self.classifier(representacao)
        if not usar_retorno_dict:
            return (logits,) + tuple(outputs[1:])
        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ClassificadorSERCompativel:
    """Adaptador com o mesmo contrato de chamada da pipeline Transformers."""

    def __init__(
        self,
        modelo: Wav2Vec2ParaClassificacaoLegada,
        extrator: Any,
        dispositivo: torch.device,
        dtype: Optional[torch.dtype],
    ) -> None:
        self.modelo = modelo
        self.extrator = extrator
        self.dispositivo = dispositivo
        self.dtype = dtype

    def __call__(self, entrada: Mapping[str, Any]) -> List[Dict[str, float | str]]:
        if "array" not in entrada or "sampling_rate" not in entrada:
            raise KeyError("SER requer 'array' e 'sampling_rate' na entrada")

        audio = np.asarray(entrada["array"], dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise ValueError("SER requer audio monofonico nao vazio")

        entradas_modelo = self.extrator(
            audio,
            sampling_rate=int(entrada["sampling_rate"]),
            return_tensors="pt",
            padding=True,
        )
        tensores: Dict[str, torch.Tensor] = {}
        for nome, tensor in entradas_modelo.items():
            tensor_no_dispositivo = tensor.to(self.dispositivo)
            if tensor_no_dispositivo.is_floating_point() and self.dtype is not None:
                tensor_no_dispositivo = tensor_no_dispositivo.to(dtype=self.dtype)
            tensores[nome] = tensor_no_dispositivo

        with torch.inference_mode():
            logits = self.modelo(**tensores).logits
            probabilidades = torch.softmax(logits.float(), dim=-1)[0].cpu()

        resultados = [
            {
                "label": str(self.modelo.config.id2label[indice]),
                "score": float(probabilidade),
            }
            for indice, probabilidade in enumerate(probabilidades.tolist())
        ]
        return sorted(resultados, key=lambda item: float(item["score"]), reverse=True)


def carregar_ser_compativel(
    identificador_modelo: str,
    device_id: int,
    dtype: Optional[torch.dtype],
    revision: Optional[str] = None,
) -> ClassificadorSERCompativel:
    """Carrega o checkpoint verificando se todos os pesos foram recuperados."""
    config = AutoConfig.from_pretrained(identificador_modelo, revision=revision)
    extrator = AutoFeatureExtractor.from_pretrained(identificador_modelo, revision=revision)
    modelo, informacoes = Wav2Vec2ParaClassificacaoLegada.from_pretrained(
        identificador_modelo,
        config=config,
        revision=revision,
        output_loading_info=True,
    )
    pesos_ausentes = informacoes.get("missing_keys", [])
    pesos_inesperados = informacoes.get("unexpected_keys", [])
    pesos_incompativeis = informacoes.get("mismatched_keys", [])
    if pesos_ausentes or pesos_inesperados or pesos_incompativeis:
        raise RuntimeError(
            "Checkpoint SER nao foi carregado integralmente: "
            f"ausentes={pesos_ausentes}, inesperados={pesos_inesperados}, "
            f"incompativeis={pesos_incompativeis}"
        )

    dispositivo = torch.device("cuda", device_id) if device_id >= 0 else torch.device("cpu")
    modelo = modelo.to(dispositivo)
    if dtype is not None and device_id >= 0:
        modelo = modelo.to(dtype=dtype)
    modelo.eval()
    return ClassificadorSERCompativel(
        modelo=modelo,
        extrator=extrator,
        dispositivo=dispositivo,
        dtype=dtype if device_id >= 0 else None,
    )
