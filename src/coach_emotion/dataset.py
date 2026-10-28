from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from math import ceil, isfinite
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from coach_emotion.symbolic_rules import normalizar_texto


def _normalizar_categoria(categoria: str) -> str:
    return normalizar_texto(categoria.strip())


ROTULOS_MODELO_0_A_6: Dict[int, str] = {
    0: "neutro sem intervencao",
    1: "neutro com intervencao",
    2: "alegria",
    3: "surpresa",
    4: "raiva",
    5: "tristeza",
    6: "medo",
}

EMOCAO_FINAL_POR_CODIGO: Dict[int, str] = {
    0: "neu",
    1: "neu",
    2: "hap",
    3: "sur",
    4: "ang",
    5: "sad",
    6: "fea",
}

EMOCAO_AUDIO_POR_CODIGO: Dict[str, int] = {
    "hap": 2,
    "sur": 3,
    "ang": 4,
    "dis": 4,
    "sad": 5,
    "fea": 6,
}

LIMIAR_TEXTO_FORTE = 1.6
LIMIAR_SENTIMENTO_FINAL = 1.0
MARGEM_MINIMA_SENTIMENTO = 0.15
LIMIAR_TEXTO_FORTE_POR_CODIGO: Dict[int, float] = {
    5: 1.8,
    6: 2.3,
}
LIMIAR_SENTIMENTO_FINAL_POR_CODIGO: Dict[int, float] = {
    6: 1.9,
}
MARGEM_MINIMA_SENTIMENTO_POR_CODIGO: Dict[int, float] = {
    6: 0.3,
}

# Compensa a vantagem estrutural do neutro: ele e a decisao de fallback quando
# nenhuma evidencia emocional supera os limiares. Os pesos so atuam sobre
# evidencias ja encontradas; nao convertem silencio em emocao.
PESOS_FUSAO_NAO_NEUTRA_PADRAO: Dict[int, float] = {
    2: 1.35,  # alegria
    3: 1.35,  # surpresa
    4: 1.30,  # raiva
    5: 1.35,  # tristeza
    6: 1.20,  # medo: aumento menor por causa de falsos positivos anteriores
}

TERMOS_SENTIMENTO: Dict[int, Tuple[Tuple[str, float], ...]] = {
    2: tuple(
        (normalizar_texto(termo), peso)
        for termo, peso in [
            ("muito bem", 1.6),
            ("que bom", 1.5),
            ("parabens", 1.6),
            ("boa", 0.8),
            ("boa bola", 1.2),
            ("isso ai", 1.2),
            ("isso", 0.7),
            ("valeu", 1.0),
            ("bora", 0.8),
            ("vamos", 0.7),
            ("gol", 1.8),
            ("comemora", 1.8),
        ]
    ),
    3: tuple(
        (normalizar_texto(termo), peso)
        for termo, peso in [
            ("eita", 1.2),
            ("opa", 1.0),
            ("uau", 1.3),
            ("nossa", 0.9),
            ("caramba", 0.9),
        ]
    ),
    4: tuple(
        (normalizar_texto(termo), peso)
        for termo, peso in [
            ("nao pode", 1.3),
            ("ta dormindo", 2.0),
            ("acorda", 1.3),
            ("absurdo", 1.8),
            ("ridiculo", 2.0),
            ("vergonha", 2.0),
            ("inadmissivel", 2.0),
            ("brincadeira", 1.6),
            ("desse jeito nao", 1.8),
            ("que passe foi esse", 2.1),
        ]
    ),
    5: tuple(
        (normalizar_texto(termo), peso)
        for termo, peso in [
            ("poxa", 1.3),
            ("que pena", 1.8),
            ("nao acredito", 1.8),
            ("que azar", 1.8),
            ("assim nao da", 1.8),
            ("ai ai ai", 1.6),
            ("era pra", 1.1),
            ("era para", 1.1),
            ("errou", 1.1),
            ("perdeu", 1.1),
        ]
    ),
    6: tuple(
        (normalizar_texto(termo), peso)
        for termo, peso in [
            ("ta doendo", 2.4),
            ("aonde ta doendo", 2.6),
            ("doendo", 1.8),
            ("bateu a cabeca", 2.8),
            ("bateu a cabeça", 2.8),
            ("bateu a cabeca de novo", 3.0),
            ("machucou", 2.4),
            ("machucado", 2.0),
            ("quer trocar", 2.0),
            ("gelinho", 1.3),
            ("bateu", 0.6),
            ("cabeca", 0.8),
        ]
    ),
}

CATEGORIAS_SENTIMENTO: Dict[int, Dict[str, float]] = {
    2: {
        _normalizar_categoria("REFORCO_POSITIVO_ALTO"): 0.9,
        _normalizar_categoria("REFORCO_POSITIVO"): 0.8,
        _normalizar_categoria("MOTIVACAO_E_INCENTIVO"): 0.6,
        _normalizar_categoria("COMEMORACAO"): 1.0,
    },
    4: {
        _normalizar_categoria("CRITICA_DURA"): 0.8,
        _normalizar_categoria("CRITICA_CONSTRUTIVA"): 0.4,
        _normalizar_categoria("RECLAMACAO_ARBITRAGEM"): 0.8,
    },
    5: {
        _normalizar_categoria("FRUSTRACAO_E_LAMENTO"): 0.8,
    },
}

CATEGORIAS_TATICAS_NEUTRAS = {
    _normalizar_categoria("INSTRUCAO_TATICA_OFENSIVA"),
    _normalizar_categoria("INSTRUCAO_TATICA_DEFENSIVA"),
    _normalizar_categoria("INSTRUCAO_TATICA_GERAL"),
}


@dataclass(frozen=True)
class JanelaModelo:
    indice_origem: int
    inicio_s: float
    fim_s: float
    emocao: str
    categoria_ctx: str
    escore: float
    confianca_emo: float
    texto: str


@dataclass(frozen=True)
class LinhaDatasetModelo:
    indice_intervalo: int
    inicio_s: float
    fim_s: float
    intervalo: str
    avaliacao_modelo: int
    codigo_emocao: int
    rotulo_modelo: str
    emocao_agregada_bruta: str
    emocao_agregada: str
    categoria_ctx_agregada: str
    escore_medio: float
    confianca_media: float
    janelas_agregadas: int
    motivo_calibracao: str


@dataclass
class ConfigDatasetModelo:
    duracao_intervalo_s: float = 5.0
    categoria_neutra: str = "neutro"
    confianca_minima_emocao: float = 0.18
    escore_neutro_max_abs: float = 2.0
    limiar_evidencia_compensavel: float = 0.9
    pesos_fusao_nao_neutra: Dict[int, float] = field(
        default_factory=lambda: dict(PESOS_FUSAO_NAO_NEUTRA_PADRAO)
    )

    def __post_init__(self) -> None:
        if not isfinite(self.duracao_intervalo_s) or self.duracao_intervalo_s <= 0:
            raise ValueError("duracao_intervalo_s deve ser positiva")
        if not 0.0 <= self.confianca_minima_emocao <= 1.0:
            raise ValueError("confianca_minima_emocao deve ficar entre 0 e 1")
        if not isfinite(self.escore_neutro_max_abs) or self.escore_neutro_max_abs < 0:
            raise ValueError("escore_neutro_max_abs nao pode ser negativo")
        if (
            not isfinite(self.limiar_evidencia_compensavel)
            or self.limiar_evidencia_compensavel <= 0
        ):
            raise ValueError("limiar_evidencia_compensavel deve ser positivo")
        codigos_validos = set(PESOS_FUSAO_NAO_NEUTRA_PADRAO)
        codigos_informados = set(self.pesos_fusao_nao_neutra)
        codigos_invalidos = codigos_informados - codigos_validos
        if codigos_invalidos:
            raise ValueError(
                f"pesos_fusao_nao_neutra contem codigos invalidos: {sorted(codigos_invalidos)}"
            )
        for codigo, peso in self.pesos_fusao_nao_neutra.items():
            if not isfinite(peso) or peso <= 0:
                raise ValueError(f"peso de fusao do codigo {codigo} deve ser positivo")


def _ha_sobreposicao(
    inicio_a: float,
    fim_a: float,
    inicio_b: float,
    fim_b: float,
) -> bool:
    return min(fim_a, fim_b) - max(inicio_a, inicio_b) > 0.0


def _formatar_mm_ss(segundos_totais: float) -> str:
    total_segundos = max(0, int(round(segundos_totais)))
    minutos, segundos = divmod(total_segundos, 60)
    return f"{minutos}:{segundos:02d}"


def _formatar_intervalo(inicio_s: float, fim_s: float) -> str:
    return f"{_formatar_mm_ss(inicio_s)} - {_formatar_mm_ss(fim_s)}:"


def _codigo_para_avaliacao(codigo_emocao: int) -> int:
    if codigo_emocao in {0, 1}:
        return 0
    if codigo_emocao in {2, 3}:
        return 1
    return 2


class GeradorDatasetModelo:
    def __init__(self, config: Optional[ConfigDatasetModelo] = None) -> None:
        self.config = config or ConfigDatasetModelo()

    def gerar(
        self,
        resultados_pipeline: Sequence[Mapping[str, Any]],
        duracao_total_s: Optional[float] = None,
    ) -> List[LinhaDatasetModelo]:
        janelas = self._converter_janelas(resultados_pipeline)
        if not janelas:
            return []

        fim_total = duracao_total_s
        if fim_total is None:
            fim_total = max(janela.fim_s for janela in janelas)
        if not isfinite(fim_total) or fim_total <= 0:
            raise ValueError("duracao_total_s deve ser positiva")

        quantidade_intervalos = int(ceil(fim_total / self.config.duracao_intervalo_s))
        linhas: List[LinhaDatasetModelo] = []

        for indice_intervalo in range(quantidade_intervalos):
            inicio_s = indice_intervalo * self.config.duracao_intervalo_s
            fim_s = min(inicio_s + self.config.duracao_intervalo_s, fim_total)
            intervalo = _formatar_intervalo(inicio_s, fim_s)
            janelas_intervalo = self._coletar_janelas_intervalo(janelas, inicio_s, fim_s)

            if janelas_intervalo:
                texto_intervalo = self._agregar_texto_intervalo(janelas_intervalo)
                emocao_agregada_bruta = self._agregar_emocao(janelas_intervalo)
                categoria_agregada = self._agregar_categoria(janelas_intervalo)
                escore_medio = float(mean(item.escore for item in janelas_intervalo))
                confianca_media = float(mean(item.confianca_emo for item in janelas_intervalo))
                codigo_emocao, motivo_calibracao = self._resolver_codigo_sentimento(
                    texto_intervalo=texto_intervalo,
                    emocao_agregada_bruta=emocao_agregada_bruta,
                    categoria_agregada=categoria_agregada,
                    escore_medio=escore_medio,
                    confianca_media=confianca_media,
                )
                emocao_agregada = EMOCAO_FINAL_POR_CODIGO[codigo_emocao]
            else:
                emocao_agregada_bruta = "neu"
                emocao_agregada = "neu"
                categoria_agregada = self.config.categoria_neutra
                escore_medio = 0.0
                confianca_media = 0.0
                codigo_emocao = self._codigo_neutro_por_contexto(categoria_agregada)
                motivo_calibracao = "sem_janelas"

            avaliacao_modelo = _codigo_para_avaliacao(codigo_emocao)

            linhas.append(
                LinhaDatasetModelo(
                    indice_intervalo=indice_intervalo,
                    inicio_s=round(inicio_s, 3),
                    fim_s=round(fim_s, 3),
                    intervalo=intervalo,
                    avaliacao_modelo=avaliacao_modelo,
                    codigo_emocao=codigo_emocao,
                    rotulo_modelo=ROTULOS_MODELO_0_A_6[codigo_emocao],
                    emocao_agregada_bruta=emocao_agregada_bruta,
                    emocao_agregada=emocao_agregada,
                    categoria_ctx_agregada=categoria_agregada,
                    escore_medio=round(escore_medio, 3),
                    confianca_media=round(confianca_media, 3),
                    janelas_agregadas=len(janelas_intervalo),
                    motivo_calibracao=motivo_calibracao,
                )
            )

        return linhas

    def exportar(
        self,
        linhas: Sequence[LinhaDatasetModelo],
        caminho_saida: str | Path,
        incluir_detalhes: bool = False,
    ) -> Path:
        caminho = Path(caminho_saida)
        caminho.parent.mkdir(parents=True, exist_ok=True)

        registros = []
        for linha in linhas:
            registro = {
                "intervalo": linha.intervalo,
                "avaliacao_modelo": linha.avaliacao_modelo,
                "codigo_emocao": linha.codigo_emocao,
                "rotulo_modelo": linha.rotulo_modelo,
                "inicio_s": linha.inicio_s,
                "fim_s": linha.fim_s,
            }
            if incluir_detalhes:
                registro.update(
                    {
                        "emocao_agregada_bruta": linha.emocao_agregada_bruta,
                        "emocao_agregada": linha.emocao_agregada,
                        "categoria_ctx_agregada": linha.categoria_ctx_agregada,
                        "escore_medio": linha.escore_medio,
                        "confianca_media": linha.confianca_media,
                        "janelas_agregadas": linha.janelas_agregadas,
                        "motivo_calibracao": linha.motivo_calibracao,
                    }
                )
            registros.append(registro)

        dataframe = pd.DataFrame(registros)
        separador = "\t" if caminho.suffix.lower() == ".tsv" else ","
        dataframe.to_csv(caminho, sep=separador, index=False, encoding="utf-8-sig")
        return caminho

    def _converter_janelas(
        self,
        resultados_pipeline: Sequence[Mapping[str, Any]],
    ) -> List[JanelaModelo]:
        janelas: List[JanelaModelo] = []
        for indice, item in enumerate(resultados_pipeline):
            if "inicio_s" not in item or "fim_s" not in item:
                raise KeyError("Cada resultado do pipeline deve conter 'inicio_s' e 'fim_s'")
            if "emocao" not in item:
                raise KeyError("Cada resultado do pipeline deve conter 'emocao'")

            inicio_s = float(item["inicio_s"])
            fim_s = float(item["fim_s"])
            if not isfinite(inicio_s) or not isfinite(fim_s) or inicio_s < 0 or fim_s <= inicio_s:
                raise ValueError(f"Janela invalida no indice {indice}: fim_s <= inicio_s")

            janelas.append(
                JanelaModelo(
                    indice_origem=indice,
                    inicio_s=inicio_s,
                    fim_s=fim_s,
                    emocao=str(item["emocao"]).strip().lower(),
                    categoria_ctx=str(item.get("categoria_ctx", "neutro")).strip(),
                    escore=float(item.get("escore", 0.0)),
                    confianca_emo=float(item.get("confianca_emo", 0.0)),
                    texto=str(item.get("texto", "")).strip(),
                )
            )

        janelas.sort(key=lambda janela: (janela.inicio_s, janela.fim_s, janela.indice_origem))
        return janelas

    def _coletar_janelas_intervalo(
        self,
        janelas: Sequence[JanelaModelo],
        inicio_s: float,
        fim_s: float,
    ) -> List[JanelaModelo]:
        return [
            janela
            for janela in janelas
            if _ha_sobreposicao(janela.inicio_s, janela.fim_s, inicio_s, fim_s)
        ]

    def _agregar_emocao(self, janelas: Sequence[JanelaModelo]) -> str:
        votos: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for ordem, janela in enumerate(janelas):
            votos[janela.emocao].append((ordem, janela.confianca_emo))

        def chave(item: Tuple[str, List[Tuple[int, float]]]) -> Tuple[int, float, int]:
            _, ocorrencias = item
            quantidade = len(ocorrencias)
            confianca_media = mean(confianca for _, confianca in ocorrencias)
            primeira_aparicao = min(ordem for ordem, _ in ocorrencias)
            return (quantidade, confianca_media, -primeira_aparicao)

        melhor_rotulo, _ = max(votos.items(), key=chave)
        return melhor_rotulo

    def _agregar_categoria(self, janelas: Sequence[JanelaModelo]) -> str:
        votos: Dict[str, List[int]] = defaultdict(list)
        for ordem, janela in enumerate(janelas):
            votos[janela.categoria_ctx].append(ordem)

        def chave(item: Tuple[str, List[int]]) -> Tuple[int, int]:
            _, ocorrencias = item
            return (len(ocorrencias), -min(ocorrencias))

        melhor_categoria, _ = max(votos.items(), key=chave)
        return melhor_categoria

    def _agregar_texto_intervalo(self, janelas: Sequence[JanelaModelo]) -> str:
        textos_unicos: List[str] = []
        vistos: set[str] = set()
        for janela in janelas:
            texto = janela.texto.strip()
            if not texto:
                continue
            chave = normalizar_texto(texto)
            if not chave or chave in vistos:
                continue
            vistos.add(chave)
            textos_unicos.append(texto)
        return " ".join(textos_unicos).strip()

    def _resolver_codigo_sentimento(
        self,
        texto_intervalo: str,
        emocao_agregada_bruta: str,
        categoria_agregada: str,
        escore_medio: float,
        confianca_media: float,
    ) -> Tuple[int, str]:
        scores_texto = self._pontuar_texto_sentimento(texto_intervalo)
        scores_contexto = self._pontuar_contexto_sentimento(categoria_agregada)
        scores_audio = self._pontuar_audio_sentimento(
            emocao_agregada_bruta,
            confianca_media,
        )

        scores_texto = self._ponderar_evidencias_nao_neutras(scores_texto)
        scores_contexto = self._ponderar_evidencias_nao_neutras(scores_contexto)
        scores_audio = self._ponderar_evidencias_nao_neutras(scores_audio)

        melhor_codigo_texto, melhor_score_texto = self._melhor_codigo(scores_texto)
        if melhor_codigo_texto is not None and melhor_score_texto >= self._limiar_texto_forte(
            melhor_codigo_texto
        ):
            return melhor_codigo_texto, f"texto_forte_{melhor_codigo_texto}"

        scores_finais: Dict[int, float] = defaultdict(float)
        for origem in (scores_texto, scores_contexto, scores_audio):
            for codigo, peso in origem.items():
                scores_finais[codigo] += peso

        melhor_codigo_final, melhor_score_final = self._melhor_codigo(scores_finais)
        segundo_score_final = self._segundo_melhor_score(
            scores_finais,
            ignorar_codigo=melhor_codigo_final,
        )

        if (
            melhor_codigo_final is not None
            and melhor_score_final >= self._limiar_sentimento_final(melhor_codigo_final)
            and (melhor_score_final - segundo_score_final)
            >= self._margem_minima_sentimento(melhor_codigo_final)
        ):
            if scores_texto.get(melhor_codigo_final, 0.0) > 0.0:
                return melhor_codigo_final, f"texto_contexto_{melhor_codigo_final}"
            if scores_contexto.get(melhor_codigo_final, 0.0) > 0.0:
                return melhor_codigo_final, f"contexto_{melhor_codigo_final}"
            return melhor_codigo_final, f"audio_{melhor_codigo_final}"

        codigo_audio = EMOCAO_AUDIO_POR_CODIGO.get(emocao_agregada_bruta)
        if (
            codigo_audio is not None
            and confianca_media >= self.config.confianca_minima_emocao
            and abs(escore_medio) > self.config.escore_neutro_max_abs
        ):
            return codigo_audio, f"audio_confiante_{codigo_audio}"

        if self._tem_sinal_negativo_suave(
            scores_texto=scores_texto,
            scores_contexto=scores_contexto,
            scores_audio=scores_audio,
            emocao_agregada_bruta=emocao_agregada_bruta,
            escore_medio=escore_medio,
            confianca_media=confianca_media,
        ):
            return 1, "negativo_suavizado"

        return self._codigo_neutro_por_contexto(categoria_agregada), "neutro_final"

    def _pontuar_texto_sentimento(self, texto_intervalo: str) -> Dict[int, float]:
        texto_normalizado = normalizar_texto(texto_intervalo)
        if not texto_normalizado:
            return {}

        scores: Dict[int, float] = defaultdict(float)
        for codigo, termos in TERMOS_SENTIMENTO.items():
            for termo_normalizado, peso in termos:
                padrao = rf"\b{re.escape(termo_normalizado)}\b"
                if re.search(padrao, texto_normalizado):
                    scores[codigo] += peso
        return dict(scores)

    def _pontuar_contexto_sentimento(self, categoria_agregada: str) -> Dict[int, float]:
        categoria_normalizada = _normalizar_categoria(categoria_agregada)
        scores: Dict[int, float] = {}
        for codigo, categorias in CATEGORIAS_SENTIMENTO.items():
            peso = categorias.get(categoria_normalizada)
            if peso is not None:
                scores[codigo] = peso
        return scores

    def _pontuar_audio_sentimento(
        self,
        emocao_agregada_bruta: str,
        confianca_media: float,
    ) -> Dict[int, float]:
        codigo_audio = EMOCAO_AUDIO_POR_CODIGO.get(emocao_agregada_bruta)
        if codigo_audio is None:
            return {}

        intensidade = max(
            0.0,
            (confianca_media - self.config.confianca_minima_emocao) * 10.0,
        )
        if intensidade <= 0.0:
            return {}
        return {codigo_audio: min(0.6, intensidade)}

    def _ponderar_evidencias_nao_neutras(
        self,
        scores: Mapping[int, float],
    ) -> Dict[int, float]:
        """Compensa classes sem elevar palavras isoladas e muito fracas."""
        return {
            codigo: (
                float(score) * self.config.pesos_fusao_nao_neutra.get(codigo, 1.0)
                if float(score) >= self.config.limiar_evidencia_compensavel
                else float(score)
            )
            for codigo, score in scores.items()
        }

    def _codigo_neutro_por_contexto(self, categoria_agregada: str) -> int:
        categoria_normalizada = _normalizar_categoria(categoria_agregada)
        if categoria_normalizada == _normalizar_categoria(self.config.categoria_neutra):
            return 0
        if categoria_normalizada in CATEGORIAS_TATICAS_NEUTRAS:
            return 1
        return 1

    def _tem_sinal_negativo_suave(
        self,
        scores_texto: Mapping[int, float],
        scores_contexto: Mapping[int, float],
        scores_audio: Mapping[int, float],
        emocao_agregada_bruta: str,
        escore_medio: float,
        confianca_media: float,
    ) -> bool:
        codigos_negativos = {4, 5, 6}
        if any(scores_texto.get(codigo, 0.0) > 0.0 for codigo in codigos_negativos):
            return True
        if any(scores_contexto.get(codigo, 0.0) > 0.0 for codigo in codigos_negativos):
            return True
        # Audio isolado nao comprova intervencao verbal. Ele pode decidir uma
        # emocao apenas pela rota audio_confiante, que exige escore agregado alto.
        del scores_audio, emocao_agregada_bruta, escore_medio, confianca_media
        return False

    def _limiar_texto_forte(self, codigo: int) -> float:
        return LIMIAR_TEXTO_FORTE_POR_CODIGO.get(codigo, LIMIAR_TEXTO_FORTE)

    def _limiar_sentimento_final(self, codigo: int) -> float:
        return LIMIAR_SENTIMENTO_FINAL_POR_CODIGO.get(codigo, LIMIAR_SENTIMENTO_FINAL)

    def _margem_minima_sentimento(self, codigo: int) -> float:
        return MARGEM_MINIMA_SENTIMENTO_POR_CODIGO.get(codigo, MARGEM_MINIMA_SENTIMENTO)

    def _melhor_codigo(self, scores: Mapping[int, float]) -> Tuple[Optional[int], float]:
        if not scores:
            return None, 0.0

        melhor_codigo = max(
            scores.items(),
            key=lambda item: (item[1], -item[0]),
        )[0]
        return melhor_codigo, float(scores[melhor_codigo])

    def _segundo_melhor_score(
        self,
        scores: Mapping[int, float],
        ignorar_codigo: Optional[int],
    ) -> float:
        candidatos = [float(score) for codigo, score in scores.items() if codigo != ignorar_codigo]
        if not candidatos:
            return 0.0
        return max(candidatos)
