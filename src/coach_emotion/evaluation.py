from __future__ import annotations

import csv
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)

from coach_emotion.config import MAPA_ROTULOS

PoliticaNulos = Literal["ignore", "map_to_neu"]
EstrategiaClasse = Literal["majority_vote"]
EstrategiaEscore = Literal["mean"]

COLUNAS_CSV: List[str] = [
    "intervalo",
    "codigo_emocao",
    "rotulo_humano",
    "metrica_humana",
]
COLUNAS_CSV_COMPACTO: List[str] = [
    "intervalo",
    "metrica_humana",
    "codigo_emocao",
]
ORDEM_CLASSES_PADRAO: List[str] = list(dict.fromkeys(MAPA_ROTULOS.values()))
ROTULO_NEUTRO_PADRAO: str = MAPA_ROTULOS.get("neutral", "neu")

MAPA_ROTULOS_HUMANOS_PADRAO: Dict[str, str] = {
    "alegria": "happiness",
    "felicidade": "happiness",
    "feliz": "happiness",
    "raiva": "anger",
    "tristeza": "sadness",
    "triste": "sadness",
    "medo": "fear",
    "nojo": "disgust",
    "neutro": "neutral",
    "neutral": "neutral",
    "neutro sem intervencao": "neutral",
    "neutro com intervencao": "neutral",
    "surpresa": "surprise",
    "contexto de jogo": "neutral",
}

MAPA_CODIGOS_ROTULOS_HUMANOS_PADRAO: Dict[int, str] = {
    0: "neutro sem intervencao",
    1: "neutro com intervencao",
    2: "alegria",
    3: "surpresa",
    4: "raiva",
    5: "tristeza",
    6: "medo",
}


def _normalizar_chave(texto: str) -> str:
    texto_normalizado = unicodedata.normalize("NFKD", texto.strip().casefold())
    texto_sem_acentos = "".join(
        caractere for caractere in texto_normalizado if not unicodedata.combining(caractere)
    )
    return " ".join(texto_sem_acentos.split())


def _mapear_para_rotulo_sistema(rotulo: str) -> str:
    chave = _normalizar_chave(rotulo)
    if chave in ORDEM_CLASSES_PADRAO:
        return chave
    if chave in MAPA_ROTULOS:
        return MAPA_ROTULOS[chave]
    raise ValueError(f"Rotulo nao reconhecido pelo sistema: {rotulo!r}")


def _tempo_para_segundos(texto_tempo: str) -> float:
    texto_limpo = texto_tempo.strip().replace(";", ":").rstrip(":")
    partes = texto_limpo.split(":")

    if len(partes) not in (2, 3):
        raise ValueError(f"Tempo invalido: {texto_tempo!r}")
    if any(not parte.isdigit() for parte in partes):
        raise ValueError(f"Tempo invalido: {texto_tempo!r}")

    valores = [int(parte) for parte in partes]
    if len(valores) == 2:
        minutos, segundos = valores
        if not 0 <= segundos < 60:
            raise ValueError(f"Segundos fora do intervalo em {texto_tempo!r}")
        return float(minutos * 60 + segundos)

    horas, minutos, segundos = valores
    if not 0 <= minutos < 60 or not 0 <= segundos < 60:
        raise ValueError(f"Horario fora do intervalo em {texto_tempo!r}")
    return float(horas * 3600 + minutos * 60 + segundos)


def _parse_intervalo(intervalo_bruto: str) -> Tuple[float, float]:
    partes = intervalo_bruto.split("-")
    if len(partes) != 2:
        raise ValueError(f"Intervalo invalido: {intervalo_bruto!r}")

    inicio_s = _tempo_para_segundos(partes[0])
    fim_s = _tempo_para_segundos(partes[1])
    if fim_s <= inicio_s:
        raise ValueError(f"Intervalo com fim menor ou igual ao inicio: {intervalo_bruto!r}")
    return inicio_s, fim_s


def _parse_fim_intervalo(intervalo_bruto: str) -> float:
    texto_limpo = intervalo_bruto.strip()
    if not texto_limpo:
        raise ValueError("Fim de intervalo vazio")
    return _tempo_para_segundos(texto_limpo)


def _ha_sobreposicao(
    inicio_a: float,
    fim_a: float,
    inicio_b: float,
    fim_b: float,
) -> bool:
    return min(fim_a, fim_b) - max(inicio_a, inicio_b) > 0.0


@dataclass(frozen=True)
class IntervaloGroundTruth:
    indice_csv: int
    intervalo_bruto: str
    inicio_s: float
    fim_s: float
    codigo_emocao: int
    rotulo_humano: Optional[str]
    rotulo_sistema: Optional[str]
    metrica_humana: Optional[float]
    permite_mapear_nulo_para_neutro: bool = True


@dataclass(frozen=True)
class JanelaPredita:
    indice_origem: int
    inicio_s: float
    fim_s: float
    emocao: str
    escore: float
    confianca_emo: float


@dataclass
class ConfigAvaliacao:
    politica_nulos: PoliticaNulos = "ignore"
    estrategia_classe: EstrategiaClasse = "majority_vote"
    estrategia_escore: EstrategiaEscore = "mean"
    rotulo_neutro_padrao: str = ROTULO_NEUTRO_PADRAO
    encoding_csv: str = "utf-8-sig"
    duracao_intervalo_padrao_s: float = 5.0

    def __post_init__(self) -> None:
        if self.politica_nulos not in ("ignore", "map_to_neu"):
            raise ValueError("politica_nulos deve ser 'ignore' ou 'map_to_neu'")
        if self.estrategia_classe != "majority_vote":
            raise ValueError("estrategia_classe suportada atualmente: 'majority_vote'")
        if self.estrategia_escore != "mean":
            raise ValueError("estrategia_escore suportada atualmente: 'mean'")
        if self.duracao_intervalo_padrao_s <= 0:
            raise ValueError("duracao_intervalo_padrao_s deve ser positiva")

        self.rotulo_neutro_padrao = _mapear_para_rotulo_sistema(self.rotulo_neutro_padrao)


@dataclass(frozen=True)
class MetricasClasse:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class IntervaloAvaliado:
    indice_csv: int
    inicio_s: float
    fim_s: float
    rotulo_humano: Optional[str]
    rotulo_real: Optional[str]
    rotulo_predito: Optional[str]
    metrica_humana: Optional[float]
    escore_predito: Optional[float]
    janelas_agregadas: int
    incluido_nas_metricas: bool
    motivo_exclusao: Optional[str] = None


@dataclass(frozen=True)
class ResultadoAvaliacao:
    total_intervalos_csv: int
    intervalos_rotulados_originais: int
    intervalos_nulos_originais: int
    intervalos_ignorados_nulos: int
    intervalos_sem_predicao: int
    intervalos_sem_metrica_humana: int
    total_intervalos_metricas: int
    classes_avaliadas: List[str]
    matriz_confusao: List[List[int]]
    metricas_por_classe: Dict[str, MetricasClasse]
    macro_precision: float
    macro_recall: float
    macro_f1: float
    relatorio_classificacao: str
    mae_escore: Optional[float]
    rmse_escore: Optional[float]
    rotulos_reais: List[str]
    rotulos_preditos: List[str]
    escores_reais: List[float]
    escores_preditos: List[float]
    intervalos: List[IntervaloAvaliado]


class AvaliadorAcuracia:
    def __init__(
        self,
        caminho_csv: str | Path,
        config: Optional[ConfigAvaliacao] = None,
        mapa_rotulos_humanos: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.caminho_csv = Path(caminho_csv)
        self.config = config or ConfigAvaliacao()
        self._ground_truth_cache: Optional[List[IntervaloGroundTruth]] = None

        mapa_base = dict(MAPA_ROTULOS_HUMANOS_PADRAO)
        if mapa_rotulos_humanos:
            mapa_base.update(mapa_rotulos_humanos)

        self.mapa_rotulos_humanos: Dict[str, str] = {
            _normalizar_chave(chave): _mapear_para_rotulo_sistema(valor)
            for chave, valor in mapa_base.items()
        }
        self.mapa_codigos_humanos: Dict[int, str] = dict(MAPA_CODIGOS_ROTULOS_HUMANOS_PADRAO)

    def carregar_ground_truth(self) -> List[IntervaloGroundTruth]:
        if self._ground_truth_cache is not None:
            return list(self._ground_truth_cache)

        if not self.caminho_csv.exists():
            raise FileNotFoundError(f"CSV nao encontrado: {self.caminho_csv}")

        delimitador_csv, colunas_csv = self._detectar_estrutura_csv()
        dataframe = pd.read_csv(
            self.caminho_csv,
            header=None,
            names=colunas_csv,
            encoding=self.config.encoding_csv,
            sep=delimitador_csv,
        )

        if dataframe.empty:
            raise ValueError("CSV de ground truth esta vazio")

        intervalos: List[IntervaloGroundTruth] = []
        fim_anterior = 0.0
        for indice, linha in dataframe.iterrows():
            intervalo_bruto = str(linha["intervalo"]).strip()
            inicio_s, fim_s = self._resolver_intervalo(
                intervalo_bruto=intervalo_bruto,
                indice=int(indice),
                fim_anterior=fim_anterior,
            )
            fim_anterior = fim_s

            codigo_emocao = self._converter_codigo_emocao(linha.get("codigo_emocao"))
            metrica_humana = self._converter_metrica_humana(linha.get("metrica_humana"))

            rotulo_humano = self._extrair_rotulo_humano(linha)
            permite_mapear_nulo = "rotulo_humano" in dataframe.columns
            rotulo_sistema: Optional[str]
            if rotulo_humano is None:
                rotulo_sistema = None
            else:
                rotulo_sistema = self._mapear_rotulo_humano(rotulo_humano)

            intervalos.append(
                IntervaloGroundTruth(
                    indice_csv=int(indice),
                    intervalo_bruto=intervalo_bruto,
                    inicio_s=inicio_s,
                    fim_s=fim_s,
                    codigo_emocao=codigo_emocao,
                    rotulo_humano=rotulo_humano,
                    rotulo_sistema=rotulo_sistema,
                    metrica_humana=metrica_humana,
                    permite_mapear_nulo_para_neutro=permite_mapear_nulo,
                )
            )

        self._ground_truth_cache = intervalos
        return list(intervalos)

    def avaliar(self, resultados_pipeline: Sequence[Mapping[str, Any]]) -> ResultadoAvaliacao:
        ground_truth = self.carregar_ground_truth()
        predicoes = self._converter_predicoes(resultados_pipeline)

        intervalos_alinhados: List[IntervaloAvaliado] = []
        rotulos_reais: List[str] = []
        rotulos_preditos: List[str] = []
        escores_reais: List[float] = []
        escores_preditos: List[float] = []

        intervalos_ignorados_nulos = 0
        intervalos_sem_predicao = 0
        intervalos_sem_metrica_humana = 0

        for intervalo in ground_truth:
            rotulo_real = intervalo.rotulo_sistema
            if (
                rotulo_real is None
                and intervalo.permite_mapear_nulo_para_neutro
                and self.config.politica_nulos == "map_to_neu"
            ):
                rotulo_real = self.config.rotulo_neutro_padrao

            predicoes_intervalo = self._coletar_predicoes_intervalo(
                predicoes,
                intervalo.inicio_s,
                intervalo.fim_s,
            )

            rotulo_predito: Optional[str] = None
            escore_predito: Optional[float] = None
            motivo_exclusao: Optional[str] = None
            incluido_nas_metricas = True

            if rotulo_real is None:
                intervalos_ignorados_nulos += 1
                incluido_nas_metricas = False
                motivo_exclusao = "rotulo_humano_nulo"
            elif not predicoes_intervalo:
                intervalos_sem_predicao += 1
                incluido_nas_metricas = False
                motivo_exclusao = "sem_predicao_sobreposta"
            else:
                rotulo_predito = self._agregar_classe(predicoes_intervalo)
                escore_predito = self._agregar_escore(predicoes_intervalo)
                rotulos_reais.append(rotulo_real)
                rotulos_preditos.append(rotulo_predito)
                if intervalo.metrica_humana is not None:
                    escores_reais.append(intervalo.metrica_humana)
                    escores_preditos.append(escore_predito)
                else:
                    intervalos_sem_metrica_humana += 1

            if rotulo_real is None and predicoes_intervalo and intervalo.metrica_humana is not None:
                escore_predito = self._agregar_escore(predicoes_intervalo)
                escores_reais.append(intervalo.metrica_humana)
                escores_preditos.append(escore_predito)
            elif rotulo_real is None and predicoes_intervalo and intervalo.metrica_humana is None:
                intervalos_sem_metrica_humana += 1

            intervalos_alinhados.append(
                IntervaloAvaliado(
                    indice_csv=intervalo.indice_csv,
                    inicio_s=intervalo.inicio_s,
                    fim_s=intervalo.fim_s,
                    rotulo_humano=intervalo.rotulo_humano,
                    rotulo_real=rotulo_real,
                    rotulo_predito=rotulo_predito,
                    metrica_humana=intervalo.metrica_humana,
                    escore_predito=escore_predito,
                    janelas_agregadas=len(predicoes_intervalo),
                    incluido_nas_metricas=incluido_nas_metricas,
                    motivo_exclusao=motivo_exclusao,
                )
            )

        classes_avaliadas = self._ordenar_classes(rotulos_reais, rotulos_preditos)
        metricas_por_classe: Dict[str, MetricasClasse] = {}
        matriz: List[List[int]] = []
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        relatorio_classificacao = "Sem intervalos suficientes para calcular metricas."

        if classes_avaliadas:
            matriz_bruta = confusion_matrix(
                rotulos_reais,
                rotulos_preditos,
                labels=classes_avaliadas,
            )
            matriz = matriz_bruta.tolist()

            precisions, recalls, f1s, supports = precision_recall_fscore_support(
                rotulos_reais,
                rotulos_preditos,
                labels=classes_avaliadas,
                zero_division=0,
            )

            for indice, classe in enumerate(classes_avaliadas):
                metricas_por_classe[classe] = MetricasClasse(
                    precision=float(precisions[indice]),
                    recall=float(recalls[indice]),
                    f1=float(f1s[indice]),
                    support=int(supports[indice]),
                )

            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                rotulos_reais,
                rotulos_preditos,
                labels=classes_avaliadas,
                average="macro",
                zero_division=0,
            )
            relatorio_classificacao = classification_report(
                rotulos_reais,
                rotulos_preditos,
                labels=classes_avaliadas,
                digits=3,
                zero_division=0,
            )

        mae_escore: Optional[float] = None
        rmse_escore: Optional[float] = None
        if escores_reais:
            mae_escore = float(mean_absolute_error(escores_reais, escores_preditos))
            mse = float(mean_squared_error(escores_reais, escores_preditos))
            rmse_escore = sqrt(mse)

        intervalos_rotulados_originais = sum(
            1 for intervalo in ground_truth if intervalo.rotulo_sistema is not None
        )
        intervalos_nulos_originais = len(ground_truth) - intervalos_rotulados_originais

        return ResultadoAvaliacao(
            total_intervalos_csv=len(ground_truth),
            intervalos_rotulados_originais=intervalos_rotulados_originais,
            intervalos_nulos_originais=intervalos_nulos_originais,
            intervalos_ignorados_nulos=intervalos_ignorados_nulos,
            intervalos_sem_predicao=intervalos_sem_predicao,
            intervalos_sem_metrica_humana=intervalos_sem_metrica_humana,
            total_intervalos_metricas=len(rotulos_reais),
            classes_avaliadas=classes_avaliadas,
            matriz_confusao=matriz,
            metricas_por_classe=metricas_por_classe,
            macro_precision=float(macro_precision),
            macro_recall=float(macro_recall),
            macro_f1=float(macro_f1),
            relatorio_classificacao=relatorio_classificacao,
            mae_escore=mae_escore,
            rmse_escore=rmse_escore,
            rotulos_reais=rotulos_reais,
            rotulos_preditos=rotulos_preditos,
            escores_reais=escores_reais,
            escores_preditos=escores_preditos,
            intervalos=intervalos_alinhados,
        )

    def formatar_relatorio(self, resultado: ResultadoAvaliacao) -> str:
        linhas: List[str] = []
        linhas.append("=== AVALIACAO DE ACURACIA ===")
        linhas.append(f"CSV: {self.caminho_csv}")
        linhas.append(
            "Intervalos CSV: "
            f"{resultado.total_intervalos_csv} "
            f"(rotulados={resultado.intervalos_rotulados_originais}, "
            f"nulos={resultado.intervalos_nulos_originais})"
        )
        linhas.append(f"Intervalos usados nas metricas: {resultado.total_intervalos_metricas}")
        linhas.append(
            "Exclusoes: "
            f"nulos_ignorados={resultado.intervalos_ignorados_nulos}, "
            f"sem_predicao={resultado.intervalos_sem_predicao}, "
            f"sem_escore_humano={resultado.intervalos_sem_metrica_humana}"
        )
        linhas.append("")
        linhas.append("Matriz de confusao")
        linhas.append(self._formatar_matriz_confusao(resultado))
        linhas.append("")
        linhas.append(
            "Macro metrics: "
            f"precision={resultado.macro_precision:.3f} | "
            f"recall={resultado.macro_recall:.3f} | "
            f"f1={resultado.macro_f1:.3f}"
        )
        linhas.append("")
        linhas.append("Metricas por classe")
        linhas.extend(self._formatar_metricas_por_classe(resultado))

        if resultado.mae_escore is not None and resultado.rmse_escore is not None:
            linhas.append("")
            linhas.append("Metricas de escore")
            linhas.append(f"MAE={resultado.mae_escore:.3f} | RMSE={resultado.rmse_escore:.3f}")

        linhas.append("")
        linhas.append("Classification report")
        linhas.append(resultado.relatorio_classificacao.rstrip())
        return "\n".join(linhas)

    def avaliar_e_formatar(self, resultados_pipeline: Sequence[Mapping[str, Any]]) -> str:
        return self.formatar_relatorio(self.avaliar(resultados_pipeline))

    def obter_intervalos_ground_truth(self) -> List[IntervaloGroundTruth]:
        return self.carregar_ground_truth()

    def obter_faixa_temporal_csv(self) -> Tuple[float, float]:
        intervalos = self.carregar_ground_truth()
        if not intervalos:
            raise ValueError("Ground truth vazio")
        return intervalos[0].inicio_s, intervalos[-1].fim_s

    def _mapear_rotulo_humano(self, rotulo_humano: str) -> str:
        chave = _normalizar_chave(rotulo_humano)
        if chave not in self.mapa_rotulos_humanos:
            raise ValueError(f"Rotulo humano nao mapeado para o sistema: {rotulo_humano!r}")
        return self.mapa_rotulos_humanos[chave]

    def _detectar_estrutura_csv(self) -> Tuple[str, List[str]]:
        texto = self.caminho_csv.read_text(
            encoding=self.config.encoding_csv,
            errors="replace",
        )
        linhas = texto.splitlines()

        candidatos = [",", "\t"]
        melhor_delimitador = ","
        maior_quantidade_colunas = 0

        for delimitador in candidatos:
            leitor = csv.reader(linhas, delimiter=delimitador)
            quantidade_maxima = 0
            for linha in leitor:
                if not linha:
                    continue
                quantidade_maxima = max(quantidade_maxima, len(linha))

            if quantidade_maxima > maior_quantidade_colunas:
                melhor_delimitador = delimitador
                maior_quantidade_colunas = quantidade_maxima

        if maior_quantidade_colunas == len(COLUNAS_CSV):
            return melhor_delimitador, list(COLUNAS_CSV)
        if maior_quantidade_colunas == len(COLUNAS_CSV_COMPACTO):
            return melhor_delimitador, list(COLUNAS_CSV_COMPACTO)
        raise ValueError(
            "CSV invalido: esperado formato com 3 ou 4 colunas, "
            f"mas foi detectado maximo de {maior_quantidade_colunas}"
        )

    def _resolver_intervalo(
        self,
        intervalo_bruto: str,
        indice: int,
        fim_anterior: float,
    ) -> Tuple[float, float]:
        texto = intervalo_bruto.strip()
        if "-" in texto:
            inicio_s, fim_s = _parse_intervalo(texto)
            return inicio_s, fim_s

        duracao = self.config.duracao_intervalo_padrao_s
        if not texto:
            inicio_s = fim_anterior
            fim_s = inicio_s + duracao
            return inicio_s, fim_s

        fim_s = _parse_fim_intervalo(texto)
        inicio_s = fim_anterior if fim_s > fim_anterior else fim_s - duracao
        if fim_s <= inicio_s:
            raise ValueError(f"Intervalo compacto invalido na linha {indice}: {intervalo_bruto!r}")
        return inicio_s, fim_s

    def _converter_codigo_emocao(self, valor: Any) -> int:
        if pd.isna(valor) or str(valor).strip() == "":
            return 0
        return int(float(valor))

    def _converter_metrica_humana(self, valor: Any) -> Optional[float]:
        if pd.isna(valor) or str(valor).strip() == "":
            return None
        return float(valor)

    def _extrair_rotulo_humano(self, linha: pd.Series) -> Optional[str]:
        if "rotulo_humano" not in linha.index:
            valor_codigo = linha.get("codigo_emocao")
            if pd.isna(valor_codigo) or str(valor_codigo).strip() == "":
                return None
            codigo = self._converter_codigo_emocao(valor_codigo)
            if codigo not in self.mapa_codigos_humanos:
                return None
            return self.mapa_codigos_humanos[codigo]
        valor = linha["rotulo_humano"]
        if pd.isna(valor) or not str(valor).strip():
            return None
        return str(valor).strip()

    def _converter_predicoes(
        self,
        resultados_pipeline: Sequence[Mapping[str, Any]],
    ) -> List[JanelaPredita]:
        predicoes: List[JanelaPredita] = []

        for indice, item in enumerate(resultados_pipeline):
            if "inicio_s" not in item or "fim_s" not in item:
                raise KeyError("Cada resultado do pipeline deve conter 'inicio_s' e 'fim_s'")
            if "emocao" not in item:
                raise KeyError("Cada resultado do pipeline deve conter a chave 'emocao'")

            inicio_s = float(item["inicio_s"])
            fim_s = float(item["fim_s"])
            if fim_s <= inicio_s:
                raise ValueError(f"Janela invalida no indice {indice}: fim_s <= inicio_s")

            emocao = _mapear_para_rotulo_sistema(str(item["emocao"]))
            escore = float(item.get("escore", 0.0))
            confianca_emo = float(item.get("confianca_emo", 0.0))

            predicoes.append(
                JanelaPredita(
                    indice_origem=indice,
                    inicio_s=inicio_s,
                    fim_s=fim_s,
                    emocao=emocao,
                    escore=escore,
                    confianca_emo=confianca_emo,
                )
            )

        predicoes.sort(key=lambda janela: (janela.inicio_s, janela.fim_s, janela.indice_origem))
        return predicoes

    def _coletar_predicoes_intervalo(
        self,
        predicoes: Sequence[JanelaPredita],
        inicio_s: float,
        fim_s: float,
    ) -> List[JanelaPredita]:
        return [
            predicao
            for predicao in predicoes
            if _ha_sobreposicao(predicao.inicio_s, predicao.fim_s, inicio_s, fim_s)
        ]

    def _agregar_classe(self, predicoes: Sequence[JanelaPredita]) -> str:
        if self.config.estrategia_classe != "majority_vote":
            raise ValueError("Estrategia de classe nao suportada")

        votos: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        for ordem, predicao in enumerate(predicoes):
            votos[predicao.emocao].append((ordem, predicao.confianca_emo))

        def chave_ordenacao(item: Tuple[str, List[Tuple[int, float]]]) -> Tuple[int, float, int]:
            _, ocorrencias = item
            quantidade = len(ocorrencias)
            confianca_media = mean(confianca for _, confianca in ocorrencias)
            primeira_aparicao = min(ordem for ordem, _ in ocorrencias)
            return (quantidade, confianca_media, -primeira_aparicao)

        melhor_rotulo, _ = max(votos.items(), key=chave_ordenacao)
        return melhor_rotulo

    def _agregar_escore(self, predicoes: Sequence[JanelaPredita]) -> float:
        if self.config.estrategia_escore != "mean":
            raise ValueError("Estrategia de escore nao suportada")
        return float(mean(predicao.escore for predicao in predicoes))

    def _ordenar_classes(
        self,
        rotulos_reais: Sequence[str],
        rotulos_preditos: Sequence[str],
    ) -> List[str]:
        classes_encontradas = set(rotulos_reais) | set(rotulos_preditos)
        classes_ordenadas = [
            classe for classe in ORDEM_CLASSES_PADRAO if classe in classes_encontradas
        ]
        extras = sorted(classes_encontradas - set(classes_ordenadas))
        return classes_ordenadas + extras

    def _formatar_matriz_confusao(self, resultado: ResultadoAvaliacao) -> str:
        if not resultado.classes_avaliadas:
            return "Sem dados avaliaveis."

        rotulo_cabecalho = "real/pred"
        largura = (
            max(
                len(rotulo_cabecalho),
                max(len(classe) for classe in resultado.classes_avaliadas),
                5,
            )
            + 2
        )

        linhas = []
        cabecalho = rotulo_cabecalho.ljust(largura)
        cabecalho += "".join(classe.rjust(largura) for classe in resultado.classes_avaliadas)
        linhas.append(cabecalho)

        for classe, linha_matriz in zip(
            resultado.classes_avaliadas,
            resultado.matriz_confusao,
        ):
            linha = classe.ljust(largura)
            linha += "".join(str(valor).rjust(largura) for valor in linha_matriz)
            linhas.append(linha)

        return "\n".join(linhas)

    def _formatar_metricas_por_classe(
        self,
        resultado: ResultadoAvaliacao,
    ) -> List[str]:
        if not resultado.metricas_por_classe:
            return ["Sem dados avaliaveis."]

        linhas = [
            "classe    precision  recall  f1     support",
        ]
        for classe in resultado.classes_avaliadas:
            metricas = resultado.metricas_por_classe[classe]
            linhas.append(
                f"{classe:<8}  "
                f"{metricas.precision:>8.3f}  "
                f"{metricas.recall:>6.3f}  "
                f"{metricas.f1:>5.3f}  "
                f"{metricas.support:>7d}"
            )
        return linhas
