# -*- coding: utf-8 -*-
"""
Módulo de Vocabulário de Domínio (TAD) - Futebol

Melhorias aplicadas:
- Estrutura com dataclasses para validação
- Chave padronizada "termos" (antes era "palavras")
- Termos adicionais e mais granulares
- Função de pré-processamento para otimização
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class CategoriaContexto:
    """Representa uma categoria de fala do técnico"""

    termos: List[str]
    pontuacao_base: float
    descricao: str = ""
    exemplos: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Valida pontuação"""
        if not 0 <= self.pontuacao_base <= 10:
            raise ValueError(f"Pontuação deve estar entre 0-10, recebido: {self.pontuacao_base}")


# ------------------------------
# Vocabulário Principal
# ------------------------------
VOCABULARIO_FUTEBOL: Dict[str, CategoriaContexto] = {
    # === INTERAÇÕES POSITIVAS ===
    "REFORCO_POSITIVO_ALTO": CategoriaContexto(
        termos=[
            "golaço",
            "perfeito",
            "excelente",
            "que jogada",
            "fenomenal",
            "craque",
            "espetacular",
            "sensacional",
            "show",
        ],
        pontuacao_base=10,
        descricao="Elogio extremo, comemoração efusiva",
        exemplos=["Que golaço!", "Perfeito, fenomenal!"],
    ),
    "REFORCO_POSITIVO": CategoriaContexto(
        termos=[
            "boa",
            "isso",
            "muito bem",
            "é isso aí",
            "ótimo",
            "gostei",
            "parabéns",
            "belo passe",
            "belo chute",
            "continue assim",
            "tá bom",
            "valeu",
            "assim mesmo",
        ],
        pontuacao_base=9,
        descricao="Reforço positivo geral",
        exemplos=["Muito bem!", "Isso aí, boa!"],
    ),
    "MOTIVACAO_E_INCENTIVO": CategoriaContexto(
        termos=[
            "vamos",
            "pra cima",
            "acredita",
            "não desiste",
            "força",
            "coragem",
            "cabeça erguida",
            "é nosso",
            "vamos virar",
            "bora",
            "segue",
            "foco",
            "concentra",
        ],
        pontuacao_base=8,
        descricao="Incentivo, motivação",
        exemplos=["Vamos, pra cima!", "Força, não desiste"],
    ),
    # === INTERAÇÕES NEUTRAS (TÁTICAS) ===
    "INSTRUCAO_TÁTICA_OFENSIVA": CategoriaContexto(
        termos=[
            "chuta",
            "passa",
            "cruza",
            "dribla",
            "infiltra",
            "tabela",
            "abre o jogo",
            "vira o jogo",
            "puxa",
            "arranca",
            "finaliza",
            "toca",
            "lança",
            "avança",
        ],
        pontuacao_base=6,
        descricao="Instruções ofensivas",
        exemplos=["Cruza!", "Abre o jogo, vira!"],
    ),
    "INSTRUCAO_TÁTICA_DEFENSIVA": CategoriaContexto(
        termos=[
            "marca",
            "volta",
            "fecha",
            "sobe a linha",
            "pressão",
            "cobre",
            "fica nele",
            "não deixa virar",
            "recua",
            "dobra a marcação",
            "compacta",
        ],
        pontuacao_base=5,
        descricao="Instruções defensivas",
        exemplos=["Marca! Fecha!", "Sobe a linha"],
    ),
    "INSTRUCAO_TÁTICA_GERAL": CategoriaContexto(
        termos=[
            "quero",
            "faz",
            "aqui",
            "lá",
            "linha",
            "área",
            "lateral",
            "meio",
            "zagueiro",
            "atenção",
            "olha",
            "posiciona",
            "gira",
            "movimenta",
            "espaço",
            "tempo",
        ],
        pontuacao_base=5,
        descricao="Instruções gerais e posicionamento",
        exemplos=["Atenção na linha!", "Posiciona aqui"],
    ),
    # === INTERAÇÕES NEGATIVAS ===
    "CRITICA_CONSTRUTIVA": CategoriaContexto(
        termos=[
            "mais atenção",
            "não pode",
            "errado",
            "de novo não",
            "acorda",
            "mais força",
            "mais rápido",
            "não assim",
            "melhora",
            "capricha",
            "pensa",
        ],
        pontuacao_base=3,
        descricao="Crítica leve, correção",
        exemplos=["Não pode assim!", "Mais atenção"],
    ),
    "FRUSTRAÇÃO_E_LAMENTO": CategoriaContexto(
        termos=[
            "ai ai ai",
            "não acredito",
            "pelo amor de deus",
            "de novo",
            "assim não dá",
            "que azar",
            "poxa",
            "caramba",
            "nossa",
        ],
        pontuacao_base=2,
        descricao="Frustração, lamento",
        exemplos=["Ai ai ai!", "Não acredito nisso"],
    ),
    "CRITICA_DURA": CategoriaContexto(
        termos=[
            "tá dormindo",
            "que passe foi esse",
            "inadmissível",
            "brincadeira",
            "desse jeito não",
            "absurdo",
            "ridículo",
            "vergonha",
        ],
        pontuacao_base=1,
        descricao="Crítica severa, repreensão",
        exemplos=["Tá dormindo?", "Inadmissível!"],
    ),
    # === CONTEXTO DE JOGO ===
    "RECLAMACAO_ARBITRAGEM": CategoriaContexto(
        termos=[
            "falta",
            "juiz",
            "roubou",
            "cartão",
            "apita direito",
            "impedimento",
            "foi nada",
            "absurdo",
            "bandeira",
            "var",
        ],
        pontuacao_base=2,
        descricao="Reclamação com árbitro",
        exemplos=["Falta, juiz!", "Foi nada disso"],
    ),
    "COMEMORACAO": CategoriaContexto(
        termos=["gol", "é gol", "comemora", "goool", "é dele"],
        pontuacao_base=10,
        descricao="Comemoração de gol",
        exemplos=["GOOOL!", "É gol, comemora!"],
    ),
}


# ------------------------------
# Funções Auxiliares
# ------------------------------
def remover_acentos(txt: str) -> str:
    """Remove acentuação mantendo caracteres base"""
    nfkd = unicodedata.normalize("NFKD", txt)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])


def normalizar_texto(txt: str) -> str:
    """Normaliza texto para comparação"""
    txt = txt.lower().strip()
    txt = remover_acentos(txt)
    # Remove pontuação mas mantém espaços
    txt = re.sub(r"[^\w\s]", " ", txt, flags=re.UNICODE)
    # Colapsa múltiplos espaços
    txt = re.sub(r"\s+", " ", txt, flags=re.UNICODE).strip()
    return txt


def preprocessar_vocabulario() -> Dict[str, Dict]:
    """
    Pré-processa vocabulário normalizando todos os termos.
    OTIMIZAÇÃO: Executa normalização UMA vez ao invés de repetir a cada query.
    """
    vocab_normalizado = {}

    for categoria, config in VOCABULARIO_FUTEBOL.items():
        termos_normalizados = [normalizar_texto(termo) for termo in config.termos]

        vocab_normalizado[categoria] = {
            "termos": termos_normalizados,
            "pontuacao_base": config.pontuacao_base,
            "descricao": config.descricao,
        }

    return vocab_normalizado


# Gera versão otimizada ao importar o módulo
VOCABULARIO_NORMALIZADO = preprocessar_vocabulario()


# ------------------------------
# Estatísticas do vocabulário
# ------------------------------
def estatisticas_vocabulario():
    """Retorna estatísticas do vocabulário carregado"""
    total_termos = sum(len(cfg.termos) for cfg in VOCABULARIO_FUTEBOL.values())
    total_categorias = len(VOCABULARIO_FUTEBOL)

    return {
        "total_categorias": total_categorias,
        "total_termos": total_termos,
        "media_termos_por_categoria": round(total_termos / total_categorias, 1),
        "categorias": list(VOCABULARIO_FUTEBOL.keys()),
    }


if __name__ == "__main__":
    # Teste do vocabulário
    print("=== Vocabulário de Futebol ===")
    stats = estatisticas_vocabulario()
    print(f"Categorias: {stats['total_categorias']}")
    print(f"Termos totais: {stats['total_termos']}")
    print(f"Média por categoria: {stats['media_termos_por_categoria']}")
    print("\nCategorias disponíveis:")
    for cat in stats["categorias"]:
        print(f"  - {cat}: {VOCABULARIO_FUTEBOL[cat].pontuacao_base} pts")
