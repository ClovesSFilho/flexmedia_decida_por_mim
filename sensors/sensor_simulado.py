"""
Sensor simulado do Totem Inteligente FlexMedia.

Gera dados realistas de interação com regras de correlação que simulam
comportamento humano em um shopping center:

- Categoria influenciada por faixa horária e faixa etária
- Preferência vinculada à categoria escolhida
- Loja recomendada coerente com categoria + preferência
- Tempo de interação influencia probabilidade de aceitação
- Volume de sessões varia por dia da semana
- Motivo de rejeição registrado quando usuário recusa

Gera ~500 sessões distribuídas ao longo de 30 dias.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURAÇÕES E REGRAS DE CORRELAÇÃO
# =============================================================================

# Preferências por categoria (cada categoria tem suas próprias preferências)
PREFERENCIAS_POR_CATEGORIA = {
    "comer":      ["doce", "salgado", "saudavel", "fast-food", "cafe"],
    "comprar":    ["roupa", "eletronico", "presente", "livro", "cosmetico"],
    "descansar":  ["banco", "cafe-tranquilo", "espaco-zen", "jardim"],
    "lazer":      ["cinema", "arcade", "evento", "livraria", "exposicao"],
}

# Lojas recomendadas por categoria + preferência
LOJAS_POR_COMBINACAO = {
    ("comer", "doce"):          ["Cacau Show", "Starbucks", "Kopenhagen"],
    ("comer", "salgado"):       ["McDonalds", "Subway", "Burger King"],
    ("comer", "saudavel"):      ["Mundo Verde", "Green Station", "Freshii"],
    ("comer", "fast-food"):     ["McDonalds", "Burger King", "Popeyes"],
    ("comer", "cafe"):          ["Starbucks", "Cafe do Mercado", "Havanna"],
    ("comprar", "roupa"):       ["Renner", "Zara", "C&A"],
    ("comprar", "eletronico"):  ["Fast Shop", "Kalunga", "iPlace"],
    ("comprar", "presente"):    ["O Boticario", "Vivara", "Pandora"],
    ("comprar", "livro"):       ["Livraria Cultura", "Saraiva", "Leitura"],
    ("comprar", "cosmetico"):   ["O Boticario", "Sephora", "MAC"],
    ("descansar", "banco"):             ["Praca Central", "Area de Descanso A"],
    ("descansar", "cafe-tranquilo"):    ["Cafe do Mercado", "Havanna"],
    ("descansar", "espaco-zen"):        ["Spa Express", "Espaco Relax"],
    ("descansar", "jardim"):            ["Jardim Interno", "Terraco Verde"],
    ("lazer", "cinema"):        ["Cinemark", "Cinepolis", "UCI"],
    ("lazer", "arcade"):        ["GameStation", "Playland", "Fliperama"],
    ("lazer", "evento"):        ["Espaco Eventos", "Palco Central"],
    ("lazer", "livraria"):      ["Livraria Cultura", "Leitura"],
    ("lazer", "exposicao"):     ["Galeria Arte", "Espaco Cultural"],
}

# Peso de cada categoria por faixa horária (simula comportamento real)
PESO_CATEGORIA_POR_HORARIO = {
    "manha":  {"comer": 0.20, "comprar": 0.45, "descansar": 0.10, "lazer": 0.25},
    "almoco": {"comer": 0.55, "comprar": 0.15, "descansar": 0.15, "lazer": 0.15},
    "tarde":  {"comer": 0.20, "comprar": 0.35, "descansar": 0.15, "lazer": 0.30},
    "noite":  {"comer": 0.25, "comprar": 0.15, "descansar": 0.10, "lazer": 0.50},
}

# Peso de cada categoria por faixa etária
PESO_CATEGORIA_POR_IDADE = {
    "jovem":  {"comer": 0.30, "comprar": 0.20, "descansar": 0.05, "lazer": 0.45},
    "adulto": {"comer": 0.25, "comprar": 0.40, "descansar": 0.10, "lazer": 0.25},
    "idoso":  {"comer": 0.20, "comprar": 0.15, "descansar": 0.45, "lazer": 0.20},
}

# Volume de sessões por dia da semana (fim de semana = mais movimento)
VOLUME_POR_DIA = {
    "segunda": 12, "terca": 10, "quarta": 9, "quinta": 12,
    "sexta": 18, "sabado": 25, "domingo": 22,
}

DIAS_SEMANA_PT = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]

FAIXAS_ETARIAS = ["jovem", "adulto", "idoso"]
PESO_FAIXA_ETARIA = [0.35, 0.45, 0.20]  # distribuição no shopping

MOTIVOS_REJEICAO = [
    "nao interessou",
    "muito longe",
    "ja conheco",
    "preco alto",
    "sem tempo",
    "prefere outro",
]

# Faixas horárias e seus intervalos
FAIXAS_HORARIAS = {
    "manha":  (9, 11),
    "almoco": (11, 14),
    "tarde":  (14, 18),
    "noite":  (18, 22),
}


# =============================================================================
# FUNÇÕES DE GERAÇÃO
# =============================================================================

def escolher_com_pesos(opcoes: dict) -> str:
    """Escolhe uma chave do dicionário usando os valores como pesos."""
    chaves = list(opcoes.keys())
    pesos = list(opcoes.values())
    return random.choices(chaves, weights=pesos, k=1)[0]


def combinar_pesos_categoria(faixa_horaria: str, faixa_etaria: str) -> dict:
    """
    Combina os pesos de categoria por horário e por idade.
    Faz a média ponderada (60% horário, 40% idade) para criar
    um perfil mais realista.
    """
    pesos_horario = PESO_CATEGORIA_POR_HORARIO[faixa_horaria]
    pesos_idade = PESO_CATEGORIA_POR_IDADE[faixa_etaria]

    combinado = {}
    for cat in pesos_horario:
        combinado[cat] = pesos_horario[cat] * 0.6 + pesos_idade[cat] * 0.4

    return combinado


def calcular_probabilidade_aceitacao(
    tempo_interacao: int,
    categoria: str,
    faixa_etaria: str
) -> float:
    """
    Calcula a probabilidade de aceitação da recomendação.

    Fatores que aumentam a chance de aceitar:
    - Tempo de interação alto (>10s = mais engajado)
    - Categoria "comer" no horário de almoço (necessidade imediata)
    - Faixa etária adulto (mais decisivo)

    Fatores que diminuem:
    - Tempo curto (<5s = passando rápido)
    - Faixa etária jovem (mais indeciso)
    """
    # base
    prob = 0.45

    # tempo de interação é o fator mais forte
    if tempo_interacao >= 15:
        prob += 0.30
    elif tempo_interacao >= 10:
        prob += 0.20
    elif tempo_interacao >= 7:
        prob += 0.10
    elif tempo_interacao <= 4:
        prob -= 0.15

    # categoria
    if categoria == "comer":
        prob += 0.08  # necessidade imediata
    elif categoria == "descansar":
        prob += 0.05  # geralmente aceita sugestão de descanso
    elif categoria == "lazer":
        prob -= 0.03  # mais seletivo

    # faixa etária
    if faixa_etaria == "adulto":
        prob += 0.05
    elif faixa_etaria == "jovem":
        prob -= 0.05
    elif faixa_etaria == "idoso":
        prob += 0.03

    # limitar entre 0.1 e 0.95
    return max(0.1, min(0.95, prob))


def gerar_sessao(data_base: datetime, faixa_horaria: str) -> dict:
    """Gera uma sessão completa com interações e recomendações."""

    # perfil do visitante
    faixa_etaria = random.choices(FAIXAS_ETARIAS, weights=PESO_FAIXA_ETARIA, k=1)[0]
    dia_semana = DIAS_SEMANA_PT[data_base.weekday()]

    # horário de início dentro da faixa
    hora_min, hora_max = FAIXAS_HORARIAS[faixa_horaria]
    hora = random.randint(hora_min, hora_max - 1)
    minuto = random.randint(0, 59)
    segundo = random.randint(0, 59)
    inicio = data_base.replace(hour=hora, minute=minuto, second=segundo)

    # cada sessão tem 1 a 3 interações (maioria tem 1)
    num_interacoes = random.choices([1, 2, 3], weights=[0.60, 0.30, 0.10], k=1)[0]

    # pesos combinados para escolha de categoria
    pesos_categoria = combinar_pesos_categoria(faixa_horaria, faixa_etaria)

    interacoes = []
    tempo_acumulado = 0

    for i in range(num_interacoes):
        # categoria baseada em horário + idade
        categoria = escolher_com_pesos(pesos_categoria)

        # preferência coerente com a categoria
        preferencia = random.choice(PREFERENCIAS_POR_CATEGORIA[categoria])

        # tempo de interação (distribuição realista)
        if faixa_etaria == "idoso":
            tempo = random.randint(8, 25)  # idosos demoram mais
        elif faixa_etaria == "jovem":
            tempo = random.randint(2, 15)  # jovens são mais rápidos
        else:
            tempo = random.randint(4, 20)

        # timestamp da interação
        ts = inicio + timedelta(seconds=tempo_acumulado)
        tempo_acumulado += tempo + random.randint(2, 8)  # pausa entre interações

        # loja recomendada coerente
        chave = (categoria, preferencia)
        lojas_possiveis = LOJAS_POR_COMBINACAO.get(chave, ["Loja Generica"])
        loja = random.choice(lojas_possiveis)

        # probabilidade de aceitação baseada em múltiplos fatores
        prob_aceitar = calcular_probabilidade_aceitacao(tempo, categoria, faixa_etaria)
        aceitou = 1 if random.random() < prob_aceitar else 0

        # motivo de rejeição (só quando não aceita)
        motivo = None
        if aceitou == 0:
            motivo = random.choice(MOTIVOS_REJEICAO)

        interacoes.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "categoria": categoria,
            "preferencia": preferencia,
            "tempo_interacao": tempo,
            "loja_recomendada": loja,
            "aceitou": aceitou,
            "motivo_rejeicao": motivo,
        })

    fim = inicio + timedelta(seconds=tempo_acumulado)

    return {
        "sessao": {
            "inicio_sessao": inicio.strftime("%Y-%m-%d %H:%M:%S"),
            "fim_sessao": fim.strftime("%Y-%m-%d %H:%M:%S"),
            "faixa_etaria": faixa_etaria,
            "dia_semana": dia_semana,
            "faixa_horaria": faixa_horaria,
        },
        "interacoes": interacoes,
    }


# =============================================================================
# INSERÇÃO NO BANCO
# =============================================================================

def popular_banco(caminho_db: str = None, dias: int = 30, seed: int = 42) -> None:
    """Gera sessões ao longo de N dias e insere no banco."""

    random.seed(seed)  # reprodutibilidade

    if caminho_db is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        caminho_db = os.path.join(base_dir, "data", "interacoes.db")

    if not os.path.exists(caminho_db):
        print("ERRO: banco de dados não encontrado. Execute create_database.py primeiro.")
        return

    conn = sqlite3.connect(caminho_db)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    data_inicio = datetime(2025, 5, 1)  # início da simulação
    total_sessoes = 0
    total_interacoes = 0

    print("=" * 50)
    print("GERANDO DADOS SIMULADOS")
    print("=" * 50)

    for dia in range(dias):
        data_atual = data_inicio + timedelta(days=dia)
        dia_semana = DIAS_SEMANA_PT[data_atual.weekday()]

        # volume do dia baseado no dia da semana
        num_sessoes_dia = VOLUME_POR_DIA[dia_semana]
        # adiciona variação aleatória de ±20%
        num_sessoes_dia = max(5, int(num_sessoes_dia * random.uniform(0.8, 1.2)))

        for _ in range(num_sessoes_dia):
            # distribuir sessões pelas faixas horárias
            faixa = random.choices(
                ["manha", "almoco", "tarde", "noite"],
                weights=[0.20, 0.25, 0.30, 0.25],
                k=1
            )[0]

            sessao_dados = gerar_sessao(data_atual, faixa)

            # inserir sessão
            s = sessao_dados["sessao"]
            cursor.execute("""
                INSERT INTO sessoes
                (inicio_sessao, fim_sessao, faixa_etaria, dia_semana, faixa_horaria)
                VALUES (?, ?, ?, ?, ?)
            """, (
                s["inicio_sessao"], s["fim_sessao"],
                s["faixa_etaria"], s["dia_semana"], s["faixa_horaria"]
            ))
            sessao_id = cursor.lastrowid

            # inserir interações e recomendações
            for inter in sessao_dados["interacoes"]:
                cursor.execute("""
                    INSERT INTO interacoes
                    (sessao_id, timestamp, categoria, preferencia, tempo_interacao)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    sessao_id, inter["timestamp"],
                    inter["categoria"], inter["preferencia"],
                    inter["tempo_interacao"]
                ))
                interacao_id = cursor.lastrowid

                cursor.execute("""
                    INSERT INTO recomendacoes
                    (interacao_id, loja_recomendada, aceitou, motivo_rejeicao)
                    VALUES (?, ?, ?, ?)
                """, (
                    interacao_id, inter["loja_recomendada"],
                    inter["aceitou"], inter["motivo_rejeicao"]
                ))

                total_interacoes += 1

            total_sessoes += 1

    conn.commit()
    conn.close()

    print(f"\nPeríodo simulado: {data_inicio.strftime('%d/%m/%Y')} a "
          f"{(data_inicio + timedelta(days=dias-1)).strftime('%d/%m/%Y')}")
    print(f"Total de sessões geradas:    {total_sessoes}")
    print(f"Total de interações geradas: {total_interacoes}")
    print(f"\nDados salvos em: {caminho_db}")


if __name__ == "__main__":
    popular_banco()