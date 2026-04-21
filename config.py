"""
Configurações e constantes compartilhadas do Totem Inteligente FlexMedia.

Este arquivo é a ÚNICA fonte de verdade para mapeamentos de lojas,
preferências, emojis e motivos de rejeição. Todos os módulos importam
daqui em vez de duplicar dicionários.

Alterações em lojas, preferências ou emojis devem ser feitas SOMENTE aqui.
"""

# =============================================================================
# CATEGORIAS E PREFERÊNCIAS
# =============================================================================

CATEGORIAS_INFO = {
    "comer":     {"emoji": "🍽️", "label": "Comer",     "desc": "Restaurantes e lanches"},
    "comprar":   {"emoji": "🛍️", "label": "Comprar",   "desc": "Lojas e produtos"},
    "descansar": {"emoji": "🧘", "label": "Descansar", "desc": "Relaxar e recarregar"},
    "lazer":     {"emoji": "🎬", "label": "Lazer",     "desc": "Diversão e cultura"},
}

PREFERENCIAS_POR_CATEGORIA = {
    "comer":     ["doce", "salgado", "saudavel", "fast-food", "cafe"],
    "comprar":   ["roupa", "eletronico", "presente", "livro", "cosmetico"],
    "descansar": ["banco", "cafe-tranquilo", "espaco-zen", "jardim"],
    "lazer":     ["cinema", "arcade", "evento", "livraria", "exposicao"],
}

PREFERENCIAS_LABELS = {
    "comer":     {"doce": "🍫 Doce", "salgado": "🍔 Salgado", "saudavel": "🥗 Saudável",
                  "fast-food": "🍟 Fast-food", "cafe": "☕ Café"},
    "comprar":   {"roupa": "👕 Roupa", "eletronico": "📱 Eletrônico", "presente": "🎁 Presente",
                  "livro": "📚 Livro", "cosmetico": "💄 Cosmético"},
    "descansar": {"banco": "🪑 Banco", "cafe-tranquilo": "☕ Café tranquilo",
                  "espaco-zen": "🧘 Espaço zen", "jardim": "🌿 Jardim"},
    "lazer":     {"cinema": "🎬 Cinema", "arcade": "🕹️ Arcade", "evento": "🎤 Evento",
                  "livraria": "📖 Livraria", "exposicao": "🖼️ Exposição"},
}


# =============================================================================
# LOJAS POR COMBINAÇÃO (categoria, preferência) → lista de lojas
# =============================================================================

LOJAS_POR_COMBINACAO = {
    ("comer", "doce"):          ["Cacau Show", "Starbucks", "Kopenhagen"],
    ("comer", "salgado"):       ["McDonalds", "Subway", "Burger King"],
    ("comer", "saudavel"):      ["Mundo Verde", "Green Station", "Hortifruti"],
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


# =============================================================================
# EMOJIS POR LOJA
# =============================================================================

EMOJIS_LOJAS = {
    "Cacau Show": "🍫", "Starbucks": "☕", "Kopenhagen": "🍬",
    "McDonalds": "🍔", "Subway": "🥪", "Burger King": "🍔",
    "Mundo Verde": "🥗", "Green Station": "🥗", "Hortifruti": "🥗",
    "Popeyes": "🍗", "Cafe do Mercado": "☕", "Havanna": "☕",
    "Renner": "👕", "Zara": "👗", "C&A": "👖",
    "Fast Shop": "📱", "Kalunga": "🖥️", "iPlace": "🍎",
    "O Boticario": "🌸", "Vivara": "💍", "Pandora": "💎",
    "Livraria Cultura": "📚", "Saraiva": "📖", "Leitura": "📗",
    "Sephora": "💄", "MAC": "💋",
    "Praca Central": "🪑", "Area de Descanso A": "🛋️",
    "Spa Express": "💆", "Espaco Relax": "🧘",
    "Jardim Interno": "🌿", "Terraco Verde": "🌳",
    "Cinemark": "🎬", "Cinepolis": "🎥", "UCI": "🎞️",
    "GameStation": "🕹️", "Playland": "🎮", "Fliperama": "👾",
    "Espaco Eventos": "🎤", "Palco Central": "🎭",
    "Galeria Arte": "🖼️", "Espaco Cultural": "🎨",
}


# =============================================================================
# MOTIVOS DE REJEIÇÃO
# =============================================================================

MOTIVOS_REJEICAO = [
    "nao interessou",
    "muito longe",
    "ja conheco",
    "preco alto",
    "sem tempo",
    "prefere outro",
]

MOTIVOS_REJEICAO_LABELS = {
    "nao interessou": "Não interessou",
    "muito longe":    "Muito longe",
    "ja conheco":     "Já conheço",
    "preco alto":     "Preço alto",
    "sem tempo":      "Sem tempo",
    "prefere outro":  "Prefere outro",
}


# =============================================================================
# PESOS DE SIMULAÇÃO (usado pelo sensor_simulado.py)
# =============================================================================

PESO_CATEGORIA_POR_HORARIO = {
    "manha":  {"comer": 0.20, "comprar": 0.45, "descansar": 0.10, "lazer": 0.25},
    "almoco": {"comer": 0.55, "comprar": 0.15, "descansar": 0.15, "lazer": 0.15},
    "tarde":  {"comer": 0.20, "comprar": 0.35, "descansar": 0.15, "lazer": 0.30},
    "noite":  {"comer": 0.25, "comprar": 0.15, "descansar": 0.10, "lazer": 0.50},
}

PESO_CATEGORIA_POR_IDADE = {
    "jovem":  {"comer": 0.30, "comprar": 0.20, "descansar": 0.05, "lazer": 0.45},
    "adulto": {"comer": 0.25, "comprar": 0.40, "descansar": 0.10, "lazer": 0.25},
    "idoso":  {"comer": 0.20, "comprar": 0.15, "descansar": 0.45, "lazer": 0.20},
}

VOLUME_POR_DIA = {
    "segunda": 12, "terca": 10, "quarta": 9, "quinta": 12,
    "sexta": 18, "sabado": 25, "domingo": 22,
}

DIAS_SEMANA_PT = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
FAIXAS_ETARIAS = ["jovem", "adulto", "idoso"]
PESO_FAIXA_ETARIA = [0.35, 0.45, 0.20]

FAIXAS_HORARIAS = {
    "manha":  (9, 11),
    "almoco": (11, 14),
    "tarde":  (14, 18),
    "noite":  (18, 22),
}


# =============================================================================
# FUNÇÕES UTILITÁRIAS COMPARTILHADAS
# =============================================================================

from datetime import datetime


def obter_faixa_horaria(hora: int = None) -> str:
    """Retorna a faixa horária baseada na hora fornecida ou atual."""
    if hora is None:
        hora = datetime.now().hour
    if 9 <= hora < 11:
        return "manha"
    elif 11 <= hora < 14:
        return "almoco"
    elif 14 <= hora < 18:
        return "tarde"
    else:
        return "noite"


def obter_dia_semana(data: datetime = None) -> str:
    """Retorna o dia da semana em português."""
    if data is None:
        data = datetime.now()
    return DIAS_SEMANA_PT[data.weekday()]
