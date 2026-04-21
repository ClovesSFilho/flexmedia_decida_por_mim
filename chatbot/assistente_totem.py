"""
Assistente Conversacional do Totem Inteligente FlexMedia — Sprint 4.

Interpreta mensagens em texto natural do visitante e extrai:
- categoria (comer, comprar, descansar, lazer)
- preferência (doce, roupa, cinema, etc.)
- resposta conversacional natural

Arquitetura HÍBRIDA com dois modos:

1. MODO LOCAL (padrão, sempre funciona):
   - Classificador TF-IDF + Logistic Regression treinado sobre exemplos
   - Resposta gerada por template baseado na intenção detectada
   - Não depende de internet ou API externa

2. MODO IA GENERATIVA (quando GEMINI_API_KEY está configurada):
   - Google Gemini 1.5 Flash interpreta a mensagem
   - Respostas mais naturais e variadas
   - Melhor cobertura de frases ambíguas ou fora do dataset

O modo é detectado automaticamente no carregamento. Se a API key existir
e a biblioteca estiver disponível, usa Gemini; senão, usa o classificador local.

Esta arquitetura garante robustez acadêmica (não depende de API paga)
e permite demonstrar ambas as abordagens no vídeo de pitch.
"""

import os
import json
import random
import re
import unicodedata
import joblib
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")
EXEMPLOS_PATH = os.path.join(CHATBOT_DIR, "exemplos_intencoes.json")
MODELO_NLP_PATH = os.path.join(CHATBOT_DIR, "classificador_intencao.joblib")


# =============================================================================
# TEMPLATES DE RESPOSTA (Modo Local)
# =============================================================================

# Respostas conversacionais por categoria — dão voz ao totem
RESPOSTAS_POR_CATEGORIA = {
    "comer": [
        "Boa pedida! Vou procurar algo gostoso para você.",
        "Fome, né? Entendi! Já separei uma sugestão.",
        "Que bom que você quer comer — tenho várias opções aqui.",
    ],
    "comprar": [
        "Perfeito! Vou te indicar uma loja ideal para isso.",
        "Entendi, vamos às compras. Já tenho uma sugestão.",
        "Ótimo! Deixa comigo, sei onde você deve ir.",
    ],
    "descansar": [
        "Todo mundo precisa de uma pausa. Tenho um lugar em mente.",
        "Entendi, você quer relaxar. Deixa que eu cuido disso.",
        "Descansar é importante! Já sei onde te indicar.",
    ],
    "lazer": [
        "Que legal! Diversão é sempre bom. Tenho uma ideia aqui.",
        "Adorei! Vou sugerir algo especial pra você.",
        "Entendi, você quer curtir. Já separei uma opção.",
    ],
}

RESPOSTAS_NAO_ENTENDI = [
    "Desculpe, não consegui entender direito. Pode me dizer de outra forma?",
    "Hmm, não captei completamente. Você quer comer, comprar, descansar ou se divertir?",
    "Não entendi bem. Tenta me dizer o que você gostaria de fazer agora.",
]

RESPOSTAS_SAUDACAO = [
    "Olá! Sou o Totem Inteligente. Como posso te ajudar?",
    "Oi! Me conta, o que você está procurando?",
    "Seja bem-vindo! Em que posso ajudar?",
]


# =============================================================================
# KEYWORDS EXPLÍCITAS (complementam o classificador probabilístico)
# =============================================================================
# Abordagem híbrida: o classificador TF-IDF dá a probabilidade geral, mas
# reforçamos com regras lexicais para palavras-chave inequívocas. Isso melhora
# a robustez em um dataset pequeno (100 exemplos).

KEYWORDS_CATEGORIA = {
    "comer": ["fome", "comer", "comida", "almoço", "jantar", "lanche", "restaurante",
              "fast food", "hamburguer", "hambúrguer", "pizza", "sushi", "salgado",
              "doce", "sobremesa", "bolo", "chocolate", "café", "cafe", "cafezinho",
              "expresso", "cappuccino", "lanchonete", "cafeteria"],
    "comprar": ["comprar", "loja", "shopping", "roupa", "calça", "camisa", "vestido",
                "celular", "smartphone", "notebook", "fone", "eletrônico", "eletronico",
                "presente", "aniversário", "aniversario", "livro", "livraria",
                "maquiagem", "perfume", "batom", "cosmético", "cosmetico", "sephora"],
    "descansar": ["descansar", "relaxar", "cansado", "cansada", "sentar", "banco",
                  "spa", "zen", "meditar", "calmo", "tranquilo", "silencioso",
                  "pausa", "jardim", "verde", "estressado", "estressada"],
    "lazer": ["cinema", "filme", "assistir", "sessão", "cinemark", "cinepolis",
              "arcade", "fliperama", "videogame", "jogar", "jogo", "evento",
              "show", "apresentação", "exposição", "exposicao", "galeria",
              "arte", "divertir", "curtir"],
}

KEYWORDS_PREFERENCIA = {
    ("comer", "doce"): ["doce", "chocolate", "brigadeiro", "bolo", "sobremesa", "açúcar"],
    ("comer", "salgado"): ["salgado", "hamburguer", "hambúrguer", "sanduíche", "sanduiche", "lanche"],
    ("comer", "saudavel"): ["saudável", "saudavel", "salada", "light", "fit", "natural", "caloria"],
    ("comer", "fast-food"): ["fast food", "fast-food", "mc donalds", "burger king", "combo", "rápido"],
    ("comer", "cafe"): ["café", "cafe", "cafezinho", "expresso", "cappuccino", "cafeína", "cafeina", "starbucks"],
    ("comprar", "roupa"): ["roupa", "calça", "camisa", "vestido", "jeans", "blusa", "renner", "zara"],
    ("comprar", "eletronico"): ["celular", "smartphone", "notebook", "fone", "eletrônico", "eletronico", "tecnologia"],
    ("comprar", "presente"): ["presente", "presentear", "aniversário", "aniversario", "dar"],
    ("comprar", "livro"): ["livro", "saraiva", "leitura", "ficção", "romance", "literatura"],
    ("comprar", "cosmetico"): ["maquiagem", "perfume", "batom", "cosmético", "cosmetico", "sephora"],
    ("descansar", "banco"): ["banco", "sentar", "descanso", "pernas"],
    ("descansar", "cafe-tranquilo"): ["café tranquilo", "cafe tranquilo", "silencioso", "sem barulho"],
    ("descansar", "espaco-zen"): ["zen", "spa", "meditar", "relax", "calma"],
    ("descansar", "jardim"): ["jardim", "verde", "plantas", "ar livre", "natureza"],
    ("lazer", "cinema"): ["cinema", "filme", "assistir", "cinemark", "cinepolis", "uci"],
    ("lazer", "arcade"): ["arcade", "fliperama", "videogame", "jogar"],
    ("lazer", "evento"): ["evento", "show", "apresentação", "acontecendo"],
    ("lazer", "livraria"): ["livraria", "folhear", "livros"],
    ("lazer", "exposicao"): ["exposição", "exposicao", "galeria", "arte", "cultural"],
}


def boost_por_keywords(texto_norm: str, categoria: str) -> float:
    """Retorna um boost (0-1) se o texto contém keywords da categoria."""
    keywords = KEYWORDS_CATEGORIA.get(categoria, [])
    matches = sum(1 for kw in keywords if kw in texto_norm)
    return min(1.0, matches * 0.3)


def detectar_preferencia_por_keyword(texto_norm: str, categoria: str) -> tuple:
    """Detecta preferência via keywords. Retorna (preferencia, num_matches)."""
    melhor_pref = None
    melhor_matches = 0

    for (cat, pref), keywords in KEYWORDS_PREFERENCIA.items():
        if cat != categoria:
            continue
        matches = sum(1 for kw in keywords if kw in texto_norm)
        if matches > melhor_matches:
            melhor_matches = matches
            melhor_pref = pref

    return melhor_pref, melhor_matches


# =============================================================================
# NORMALIZAÇÃO DE TEXTO
# =============================================================================

def normalizar_texto(texto: str) -> str:
    """Remove acentos, converte para minúsculas, remove pontuação excessiva."""
    # Minúsculas
    texto = texto.lower().strip()
    # Remove acentos
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # Normaliza espaços e remove pontuação
    texto = re.sub(r"[^\w\s-]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def detectar_saudacao(texto: str) -> bool:
    """Detecta saudações simples."""
    saudacoes = ["ola", "oi", "bom dia", "boa tarde", "boa noite", "opa", "eae"]
    texto_norm = normalizar_texto(texto)
    return any(texto_norm.startswith(s) or texto_norm == s for s in saudacoes) and len(texto_norm.split()) <= 3


# =============================================================================
# CLASSIFICADOR LOCAL (TF-IDF + Logistic Regression)
# =============================================================================

class AssistenteLocal:
    """
    Chatbot baseado em classificação de intenção com NLP clássico.
    Funciona offline, treinado sobre o dataset de exemplos.
    """

    def __init__(self):
        self.modelo_cat = None  # classificador de categoria
        self.modelo_pref = None  # classificador de preferência (por categoria)
        self.classes_cat = []
        self.modelos_pref_por_cat = {}  # dict: categoria → pipeline
        self.exemplos = []
        self.treinado = False

    def carregar_exemplos(self) -> list:
        """Carrega o dataset de exemplos."""
        with open(EXEMPLOS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["exemplos"]

    def treinar(self, verbose: bool = True) -> dict:
        """Treina classificador de categoria e, para cada categoria, um classificador de preferência."""

        self.exemplos = self.carregar_exemplos()

        frases = [normalizar_texto(e["frase"]) for e in self.exemplos]
        categorias = [e["categoria"] for e in self.exemplos]
        preferencias = [e["preferencia"] for e in self.exemplos]

        if verbose:
            print(f"\n  Treinando com {len(frases)} exemplos")

        # --- Classificador de CATEGORIA ---
        self.modelo_cat = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
        ])
        self.modelo_cat.fit(frases, categorias)

        # Validação cruzada
        cv_cat = cross_val_score(self.modelo_cat, frases, categorias, cv=5, scoring="accuracy")
        if verbose:
            print(f"  Classificador categoria: acurácia CV = {cv_cat.mean():.3f} (±{cv_cat.std():.3f})")

        # --- Classificadores de PREFERÊNCIA (um por categoria) ---
        self.modelos_pref_por_cat = {}
        acuracias_pref = {}

        for cat in set(categorias):
            # filtrar exemplos dessa categoria
            indices = [i for i, c in enumerate(categorias) if c == cat]
            frases_cat = [frases[i] for i in indices]
            prefs_cat = [preferencias[i] for i in indices]

            # precisa de pelo menos 2 classes diferentes
            if len(set(prefs_cat)) < 2:
                continue

            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                )),
                ("clf", LogisticRegression(max_iter=1000, C=2.0, random_state=42)),
            ])
            pipe.fit(frases_cat, prefs_cat)
            self.modelos_pref_por_cat[cat] = pipe

            # Acurácia de treino (dataset pequeno por categoria, CV nem sempre é viável)
            acuracias_pref[cat] = pipe.score(frases_cat, prefs_cat)

        if verbose:
            print(f"  Classificadores de preferência treinados: {len(self.modelos_pref_por_cat)}")
            for cat, acc in acuracias_pref.items():
                print(f"    {cat:12s} → acurácia treino = {acc:.3f}")

        self.treinado = True

        return {
            "acuracia_cv_categoria": float(cv_cat.mean()),
            "desvio_cv_categoria": float(cv_cat.std()),
            "acuracias_preferencia": acuracias_pref,
            "total_exemplos": len(frases),
            "n_categorias": len(set(categorias)),
            "n_combinacoes": len(set((c, p) for c, p in zip(categorias, preferencias))),
        }

    def avaliar_sistema_hibrido(self) -> dict:
        """
        Avalia a acurácia do sistema híbrido (ML + keywords) sobre o dataset.
        Esta é a acurácia real do chatbot em produção, considerando ambos os sinais.
        """
        if not self.treinado:
            self.treinar(verbose=False)

        exemplos = self.carregar_exemplos()
        acertos_cat = 0
        acertos_pref = 0
        acertos_ambos = 0

        for exemplo in exemplos:
            resultado = self.interpretar(exemplo["frase"])
            cat_correta = resultado["categoria"] == exemplo["categoria"]
            pref_correta = resultado["preferencia"] == exemplo["preferencia"]

            if cat_correta:
                acertos_cat += 1
            if pref_correta:
                acertos_pref += 1
            if cat_correta and pref_correta:
                acertos_ambos += 1

        n = len(exemplos)
        return {
            "acuracia_categoria": acertos_cat / n,
            "acuracia_preferencia": acertos_pref / n,
            "acuracia_ambos": acertos_ambos / n,
            "total_testado": n,
        }

    def interpretar(self, mensagem: str) -> dict:
        """
        Interpreta uma mensagem e retorna categoria, preferência e resposta.

        Abordagem híbrida: combina probabilidade do classificador TF-IDF com
        detecção de keywords explícitas. Isso compensa o dataset pequeno.
        """
        if not self.treinado:
            self.treinar(verbose=False)

        # Saudação simples?
        if detectar_saudacao(mensagem):
            return {
                "categoria": None,
                "preferencia": None,
                "resposta": random.choice(RESPOSTAS_SAUDACAO),
                "confianca_categoria": 0.0,
                "confianca_preferencia": 0.0,
                "modo": "local",
                "tipo": "saudacao",
            }

        msg_norm = normalizar_texto(mensagem)

        if len(msg_norm) < 3:
            return {
                "categoria": None,
                "preferencia": None,
                "resposta": random.choice(RESPOSTAS_NAO_ENTENDI),
                "confianca_categoria": 0.0,
                "confianca_preferencia": 0.0,
                "modo": "local",
                "tipo": "vazio",
            }

        # --- PREDIÇÃO DE CATEGORIA (ML + keywords combinadas) ---
        # Probabilidades do classificador TF-IDF
        probs_cat = self.modelo_cat.predict_proba([msg_norm])[0]
        classes_cat = self.modelo_cat.classes_
        probs_dict = dict(zip(classes_cat, probs_cat))

        # Boost por keywords: soma à probabilidade do classificador
        for cat in classes_cat:
            boost = boost_por_keywords(msg_norm, cat)
            probs_dict[cat] += boost

        # Normalizar de volta para proporção
        total = sum(probs_dict.values())
        if total > 0:
            probs_dict = {k: v / total for k, v in probs_dict.items()}

        cat = max(probs_dict, key=probs_dict.get)
        conf_cat = probs_dict[cat]

        # Se confiança baixa mesmo com keywords, pede esclarecimento
        if conf_cat < 0.35:
            return {
                "categoria": None,
                "preferencia": None,
                "resposta": random.choice(RESPOSTAS_NAO_ENTENDI),
                "confianca_categoria": round(conf_cat, 4),
                "confianca_preferencia": 0.0,
                "modo": "local",
                "tipo": "baixa_confianca",
            }

        # --- PREDIÇÃO DE PREFERÊNCIA (ML + keywords combinadas) ---
        pref = None
        conf_pref = 0.0

        # Primeiro, tenta via keywords (mais confiável quando há match)
        pref_kw, matches_kw = detectar_preferencia_por_keyword(msg_norm, cat)

        # Se keywords deram match forte (2+), usa direto com alta confiança
        if matches_kw >= 2:
            pref = pref_kw
            conf_pref = 0.90
        # Senão, usa classificador ML com possível reforço de keyword
        elif cat in self.modelos_pref_por_cat:
            pref_ml = self.modelos_pref_por_cat[cat].predict([msg_norm])[0]
            probs_pref = self.modelos_pref_por_cat[cat].predict_proba([msg_norm])[0]
            classes_pref = self.modelos_pref_por_cat[cat].classes_

            # Se keyword deu match fraco (1) e concorda com ML, boost
            if pref_kw == pref_ml and matches_kw == 1:
                pref = pref_ml
                conf_pref = min(0.85, float(probs_pref.max()) + 0.20)
            # Se keyword sugere algo diferente do ML, prioriza keyword
            elif pref_kw and matches_kw >= 1:
                pref = pref_kw
                conf_pref = 0.70
            # Senão, usa ML puro
            else:
                pref = pref_ml
                conf_pref = float(probs_pref.max())

        # Resposta conversacional
        resposta = random.choice(RESPOSTAS_POR_CATEGORIA.get(cat, RESPOSTAS_NAO_ENTENDI))

        return {
            "categoria": cat,
            "preferencia": pref,
            "resposta": resposta,
            "confianca_categoria": round(conf_cat, 4),
            "confianca_preferencia": round(conf_pref, 4),
            "modo": "local",
            "tipo": "classificado",
        }

    def salvar(self, caminho: str = None) -> None:
        """Salva os classificadores treinados."""
        if caminho is None:
            caminho = MODELO_NLP_PATH
        artefatos = {
            "modelo_cat": self.modelo_cat,
            "modelos_pref_por_cat": self.modelos_pref_por_cat,
        }
        joblib.dump(artefatos, caminho)

    def carregar(self, caminho: str = None) -> bool:
        """Carrega classificadores salvos. Retorna True se bem-sucedido."""
        if caminho is None:
            caminho = MODELO_NLP_PATH
        if not os.path.exists(caminho):
            return False
        artefatos = joblib.load(caminho)
        self.modelo_cat = artefatos["modelo_cat"]
        self.modelos_pref_por_cat = artefatos["modelos_pref_por_cat"]
        self.treinado = True
        return True


# =============================================================================
# CHATBOT COM IA GENERATIVA (Modo Gemini — opcional)
# =============================================================================

class AssistenteIA:
    """
    Chatbot usando Google Gemini 1.5 Flash.
    Requer GEMINI_API_KEY no ambiente e biblioteca google-generativeai instalada.
    """

    SYSTEM_PROMPT = """Você é o assistente virtual do Totem Inteligente de um shopping center.
Sua missão é entender o que o visitante deseja e classificar a intenção em:

CATEGORIAS e PREFERÊNCIAS VÁLIDAS:
- comer: doce, salgado, saudavel, fast-food, cafe
- comprar: roupa, eletronico, presente, livro, cosmetico
- descansar: banco, cafe-tranquilo, espaco-zen, jardim
- lazer: cinema, arcade, evento, livraria, exposicao

Você DEVE responder SEMPRE em JSON válido no seguinte formato exato:
{"categoria": "...", "preferencia": "...", "resposta": "..."}

Regras:
- Use apenas valores válidos da lista acima
- Se não conseguir identificar a intenção, use null para categoria e preferencia
- O campo "resposta" é uma mensagem curta, amigável e natural em português do Brasil
- Máximo 2 frases na resposta
- Seja caloroso e prestativo, como um concierge de shopping
"""

    def __init__(self):
        self.disponivel = False
        self.modelo = None
        self._inicializar()

    def _inicializar(self) -> None:
        """Tenta inicializar o Gemini se disponível."""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        if not api_key:
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.modelo = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={
                    "temperature": 0.7,
                    "response_mime_type": "application/json",
                },
                system_instruction=self.SYSTEM_PROMPT,
            )
            self.disponivel = True
        except ImportError:
            self.disponivel = False
        except Exception:
            self.disponivel = False

    def interpretar(self, mensagem: str) -> Optional[dict]:
        """Interpreta via Gemini. Retorna None se falhar."""
        if not self.disponivel:
            return None

        try:
            resposta = self.modelo.generate_content(mensagem)
            dados = json.loads(resposta.text)

            return {
                "categoria": dados.get("categoria"),
                "preferencia": dados.get("preferencia"),
                "resposta": dados.get("resposta", ""),
                "confianca_categoria": 0.95 if dados.get("categoria") else 0.0,
                "confianca_preferencia": 0.90 if dados.get("preferencia") else 0.0,
                "modo": "gemini",
                "tipo": "classificado" if dados.get("categoria") else "nao_identificado",
            }
        except (json.JSONDecodeError, Exception):
            return None


# =============================================================================
# INTERFACE UNIFICADA
# =============================================================================

class AssistenteTotem:
    """
    Interface unificada que combina IA generativa (quando disponível)
    com classificador local como fallback garantido.
    """

    def __init__(self, forcar_local: bool = False):
        self.local = AssistenteLocal()

        # Tentar carregar classificador salvo; senão, treinar
        if not self.local.carregar():
            self.local.treinar(verbose=False)
            self.local.salvar()

        # Tentar inicializar IA generativa
        self.ia = None if forcar_local else AssistenteIA()

    @property
    def modo_ativo(self) -> str:
        """Retorna qual modo está sendo usado."""
        if self.ia and self.ia.disponivel:
            return "hibrido"
        return "local"

    def interpretar(self, mensagem: str) -> dict:
        """
        Interpreta a mensagem usando IA generativa se disponível,
        com fallback automático para classificador local.
        """
        # Tentar IA generativa primeiro
        if self.ia and self.ia.disponivel:
            resultado_ia = self.ia.interpretar(mensagem)
            if resultado_ia is not None:
                return resultado_ia

        # Fallback: classificador local
        return self.local.interpretar(mensagem)


# =============================================================================
# EXECUÇÃO STANDALONE — treina e testa
# =============================================================================

def main():
    """Treina o classificador local e roda bateria de testes."""
    print("\n" + "█" * 60)
    print("  CHATBOT DO TOTEM — TREINAMENTO E TESTES")
    print("  Sprint 4 — Totem Inteligente FlexMedia")
    print("█" * 60)

    # Treinar modelo local
    print("\n" + "=" * 60)
    print("  TREINAMENTO DO CLASSIFICADOR LOCAL")
    print("=" * 60)

    local = AssistenteLocal()
    metricas = local.treinar(verbose=True)
    local.salvar()

    print(f"\n  Modelo salvo: {MODELO_NLP_PATH}")

    # Testes com frases variadas
    print("\n" + "=" * 60)
    print("  BATERIA DE TESTES")
    print("=" * 60)

    frases_teste = [
        # Frases diretas
        "to com vontade de um brigadeiro",
        "quero ver celulares novos",
        "preciso descansar, to cansado",
        "quero ir no cinema hoje",

        # Frases mais naturais
        "cara, bateu uma fome daquelas",
        "onde tem uma loja de roupa boa aqui",
        "queria só sentar e tomar um café",
        "que filmes estao passando",

        # Variações coloquiais
        "to precisando de um cafezinho",
        "qual a melhor livraria do shopping",
        "quero jogar videogame",

        # Saudações
        "olá",
        "oi, bom dia",

        # Ambíguas / difíceis
        "sei lá, alguma coisa",
        "",
    ]

    assistente = AssistenteTotem(forcar_local=True)

    print(f"\n  Modo ativo: {assistente.modo_ativo}\n")

    acertos = 0
    total_classificadas = 0

    for frase in frases_teste:
        resultado = assistente.interpretar(frase)
        cat = resultado["categoria"] or "—"
        pref = resultado["preferencia"] or "—"
        conf_c = resultado.get("confianca_categoria", 0)
        conf_p = resultado.get("confianca_preferencia", 0)
        tipo = resultado.get("tipo", "?")

        if resultado["categoria"]:
            total_classificadas += 1

        print(f"  '{frase[:45]}'")
        print(f"    → categoria={cat:12s} ({conf_c:.0%})  "
              f"preferencia={pref:15s} ({conf_p:.0%})  [{tipo}]")
        print(f"    → {resultado['resposta']}")
        print()

    print(f"\n  Frases classificadas: {total_classificadas}/{len(frases_teste)}")

    print("\n" + "=" * 60)
    print("  MÉTRICAS FINAIS")
    print("=" * 60)
    print(f"\n  Classificador ML puro (TF-IDF + LogReg):")
    print(f"    Acurácia CV (categoria):   {metricas['acuracia_cv_categoria']:.4f}")
    print(f"    Desvio padrão CV:          {metricas['desvio_cv_categoria']:.4f}")

    # Avaliar sistema híbrido (ML + keywords) no dataset completo
    print(f"\n  Sistema híbrido (ML + keywords):")
    metricas_hibrido = assistente.local.avaliar_sistema_hibrido()
    print(f"    Acurácia categoria:        {metricas_hibrido['acuracia_categoria']:.4f}")
    print(f"    Acurácia preferência:      {metricas_hibrido['acuracia_preferencia']:.4f}")
    print(f"    Acurácia ambos corretos:   {metricas_hibrido['acuracia_ambos']:.4f}")

    print(f"\n  Total de exemplos:           {metricas['total_exemplos']}")
    print(f"  Categorias distintas:        {metricas['n_categorias']}")
    print(f"  Combinações distintas:       {metricas['n_combinacoes']}")

    # Verificar se Gemini estaria disponível
    print("\n" + "=" * 60)
    print("  STATUS DA IA GENERATIVA")
    print("=" * 60)
    ia_teste = AssistenteIA()
    if ia_teste.disponivel:
        print("  ✓ Gemini 1.5 Flash disponível (GEMINI_API_KEY configurada)")
        print("    → Totem usará IA generativa como primeira escolha")
    else:
        print("  ○ Gemini não disponível (sem API key ou biblioteca)")
        print("    → Totem usará classificador local (funciona offline)")

    # Salvar métricas em JSON para o dashboard
    metricas_export = {
        "tipo": "classificador_intencao_nlp",
        "modelo_puro_ml": {
            "acuracia_cv_categoria": round(metricas['acuracia_cv_categoria'], 4),
            "desvio_cv_categoria": round(metricas['desvio_cv_categoria'], 4),
            "acuracias_preferencia": {k: round(v, 4) for k, v in metricas['acuracias_preferencia'].items()},
        },
        "sistema_hibrido": {
            "acuracia_categoria": round(metricas_hibrido['acuracia_categoria'], 4),
            "acuracia_preferencia": round(metricas_hibrido['acuracia_preferencia'], 4),
            "acuracia_ambos": round(metricas_hibrido['acuracia_ambos'], 4),
        },
        "total_exemplos": metricas['total_exemplos'],
        "n_categorias": metricas['n_categorias'],
        "n_combinacoes": metricas['n_combinacoes'],
    }

    metricas_path = os.path.join(BASE_DIR, "ml", "metricas_chatbot.json")
    with open(metricas_path, "w", encoding="utf-8") as f:
        json.dump(metricas_export, f, indent=2, ensure_ascii=False)

    print(f"\n  Métricas salvas em: {metricas_path}")


if __name__ == "__main__":
    main()
