"""
Interface Interativa do Totem Inteligente FlexMedia — Sprint 4.

Simula a experiência completa do visitante no totem do shopping:
1. Detecção de presença (integração com módulo de visão — Etapa 2)
2. Seleção de intenção e preferência (botões ou chat — Etapa 3)
3. Recomendação personalizada via modelo de ML treinado
4. Registro de feedback (aceitou/rejeitou) no banco SQLite
5. Histórico de interações da sessão atual

O modelo treinado na Sprint 3 (Logistic Regression, F1=0.80) é usado
de forma invertida: dado o perfil do visitante, testamos todas as lojas
possíveis e recomendamos a que tem maior probabilidade de aceitação.

Executar com: streamlit run totem/app_totem.py
"""

import streamlit as st
import sqlite3
import os
import sys
import numpy as np
import pandas as pd
import joblib
import cv2
from datetime import datetime

# Adicionar raiz do projeto ao path para imports
BASE_DIR_MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR_MODULE not in sys.path:
    sys.path.insert(0, BASE_DIR_MODULE)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "interacoes.db")
MODELO_PATH = os.path.join(BASE_DIR, "ml", "modelo_completo.joblib")

st.set_page_config(
    page_title="Totem FlexMedia",
    page_icon="🏬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# CSS customizado — visual de totem real (tela escura, elementos grandes)
st.markdown("""
<style>
    /* Fundo escuro estilo totem */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* Esconder sidebar e menu por padrão */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }

    /* Título central */
    .totem-titulo {
        text-align: center;
        color: #f8fafc;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .totem-subtitulo {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Cards de categoria */
    .categoria-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        transition: all 0.2s ease;
        cursor: pointer;
    }
    .categoria-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.05));
    }
    .categoria-emoji {
        font-size: 2.5rem;
        display: block;
        margin-bottom: 0.5rem;
    }
    .categoria-label {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 600;
    }

    /* Resultado da recomendação */
    .recomendacao-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }
    .recomendacao-loja {
        color: #10b981;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .recomendacao-prob {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* Feedback de agradecimento */
    .agradecimento-box {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(99, 102, 241, 0.05));
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    /* Estilo geral de texto */
    .stMarkdown p, .stMarkdown li { color: #cbd5e1; }
    h1, h2, h3 { color: #f1f5f9 !important; }

    /* Botões maiores */
    .stButton > button {
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.15s ease;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DADOS DE REFERÊNCIA (espelhados do sensor_simulado.py)
# =============================================================================

CATEGORIAS = {
    "comer":     {"emoji": "🍽️", "label": "Comer",     "desc": "Restaurantes e lanches"},
    "comprar":   {"emoji": "🛍️", "label": "Comprar",   "desc": "Lojas e produtos"},
    "descansar": {"emoji": "🧘", "label": "Descansar", "desc": "Relaxar e recarregar"},
    "lazer":     {"emoji": "🎬", "label": "Lazer",     "desc": "Diversão e cultura"},
}

PREFERENCIAS_POR_CATEGORIA = {
    "comer":     {"doce": "🍫 Doce", "salgado": "🍔 Salgado", "saudavel": "🥗 Saudável",
                  "fast-food": "🍟 Fast-food", "cafe": "☕ Café"},
    "comprar":   {"roupa": "👕 Roupa", "eletronico": "📱 Eletrônico", "presente": "🎁 Presente",
                  "livro": "📚 Livro", "cosmetico": "💄 Cosmético"},
    "descansar": {"banco": "🪑 Banco", "cafe-tranquilo": "☕ Café tranquilo",
                  "espaco-zen": "🧘 Espaço zen", "jardim": "🌿 Jardim"},
    "lazer":     {"cinema": "🎬 Cinema", "arcade": "🕹️ Arcade", "evento": "🎤 Evento",
                  "livraria": "📖 Livraria", "exposicao": "🖼️ Exposição"},
}

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

MOTIVOS_REJEICAO = [
    "Não interessou",
    "Muito longe",
    "Já conheço",
    "Preço alto",
    "Sem tempo",
    "Prefere outro",
]

EMOJIS_LOJAS = {
    "Cacau Show": "🍫", "Starbucks": "☕", "Kopenhagen": "🍬",
    "McDonalds": "🍔", "Subway": "🥪", "Burger King": "🍔",
    "Mundo Verde": "🥗", "Green Station": "🥗", "Freshii": "🥗",
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
# CARREGAMENTO DO MODELO
# =============================================================================

@st.cache_resource
def carregar_modelo():
    """Carrega o modelo treinado com encoders embutidos."""
    if not os.path.exists(MODELO_PATH):
        st.error(
            "Modelo não encontrado. Execute primeiro:\n"
            "```python ml/modelo_ml.py```\n"
            "E depois o script de geração do modelo completo."
        )
        st.stop()

    artefatos = joblib.load(MODELO_PATH)
    return artefatos["modelo"], artefatos["encoders"], artefatos["colunas"]


modelo, encoders, colunas_modelo = carregar_modelo()


@st.cache_resource
def carregar_detector_presenca():
    """Carrega o detector de presença (Visão Computacional)."""
    from vision.detector_presenca import DetectorPresenca, extrair_features
    detector = DetectorPresenca(metodo="hog")

    # Carregar classificador ML de presença (se disponível)
    classificador_path = os.path.join(BASE_DIR, "ml", "classificador_presenca.joblib")
    classificador = None
    if os.path.exists(classificador_path):
        classificador = joblib.load(classificador_path)

    return detector, classificador, extrair_features


detector_presenca, classificador_presenca, extrair_features_fn = carregar_detector_presenca()


@st.cache_resource
def carregar_assistente_chatbot():
    """Carrega o assistente conversacional (NLP + IA opcional)."""
    from chatbot.assistente_totem import AssistenteTotem
    return AssistenteTotem()


assistente_chatbot = carregar_assistente_chatbot()


@st.cache_resource
def carregar_motor_recomendacao():
    """Carrega o motor de recomendação centralizado."""
    from ml.modelo_recomendacao import MotorRecomendacao
    return MotorRecomendacao()


motor_recomendacao = carregar_motor_recomendacao()


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def obter_faixa_horaria() -> str:
    """Retorna a faixa horária atual baseada no relógio."""
    hora = datetime.now().hour
    if 9 <= hora < 11:
        return "manha"
    elif 11 <= hora < 14:
        return "almoco"
    elif 14 <= hora < 18:
        return "tarde"
    else:
        return "noite"


def obter_dia_semana() -> str:
    """Retorna o dia da semana atual em português."""
    dias = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
    return dias[datetime.now().weekday()]


def recomendar_loja(faixa_etaria: str, categoria: str, preferencia: str) -> dict:
    """
    Sistema de recomendação: delega para o MotorRecomendacao centralizado
    em ml/modelo_recomendacao.py.

    Retorna dict no formato esperado pela interface do totem.
    """
    resultado_motor = motor_recomendacao.recomendar(
        faixa_etaria=faixa_etaria,
        categoria=categoria,
        preferencia=preferencia,
        tempo_interacao=12,
    )

    if not resultado_motor["loja"]:
        lojas_fallback = LOJAS_POR_COMBINACAO.get((categoria, preferencia), ["Loja Genérica"])
        return {
            "loja": lojas_fallback[0],
            "probabilidade": 0.5,
            "alternativas": lojas_fallback[1:] if len(lojas_fallback) > 1 else [],
            "todas_probs": [],
        }

    ranking = resultado_motor["ranking"]
    return {
        "loja": resultado_motor["loja"],
        "probabilidade": resultado_motor["probabilidade"],
        "alternativas": [r["loja"] for r in ranking[1:]],
        "todas_probs": [
            {"loja": r["loja"], "probabilidade": r["probabilidade"]}
            for r in ranking
        ],
        "ranking_completo": ranking,
        "contexto": resultado_motor["contexto"],
    }


def obter_ou_criar_sessao(faixa_etaria: str) -> int:
    """
    Retorna o sessao_id atual. Cria uma nova sessão somente se não existir
    uma sessão ativa no session_state (evita criar sessão a cada clique).
    """
    if "sessao_id_banco" in st.session_state and st.session_state.sessao_id_banco is not None:
        return st.session_state.sessao_id_banco

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    agora = datetime.now()
    dia_semana = obter_dia_semana()
    faixa_horaria = obter_faixa_horaria()

    cursor.execute("""
        INSERT INTO sessoes (inicio_sessao, fim_sessao, faixa_etaria, dia_semana, faixa_horaria)
        VALUES (?, ?, ?, ?, ?)
    """, (
        agora.strftime("%Y-%m-%d %H:%M:%S"),
        agora.strftime("%Y-%m-%d %H:%M:%S"),
        faixa_etaria, dia_semana, faixa_horaria,
    ))
    sessao_id = cursor.lastrowid
    conn.commit()
    conn.close()

    st.session_state.sessao_id_banco = sessao_id
    return sessao_id


def registrar_interacao(
    faixa_etaria: str,
    categoria: str,
    preferencia: str,
    loja: str,
    aceitou: int,
    motivo_rejeicao: str = None,
    tempo_interacao: int = 10,
) -> None:
    """
    Registra a interação no banco SQLite.
    Reutiliza a sessão existente em vez de criar uma nova a cada clique.
    """
    sessao_id = obter_ou_criar_sessao(faixa_etaria)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    agora = datetime.now()

    # Atualizar fim_sessao da sessão ativa
    cursor.execute("""
        UPDATE sessoes SET fim_sessao = ? WHERE id = ?
    """, (agora.strftime("%Y-%m-%d %H:%M:%S"), sessao_id))

    # Criar interação vinculada à sessão existente
    cursor.execute("""
        INSERT INTO interacoes (sessao_id, timestamp, categoria, preferencia, tempo_interacao)
        VALUES (?, ?, ?, ?, ?)
    """, (
        sessao_id,
        agora.strftime("%Y-%m-%d %H:%M:%S"),
        categoria, preferencia, tempo_interacao,
    ))
    interacao_id = cursor.lastrowid

    # Criar recomendação
    cursor.execute("""
        INSERT INTO recomendacoes (interacao_id, loja_recomendada, aceitou, motivo_rejeicao)
        VALUES (?, ?, ?, ?)
    """, (
        interacao_id, loja, aceitou,
        motivo_rejeicao.lower() if motivo_rejeicao else None,
    ))

    conn.commit()
    conn.close()


# =============================================================================
# INICIALIZAÇÃO DO SESSION STATE
# =============================================================================

if "etapa" not in st.session_state:
    st.session_state.etapa = "deteccao_presenca"
if "faixa_etaria" not in st.session_state:
    st.session_state.faixa_etaria = None
if "categoria" not in st.session_state:
    st.session_state.categoria = None
if "preferencia" not in st.session_state:
    st.session_state.preferencia = None
if "recomendacao" not in st.session_state:
    st.session_state.recomendacao = None
if "historico" not in st.session_state:
    st.session_state.historico = []
if "inicio_interacao" not in st.session_state:
    st.session_state.inicio_interacao = None
if "presenca_detectada" not in st.session_state:
    st.session_state.presenca_detectada = False
if "sessao_id_banco" not in st.session_state:
    st.session_state.sessao_id_banco = None


def resetar_fluxo():
    """Reseta o estado para uma nova interação."""
    st.session_state.etapa = "deteccao_presenca"
    st.session_state.faixa_etaria = None
    st.session_state.categoria = None
    st.session_state.preferencia = None
    st.session_state.recomendacao = None
    st.session_state.inicio_interacao = None
    st.session_state.presenca_detectada = False
    st.session_state.chat_historico = []
    st.session_state.chat_modo = False
    st.session_state.sessao_id_banco = None  # próxima interação cria sessão nova
    st.session_state.visao_resultado = None
    st.session_state.visao_img_rgb = None


# =============================================================================
# TELA 0: DETECÇÃO DE PRESENÇA (Visão Computacional)
# =============================================================================

if st.session_state.etapa == "deteccao_presenca":

    st.markdown('<p class="totem-titulo">🏬 Totem Inteligente FlexMedia</p>', unsafe_allow_html=True)
    st.markdown('<p class="totem-subtitulo">Aproxime-se do totem para começar</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📷 Detecção de Presença")
    st.markdown("*O totem usa visão computacional para detectar visitantes.*")

    # Inicializar estado de detecção
    if "visao_resultado" not in st.session_state:
        st.session_state.visao_resultado = None
    if "visao_img_rgb" not in st.session_state:
        st.session_state.visao_img_rgb = None

    # Opção 1: Upload de imagem (simulação de câmera)
    foto = st.file_uploader(
        "Simular câmera do totem (envie uma foto)",
        type=["jpg", "jpeg", "png"],
        key="camera_upload",
    )

    # Opção 2: Usar imagem de exemplo do dataset
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        usar_exemplo_com = st.button("📸 Simular: visitante presente", use_container_width=True)
    with col_ex2:
        usar_exemplo_sem = st.button("📸 Simular: ninguém presente", use_container_width=True)

    # Processar imagem se algum botão foi clicado NESTE rerun
    imagem_para_analisar = None

    if foto is not None:
        file_bytes = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        imagem_para_analisar = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif usar_exemplo_com:
        exemplo_path = os.path.join(BASE_DIR, "vision", "dataset", "com_presenca", "presenca_01.jpg")
        if os.path.exists(exemplo_path):
            imagem_para_analisar = cv2.imread(exemplo_path)
    elif usar_exemplo_sem:
        exemplo_path = os.path.join(BASE_DIR, "vision", "dataset", "sem_presenca", "vazio_01.jpg")
        if os.path.exists(exemplo_path):
            imagem_para_analisar = cv2.imread(exemplo_path)

    # Se temos imagem nova para analisar, rodar detecção e salvar no session_state
    if imagem_para_analisar is not None:
        resultado_visao = detector_presenca.detectar(imagem_para_analisar)

        confianca_ml = None
        if classificador_presenca is not None:
            features = extrair_features_fn(imagem_para_analisar)
            pred_ml = classificador_presenca.predict([features])[0]
            if hasattr(classificador_presenca, "predict_proba"):
                confianca_ml = classificador_presenca.predict_proba([features])[0][1]
            else:
                confianca_ml = float(pred_ml)

        img_anotada = detector_presenca.anotar_imagem(imagem_para_analisar, resultado_visao)
        img_rgb = cv2.cvtColor(img_anotada, cv2.COLOR_BGR2RGB)

        # Salvar resultado no session_state para persistir entre reruns
        st.session_state.visao_resultado = {
            "presenca": resultado_visao["presenca"],
            "confianca_hog": resultado_visao["confianca_max"],
            "confianca_ml": confianca_ml,
        }
        st.session_state.visao_img_rgb = img_rgb

    # Mostrar resultado salvo (persiste entre reruns)
    if st.session_state.visao_img_rgb is not None:
        st.image(st.session_state.visao_img_rgb, caption="Análise de Visão Computacional", use_container_width=True)

    if st.session_state.visao_resultado is not None:
        res = st.session_state.visao_resultado
        if res["presenca"]:
            st.success(
                f"✅ **Visitante detectado!**  \n"
                f"Confiança HOG: {res['confianca_hog']:.0%}"
                + (f" | Confiança ML: {res['confianca_ml']:.0%}" if res["confianca_ml"] is not None else "")
            )
            st.session_state.presenca_detectada = True

            if st.button("▶️ Iniciar interação", use_container_width=True, type="primary"):
                st.session_state.etapa = "boas_vindas"
                st.session_state.inicio_interacao = datetime.now()
                st.session_state.visao_resultado = None
                st.session_state.visao_img_rgb = None
                st.rerun()
        else:
            st.warning(
                "⏳ **Nenhuma presença detectada.**  \n"
                "Aproxime-se do totem ou tente outra imagem."
            )

    # Botão para pular detecção (modo manual)
    st.markdown("---")
    st.markdown("*Ou entre diretamente:*")
    if st.button("⏭️ Pular detecção (modo manual)", use_container_width=True):
        st.session_state.etapa = "boas_vindas"
        st.session_state.inicio_interacao = datetime.now()
        st.rerun()


# =============================================================================
# TELA 1: BOAS-VINDAS + FAIXA ETÁRIA
# =============================================================================

elif st.session_state.etapa == "boas_vindas":

    st.markdown('<p class="totem-titulo">🏬 Totem Inteligente FlexMedia</p>', unsafe_allow_html=True)
    st.markdown('<p class="totem-subtitulo">Bem-vindo ao shopping! Deixe-nos ajudar você.</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Qual é a sua faixa etária?")
    st.markdown("*Isso nos ajuda a personalizar sua experiência.*")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🧑 Jovem\n(18-29)", use_container_width=True):
            st.session_state.faixa_etaria = "jovem"
            st.session_state.etapa = "selecao_categoria"
            st.session_state.inicio_interacao = datetime.now()
            st.rerun()

    with col2:
        if st.button("👨 Adulto\n(30-59)", use_container_width=True):
            st.session_state.faixa_etaria = "adulto"
            st.session_state.etapa = "selecao_categoria"
            st.session_state.inicio_interacao = datetime.now()
            st.rerun()

    with col3:
        if st.button("👴 Idoso\n(60+)", use_container_width=True):
            st.session_state.faixa_etaria = "idoso"
            st.session_state.etapa = "selecao_categoria"
            st.session_state.inicio_interacao = datetime.now()
            st.rerun()

    # Informações contextuais
    st.markdown("---")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"📅 **{obter_dia_semana().capitalize()}**")
    with col_info2:
        st.markdown(f"🕐 **{obter_faixa_horaria().replace('almoco', 'almoço').capitalize()}**")


# =============================================================================
# TELA 2: SELEÇÃO DE CATEGORIA
# =============================================================================

elif st.session_state.etapa == "selecao_categoria":

    st.markdown('<p class="totem-titulo">🏬 O que você gostaria de fazer?</p>', unsafe_allow_html=True)
    st.markdown('<p class="totem-subtitulo">Escolha uma das opções abaixo</p>', unsafe_allow_html=True)

    st.markdown("---")

    # OPÇÃO 1: Conversa com o totem (chatbot)
    st.markdown("#### 💬 Prefere conversar comigo?")
    st.markdown("*Me diga o que você procura em suas próprias palavras:*")

    if "chat_historico" not in st.session_state:
        st.session_state.chat_historico = []

    # Mostrar histórico de chat
    for msg in st.session_state.chat_historico:
        if msg["role"] == "user":
            st.markdown(f"**🧑 Você:** {msg['texto']}")
        else:
            st.markdown(f"**🤖 Totem:** {msg['texto']}")

    mensagem_usuario = st.chat_input("Ex: 'to com fome, quero algo doce'")

    if mensagem_usuario:
        # Adicionar ao histórico
        st.session_state.chat_historico.append({"role": "user", "texto": mensagem_usuario})

        # Interpretar
        resultado_chat = assistente_chatbot.interpretar(mensagem_usuario)

        # Adicionar resposta ao histórico
        st.session_state.chat_historico.append({"role": "bot", "texto": resultado_chat["resposta"]})

        # Se identificou categoria e preferência, avança direto para recomendação
        if resultado_chat["categoria"] and resultado_chat["preferencia"]:
            st.session_state.categoria = resultado_chat["categoria"]
            st.session_state.preferencia = resultado_chat["preferencia"]
            st.session_state.chat_modo = True
            st.session_state.chat_confianca = {
                "categoria": resultado_chat["confianca_categoria"],
                "preferencia": resultado_chat["confianca_preferencia"],
                "modo": resultado_chat["modo"],
            }
            st.session_state.etapa = "recomendacao"
            st.rerun()
        # Se identificou só a categoria, mostra preferências
        elif resultado_chat["categoria"]:
            st.session_state.categoria = resultado_chat["categoria"]
            st.session_state.etapa = "selecao_preferencia"
            st.rerun()
        else:
            st.rerun()

    st.markdown("---")
    st.markdown("#### Ou escolha uma categoria diretamente:")

    col1, col2 = st.columns(2)

    for i, (cat_key, cat_info) in enumerate(CATEGORIAS.items()):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(
                f"{cat_info['emoji']}  {cat_info['label']}\n{cat_info['desc']}",
                use_container_width=True,
                key=f"cat_{cat_key}",
            ):
                st.session_state.categoria = cat_key
                st.session_state.etapa = "selecao_preferencia"
                st.rerun()

    st.markdown("---")
    if st.button("← Voltar", key="voltar_cat"):
        resetar_fluxo()
        st.rerun()


# =============================================================================
# TELA 3: SELEÇÃO DE PREFERÊNCIA
# =============================================================================

elif st.session_state.etapa == "selecao_preferencia":

    cat = st.session_state.categoria
    cat_info = CATEGORIAS[cat]

    st.markdown(
        f'<p class="totem-titulo">{cat_info["emoji"]} {cat_info["label"]}</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="totem-subtitulo">Que tipo especificamente?</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    prefs = PREFERENCIAS_POR_CATEGORIA[cat]

    # Layout em 2 ou 3 colunas dependendo do número de preferências
    num_cols = 2 if len(prefs) <= 4 else 3
    cols = st.columns(num_cols)

    for i, (pref_key, pref_label) in enumerate(prefs.items()):
        col = cols[i % num_cols]
        with col:
            if st.button(pref_label, use_container_width=True, key=f"pref_{pref_key}"):
                st.session_state.preferencia = pref_key
                st.session_state.etapa = "recomendacao"
                st.rerun()

    st.markdown("---")
    if st.button("← Voltar", key="voltar_pref"):
        st.session_state.etapa = "selecao_categoria"
        st.rerun()


# =============================================================================
# TELA 4: RECOMENDAÇÃO
# =============================================================================

elif st.session_state.etapa == "recomendacao":

    # Gerar recomendação se ainda não existe
    if st.session_state.recomendacao is None:
        st.session_state.recomendacao = recomendar_loja(
            st.session_state.faixa_etaria,
            st.session_state.categoria,
            st.session_state.preferencia,
        )

    rec = st.session_state.recomendacao
    loja = rec["loja"]
    prob = rec["probabilidade"]
    emoji_loja = EMOJIS_LOJAS.get(loja, "📍")

    st.markdown('<p class="totem-titulo">✨ Nossa recomendação</p>', unsafe_allow_html=True)

    # Box da recomendação
    st.markdown(f"""
    <div class="recomendacao-box">
        <span style="font-size: 3rem;">{emoji_loja}</span>
        <p class="recomendacao-loja">{loja}</p>
        <p class="recomendacao-prob">Compatibilidade com seu perfil: {prob:.0%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Contexto da recomendação
    cat_info = CATEGORIAS[st.session_state.categoria]
    pref_label = PREFERENCIAS_POR_CATEGORIA[st.session_state.categoria][st.session_state.preferencia]
    st.markdown(
        f"Baseado na sua escolha: **{cat_info['label']}** → **{pref_label}** | "
        f"Perfil: **{st.session_state.faixa_etaria.capitalize()}** | "
        f"Horário: **{obter_faixa_horaria().replace('almoco', 'almoço').capitalize()}**"
    )

    # Mostrar confiança do NLP quando veio do chatbot
    chat_conf = st.session_state.get("chat_confianca")
    if chat_conf and st.session_state.get("chat_modo"):
        modo_label = "IA Generativa (Gemini)" if chat_conf.get("modo") == "gemini" else "NLP Local (TF-IDF + keywords)"
        st.caption(
            f"💬 Intenção detectada por {modo_label} — "
            f"confiança categoria: {chat_conf['categoria']:.0%}, "
            f"confiança preferência: {chat_conf['preferencia']:.0%}"
        )

    # Explicação expandível (transparência do modelo)
    if rec.get("ranking_completo"):
        with st.expander("🔍 Por que essa recomendação?"):
            st.markdown("O sistema avaliou **todas as lojas candidatas** e escolheu "
                        "a com maior probabilidade de aceitação para o seu perfil:")
            st.markdown("")
            for i, r in enumerate(rec["ranking_completo"], 1):
                destaque = "**" if i == 1 else ""
                emoji_r = EMOJIS_LOJAS.get(r["loja"], "📍")
                st.markdown(
                    f"{i}. {destaque}{emoji_r} {r['loja']} — "
                    f"{r['probabilidade']:.0%} de compatibilidade{destaque}"
                )
            st.caption(
                "O modelo (Logistic Regression, F1=0.80) considera faixa etária, "
                "dia da semana, horário, categoria, preferência e tempo de interação."
            )

    st.markdown("---")

    # Botões de ação
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Aceitar", use_container_width=True, type="primary"):
            # Calcular tempo de interação
            tempo = 10
            if st.session_state.inicio_interacao:
                delta = (datetime.now() - st.session_state.inicio_interacao).total_seconds()
                tempo = max(2, min(30, int(delta)))

            registrar_interacao(
                faixa_etaria=st.session_state.faixa_etaria,
                categoria=st.session_state.categoria,
                preferencia=st.session_state.preferencia,
                loja=loja,
                aceitou=1,
                tempo_interacao=tempo,
            )
            st.session_state.historico.append({
                "loja": loja, "aceitou": True,
                "categoria": st.session_state.categoria,
                "preferencia": st.session_state.preferencia,
            })
            st.session_state.etapa = "agradecimento"
            st.rerun()

    with col2:
        if st.button("🔄 Ver outra", use_container_width=True):
            # Recomenda a próxima alternativa
            alternativas = rec.get("alternativas", [])
            if alternativas:
                st.session_state.recomendacao = {
                    "loja": alternativas[0],
                    "probabilidade": next(
                        (r["probabilidade"] for r in rec.get("todas_probs", [])
                         if r["loja"] == alternativas[0]),
                        0.5,
                    ),
                    "alternativas": alternativas[1:],
                    "todas_probs": rec.get("todas_probs", []),
                }
            else:
                st.session_state.recomendacao = None  # recalcula
            st.rerun()

    with col3:
        if st.button("❌ Recusar", use_container_width=True):
            st.session_state.etapa = "rejeicao"
            st.rerun()


# =============================================================================
# TELA 4b: MOTIVO DE REJEIÇÃO
# =============================================================================

elif st.session_state.etapa == "rejeicao":

    st.markdown('<p class="totem-titulo">😕 Sem problema!</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="totem-subtitulo">Pode nos dizer o motivo? Isso nos ajuda a melhorar.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    from config import MOTIVOS_REJEICAO as MOTIVOS_KEYS, MOTIVOS_REJEICAO_LABELS

    for motivo_key in MOTIVOS_KEYS:
        motivo_label = MOTIVOS_REJEICAO_LABELS.get(motivo_key, motivo_key)
        if st.button(motivo_label, use_container_width=True, key=f"motivo_{motivo_key}"):
            # Calcular tempo de interação
            tempo = 10
            if st.session_state.inicio_interacao:
                delta = (datetime.now() - st.session_state.inicio_interacao).total_seconds()
                tempo = max(2, min(30, int(delta)))

            rec = st.session_state.recomendacao
            registrar_interacao(
                faixa_etaria=st.session_state.faixa_etaria,
                categoria=st.session_state.categoria,
                preferencia=st.session_state.preferencia,
                loja=rec["loja"],
                aceitou=0,
                motivo_rejeicao=motivo_key,  # salva a key normalizada, não o label
                tempo_interacao=tempo,
            )
            st.session_state.historico.append({
                "loja": rec["loja"], "aceitou": False,
                "categoria": st.session_state.categoria,
                "preferencia": st.session_state.preferencia,
                "motivo": motivo_label,
            })
            st.session_state.etapa = "agradecimento"
            st.rerun()

    st.markdown("---")
    if st.button("← Voltar", key="voltar_rej"):
        st.session_state.etapa = "recomendacao"
        st.rerun()


# =============================================================================
# TELA 5: AGRADECIMENTO
# =============================================================================

elif st.session_state.etapa == "agradecimento":

    ultimo = st.session_state.historico[-1] if st.session_state.historico else {}

    if ultimo.get("aceitou"):
        emoji_loja = EMOJIS_LOJAS.get(ultimo["loja"], "📍")
        st.markdown(f"""
        <div class="recomendacao-box">
            <span style="font-size: 3rem;">🎉</span>
            <p class="recomendacao-loja">Ótima escolha!</p>
            <p class="recomendacao-prob">
                Aproveite sua visita ao {emoji_loja} <strong>{ultimo['loja']}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="agradecimento-box">
            <span style="font-size: 3rem;">🙏</span>
            <p class="recomendacao-loja" style="color: #818cf8;">Obrigado pelo feedback!</p>
            <p class="recomendacao-prob">
                Sua resposta nos ajuda a melhorar as recomendações.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Nova consulta", use_container_width=True, type="primary"):
            resetar_fluxo()
            st.rerun()

    with col2:
        if st.button("👋 Encerrar", use_container_width=True):
            st.session_state.etapa = "encerramento"
            st.rerun()

    # Mostrar histórico da sessão se houver mais de 1 interação
    if len(st.session_state.historico) > 1:
        st.markdown("---")
        st.markdown("#### 📋 Suas interações nesta sessão")
        for i, h in enumerate(st.session_state.historico, 1):
            status = "✅" if h["aceitou"] else "❌"
            emoji = EMOJIS_LOJAS.get(h["loja"], "📍")
            st.markdown(f"{i}. {status} {emoji} **{h['loja']}** ({h['categoria']} → {h['preferencia']})")


# =============================================================================
# TELA 6: ENCERRAMENTO
# =============================================================================

elif st.session_state.etapa == "encerramento":

    st.markdown(f"""
    <div class="agradecimento-box">
        <span style="font-size: 4rem;">👋</span>
        <p class="recomendacao-loja" style="color: #818cf8;">Até a próxima!</p>
        <p class="recomendacao-prob">
            Obrigado por usar o Totem Inteligente FlexMedia.<br>
            Bom passeio pelo shopping!
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Resumo da sessão
    if st.session_state.historico:
        total = len(st.session_state.historico)
        aceitas = sum(1 for h in st.session_state.historico if h["aceitou"])
        st.markdown("---")
        st.markdown(f"**Resumo da sessão:** {total} interações, {aceitas} aceitas")

    st.markdown("---")
    if st.button("🔄 Iniciar nova sessão", use_container_width=True, type="primary"):
        st.session_state.historico = []
        resetar_fluxo()
        st.rerun()
