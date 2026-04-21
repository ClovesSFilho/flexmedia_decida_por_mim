"""
Dashboard Interativo – Totem Inteligente FlexMedia.

Painel de controle completo para acompanhamento de métricas
do totem, com filtros interativos e visualizações avançadas.

Executar com: streamlit run dashboard/app_streamlit.py
"""

import sqlite3
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Totem FlexMedia – Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS customizado para visual limpo
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetric"] label { font-size: 0.85rem; }
    h1 { color: #1e293b; }
    h2 { color: #334155; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
    h3 { color: #475569; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CARREGAMENTO DOS DADOS
# =============================================================================

@st.cache_data
def carregar_dados() -> pd.DataFrame:
    """Carrega dados com JOINs entre as 3 tabelas."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "interacoes.db")

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT
            s.id AS sessao_id,
            s.inicio_sessao,
            s.faixa_etaria,
            s.dia_semana,
            s.faixa_horaria,
            i.id AS interacao_id,
            i.timestamp,
            i.categoria,
            i.preferencia,
            i.tempo_interacao,
            r.loja_recomendada,
            r.aceitou,
            r.motivo_rejeicao
        FROM interacoes i
        JOIN sessoes s ON i.sessao_id = s.id
        JOIN recomendacoes r ON r.interacao_id = i.id
    """, conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["data"] = df["timestamp"].dt.date
    df["hora"] = df["timestamp"].dt.hour

    return df


@st.cache_data
def carregar_metricas_ml() -> dict | None:
    """Carrega métricas de ML do arquivo JSON gerado pelo modelo."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metricas_path = os.path.join(base_dir, "ml", "metricas.json")

    if os.path.exists(metricas_path):
        with open(metricas_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


df_original = carregar_dados()
metricas_ml = carregar_metricas_ml()


# =============================================================================
# SIDEBAR – FILTROS
# =============================================================================

st.sidebar.markdown("# 📊")  # ícone local (sem dependência de internet)
st.sidebar.title("Filtros")
st.sidebar.markdown("---")

# filtro por categoria
categorias_disponiveis = sorted(df_original["categoria"].unique())
categorias_selecionadas = st.sidebar.multiselect(
    "Categoria",
    options=categorias_disponiveis,
    default=categorias_disponiveis,
)

# filtro por faixa etária
faixas_disponiveis = sorted(df_original["faixa_etaria"].unique())
faixas_selecionadas = st.sidebar.multiselect(
    "Faixa Etária",
    options=faixas_disponiveis,
    default=faixas_disponiveis,
)

# filtro por faixa horária
horarios_disponiveis = ["manha", "almoco", "tarde", "noite"]
horarios_selecionados = st.sidebar.multiselect(
    "Faixa Horária",
    options=horarios_disponiveis,
    default=horarios_disponiveis,
)

# filtro por dia da semana
dias_disponiveis = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
dias_selecionados = st.sidebar.multiselect(
    "Dia da Semana",
    options=dias_disponiveis,
    default=dias_disponiveis,
)

# filtro por período
st.sidebar.markdown("---")
datas = sorted(df_original["data"].unique())
data_inicio = st.sidebar.date_input("Data início", value=min(datas), min_value=min(datas), max_value=max(datas))
data_fim = st.sidebar.date_input("Data fim", value=max(datas), min_value=min(datas), max_value=max(datas))

# aplicar filtros
df = df_original[
    (df_original["categoria"].isin(categorias_selecionadas)) &
    (df_original["faixa_etaria"].isin(faixas_selecionadas)) &
    (df_original["faixa_horaria"].isin(horarios_selecionados)) &
    (df_original["dia_semana"].isin(dias_selecionados)) &
    (df_original["data"] >= data_inicio) &
    (df_original["data"] <= data_fim)
]

st.sidebar.markdown("---")
st.sidebar.metric("Registros filtrados", f"{len(df):,}")
st.sidebar.metric("Total no banco", f"{len(df_original):,}")


# =============================================================================
# HEADER
# =============================================================================

st.title("📊 Totem Inteligente FlexMedia")
st.markdown("**Dashboard de Monitoramento de Interações e Performance**")
st.markdown("---")

if len(df) == 0:
    st.warning("Nenhum dado encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()


# =============================================================================
# KPIs PRINCIPAIS
# =============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

total_interacoes = len(df)
total_sessoes = df["sessao_id"].nunique()
taxa_aceitacao = df["aceitou"].mean() * 100
tempo_medio = df["tempo_interacao"].mean()
categoria_top = df.groupby("categoria")["aceitou"].mean().idxmax()

col1.metric("Total de Interações", f"{total_interacoes:,}")
col2.metric("Sessões Únicas", f"{total_sessoes:,}")
col3.metric("Taxa de Aceitação", f"{taxa_aceitacao:.1f}%")
col4.metric("Tempo Médio", f"{tempo_medio:.1f}s")
col5.metric("Melhor Categoria", categoria_top.capitalize())

st.markdown("---")


# =============================================================================
# GRÁFICOS – LINHA 1: TEMPORAL + CATEGORIAS
# =============================================================================

st.header("📈 Visão Geral")

col_left, col_right = st.columns(2)

# gráfico 1: interações ao longo do tempo
with col_left:
    st.subheader("Interações ao Longo do Tempo")

    interacoes_por_dia = df.groupby("data").size().reset_index(name="total")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(interacoes_por_dia["data"], interacoes_por_dia["total"],
                    alpha=0.3, color="#2563EB")
    ax.plot(interacoes_por_dia["data"], interacoes_por_dia["total"],
            color="#2563EB", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Data")
    ax.set_ylabel("Nº de Interações")
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# gráfico 2: distribuição por categoria
with col_right:
    st.subheader("Distribuição por Categoria")

    cat_counts = df["categoria"].value_counts()
    cores_cat = ["#2563EB", "#10B981", "#F59E0B", "#EF4444"]

    fig, ax = plt.subplots(figsize=(10, 5))
    wedges, texts, autotexts = ax.pie(
        cat_counts.values, labels=cat_counts.index,
        autopct="%1.1f%%", colors=cores_cat[:len(cat_counts)],
        startangle=90, textprops={"fontsize": 12}
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")
    ax.set_title("")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# GRÁFICOS – LINHA 2: ACEITAÇÃO SEGMENTADA
# =============================================================================

st.markdown("---")
st.header("🎯 Análise de Aceitação")

col_a, col_b, col_c = st.columns(3)

# aceitação por categoria
with col_a:
    st.subheader("Por Categoria")
    taxa_cat = df.groupby("categoria")["aceitou"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    barras = ax.bar(taxa_cat.index, taxa_cat.values, color=cores_cat[:len(taxa_cat)],
                    edgecolor="white", linewidth=1.5)
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1,
                f"{barra.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Taxa de Aceitação (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# aceitação por faixa etária
with col_b:
    st.subheader("Por Faixa Etária")
    taxa_idade = df.groupby("faixa_etaria")["aceitou"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5))
    cores_idade = ["#2563EB", "#10B981", "#F59E0B"]
    barras = ax.bar(taxa_idade.index, taxa_idade.values, color=cores_idade[:len(taxa_idade)],
                    edgecolor="white", linewidth=1.5)
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1,
                f"{barra.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Taxa de Aceitação (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# aceitação por faixa horária
with col_c:
    st.subheader("Por Faixa Horária")
    taxa_hora = df.groupby("faixa_horaria")["aceitou"].mean() * 100
    taxa_hora = taxa_hora.reindex(["manha", "almoco", "tarde", "noite"])

    fig, ax = plt.subplots(figsize=(8, 5))
    cores_hora = ["#F59E0B", "#EF4444", "#2563EB", "#6366F1"]
    barras = ax.bar(taxa_hora.index, taxa_hora.values, color=cores_hora[:len(taxa_hora)],
                    edgecolor="white", linewidth=1.5)
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1,
                f"{barra.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Taxa de Aceitação (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# GRÁFICOS – LINHA 3: HEATMAP + TEMPO
# =============================================================================

st.markdown("---")
st.header("🕐 Análise Temporal")

col_heat, col_tempo = st.columns(2)

# heatmap dia × horário
with col_heat:
    st.subheader("Heatmap: Dia × Horário")

    ordem_dias = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
    ordem_horas = ["manha", "almoco", "tarde", "noite"]

    pivot = df.groupby(["dia_semana", "faixa_horaria"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=ordem_dias, columns=ordem_horas, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Nº Interações"},
                annot_kws={"size": 13, "fontweight": "bold"})
    ax.set_ylabel("Dia da Semana")
    ax.set_xlabel("Faixa Horária")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# tempo de interação vs aceitação
with col_tempo:
    st.subheader("Tempo de Interação × Aceitação")

    bins = [0, 5, 10, 15, 20, 30]
    labels = ["1-5s", "6-10s", "11-15s", "16-20s", "21s+"]
    df_tempo = df.copy()
    df_tempo["faixa_tempo"] = pd.cut(df_tempo["tempo_interacao"], bins=bins, labels=labels, right=True)

    taxa_tempo = df_tempo.groupby("faixa_tempo", observed=True)["aceitou"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    cores_tempo = ["#EF4444", "#F59E0B", "#10B981", "#2563EB", "#1D4ED8"]
    barras = ax.bar(taxa_tempo.index, taxa_tempo.values,
                    color=cores_tempo[:len(taxa_tempo)], edgecolor="white", linewidth=1.5)
    for barra in barras:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 1,
                f"{barra.get_height():.1f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Taxa de Aceitação (%)")
    ax.set_xlabel("Faixa de Tempo")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # df_tempo é cópia local, não precisa de cleanup


# =============================================================================
# GRÁFICOS – LINHA 4: LOJAS + REJEIÇÃO
# =============================================================================

st.markdown("---")
st.header("🏪 Performance das Lojas")

col_lojas, col_motivos = st.columns(2)

# top lojas por aceitação
with col_lojas:
    st.subheader("Top 10 Lojas – Taxa de Aceitação")

    top_lojas = df.groupby("loja_recomendada").agg(
        total=("aceitou", "count"),
        aceitas=("aceitou", "sum"),
    )
    top_lojas["taxa"] = (top_lojas["aceitas"] / top_lojas["total"] * 100)
    top_lojas = top_lojas[top_lojas["total"] >= 5]  # mínimo 5 recomendações
    top_lojas = top_lojas.sort_values("taxa", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 6))
    cores_loja = ["#10B981" if t >= 70 else "#F59E0B" if t >= 55
                  else "#EF4444" for t in top_lojas["taxa"]]
    ax.barh(top_lojas.index, top_lojas["taxa"], color=cores_loja)
    ax.set_xlabel("Taxa de Aceitação (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    for i, (_, row) in enumerate(top_lojas.iterrows()):
        ax.text(row["taxa"] + 1, i, f"{row['taxa']:.1f}% (n={int(row['total'])})",
                va="center", fontsize=10)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# motivos de rejeição
with col_motivos:
    st.subheader("Motivos de Rejeição")

    rejeicoes = df[df["aceitou"] == 0]["motivo_rejeicao"].value_counts()

    if len(rejeicoes) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        wedges, texts, autotexts = ax.pie(
            rejeicoes.values, labels=rejeicoes.index,
            autopct="%1.1f%%",
            colors=["#EF4444", "#F59E0B", "#6366F1", "#10B981", "#2563EB", "#94A3B8"],
            startangle=90, textprops={"fontsize": 11}
        )
        for autotext in autotexts:
            autotext.set_fontweight("bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Nenhuma rejeição nos dados filtrados.")


# =============================================================================
# SEÇÃO ML – RESULTADOS DO MODELO (DINÂMICO VIA JSON)
# =============================================================================

st.markdown("---")
st.header("🤖 Machine Learning – Resultados")

if metricas_ml:
    col_ml1, col_ml2 = st.columns(2)

    rf = metricas_ml["random_forest"]
    lr = metricas_ml["logistic_regression"]
    cv = metricas_ml["validacao_cruzada"]
    modelo_sel = metricas_ml["modelo_selecionado"]

    with col_ml1:
        st.subheader("Métricas dos Modelos")

        # tabela comparativa
        dados_tabela = {
            "Métrica": ["Acurácia", "Precisão", "Recall", "F1-Score"],
            "Random Forest": [rf["acuracia"], rf["precisao"], rf["recall"], rf["f1"]],
            "Logistic Regression": [lr["acuracia"], lr["precisao"], lr["recall"], lr["f1"]],
        }
        df_metricas = pd.DataFrame(dados_tabela)
        st.dataframe(df_metricas, hide_index=True, use_container_width=True)

        st.markdown(f"**Modelo selecionado:** {modelo_sel} (melhor F1-Score)")
        st.markdown(
            f"**Validação Cruzada (5-fold):** Média F1 = {cv['media_f1']}, "
            f"Desvio = {cv['desvio_f1']}"
        )
        st.markdown("→ Modelo estável com baixa variância entre folds.")

    with col_ml2:
        st.subheader("Importância das Features")

        features = metricas_ml["feature_importance"]

        st.markdown(
            f"O **tempo de interação** é a variável mais importante para "
            f"prever se o usuário aceitará a recomendação "
            f"({features[0]['importancia'] * 100:.1f}% de importância)."
        )

        st.markdown("**Ranking completo:**")
        for i, feat in enumerate(features, 1):
            st.markdown(
                f"{i}. `{feat['feature']}` → {feat['importancia'] * 100:.1f}%"
            )

        # carregar imagem do feature importance se existir
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fi_path = os.path.join(base_dir, "ml", "graficos", "01_feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)

else:
    st.warning(
        "Métricas de ML não encontradas. Execute `python ml/modelo_ml.py` "
        "para gerar o arquivo `ml/metricas.json`."
    )


# =============================================================================
# TABELA DE DADOS
# =============================================================================

st.markdown("---")
st.header("📋 Dados Detalhados")

with st.expander("Visualizar dados brutos (clique para expandir)"):
    colunas_exibir = [
        "timestamp", "categoria", "preferencia", "tempo_interacao",
        "loja_recomendada", "aceitou", "faixa_etaria", "dia_semana", "faixa_horaria"
    ]
    st.dataframe(
        df[colunas_exibir].sort_values("timestamp", ascending=False),
        use_container_width=True,
        height=400,
    )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Totem Inteligente FlexMedia – Dashboard de Monitoramento | "
    "Challenge FIAP 2025 | Dados simulados para fins acadêmicos"
    "</div>",
    unsafe_allow_html=True,
)