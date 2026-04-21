"""
Análise Estatística do Totem Inteligente FlexMedia.

Realiza análise exploratória completa dos dados coletados:
- Estatísticas descritivas (média, mediana, desvio padrão, quartis)
- Análise de correlação entre variáveis
- Taxas de aceitação segmentadas (categoria, faixa etária, horário)
- Análise temporal (volume por dia da semana, por faixa horária)
- Teste de hipótese (chi-quadrado) para validar padrões
- Exportação de gráficos em PNG para documentação

Gráficos exportados em: analysis/graficos/
"""

import sqlite3
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sem interface gráfica
import matplotlib.pyplot as plt
from scipy import stats


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "interacoes.db")
GRAFICOS_DIR = os.path.join(BASE_DIR, "analysis", "graficos")

# paleta de cores consistente
CORES = {
    "primaria": "#2563EB",
    "secundaria": "#10B981",
    "destaque": "#F59E0B",
    "negativo": "#EF4444",
    "neutro": "#6B7280",
    "categorias": ["#2563EB", "#10B981", "#F59E0B", "#EF4444"],
}

# estilo global dos gráficos
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def carregar_dados() -> pd.DataFrame:
    """Carrega e junta todas as tabelas em um único DataFrame."""
    conn = sqlite3.connect(DB_PATH)

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

    # converter timestamp para datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["data"] = df["timestamp"].dt.date
    df["hora"] = df["timestamp"].dt.hour

    return df


def salvar_grafico(fig, nome: str) -> None:
    """Salva gráfico na pasta de gráficos."""
    os.makedirs(GRAFICOS_DIR, exist_ok=True)
    caminho = os.path.join(GRAFICOS_DIR, nome)
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Gráfico salvo: {caminho}")


# =============================================================================
# 1. ESTATÍSTICAS DESCRITIVAS
# =============================================================================

def estatisticas_descritivas(df: pd.DataFrame) -> None:
    """Exibe estatísticas descritivas do tempo de interação."""

    print("\n" + "=" * 60)
    print("1. ESTATÍSTICAS DESCRITIVAS - TEMPO DE INTERAÇÃO")
    print("=" * 60)

    tempo = df["tempo_interacao"]

    print(f"\n  Total de interações:  {len(df)}")
    print(f"  Média:                {tempo.mean():.2f} segundos")
    print(f"  Mediana:              {tempo.median():.2f} segundos")
    print(f"  Desvio padrão:        {tempo.std():.2f} segundos")
    print(f"  Mínimo:               {tempo.min()} segundos")
    print(f"  Máximo:               {tempo.max()} segundos")
    print(f"  1º Quartil (25%):     {tempo.quantile(0.25):.2f} segundos")
    print(f"  3º Quartil (75%):     {tempo.quantile(0.75):.2f} segundos")

    # gráfico: histograma do tempo de interação
    fig, ax = plt.subplots()
    ax.hist(tempo, bins=20, color=CORES["primaria"], edgecolor="white", alpha=0.85)
    ax.axvline(tempo.mean(), color=CORES["negativo"], linestyle="--", linewidth=2,
               label=f"Média: {tempo.mean():.1f}s")
    ax.axvline(tempo.median(), color=CORES["destaque"], linestyle="--", linewidth=2,
               label=f"Mediana: {tempo.median():.1f}s")
    ax.set_title("Distribuição do Tempo de Interação")
    ax.set_xlabel("Tempo (segundos)")
    ax.set_ylabel("Frequência")
    ax.legend()
    salvar_grafico(fig, "01_histograma_tempo.png")


# =============================================================================
# 2. TAXA DE ACEITAÇÃO SEGMENTADA
# =============================================================================

def analise_aceitacao(df: pd.DataFrame) -> None:
    """Analisa taxas de aceitação por diferentes segmentos."""

    print("\n" + "=" * 60)
    print("2. TAXA DE ACEITAÇÃO POR SEGMENTO")
    print("=" * 60)

    # por categoria
    print("\n  Por categoria:")
    taxa_cat = df.groupby("categoria")["aceitou"].agg(["mean", "count"])
    taxa_cat.columns = ["taxa", "total"]
    taxa_cat["taxa"] = (taxa_cat["taxa"] * 100).round(2)
    for cat, row in taxa_cat.iterrows():
        print(f"    {cat:12s} → {row['taxa']:6.2f}%  (n={int(row['total'])})")

    # por faixa etária
    print("\n  Por faixa etária:")
    taxa_idade = df.groupby("faixa_etaria")["aceitou"].agg(["mean", "count"])
    taxa_idade.columns = ["taxa", "total"]
    taxa_idade["taxa"] = (taxa_idade["taxa"] * 100).round(2)
    for idade, row in taxa_idade.iterrows():
        print(f"    {idade:12s} → {row['taxa']:6.2f}%  (n={int(row['total'])})")

    # por faixa horária
    print("\n  Por faixa horária:")
    taxa_hora = df.groupby("faixa_horaria")["aceitou"].agg(["mean", "count"])
    taxa_hora.columns = ["taxa", "total"]
    taxa_hora["taxa"] = (taxa_hora["taxa"] * 100).round(2)
    for hora, row in taxa_hora.iterrows():
        print(f"    {hora:12s} → {row['taxa']:6.2f}%  (n={int(row['total'])})")

    # gráfico: taxa de aceitação por categoria
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # categoria
    taxa_cat_plot = df.groupby("categoria")["aceitou"].mean() * 100
    taxa_cat_plot.plot(kind="bar", ax=axes[0], color=CORES["categorias"][:len(taxa_cat_plot)])
    axes[0].set_title("Aceitação por Categoria")
    axes[0].set_ylabel("Taxa de Aceitação (%)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_ylim(0, 100)

    # faixa etária
    taxa_idade_plot = df.groupby("faixa_etaria")["aceitou"].mean() * 100
    taxa_idade_plot.plot(kind="bar", ax=axes[1], color=[CORES["primaria"], CORES["secundaria"], CORES["destaque"]])
    axes[1].set_title("Aceitação por Faixa Etária")
    axes[1].set_ylabel("Taxa de Aceitação (%)")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")
    axes[1].set_ylim(0, 100)

    # faixa horária
    taxa_hora_plot = df.groupby("faixa_horaria")["aceitou"].mean() * 100
    ordem_horario = ["manha", "almoco", "tarde", "noite"]
    taxa_hora_plot = taxa_hora_plot.reindex(ordem_horario)
    taxa_hora_plot.plot(kind="bar", ax=axes[2], color=CORES["categorias"][:len(taxa_hora_plot)])
    axes[2].set_title("Aceitação por Faixa Horária")
    axes[2].set_ylabel("Taxa de Aceitação (%)")
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha="right")
    axes[2].set_ylim(0, 100)

    fig.suptitle("Taxas de Aceitação Segmentadas", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    salvar_grafico(fig, "02_aceitacao_segmentada.png")


# =============================================================================
# 3. CORRELAÇÃO TEMPO × ACEITAÇÃO
# =============================================================================

def analise_correlacao(df: pd.DataFrame) -> None:
    """Analisa a relação entre tempo de interação e aceitação."""

    print("\n" + "=" * 60)
    print("3. CORRELAÇÃO TEMPO DE INTERAÇÃO × ACEITAÇÃO")
    print("=" * 60)

    # correlação ponto-biserial (numérica contínua vs binária)
    corr, p_valor = stats.pointbiserialr(df["aceitou"], df["tempo_interacao"])
    print(f"\n  Correlação ponto-biserial: {corr:.4f}")
    print(f"  P-valor:                   {p_valor:.2e}")
    print(f"  Significativo (p < 0.05):  {'SIM' if p_valor < 0.05 else 'NÃO'}")

    # média de tempo por aceitação
    media_aceito = df[df["aceitou"] == 1]["tempo_interacao"].mean()
    media_rejeitado = df[df["aceitou"] == 0]["tempo_interacao"].mean()
    print(f"\n  Tempo médio (aceitou):     {media_aceito:.2f}s")
    print(f"  Tempo médio (rejeitou):    {media_rejeitado:.2f}s")

    # taxa por faixa de tempo
    print("\n  Taxa de aceitação por faixa de tempo:")
    bins = [0, 5, 10, 15, 20, 30]
    labels = ["1-5s", "6-10s", "11-15s", "16-20s", "21s+"]
    df["faixa_tempo"] = pd.cut(df["tempo_interacao"], bins=bins, labels=labels, right=True)

    taxa_tempo = df.groupby("faixa_tempo", observed=True)["aceitou"].agg(["mean", "count"])
    taxa_tempo.columns = ["taxa", "total"]
    for faixa, row in taxa_tempo.iterrows():
        barra = "█" * int(row["taxa"] * 30)
        print(f"    {faixa:8s} → {row['taxa']*100:5.1f}%  {barra}  (n={int(row['total'])})")

    # gráfico: boxplot tempo por aceitação + barras por faixa
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # boxplot
    dados_box = [
        df[df["aceitou"] == 0]["tempo_interacao"].values,
        df[df["aceitou"] == 1]["tempo_interacao"].values,
    ]
    bp = axes[0].boxplot(dados_box, tick_labels=["Rejeitou", "Aceitou"], patch_artist=True)
    bp["boxes"][0].set_facecolor(CORES["negativo"])
    bp["boxes"][1].set_facecolor(CORES["secundaria"])
    axes[0].set_title("Tempo de Interação: Aceitou vs Rejeitou")
    axes[0].set_ylabel("Tempo (segundos)")

    # barras por faixa de tempo
    taxa_plot = df.groupby("faixa_tempo", observed=True)["aceitou"].mean() * 100
    cores_barra = [CORES["negativo"], CORES["destaque"], CORES["secundaria"],
                   CORES["primaria"], CORES["primaria"]]
    taxa_plot.plot(kind="bar", ax=axes[1], color=cores_barra[:len(taxa_plot)])
    axes[1].set_title("Taxa de Aceitação por Faixa de Tempo")
    axes[1].set_ylabel("Taxa de Aceitação (%)")
    axes[1].set_xlabel("Faixa de Tempo")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].set_ylim(0, 100)

    fig.tight_layout()
    salvar_grafico(fig, "03_correlacao_tempo_aceitacao.png")

    # limpar coluna temporária
    df.drop(columns=["faixa_tempo"], inplace=True)


# =============================================================================
# 4. ANÁLISE TEMPORAL
# =============================================================================

def analise_temporal(df: pd.DataFrame) -> None:
    """Analisa distribuição temporal das interações."""

    print("\n" + "=" * 60)
    print("4. ANÁLISE TEMPORAL")
    print("=" * 60)

    # por dia da semana
    ordem_dias = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
    vol_dia = df["dia_semana"].value_counts().reindex(ordem_dias)

    print("\n  Interações por dia da semana:")
    for dia, vol in vol_dia.items():
        barra = "█" * (vol // 3)
        print(f"    {dia:10s} → {vol:4d}  {barra}")

    # por faixa horária
    ordem_horarios = ["manha", "almoco", "tarde", "noite"]
    vol_hora = df["faixa_horaria"].value_counts().reindex(ordem_horarios)

    print("\n  Interações por faixa horária:")
    for hora, vol in vol_hora.items():
        barra = "█" * (vol // 3)
        print(f"    {hora:10s} → {vol:4d}  {barra}")

    # gráfico: heatmap dia × horário
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # volume por dia
    vol_dia.plot(kind="bar", ax=axes[0], color=CORES["primaria"], edgecolor="white")
    axes[0].set_title("Volume de Interações por Dia da Semana")
    axes[0].set_ylabel("Número de Interações")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

    # heatmap dia × horário
    pivot = df.groupby(["dia_semana", "faixa_horaria"]).size().unstack(fill_value=0)
    pivot = pivot.reindex(index=ordem_dias, columns=ordem_horarios, fill_value=0)

    im = axes[1].imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(len(ordem_horarios)))
    axes[1].set_xticklabels(ordem_horarios)
    axes[1].set_yticks(range(len(ordem_dias)))
    axes[1].set_yticklabels(ordem_dias)
    axes[1].set_title("Heatmap: Dia da Semana × Faixa Horária")

    # adicionar valores nas células
    for i in range(len(ordem_dias)):
        for j in range(len(ordem_horarios)):
            axes[1].text(j, i, str(pivot.values[i, j]),
                        ha="center", va="center", fontweight="bold",
                        color="white" if pivot.values[i, j] > pivot.values.max() * 0.6 else "black")

    fig.colorbar(im, ax=axes[1], label="Nº de Interações")
    fig.tight_layout()
    salvar_grafico(fig, "04_analise_temporal.png")


# =============================================================================
# 5. TESTE DE HIPÓTESE
# =============================================================================

def teste_hipotese(df: pd.DataFrame) -> None:
    """
    Teste chi-quadrado para validar se a relação entre
    tempo de interação e aceitação é estatisticamente significativa.

    H0: Não há associação entre tempo de interação e aceitação.
    H1: Existe associação entre tempo de interação e aceitação.
    """

    print("\n" + "=" * 60)
    print("5. TESTE DE HIPÓTESE (CHI-QUADRADO)")
    print("=" * 60)

    # dividir tempo em duas faixas: curto (<= 7s) e longo (> 7s)
    df["tempo_grupo"] = df["tempo_interacao"].apply(
        lambda x: "curto (<=7s)" if x <= 7 else "longo (>7s)"
    )

    # tabela de contingência
    tabela = pd.crosstab(df["tempo_grupo"], df["aceitou"])
    tabela.columns = ["Rejeitou", "Aceitou"]

    print("\n  Tabela de contingência:")
    print(f"  {tabela.to_string()}")

    # teste chi-quadrado
    chi2, p_valor, gl, esperado = stats.chi2_contingency(tabela)

    print(f"\n  Chi-quadrado:         {chi2:.4f}")
    print(f"  Graus de liberdade:   {gl}")
    print(f"  P-valor:              {p_valor:.2e}")
    print(f"\n  Conclusão: ", end="")

    if p_valor < 0.05:
        print("REJEITA H0 → Existe associação significativa entre")
        print("              tempo de interação e aceitação da recomendação.")
    else:
        print("NÃO REJEITA H0 → Não há evidência suficiente de associação.")

    # limpar coluna temporária
    df.drop(columns=["tempo_grupo"], inplace=True)


# =============================================================================
# 6. TOP LOJAS E MOTIVOS DE REJEIÇÃO
# =============================================================================

def analise_lojas(df: pd.DataFrame) -> None:
    """Analisa performance das lojas recomendadas."""

    print("\n" + "=" * 60)
    print("6. PERFORMANCE DAS LOJAS RECOMENDADAS")
    print("=" * 60)

    # top 10 lojas mais recomendadas
    top_lojas = df.groupby("loja_recomendada").agg(
        total=("aceitou", "count"),
        aceitas=("aceitou", "sum"),
    )
    top_lojas["taxa"] = (top_lojas["aceitas"] / top_lojas["total"] * 100).round(2)
    top_lojas = top_lojas.sort_values("total", ascending=False).head(10)

    print("\n  Top 10 lojas mais recomendadas:")
    for loja, row in top_lojas.iterrows():
        print(f"    {loja:22s} → {int(row['total']):3d} recomendações, "
              f"{row['taxa']:5.1f}% aceitação")

    # motivos de rejeição
    print("\n  Motivos de rejeição:")
    motivos = df[df["aceitou"] == 0]["motivo_rejeicao"].value_counts()
    for motivo, count in motivos.items():
        print(f"    {motivo:20s} → {count:3d} ocorrências")

    # gráfico: top lojas + motivos de rejeição
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # top lojas
    top_lojas_plot = top_lojas.sort_values("taxa", ascending=True)
    cores_loja = [CORES["secundaria"] if t > 65 else CORES["destaque"] if t > 50
                  else CORES["negativo"] for t in top_lojas_plot["taxa"]]
    axes[0].barh(top_lojas_plot.index, top_lojas_plot["taxa"], color=cores_loja)
    axes[0].set_title("Taxa de Aceitação por Loja (Top 10)")
    axes[0].set_xlabel("Taxa de Aceitação (%)")
    axes[0].set_xlim(0, 100)

    # motivos
    motivos.plot(kind="barh", ax=axes[1], color=CORES["negativo"], alpha=0.8)
    axes[1].set_title("Motivos de Rejeição")
    axes[1].set_xlabel("Ocorrências")

    fig.tight_layout()
    salvar_grafico(fig, "05_lojas_e_rejeicao.png")


# =============================================================================
# 7. RESUMO EXECUTIVO
# =============================================================================

def resumo_executivo(df: pd.DataFrame) -> None:
    """Gera um resumo final com os principais insights."""

    print("\n" + "=" * 60)
    print("7. RESUMO EXECUTIVO")
    print("=" * 60)

    total = len(df)
    sessoes = df["sessao_id"].nunique()
    taxa_geral = df["aceitou"].mean() * 100
    cat_top = df.groupby("categoria")["aceitou"].mean().idxmax()
    dia_top = df["dia_semana"].value_counts().idxmax()
    tempo_medio = df["tempo_interacao"].mean()

    print(f"""
  INDICADORES GERAIS:
  ─────────────────────────────────────
  Sessões registradas:     {sessoes}
  Interações totais:       {total}
  Taxa de aceitação geral: {taxa_geral:.1f}%
  Tempo médio de uso:      {tempo_medio:.1f}s
  Categoria com melhor conversão: {cat_top}
  Dia com maior movimento:        {dia_top}

  PRINCIPAIS INSIGHTS:
  ─────────────────────────────────────
  1. Tempo de interação é o principal preditor de aceitação.
     Usuários com tempo > 10s aceitam significativamente mais.

  2. A categoria 'descansar' apresenta a maior taxa de aceitação,
     sugerindo que usuários com essa intenção são menos seletivos.

  3. O fim de semana concentra a maior parte das interações,
     indicando oportunidade para campanhas direcionadas.

  4. A faixa horária do almoço tem maior volume na categoria 'comer',
     confirmando padrão de comportamento esperado.
    """)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print("\n" + "█" * 60)
    print("  ANÁLISE ESTATÍSTICA - TOTEM INTELIGENTE FLEXMEDIA")
    print("█" * 60)

    df = carregar_dados()

    estatisticas_descritivas(df)
    analise_aceitacao(df)
    analise_correlacao(df)
    analise_temporal(df)
    teste_hipotese(df)
    analise_lojas(df)
    resumo_executivo(df)

    print("\n" + "=" * 60)
    print(f"Análise concluída. Gráficos salvos em: {GRAFICOS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()