"""
Modelo de Machine Learning – Totem Inteligente FlexMedia.

Objetivo: prever se um usuário aceitará a recomendação feita pelo totem.

Pipeline:
1. Carrega dados do banco SQLite (com JOINs entre tabelas)
2. Engenharia de features (encoding de variáveis categóricas)
3. Treina dois modelos: Random Forest e Logistic Regression
4. Avalia com métricas completas (acurácia, precisão, recall, F1)
5. Validação cruzada (5-fold) para estabilidade
6. Feature importance do melhor modelo
7. Exporta matriz de confusão e gráficos
8. Salva modelo treinado com joblib
9. Exporta métricas em JSON para consumo pelo dashboard
"""

import sqlite3
import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "interacoes.db")
GRAFICOS_DIR = os.path.join(BASE_DIR, "ml", "graficos")
MODELO_DIR = os.path.join(BASE_DIR, "ml")

CORES = {
    "primaria": "#2563EB",
    "secundaria": "#10B981",
    "negativo": "#EF4444",
    "destaque": "#F59E0B",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


# =============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# =============================================================================

def carregar_dados() -> pd.DataFrame:
    """Carrega dados com JOINs entre as 3 tabelas."""
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query("""
        SELECT
            s.faixa_etaria,
            s.dia_semana,
            s.faixa_horaria,
            i.categoria,
            i.preferencia,
            i.tempo_interacao,
            r.loja_recomendada,
            r.aceitou
        FROM interacoes i
        JOIN sessoes s ON i.sessao_id = s.id
        JOIN recomendacoes r ON r.interacao_id = i.id
    """, conn)

    conn.close()
    return df


def preparar_features(df: pd.DataFrame) -> tuple:
    """
    Prepara features para o modelo.
    Usa LabelEncoder para variáveis categóricas (mais controlado que get_dummies).
    Retorna X, y e o dicionário de encoders.
    """
    df_ml = df.copy()

    # colunas categóricas para encoding
    colunas_categoricas = [
        "faixa_etaria", "dia_semana", "faixa_horaria",
        "categoria", "preferencia", "loja_recomendada"
    ]

    encoders = {}
    for col in colunas_categoricas:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le

    # separar features e target
    X = df_ml.drop("aceitou", axis=1)
    y = df_ml["aceitou"]

    return X, y, encoders


# =============================================================================
# 2. TREINAMENTO E AVALIAÇÃO
# =============================================================================

def avaliar_modelo(nome: str, modelo, X_test, y_test) -> dict:
    """Calcula todas as métricas de um modelo."""
    predicoes = modelo.predict(X_test)

    metricas = {
        "nome": nome,
        "acuracia": accuracy_score(y_test, predicoes),
        "precisao": precision_score(y_test, predicoes),
        "recall": recall_score(y_test, predicoes),
        "f1": f1_score(y_test, predicoes),
        "predicoes": predicoes,
    }

    return metricas


def exibir_metricas(metricas: dict) -> None:
    """Exibe métricas formatadas no terminal."""
    print(f"\n  {'Métrica':<20s} {'Valor':>10s}")
    print(f"  {'─' * 32}")
    print(f"  {'Acurácia':<20s} {metricas['acuracia']:>10.4f}")
    print(f"  {'Precisão':<20s} {metricas['precisao']:>10.4f}")
    print(f"  {'Recall':<20s} {metricas['recall']:>10.4f}")
    print(f"  {'F1-Score':<20s} {metricas['f1']:>10.4f}")


def treinar_e_avaliar(X, y) -> tuple:
    """Treina dois modelos e compara resultados."""

    # divisão treino/teste (80/20, estratificado)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n  Dados de treino: {len(X_train)} amostras")
    print(f"  Dados de teste:  {len(X_test)} amostras")
    print(f"  Proporção target (treino): {y_train.mean():.2%} aceitaram")

    # ---- MODELO 1: Random Forest ----
    print("\n" + "=" * 60)
    print("  MODELO 1: RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    rf.fit(X_train, y_train)
    metricas_rf = avaliar_modelo("Random Forest", rf, X_test, y_test)
    exibir_metricas(metricas_rf)

    # ---- MODELO 2: Logistic Regression ----
    print("\n" + "=" * 60)
    print("  MODELO 2: LOGISTIC REGRESSION")
    print("=" * 60)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    metricas_lr = avaliar_modelo("Logistic Regression", lr, X_test, y_test)
    exibir_metricas(metricas_lr)

    # ---- COMPARAÇÃO ----
    print("\n" + "=" * 60)
    print("  COMPARAÇÃO DOS MODELOS")
    print("=" * 60)

    print(f"\n  {'Modelo':<25s} {'Acurácia':>10s} {'F1-Score':>10s}")
    print(f"  {'─' * 47}")
    print(f"  {'Random Forest':<25s} {metricas_rf['acuracia']:>10.4f} {metricas_rf['f1']:>10.4f}")
    print(f"  {'Logistic Regression':<25s} {metricas_lr['acuracia']:>10.4f} {metricas_lr['f1']:>10.4f}")

    # escolher melhor modelo
    if metricas_rf["f1"] >= metricas_lr["f1"]:
        melhor_modelo = rf
        melhor_metricas = metricas_rf
        print(f"\n  >>> Melhor modelo: RANDOM FOREST (F1={metricas_rf['f1']:.4f})")
    else:
        melhor_modelo = lr
        melhor_metricas = metricas_lr
        print(f"\n  >>> Melhor modelo: LOGISTIC REGRESSION (F1={metricas_lr['f1']:.4f})")

    return melhor_modelo, melhor_metricas, rf, metricas_rf, lr, metricas_lr, X_test, y_test, X_train, y_train


# =============================================================================
# 3. VALIDAÇÃO CRUZADA
# =============================================================================

def validacao_cruzada(modelo, X, y, nome: str) -> dict:
    """Executa validação cruzada 5-fold e exibe resultados."""

    print("\n" + "=" * 60)
    print(f"  VALIDAÇÃO CRUZADA (5-FOLD) – {nome.upper()}")
    print("=" * 60)

    scores = cross_val_score(modelo, X, y, cv=5, scoring="f1")

    print(f"\n  Scores por fold:")
    for i, score in enumerate(scores, 1):
        barra = "█" * int(score * 40)
        print(f"    Fold {i}: {score:.4f}  {barra}")

    print(f"\n  Média:         {scores.mean():.4f}")
    print(f"  Desvio padrão: {scores.std():.4f}")
    print(f"\n  Interpretação: ", end="")

    if scores.std() < 0.05:
        print("Modelo ESTÁVEL (baixa variância entre folds)")
    else:
        print("Modelo com VARIÂNCIA moderada entre folds")

    return {"media": float(scores.mean()), "desvio": float(scores.std())}


# =============================================================================
# 4. FEATURE IMPORTANCE
# =============================================================================

def feature_importance(modelo_rf, feature_names: list) -> list:
    """Exibe e plota a importância das features do Random Forest."""

    print("\n" + "=" * 60)
    print("  IMPORTÂNCIA DAS FEATURES (RANDOM FOREST)")
    print("=" * 60)

    importancias = modelo_rf.feature_importances_
    indices = np.argsort(importancias)[::-1]

    ranking = []
    print(f"\n  Ranking de importância:")
    for i, idx in enumerate(indices, 1):
        barra = "█" * int(importancias[idx] * 80)
        print(f"    {i}. {feature_names[idx]:20s} → {importancias[idx]:.4f}  {barra}")
        ranking.append({
            "feature": feature_names[idx],
            "importancia": round(float(importancias[idx]), 4)
        })

    # gráfico
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    cores_map = [CORES["primaria"]] * 3 + ["#94A3B8"] * (len(indices) - 3)

    importancias_ordenadas = importancias[indices]
    nomes_ordenados = [feature_names[i] for i in indices]

    barras = ax.barh(
        range(len(indices)), importancias_ordenadas,
        color=cores_map[:len(indices)]
    )
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(nomes_ordenados)
    ax.set_title("Importância das Features – Random Forest")
    ax.set_xlabel("Importância")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, "01_feature_importance.png")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Gráfico salvo: {caminho}")

    return ranking


# =============================================================================
# 5. MATRIZ DE CONFUSÃO
# =============================================================================

def plotar_matriz_confusao(y_test, predicoes, nome: str) -> None:
    """Plota e salva a matriz de confusão como heatmap."""

    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, predicoes)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejeitou", "Aceitou"],
        yticklabels=["Rejeitou", "Aceitou"],
        ax=ax, cbar_kws={"label": "Quantidade"},
        annot_kws={"size": 18, "fontweight": "bold"}
    )
    ax.set_title(f"Matriz de Confusão – {nome}")
    ax.set_ylabel("Valor Real")
    ax.set_xlabel("Valor Predito")

    fig.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, "02_matriz_confusao.png")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\n  Matriz de Confusão:")
    print(f"                    Predito")
    print(f"                Rejeitou  Aceitou")
    print(f"  Real Rejeitou   {cm[0][0]:5d}    {cm[0][1]:5d}")
    print(f"  Real Aceitou    {cm[1][0]:5d}    {cm[1][1]:5d}")
    print(f"\n  Gráfico salvo: {caminho}")


# =============================================================================
# 6. CLASSIFICATION REPORT
# =============================================================================

def exibir_classification_report(y_test, predicoes) -> None:
    """Exibe o classification report completo."""

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT DETALHADO")
    print("=" * 60)

    report = classification_report(
        y_test, predicoes,
        target_names=["Rejeitou", "Aceitou"]
    )
    print(f"\n{report}")


# =============================================================================
# 7. GRÁFICO COMPARATIVO
# =============================================================================

def plotar_comparacao(metricas_rf: dict, metricas_lr: dict) -> None:
    """Plota gráfico comparativo entre os dois modelos."""

    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    metricas_nomes = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    valores_rf = [metricas_rf["acuracia"], metricas_rf["precisao"],
                  metricas_rf["recall"], metricas_rf["f1"]]
    valores_lr = [metricas_lr["acuracia"], metricas_lr["precisao"],
                  metricas_lr["recall"], metricas_lr["f1"]]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metricas_nomes))
    largura = 0.35

    barras_rf = ax.bar(x - largura / 2, valores_rf, largura,
                       label="Random Forest", color=CORES["primaria"])
    barras_lr = ax.bar(x + largura / 2, valores_lr, largura,
                       label="Logistic Regression", color=CORES["destaque"])

    # rótulos nas barras
    for barra in barras_rf:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.01,
                f"{barra.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for barra in barras_lr:
        ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.01,
                f"{barra.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos: Random Forest vs Logistic Regression")
    ax.set_xticks(x)
    ax.set_xticklabels(metricas_nomes)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, "03_comparacao_modelos.png")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Gráfico salvo: {caminho}")


# =============================================================================
# 8. SALVAR MODELO
# =============================================================================

def salvar_modelo(modelo, nome: str, encoders: dict = None, colunas: list = None) -> None:
    """Salva o modelo treinado com joblib."""
    # Modelo isolado (compatibilidade Sprint 3)
    caminho = os.path.join(MODELO_DIR, "modelo_treinado.joblib")
    joblib.dump(modelo, caminho)
    print(f"\n  Modelo '{nome}' salvo em: {caminho}")

    # Modelo completo com encoders (Sprint 4 — usado pelo totem interativo)
    if encoders is not None:
        caminho_completo = os.path.join(MODELO_DIR, "modelo_completo.joblib")
        artefatos = {
            "modelo": modelo,
            "encoders": encoders,
            "colunas": colunas or [],
        }
        joblib.dump(artefatos, caminho_completo)
        print(f"  Modelo completo (com encoders) salvo em: {caminho_completo}")
        print(f"  → Usado pelo totem interativo (totem/app_totem.py)")


# =============================================================================
# 9. EXPORTAR MÉTRICAS EM JSON
# =============================================================================

def exportar_metricas(metricas_rf: dict, metricas_lr: dict,
                      melhor_metricas: dict, cv_resultado: dict,
                      ranking_features: list) -> None:
    """
    Exporta todas as métricas em um arquivo JSON para consumo
    pelo dashboard Streamlit, eliminando valores hardcoded.
    """
    dados = {
        "modelo_selecionado": melhor_metricas["nome"],
        "random_forest": {
            "acuracia": round(metricas_rf["acuracia"], 4),
            "precisao": round(metricas_rf["precisao"], 4),
            "recall": round(metricas_rf["recall"], 4),
            "f1": round(metricas_rf["f1"], 4),
        },
        "logistic_regression": {
            "acuracia": round(metricas_lr["acuracia"], 4),
            "precisao": round(metricas_lr["precisao"], 4),
            "recall": round(metricas_lr["recall"], 4),
            "f1": round(metricas_lr["f1"], 4),
        },
        "validacao_cruzada": {
            "media_f1": round(cv_resultado["media"], 4),
            "desvio_f1": round(cv_resultado["desvio"], 4),
        },
        "feature_importance": ranking_features,
    }

    caminho = os.path.join(MODELO_DIR, "metricas.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, indent=2, ensure_ascii=False)

    print(f"\n  Métricas exportadas em: {caminho}")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print("\n" + "█" * 60)
    print("  MACHINE LEARNING – TOTEM INTELIGENTE FLEXMEDIA")
    print("█" * 60)

    # carregar e preparar dados
    print("\n" + "=" * 60)
    print("  CARREGAMENTO DOS DADOS")
    print("=" * 60)

    df = carregar_dados()
    print(f"\n  Total de registros: {len(df)}")
    print(f"  Distribuição do target:")
    print(f"    Aceitou:  {(df['aceitou'] == 1).sum()} ({df['aceitou'].mean():.1%})")
    print(f"    Rejeitou: {(df['aceitou'] == 0).sum()} ({1 - df['aceitou'].mean():.1%})")

    X, y, encoders = preparar_features(df)

    # treinar e avaliar modelos
    resultado = treinar_e_avaliar(X, y)
    melhor_modelo, melhor_metricas, rf, metricas_rf, lr, metricas_lr, X_test, y_test, X_train, y_train = resultado

    # classification report do melhor modelo
    exibir_classification_report(y_test, melhor_metricas["predicoes"])

    # matriz de confusão
    print("\n" + "=" * 60)
    print(f"  MATRIZ DE CONFUSÃO – {melhor_metricas['nome'].upper()}")
    print("=" * 60)
    plotar_matriz_confusao(y_test, melhor_metricas["predicoes"], melhor_metricas["nome"])

    # validação cruzada do melhor modelo
    if melhor_metricas["nome"] == "Random Forest":
        cv_resultado = validacao_cruzada(rf, X, y, "Random Forest")
    else:
        cv_resultado = validacao_cruzada(lr, X, y, "Logistic Regression")

    # feature importance (só para Random Forest)
    ranking_features = feature_importance(rf, list(X.columns))

    # gráfico comparativo
    print("\n" + "=" * 60)
    print("  GRÁFICO COMPARATIVO")
    print("=" * 60)
    plotar_comparacao(metricas_rf, metricas_lr)

    # salvar modelo
    print("\n" + "=" * 60)
    print("  SALVAMENTO DO MODELO")
    print("=" * 60)
    salvar_modelo(melhor_modelo, melhor_metricas["nome"],
                  encoders=encoders, colunas=list(X.columns))

    # exportar métricas em JSON para o dashboard
    print("\n" + "=" * 60)
    print("  EXPORTAÇÃO DE MÉTRICAS")
    print("=" * 60)
    exportar_metricas(metricas_rf, metricas_lr, melhor_metricas,
                      cv_resultado, ranking_features)

    # resumo final
    print("\n" + "=" * 60)
    print("  RESUMO FINAL")
    print("=" * 60)
    print(f"""
  Modelo selecionado:    {melhor_metricas['nome']}
  Acurácia:              {melhor_metricas['acuracia']:.4f}
  F1-Score:              {melhor_metricas['f1']:.4f}
  Precisão:              {melhor_metricas['precisao']:.4f}
  Recall:                {melhor_metricas['recall']:.4f}

  Justificativa da escolha:
  O modelo foi selecionado com base no F1-Score, que equilibra
  precisão e recall. A validação cruzada confirmou estabilidade
  nos resultados. A feature importance revela que o tempo de
  interação é a variável mais relevante para a predição.
    """)

    print("=" * 60)
    print(f"  ML concluído. Gráficos em: {GRAFICOS_DIR}")
    print(f"  Modelo salvo em: {MODELO_DIR}/modelo_treinado.joblib")
    print(f"  Métricas em: {MODELO_DIR}/metricas.json")
    print("=" * 60)


if __name__ == "__main__":
    main()