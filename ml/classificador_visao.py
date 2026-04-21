"""
Classificador de Presença via Visão Computacional — Sprint 4.

Treina um modelo de Machine Learning supervisionado para classificar
imagens como "com presença" ou "sem presença" humana, usando features
extraídas por processamento de imagem (histogramas, gradientes, HOG).

Pipeline:
1. Carrega dataset de imagens (vision/dataset/)
2. Extrai 101 features por imagem (cor, textura, pele, gradientes)
3. Treina SVM e Random Forest
4. Avalia com acurácia, precisão, recall, F1, matriz de confusão
5. Validação cruzada (5-fold)
6. Exporta modelo treinado e métricas
7. Gera gráficos (matriz de confusão, importância de features)

O modelo é integrado ao totem interativo como alternativa/complemento
ao detector baseado em regras (HOG/MediaPipe).
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Adicionar diretório raiz ao path para importar o detector
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from vision.detector_presenca import processar_dataset, extrair_features

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
# 1. CARREGAMENTO E PREPARAÇÃO
# =============================================================================

def carregar_e_preparar() -> tuple:
    """Carrega dataset, extrai features e divide em treino/teste."""

    print("\n" + "=" * 60)
    print("  CARREGAMENTO DO DATASET DE IMAGENS")
    print("=" * 60)

    X, y, arquivos = processar_dataset()

    print(f"\n  Total de imagens:  {len(arquivos)}")
    print(f"  Com presença:      {(y == 1).sum()}")
    print(f"  Sem presença:      {(y == 0).sum()}")
    print(f"  Features/imagem:   {X.shape[1]}")

    # Divisão treino/teste (70/30 por dataset pequeno)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    print(f"\n  Treino: {len(X_train)} amostras")
    print(f"  Teste:  {len(X_test)} amostras")

    return X_train, X_test, y_train, y_test, X, y


# =============================================================================
# 2. TREINAMENTO
# =============================================================================

def treinar_modelos(X_train, y_train, X_test, y_test) -> tuple:
    """Treina SVM e Random Forest, compara métricas."""

    resultados = {}

    # --- MODELO 1: SVM com normalização ---
    print("\n" + "=" * 60)
    print("  MODELO 1: SVM (Support Vector Machine)")
    print("=" * 60)

    pipe_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    pipe_svm.fit(X_train, y_train)
    preds_svm = pipe_svm.predict(X_test)

    metricas_svm = {
        "nome": "SVM",
        "acuracia": accuracy_score(y_test, preds_svm),
        "precisao": precision_score(y_test, preds_svm, zero_division=0),
        "recall": recall_score(y_test, preds_svm, zero_division=0),
        "f1": f1_score(y_test, preds_svm, zero_division=0),
        "predicoes": preds_svm,
    }

    print(f"\n  Acurácia:  {metricas_svm['acuracia']:.4f}")
    print(f"  Precisão:  {metricas_svm['precisao']:.4f}")
    print(f"  Recall:    {metricas_svm['recall']:.4f}")
    print(f"  F1-Score:  {metricas_svm['f1']:.4f}")

    resultados["svm"] = {"modelo": pipe_svm, "metricas": metricas_svm}

    # --- MODELO 2: Random Forest ---
    print("\n" + "=" * 60)
    print("  MODELO 2: RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=3,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_test)

    metricas_rf = {
        "nome": "Random Forest",
        "acuracia": accuracy_score(y_test, preds_rf),
        "precisao": precision_score(y_test, preds_rf, zero_division=0),
        "recall": recall_score(y_test, preds_rf, zero_division=0),
        "f1": f1_score(y_test, preds_rf, zero_division=0),
        "predicoes": preds_rf,
    }

    print(f"\n  Acurácia:  {metricas_rf['acuracia']:.4f}")
    print(f"  Precisão:  {metricas_rf['precisao']:.4f}")
    print(f"  Recall:    {metricas_rf['recall']:.4f}")
    print(f"  F1-Score:  {metricas_rf['f1']:.4f}")

    resultados["rf"] = {"modelo": rf, "metricas": metricas_rf}

    # --- COMPARAÇÃO ---
    print("\n" + "=" * 60)
    print("  COMPARAÇÃO DOS MODELOS")
    print("=" * 60)

    print(f"\n  {'Modelo':<20s} {'Acurácia':>10s} {'F1-Score':>10s}")
    print(f"  {'─' * 42}")
    for key in resultados:
        m = resultados[key]["metricas"]
        print(f"  {m['nome']:<20s} {m['acuracia']:>10.4f} {m['f1']:>10.4f}")

    # Selecionar melhor
    melhor_key = max(resultados, key=lambda k: resultados[k]["metricas"]["f1"])
    melhor = resultados[melhor_key]
    print(f"\n  >>> Melhor modelo: {melhor['metricas']['nome']} "
          f"(F1={melhor['metricas']['f1']:.4f})")

    return resultados, melhor


# =============================================================================
# 3. VALIDAÇÃO CRUZADA
# =============================================================================

def validacao_cruzada(modelo, X, y, nome: str) -> dict:
    """Validação cruzada 5-fold (ou 3-fold se dataset muito pequeno)."""

    print("\n" + "=" * 60)
    print(f"  VALIDAÇÃO CRUZADA — {nome.upper()}")
    print("=" * 60)

    n_folds = min(5, min((y == 0).sum(), (y == 1).sum()))
    n_folds = max(2, n_folds)  # mínimo 2 folds

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = cross_val_score(modelo, X, y, cv=cv, scoring="f1")

    print(f"\n  Folds: {n_folds}")
    for i, score in enumerate(scores, 1):
        barra = "█" * int(score * 30)
        print(f"    Fold {i}: {score:.4f}  {barra}")

    print(f"\n  Média:  {scores.mean():.4f}")
    print(f"  Desvio: {scores.std():.4f}")

    return {
        "folds": n_folds,
        "media": float(scores.mean()),
        "desvio": float(scores.std()),
        "scores": [float(s) for s in scores],
    }


# =============================================================================
# 4. GRÁFICOS
# =============================================================================

def plotar_matriz_confusao(y_test, predicoes, nome: str) -> None:
    """Gera e salva a matriz de confusão."""
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    cm = confusion_matrix(y_test, predicoes)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Sem Presença", "Com Presença"],
        yticklabels=["Sem Presença", "Com Presença"],
        ax=ax, cbar_kws={"label": "Quantidade"},
        annot_kws={"size": 18, "fontweight": "bold"},
    )
    ax.set_title(f"Matriz de Confusão — Visão Computacional\n({nome})")
    ax.set_ylabel("Valor Real")
    ax.set_xlabel("Valor Predito")

    fig.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, "04_matriz_confusao_visao.png")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Gráfico salvo: {caminho}")


def plotar_feature_importance_visao(modelo_rf, n_features: int) -> None:
    """Plota importância das features do classificador de visão."""
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    importancias = modelo_rf.feature_importances_

    # Nomes das features
    nomes = []
    nomes += [f"hist_H_{i}" for i in range(16)]
    nomes += [f"hist_S_{i}" for i in range(16)]
    nomes += [f"hist_V_{i}" for i in range(16)]
    nomes += ["grad_media", "grad_desvio"]
    nomes += ["skin_ratio"]
    n_hog = n_features - len(nomes)
    nomes += [f"hog_{i}" for i in range(n_hog)]

    # Top 15 features
    indices = np.argsort(importancias)[::-1][:15]

    fig, ax = plt.subplots(figsize=(10, 6))
    cores = [CORES["primaria"] if "hist" in nomes[i]
             else CORES["secundaria"] if "hog" in nomes[i]
             else CORES["destaque"] for i in indices]

    ax.barh(
        [nomes[i] for i in indices][::-1],
        [importancias[i] for i in indices][::-1],
        color=cores[::-1],
    )
    ax.set_title("Top 15 Features — Classificador de Presença")
    ax.set_xlabel("Importância")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, "05_feature_importance_visao.png")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Gráfico salvo: {caminho}")


def plotar_comparacao_modelos(resultados: dict) -> None:
    """Gráfico comparativo entre os modelos de visão."""
    os.makedirs(GRAFICOS_DIR, exist_ok=True)

    nomes_metricas = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    modelos = list(resultados.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(nomes_metricas))
    largura = 0.35

    for i, key in enumerate(modelos):
        m = resultados[key]["metricas"]
        valores = [m["acuracia"], m["precisao"], m["recall"], m["f1"]]
        offset = (i - 0.5) * largura
        cor = CORES["primaria"] if i == 0 else CORES["destaque"]
        barras = ax.bar(x + offset, valores, largura, label=m["nome"], color=cor)
        for barra in barras:
            ax.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.01,
                    f"{barra.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Score")
    ax.set_title("Comparação de Modelos — Classificador de Presença")
    ax.set_xticks(x)
    ax.set_xticklabels(nomes_metricas)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    caminho = os.path.join(GRAFICOS_DIR, "06_comparacao_modelos_visao.png")
    fig.savefig(caminho, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Gráfico salvo: {caminho}")


# =============================================================================
# 5. SALVAR MODELO E MÉTRICAS
# =============================================================================

def salvar_artefatos(melhor: dict, cv_resultado: dict, resultados: dict) -> None:
    """Salva o modelo treinado e as métricas em JSON."""

    # Salvar modelo
    caminho_modelo = os.path.join(MODELO_DIR, "classificador_presenca.joblib")
    joblib.dump(melhor["modelo"], caminho_modelo)
    print(f"\n  Modelo salvo: {caminho_modelo}")

    # Salvar métricas
    metricas = {
        "tipo": "classificador_presenca_visao_computacional",
        "modelo_selecionado": melhor["metricas"]["nome"],
        "modelos": {},
        "validacao_cruzada": cv_resultado,
    }

    for key in resultados:
        m = resultados[key]["metricas"]
        metricas["modelos"][key] = {
            "nome": m["nome"],
            "acuracia": round(m["acuracia"], 4),
            "precisao": round(m["precisao"], 4),
            "recall": round(m["recall"], 4),
            "f1": round(m["f1"], 4),
        }

    caminho_metricas = os.path.join(MODELO_DIR, "metricas_visao.json")
    with open(caminho_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"  Métricas salvas: {caminho_metricas}")


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

def main():
    print("\n" + "█" * 60)
    print("  CLASSIFICADOR DE PRESENÇA — VISÃO COMPUTACIONAL")
    print("  Sprint 4 — Totem Inteligente FlexMedia")
    print("█" * 60)

    # 1. Carregar e preparar
    X_train, X_test, y_train, y_test, X, y = carregar_e_preparar()

    # 2. Treinar modelos
    resultados, melhor = treinar_modelos(X_train, y_train, X_test, y_test)

    # 3. Classification report
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT DETALHADO")
    print("=" * 60)
    report = classification_report(
        y_test, melhor["metricas"]["predicoes"],
        target_names=["Sem Presença", "Com Presença"],
    )
    print(f"\n{report}")

    # 4. Validação cruzada
    cv_resultado = validacao_cruzada(melhor["modelo"], X, y, melhor["metricas"]["nome"])

    # 5. Gráficos
    print("\n" + "=" * 60)
    print("  GERANDO GRÁFICOS")
    print("=" * 60)

    plotar_matriz_confusao(y_test, melhor["metricas"]["predicoes"], melhor["metricas"]["nome"])

    # Feature importance (só RF)
    if "rf" in resultados:
        plotar_feature_importance_visao(resultados["rf"]["modelo"], X.shape[1])

    plotar_comparacao_modelos(resultados)

    # 6. Salvar artefatos
    print("\n" + "=" * 60)
    print("  SALVAMENTO")
    print("=" * 60)

    salvar_artefatos(melhor, cv_resultado, resultados)

    # Resumo
    m = melhor["metricas"]
    print(f"""
{'=' * 60}
  RESUMO — CLASSIFICADOR DE PRESENÇA
{'=' * 60}

  Modelo selecionado:  {m['nome']}
  Acurácia:            {m['acuracia']:.4f}
  Precisão:            {m['precisao']:.4f}
  Recall:              {m['recall']:.4f}
  F1-Score:            {m['f1']:.4f}

  Validação cruzada:   F1 médio = {cv_resultado['media']:.4f} (±{cv_resultado['desvio']:.4f})

  O classificador será integrado ao totem interativo como
  complemento ao detector baseado em regras (HOG/MediaPipe),
  permitindo detecção de presença a partir de frames de câmera
  ou imagens carregadas pelo visitante.
    """)


if __name__ == "__main__":
    main()
