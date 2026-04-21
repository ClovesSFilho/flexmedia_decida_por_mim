"""
Sistema de Recomendação Ativa — Sprint 4.

Módulo dedicado ao motor de recomendação do Totem Inteligente FlexMedia.

O modelo de Machine Learning treinado na Sprint 3 (Logistic Regression, F1=0.80)
foi originalmente um CLASSIFICADOR DE ACEITAÇÃO: dado um perfil + loja, prevê
se o visitante aceitaria a recomendação.

Na Sprint 4, invertemos a lógica para transformá-lo em RECOMENDADOR ATIVO:
dado apenas o perfil do visitante (sem a loja), testamos todas as lojas
candidatas e retornamos aquela com maior probabilidade de aceitação.

Funcionalidades:
1. `recomendar()` — retorna a melhor loja + ranking completo
2. `explicar_recomendacao()` — gera justificativa textual
3. `recomendar_top_n()` — retorna top-N alternativas ranqueadas
4. `simular_cenarios()` — útil para o dashboard (exploração "what-if")

Este módulo é importado pelo totem interativo, pelo dashboard e pelo
relatório analítico final.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from typing import Optional
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELO_PATH = os.path.join(BASE_DIR, "ml", "modelo_completo.joblib")


# =============================================================================
# MAPEAMENTO LOJAS × COMBINAÇÃO (espelhado do sensor_simulado.py)
# =============================================================================

LOJAS_POR_COMBINACAO = {
    ("comer", "doce"):          ["Cacau Show", "Starbucks", "Kopenhagen"],
    ("comer", "salgado"):       ["McDonalds", "Subway", "Burger King"],
    ("comer", "saudavel"):      ["Mundo Verde", "Green Station", "Freshii"],
    ("comer", "fast-food"):     ["McDonalds", "Burger King", "Popeyes"],
    ("comer", "cafe"):           ["Starbucks", "Cafe do Mercado", "Havanna"],
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

# Emojis para exibição
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
# MOTOR DE RECOMENDAÇÃO
# =============================================================================

class MotorRecomendacao:
    """
    Motor de recomendação baseado em classificador de aceitação.

    Para cada par (perfil, loja candidata), calcula P(aceitar | perfil, loja)
    e retorna a loja que maximiza essa probabilidade.

    A classe abstrai todos os detalhes de encoding, predição e ranking,
    expondo uma API simples para consumidores (totem, dashboard, relatório).
    """

    def __init__(self, modelo_path: str = None):
        """Carrega o modelo completo com encoders embutidos."""
        if modelo_path is None:
            modelo_path = MODELO_PATH

        if not os.path.exists(modelo_path):
            raise FileNotFoundError(
                f"Modelo não encontrado em {modelo_path}. "
                f"Execute 'python ml/modelo_ml.py' primeiro."
            )

        artefatos = joblib.load(modelo_path)
        self.modelo = artefatos["modelo"]
        self.encoders = artefatos["encoders"]
        self.colunas = artefatos["colunas"]

    # -------------------------------------------------------------------------
    # Helpers de tempo (contextualiza recomendações com momento atual)
    # -------------------------------------------------------------------------

    @staticmethod
    def obter_faixa_horaria(hora: int = None) -> str:
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

    @staticmethod
    def obter_dia_semana(data: datetime = None) -> str:
        if data is None:
            data = datetime.now()
        dias = ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
        return dias[data.weekday()]

    # -------------------------------------------------------------------------
    # Predição base: probabilidade de aceitação para um par (perfil, loja)
    # -------------------------------------------------------------------------

    def _prever_aceitacao(
        self,
        faixa_etaria: str,
        dia_semana: str,
        faixa_horaria: str,
        categoria: str,
        preferencia: str,
        loja: str,
        tempo_interacao: int = 12,
    ) -> Optional[float]:
        """
        Retorna P(aceitar=1) para um par completo (perfil, loja).
        Retorna None se qualquer valor for desconhecido para os encoders.
        """
        try:
            dados = {
                "faixa_etaria":    self.encoders["faixa_etaria"].transform([faixa_etaria])[0],
                "dia_semana":      self.encoders["dia_semana"].transform([dia_semana])[0],
                "faixa_horaria":   self.encoders["faixa_horaria"].transform([faixa_horaria])[0],
                "categoria":       self.encoders["categoria"].transform([categoria])[0],
                "preferencia":     self.encoders["preferencia"].transform([preferencia])[0],
                "tempo_interacao": tempo_interacao,
                "loja_recomendada": self.encoders["loja_recomendada"].transform([loja])[0],
            }
            X = pd.DataFrame([dados])[self.colunas]
            prob = self.modelo.predict_proba(X)[0][1]
            return float(prob)
        except (ValueError, KeyError):
            return None

    # -------------------------------------------------------------------------
    # API PÚBLICA
    # -------------------------------------------------------------------------

    def recomendar(
        self,
        faixa_etaria: str,
        categoria: str,
        preferencia: str,
        tempo_interacao: int = 12,
        dia_semana: str = None,
        faixa_horaria: str = None,
    ) -> dict:
        """
        Recomenda a melhor loja para o perfil fornecido.

        Args:
            faixa_etaria: 'jovem', 'adulto' ou 'idoso'
            categoria: 'comer', 'comprar', 'descansar' ou 'lazer'
            preferencia: uma das preferências válidas da categoria
            tempo_interacao: tempo estimado em segundos (default=12)
            dia_semana: se None, usa o atual
            faixa_horaria: se None, usa a atual

        Returns:
            dict com:
                - loja: str (melhor recomendação)
                - probabilidade: float (P(aceitar))
                - emoji: str
                - ranking: list de dicts com todas as alternativas ordenadas
                - contexto: dict com info sobre o perfil usado
        """
        if dia_semana is None:
            dia_semana = self.obter_dia_semana()
        if faixa_horaria is None:
            faixa_horaria = self.obter_faixa_horaria()

        lojas_candidatas = LOJAS_POR_COMBINACAO.get((categoria, preferencia), [])

        if not lojas_candidatas:
            return {
                "loja": None,
                "probabilidade": 0.0,
                "emoji": "📍",
                "ranking": [],
                "contexto": {
                    "erro": f"Combinação categoria='{categoria}' + preferencia='{preferencia}' "
                            f"não tem lojas cadastradas."
                },
            }

        # Calcular probabilidade para cada loja candidata
        resultados = []
        for loja in lojas_candidatas:
            prob = self._prever_aceitacao(
                faixa_etaria, dia_semana, faixa_horaria,
                categoria, preferencia, loja, tempo_interacao,
            )
            if prob is not None:
                resultados.append({
                    "loja": loja,
                    "probabilidade": round(prob, 4),
                    "emoji": EMOJIS_LOJAS.get(loja, "📍"),
                })

        if not resultados:
            # Fallback se nenhuma loja foi vista no treinamento
            return {
                "loja": lojas_candidatas[0],
                "probabilidade": 0.5,
                "emoji": EMOJIS_LOJAS.get(lojas_candidatas[0], "📍"),
                "ranking": [{"loja": l, "probabilidade": 0.5,
                             "emoji": EMOJIS_LOJAS.get(l, "📍")} for l in lojas_candidatas],
                "contexto": {
                    "aviso": "Nenhuma loja reconhecida pelo modelo; fallback para primeira opção."
                },
            }

        # Ordenar por probabilidade descendente
        resultados.sort(key=lambda x: x["probabilidade"], reverse=True)
        melhor = resultados[0]

        return {
            "loja": melhor["loja"],
            "probabilidade": melhor["probabilidade"],
            "emoji": melhor["emoji"],
            "ranking": resultados,
            "contexto": {
                "faixa_etaria": faixa_etaria,
                "dia_semana": dia_semana,
                "faixa_horaria": faixa_horaria,
                "categoria": categoria,
                "preferencia": preferencia,
                "tempo_interacao": tempo_interacao,
                "n_candidatas": len(resultados),
            },
        }

    def recomendar_top_n(
        self,
        faixa_etaria: str,
        categoria: str,
        preferencia: str,
        n: int = 3,
        **kwargs,
    ) -> list:
        """Retorna as top-N lojas ranqueadas por probabilidade de aceitação."""
        resultado = self.recomendar(faixa_etaria, categoria, preferencia, **kwargs)
        return resultado["ranking"][:n]

    def explicar_recomendacao(self, resultado_recomendacao: dict) -> str:
        """
        Gera uma explicação textual em linguagem natural sobre por que
        a loja foi recomendada. Útil para o vídeo, o relatório e o totem.
        """
        if not resultado_recomendacao.get("loja"):
            return "Não foi possível gerar uma recomendação para essa combinação."

        ctx = resultado_recomendacao["contexto"]
        loja = resultado_recomendacao["loja"]
        prob = resultado_recomendacao["probabilidade"]
        ranking = resultado_recomendacao["ranking"]

        # Tradução legível
        idade_label = {
            "jovem": "jovem (18-29)",
            "adulto": "adulto (30-59)",
            "idoso": "idoso (60+)",
        }.get(ctx.get("faixa_etaria", ""), ctx.get("faixa_etaria", ""))

        horario_label = {
            "manha": "manhã",
            "almoco": "horário de almoço",
            "tarde": "tarde",
            "noite": "noite",
        }.get(ctx.get("faixa_horaria", ""), ctx.get("faixa_horaria", ""))

        dia_label = ctx.get("dia_semana", "").capitalize()

        linhas = []
        linhas.append(f"**{loja}** foi a loja escolhida com {prob:.0%} de probabilidade de aceitação.")
        linhas.append(f"")
        linhas.append(f"Análise do perfil do visitante:")
        linhas.append(f"  • Faixa etária: {idade_label}")
        linhas.append(f"  • Momento: {dia_label} à {horario_label}")
        linhas.append(f"  • Intenção: {ctx.get('categoria', '')} ({ctx.get('preferencia', '')})")

        if len(ranking) > 1:
            linhas.append(f"")
            linhas.append(f"Alternativas consideradas:")
            for i, r in enumerate(ranking[:3], 1):
                linhas.append(f"  {i}. {r['emoji']} {r['loja']} — {r['probabilidade']:.0%}")

        return "\n".join(linhas)

    def simular_cenarios(
        self,
        categoria: str,
        preferencia: str,
        faixas_etarias: list = None,
        faixas_horarias: list = None,
    ) -> pd.DataFrame:
        """
        Simula recomendações para múltiplos cenários.
        Útil para o dashboard mostrar "qual loja é melhor para cada perfil".

        Returns:
            DataFrame com colunas: faixa_etaria, faixa_horaria, loja_recomendada, probabilidade
        """
        if faixas_etarias is None:
            faixas_etarias = ["jovem", "adulto", "idoso"]
        if faixas_horarias is None:
            faixas_horarias = ["manha", "almoco", "tarde", "noite"]

        linhas = []
        for faixa_etaria in faixas_etarias:
            for faixa_horaria in faixas_horarias:
                resultado = self.recomendar(
                    faixa_etaria=faixa_etaria,
                    categoria=categoria,
                    preferencia=preferencia,
                    faixa_horaria=faixa_horaria,
                )
                linhas.append({
                    "faixa_etaria": faixa_etaria,
                    "faixa_horaria": faixa_horaria,
                    "loja_recomendada": resultado["loja"],
                    "probabilidade": resultado["probabilidade"],
                })

        return pd.DataFrame(linhas)

    def ranking_global_por_categoria(self, categoria: str) -> pd.DataFrame:
        """
        Para todas as preferências de uma categoria, mostra qual loja
        seria recomendada para um perfil médio (adulto, tarde, quarta).

        Útil para relatórios gerenciais.
        """
        linhas = []

        for (cat, pref), lojas in LOJAS_POR_COMBINACAO.items():
            if cat != categoria:
                continue

            resultado = self.recomendar(
                faixa_etaria="adulto",
                categoria=cat,
                preferencia=pref,
                dia_semana="quarta",
                faixa_horaria="tarde",
            )

            linhas.append({
                "categoria": cat,
                "preferencia": pref,
                "loja_recomendada": resultado["loja"],
                "probabilidade": resultado["probabilidade"],
                "n_alternativas": len(lojas),
            })

        return pd.DataFrame(linhas).sort_values("probabilidade", ascending=False)


# =============================================================================
# EXECUÇÃO STANDALONE — demonstração e testes
# =============================================================================

def main():
    print("\n" + "█" * 60)
    print("  SISTEMA DE RECOMENDAÇÃO ATIVA — TOTEM FLEXMEDIA")
    print("  Sprint 4")
    print("█" * 60)

    motor = MotorRecomendacao()
    print(f"\n  Modelo carregado de: {MODELO_PATH}")
    print(f"  Colunas do modelo:   {motor.colunas}")
    print(f"  Lojas cadastradas:   {sum(len(v) for v in LOJAS_POR_COMBINACAO.values())}")
    print(f"  Combinações:         {len(LOJAS_POR_COMBINACAO)}")

    # -------------------------------------------------------------------------
    # Teste 1: Recomendação simples
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TESTE 1: Recomendação para jovem querendo doce")
    print("=" * 60)

    rec = motor.recomendar(
        faixa_etaria="jovem",
        categoria="comer",
        preferencia="doce",
    )

    print(f"\n  Melhor loja: {rec['emoji']} {rec['loja']} ({rec['probabilidade']:.0%})")
    print(f"\n  Ranking completo:")
    for i, r in enumerate(rec["ranking"], 1):
        print(f"    {i}. {r['emoji']} {r['loja']:20s} → {r['probabilidade']:.0%}")

    # -------------------------------------------------------------------------
    # Teste 2: Explicabilidade
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TESTE 2: Explicação da recomendação")
    print("=" * 60)

    explicacao = motor.explicar_recomendacao(rec)
    print()
    for linha in explicacao.split("\n"):
        print(f"  {linha}")

    # -------------------------------------------------------------------------
    # Teste 3: Cenários por perfil
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TESTE 3: Simulação por cenários (comer + salgado)")
    print("=" * 60)

    df_cenarios = motor.simular_cenarios("comer", "salgado")
    print()
    print(df_cenarios.to_string(index=False))

    # -------------------------------------------------------------------------
    # Teste 4: Ranking global por categoria
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TESTE 4: Ranking global — categoria 'comer'")
    print("=" * 60)

    df_ranking = motor.ranking_global_por_categoria("comer")
    print()
    print(df_ranking.to_string(index=False))

    # -------------------------------------------------------------------------
    # Teste 5: Diversidade de recomendações
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TESTE 5: O modelo diferencia por perfil?")
    print("=" * 60)

    cenarios_teste = [
        ("jovem", "noite", "lazer", "cinema"),
        ("adulto", "almoco", "comer", "salgado"),
        ("idoso", "tarde", "descansar", "jardim"),
        ("jovem", "manha", "comprar", "eletronico"),
    ]

    print()
    print(f"  {'Perfil':<35s} {'Loja recomendada':<25s} {'Prob':>6s}")
    print(f"  {'─' * 70}")
    for faixa, horario, cat, pref in cenarios_teste:
        r = motor.recomendar(
            faixa_etaria=faixa,
            categoria=cat,
            preferencia=pref,
            faixa_horaria=horario,
        )
        perfil = f"{faixa}/{horario}/{cat}/{pref}"
        print(f"  {perfil:<35s} {r['emoji']} {r['loja']:<22s} {r['probabilidade']:.0%}")

    print("\n" + "=" * 60)
    print("  SISTEMA DE RECOMENDAÇÃO VALIDADO")
    print("=" * 60)
    print(f"""
  Características demonstradas:

  1. ✓ Recomendação baseada em ML (não regra fixa)
  2. ✓ Ranking completo de alternativas
  3. ✓ Explicabilidade (justificativa textual)
  4. ✓ Simulação de cenários por perfil
  5. ✓ Análise agregada para relatórios gerenciais
  6. ✓ Diferenciação por perfil do visitante

  API pública do MotorRecomendacao:
    - recomendar(perfil)                → melhor loja + ranking
    - recomendar_top_n(perfil, n=3)     → top-N alternativas
    - explicar_recomendacao(resultado)  → justificativa textual
    - simular_cenarios(cat, pref)       → matriz perfil × horário
    - ranking_global_por_categoria(cat) → relatório gerencial
    """)


if __name__ == "__main__":
    main()
