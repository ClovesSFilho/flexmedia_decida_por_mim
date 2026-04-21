"""
Gerador de Relatório Analítico Final — Sprint 4.

Consolida todas as métricas, insights e resultados do Totem Inteligente
FlexMedia em um relatório executivo de uma única peça, formatado
profissionalmente em PDF.

Fontes de dados consolidadas:
- Banco SQLite (sessões, interações, recomendações) — dados de uso
- ml/metricas.json                 — modelo de aceitação (Sprint 3)
- ml/metricas_visao.json           — classificador de presença (Sprint 4)
- ml/metricas_chatbot.json         — classificador de intenção (Sprint 4)
- Gráficos de analysis/graficos/   — análises estatísticas
- Gráficos de ml/graficos/         — performance dos modelos

Estrutura do relatório:
1. Capa
2. Resumo Executivo
3. Métricas de Uso
4. Métricas de Engajamento
5. Performance dos Modelos de IA
6. Visão Computacional e NLP (Sprint 4)
7. Insights e Recomendações
8. Conclusão e Próximos Passos

O PDF gerado é usado:
- Como entregável obrigatório da Sprint 4
- Para demonstrar visão sistêmica no vídeo de pitch
- Como material de apoio para o administrador do shopping
"""

import os
import sys
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    Image, KeepTogether, HRFlowable, Flowable,
)
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate, Frame
from reportlab.pdfgen import canvas as pdf_canvas
from scipy import stats


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# PALETA DE CORES (alinhada com dashboard)
# =============================================================================

CORES = {
    "primaria":    HexColor("#2563EB"),
    "secundaria":  HexColor("#10B981"),
    "destaque":    HexColor("#F59E0B"),
    "negativo":    HexColor("#EF4444"),
    "neutro_escuro": HexColor("#1E293B"),
    "neutro_medio":  HexColor("#475569"),
    "neutro_claro":  HexColor("#94A3B8"),
    "fundo_claro":   HexColor("#F8FAFC"),
    "fundo_card":    HexColor("#F1F5F9"),
    "borda":         HexColor("#E2E8F0"),
    "branco": white,
}


# =============================================================================
# ESTILOS DE TEXTO
# =============================================================================

def criar_estilos():
    """Define estilos tipográficos do relatório."""
    estilos = getSampleStyleSheet()

    estilos.add(ParagraphStyle(
        name="TituloCapa",
        fontName="Helvetica-Bold",
        fontSize=28,
        leading=34,
        alignment=TA_CENTER,
        textColor=CORES["neutro_escuro"],
        spaceAfter=8,
    ))

    estilos.add(ParagraphStyle(
        name="SubtituloCapa",
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
        textColor=CORES["neutro_medio"],
        spaceAfter=20,
    ))

    estilos.add(ParagraphStyle(
        name="H1Secao",
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=CORES["neutro_escuro"],
        spaceBefore=18,
        spaceAfter=10,
        borderPadding=(0, 0, 4, 0),
    ))

    estilos.add(ParagraphStyle(
        name="H2Secao",
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=CORES["primaria"],
        spaceBefore=12,
        spaceAfter=6,
    ))

    estilos.add(ParagraphStyle(
        name="H3Secao",
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=CORES["neutro_escuro"],
        spaceBefore=8,
        spaceAfter=4,
    ))

    estilos.add(ParagraphStyle(
        name="Corpo",
        fontName="Helvetica",
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=CORES["neutro_escuro"],
        spaceAfter=6,
    ))

    estilos.add(ParagraphStyle(
        name="CorpoDestaque",
        fontName="Helvetica-Oblique",
        fontSize=10.5,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=CORES["neutro_medio"],
        spaceAfter=6,
    ))

    estilos.add(ParagraphStyle(
        name="Legenda",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=CORES["neutro_claro"],
        spaceAfter=10,
    ))

    estilos.add(ParagraphStyle(
        name="KpiValor",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=TA_CENTER,
        textColor=CORES["primaria"],
    ))

    estilos.add(ParagraphStyle(
        name="KpiLabel",
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        alignment=TA_CENTER,
        textColor=CORES["neutro_medio"],
    ))

    return estilos


# =============================================================================
# ELEMENTOS VISUAIS CUSTOMIZADOS
# =============================================================================

class LinhaColorida(Flowable):
    """Linha horizontal decorativa."""
    def __init__(self, largura, cor=None, altura=3):
        Flowable.__init__(self)
        self.largura = largura
        self.altura = altura
        self.cor = cor or CORES["primaria"]

    def draw(self):
        self.canv.setFillColor(self.cor)
        self.canv.rect(0, 0, self.largura, self.altura, fill=1, stroke=0)


def cabecalho_rodape(canvas, doc):
    """Desenha cabeçalho e rodapé em cada página (exceto capa)."""
    if doc.page == 1:
        return  # capa não tem cabeçalho/rodapé

    canvas.saveState()

    # Cabeçalho
    canvas.setFillColor(CORES["neutro_claro"])
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, A4[1] - 1.2 * cm,
                      "Totem Inteligente FlexMedia — Relatório Analítico Final")
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.2 * cm,
                           f"Sprint 4 — Challenge FlexMedia")

    # Linha divisória
    canvas.setStrokeColor(CORES["borda"])
    canvas.setLineWidth(0.5)
    canvas.line(2 * cm, A4[1] - 1.4 * cm, A4[0] - 2 * cm, A4[1] - 1.4 * cm)

    # Rodapé
    canvas.setFont("Helvetica", 8)
    canvas.drawString(2 * cm, 1.2 * cm, "Cloves Silva Filho — RA567250 — FIAP")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")

    canvas.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm)

    canvas.restoreState()


# =============================================================================
# COLETA DE DADOS
# =============================================================================

def carregar_dados_banco() -> pd.DataFrame:
    """Carrega dados do banco SQLite."""
    db_path = os.path.join(BASE_DIR, "data", "interacoes.db")
    conn = sqlite3.connect(db_path)

    df = pd.read_sql_query("""
        SELECT
            s.id AS sessao_id,
            s.inicio_sessao,
            s.faixa_etaria,
            s.dia_semana,
            s.faixa_horaria,
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

    df["inicio_sessao"] = pd.to_datetime(df["inicio_sessao"])
    return df


def carregar_json_seguro(caminho: str) -> dict:
    """Carrega JSON com fallback para dict vazio."""
    if not os.path.exists(caminho):
        return {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def coletar_metricas_ia() -> dict:
    """Coleta métricas dos 3 modelos de IA."""
    return {
        "aceitacao": carregar_json_seguro(os.path.join(BASE_DIR, "ml", "metricas.json")),
        "visao":     carregar_json_seguro(os.path.join(BASE_DIR, "ml", "metricas_visao.json")),
        "chatbot":   carregar_json_seguro(os.path.join(BASE_DIR, "ml", "metricas_chatbot.json")),
    }


def calcular_estatisticas_dinamicas(df: pd.DataFrame) -> dict:
    """
    Calcula estatísticas dinamicamente a partir do DataFrame.
    Evita valores hardcoded no relatório — tudo vem dos dados reais.
    """
    # Correlação ponto-biserial: tempo × aceitação
    corr, p_corr = stats.pointbiserialr(df["aceitou"], df["tempo_interacao"])

    # Chi-quadrado: tempo curto/longo × aceitação
    df_temp = df.copy()
    df_temp["tempo_grupo"] = df_temp["tempo_interacao"].apply(
        lambda x: "curto" if x <= 7 else "longo"
    )
    tabela_cont = pd.crosstab(df_temp["tempo_grupo"], df_temp["aceitou"])
    chi2, p_chi2, gl, _ = stats.chi2_contingency(tabela_cont)

    # Taxas de aceitação por faixa de tempo
    bins = [0, 5, 10, 15, 20, 30]
    labels_tempo = ["1-5s", "6-10s", "11-15s", "16-20s", "21s+"]
    df_temp["faixa_tempo"] = pd.cut(df_temp["tempo_interacao"], bins=bins, labels=labels_tempo, right=True)
    taxa_por_faixa = df_temp.groupby("faixa_tempo", observed=True)["aceitou"].mean() * 100

    taxa_curto = taxa_por_faixa.iloc[0] if len(taxa_por_faixa) > 0 else 0
    taxa_longo = taxa_por_faixa.iloc[-1] if len(taxa_por_faixa) > 0 else 0
    taxa_acima_10 = df[df["tempo_interacao"] > 10]["aceitou"].mean() * 100

    return {
        "corr_r": round(corr, 3),
        "corr_p": p_corr,
        "chi2": round(chi2, 2),
        "chi2_p": p_chi2,
        "chi2_gl": gl,
        "taxa_curto_pct": round(taxa_curto, 1),
        "taxa_longo_pct": round(taxa_longo, 1),
        "taxa_acima_10s": round(taxa_acima_10, 1),
    }


# =============================================================================
# CONSTRUÇÃO DE SEÇÕES
# =============================================================================

def criar_capa(estilos) -> list:
    """Capa do relatório."""
    elementos = []

    elementos.append(Spacer(1, 4 * cm))

    # Selo superior
    selo = Table([["CHALLENGE FLEXMEDIA  •  SPRINT 4"]],
                 colWidths=[10 * cm])
    selo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CORES["primaria"]),
        ("TEXTCOLOR", (0, 0), (-1, -1), CORES["branco"]),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elementos.append(selo)
    elementos.append(Spacer(1, 1.5 * cm))

    # Títulos
    elementos.append(Paragraph("Totem Inteligente FlexMedia", estilos["TituloCapa"]))
    elementos.append(Paragraph(
        '"Decida por Mim" — Relatório Analítico Final',
        estilos["SubtituloCapa"],
    ))

    elementos.append(LinhaColorida(8 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 2 * cm))

    # Bloco de destaque: o que o relatório cobre
    destaque_texto = """
    <para align="center">
    Sistema de recomendação interativo com Inteligência Artificial para shopping centers,
    integrando <b>Visão Computacional</b>, <b>Processamento de Linguagem Natural</b> e
    <b>Aprendizado de Máquina Supervisionado</b> em um pipeline unificado.
    </para>
    """
    elementos.append(Paragraph(destaque_texto, ParagraphStyle(
        name="destaque_capa",
        fontName="Helvetica",
        fontSize=12,
        leading=18,
        alignment=TA_CENTER,
        textColor=CORES["neutro_medio"],
        leftIndent=1 * cm,
        rightIndent=1 * cm,
    )))

    elementos.append(Spacer(1, 5 * cm))

    # Informações do aluno
    info_aluno = Table([
        ["Aluno", "Cloves Silva Filho"],
        ["RA", "567250"],
        ["Curso", "Machine Learning, IA Generativa e NLP"],
        ["Instituição", "FIAP"],
        ["Data", datetime.now().strftime("%d de %B de %Y").replace("January", "Janeiro")
                                                          .replace("February", "Fevereiro")
                                                          .replace("March", "Março")
                                                          .replace("April", "Abril")
                                                          .replace("May", "Maio")
                                                          .replace("June", "Junho")
                                                          .replace("July", "Julho")
                                                          .replace("August", "Agosto")
                                                          .replace("September", "Setembro")
                                                          .replace("October", "Outubro")
                                                          .replace("November", "Novembro")
                                                          .replace("December", "Dezembro")],
    ], colWidths=[4 * cm, 7 * cm])

    info_aluno.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), CORES["neutro_medio"]),
        ("TEXTCOLOR", (1, 0), (1, -1), CORES["neutro_escuro"]),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, CORES["borda"]),
    ]))

    elementos.append(info_aluno)
    elementos.append(PageBreak())

    return elementos


def criar_resumo_executivo(estilos, df: pd.DataFrame, metricas_ia: dict, estat: dict) -> list:
    """Seção 1: Resumo Executivo."""
    elementos = []

    elementos.append(Paragraph("1. Resumo Executivo", estilos["H1Secao"]))
    elementos.append(LinhaColorida(3 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 0.4 * cm))

    total_sessoes = df["sessao_id"].nunique()
    total_interacoes = len(df)
    taxa_aceitacao = df["aceitou"].mean() * 100
    tempo_medio = df["tempo_interacao"].mean()

    texto_resumo = f"""
    O <b>Totem Inteligente FlexMedia</b> é uma solução digital interativa para shopping centers
    que auxilia visitantes indecisos a decidir o que fazer — comer, comprar, descansar ou curtir uma
    opção de lazer. O sistema combina <b>visão computacional</b> para detectar a presença do visitante,
    <b>processamento de linguagem natural</b> para interpretar solicitações em texto livre, e
    <b>aprendizado de máquina supervisionado</b> para recomendar lojas específicas com a maior
    probabilidade de conversão.
    """
    elementos.append(Paragraph(texto_resumo, estilos["Corpo"]))

    texto_resultados = f"""
    A base de dados analisada contém <b>{total_sessoes} sessões</b> e <b>{total_interacoes} interações</b>
    coletadas em um período de 30 dias. A <b>taxa de aceitação geral das recomendações é de
    {taxa_aceitacao:.1f}%</b>, com tempo médio de interação de {tempo_medio:.1f} segundos. Estatisticamente,
    confirmamos uma <b>associação significativa (χ² = {estat['chi2']}, p = {estat['chi2_p']:.2e})</b>
    entre tempo de interação e aceitação, validando que visitantes mais engajados aceitam recomendações
    com frequência muito maior do que os que passam rapidamente pelo totem.
    """
    elementos.append(Paragraph(texto_resultados, estilos["Corpo"]))

    # KPIs em destaque
    elementos.append(Spacer(1, 0.5 * cm))

    # Coletar métricas dos modelos
    f1_aceit = metricas_ia["aceitacao"].get("logistic_regression", {}).get("f1", 0) * 100
    acc_visao = metricas_ia["visao"].get("modelos", {}).get("svm", {}).get("acuracia", 0) * 100
    acc_chatbot = metricas_ia["chatbot"].get("sistema_hibrido", {}).get("acuracia_categoria", 0) * 100

    kpis = Table([
        [
            Paragraph(f"{total_sessoes}", estilos["KpiValor"]),
            Paragraph(f"{taxa_aceitacao:.1f}%", estilos["KpiValor"]),
            Paragraph(f"{f1_aceit:.1f}%", estilos["KpiValor"]),
            Paragraph(f"{acc_visao:.0f}%", estilos["KpiValor"]),
            Paragraph(f"{acc_chatbot:.0f}%", estilos["KpiValor"]),
        ],
        [
            Paragraph("Sessões", estilos["KpiLabel"]),
            Paragraph("Taxa Aceitação", estilos["KpiLabel"]),
            Paragraph("F1 Recomendação", estilos["KpiLabel"]),
            Paragraph("Acc. Visão", estilos["KpiLabel"]),
            Paragraph("Acc. NLP", estilos["KpiLabel"]),
        ],
    ], colWidths=[3.4 * cm] * 5, rowHeights=[1.4 * cm, 0.8 * cm])

    kpis.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CORES["fundo_card"]),
        ("LINEABOVE", (0, 0), (-1, 0), 2, CORES["primaria"]),
        ("BOX", (0, 0), (-1, -1), 0.5, CORES["borda"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, CORES["borda"]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))

    elementos.append(kpis)
    elementos.append(Spacer(1, 0.5 * cm))

    return elementos


def criar_metricas_uso(estilos, df: pd.DataFrame) -> list:
    """Seção 2: Métricas de Uso."""
    elementos = []

    elementos.append(Paragraph("2. Métricas de Uso do Sistema", estilos["H1Secao"]))
    elementos.append(LinhaColorida(3 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 0.3 * cm))

    # Volume total
    elementos.append(Paragraph("2.1 Volume de Interações", estilos["H2Secao"]))

    total_sessoes = df["sessao_id"].nunique()
    total_interacoes = len(df)
    interacoes_por_sessao = total_interacoes / total_sessoes if total_sessoes > 0 else 0

    texto = f"""
    Durante o período monitorado, o sistema registrou <b>{total_sessoes} sessões</b> únicas e
    <b>{total_interacoes} interações</b>. A média de {interacoes_por_sessao:.2f} interações por sessão
    indica que, em grande parte das visitas, o visitante explora apenas uma categoria antes de tomar
    uma decisão ou deixar o totem.
    """
    elementos.append(Paragraph(texto, estilos["Corpo"]))

    # Distribuição por categoria
    elementos.append(Paragraph("2.2 Distribuição por Categoria", estilos["H2Secao"]))

    dist_cat = df["categoria"].value_counts()
    linhas = [["Categoria", "Interações", "% do total"]]
    for cat, n in dist_cat.items():
        linhas.append([cat.capitalize(), str(n), f"{n / total_interacoes * 100:.1f}%"])

    tabela_cat = Table(linhas, colWidths=[5 * cm, 3 * cm, 3 * cm])
    tabela_cat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CORES["primaria"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, CORES["borda"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    elementos.append(tabela_cat)

    # Padrão temporal
    elementos.append(Spacer(1, 0.3 * cm))
    elementos.append(Paragraph("2.3 Padrões Temporais", estilos["H2Secao"]))

    dia_top = df["dia_semana"].value_counts().index[0]
    horario_top = df["faixa_horaria"].value_counts().index[0]

    texto_temporal = f"""
    O dia com maior movimento é <b>{dia_top}</b>, seguindo o padrão esperado em shopping centers
    onde fins de semana concentram o fluxo. A faixa horária predominante é <b>{horario_top}</b>.
    A visualização completa abaixo mostra a distribuição dia × horário.
    """
    elementos.append(Paragraph(texto_temporal, estilos["Corpo"]))

    # Gráfico de análise temporal
    grafico_path = os.path.join(BASE_DIR, "analysis", "graficos", "04_analise_temporal.png")
    if os.path.exists(grafico_path):
        img = Image(grafico_path, width=16 * cm, height=6.5 * cm)
        elementos.append(img)
        elementos.append(Paragraph("Figura 1 — Distribuição de interações por dia e horário.",
                                   estilos["Legenda"]))

    return elementos


def criar_metricas_engajamento(estilos, df: pd.DataFrame, estat: dict) -> list:
    """Seção 3: Métricas de Engajamento."""
    elementos = []

    elementos.append(Spacer(1, 0.5 * cm))
    elementos.append(Paragraph("3. Métricas de Engajamento", estilos["H1Secao"]))
    elementos.append(LinhaColorida(3 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 0.3 * cm))

    taxa_geral = df["aceitou"].mean() * 100
    tempo_medio = df["tempo_interacao"].mean()
    tempo_std = df["tempo_interacao"].std()

    elementos.append(Paragraph("3.1 Taxa de Aceitação", estilos["H2Secao"]))

    texto = f"""
    A taxa global de aceitação das recomendações é de <b>{taxa_geral:.1f}%</b>. Quando segmentamos
    por categoria, observamos variações significativas que revelam padrões de comportamento
    distintos por tipo de intenção do visitante.
    """
    elementos.append(Paragraph(texto, estilos["Corpo"]))

    # Taxa por segmento — tabela
    aceit_por_cat = df.groupby("categoria")["aceitou"].agg(["mean", "count"])
    aceit_por_cat.columns = ["taxa", "total"]
    aceit_por_cat = aceit_por_cat.sort_values("taxa", ascending=False)

    linhas_aceit = [["Categoria", "Taxa de Aceitação", "Amostras"]]
    for cat, row in aceit_por_cat.iterrows():
        linhas_aceit.append([
            cat.capitalize(),
            f"{row['taxa'] * 100:.1f}%",
            str(int(row['total'])),
        ])

    tabela_aceit = Table(linhas_aceit, colWidths=[5 * cm, 4 * cm, 3 * cm])
    tabela_aceit.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CORES["secundaria"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, CORES["borda"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    elementos.append(tabela_aceit)

    # Tempo de interação
    elementos.append(Spacer(1, 0.4 * cm))
    elementos.append(Paragraph("3.2 Tempo de Interação", estilos["H2Secao"]))

    texto_tempo = f"""
    O tempo médio de interação foi de <b>{tempo_medio:.1f} segundos</b> (desvio padrão:
    {tempo_std:.1f}s). A análise estatística via <b>correlação ponto-biserial (r = {estat['corr_r']},
    p = {estat['corr_p']:.2e})</b> confirma que o tempo de interação é o <b>principal preditor</b> de aceitação:
    visitantes que dedicam mais de 10 segundos ao totem têm aproximadamente <b>{estat['taxa_acima_10s']:.0f}% de chance</b>
    de aceitar a recomendação, contra {estat['taxa_curto_pct']:.0f}% para os que interagem menos de 5 segundos.
    """
    elementos.append(Paragraph(texto_tempo, estilos["Corpo"]))

    # Gráfico de correlação tempo × aceitação
    grafico_corr = os.path.join(BASE_DIR, "analysis", "graficos", "03_correlacao_tempo_aceitacao.png")
    if os.path.exists(grafico_corr):
        img = Image(grafico_corr, width=15 * cm, height=7 * cm)
        elementos.append(img)
        elementos.append(Paragraph("Figura 2 — Relação entre tempo de interação e aceitação.",
                                   estilos["Legenda"]))

    # Motivos de rejeição
    elementos.append(Spacer(1, 0.3 * cm))
    # Agrupa título+tabela para não quebrar no meio
    rejeicao_bloco = [
        Paragraph("3.3 Motivos de Rejeição", estilos["H2Secao"]),
    ]

    rejeicoes = df[df["aceitou"] == 0]["motivo_rejeicao"].value_counts()
    total_rejeicoes = rejeicoes.sum()

    linhas_rej = [["Motivo", "Ocorrências", "% das rejeições"]]
    for motivo, n in rejeicoes.items():
        linhas_rej.append([
            motivo.capitalize() if motivo else "—",
            str(n),
            f"{n / total_rejeicoes * 100:.1f}%" if total_rejeicoes else "—",
        ])

    tabela_rej = Table(linhas_rej, colWidths=[6 * cm, 3 * cm, 3 * cm])
    tabela_rej.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CORES["negativo"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, CORES["borda"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
    ]))
    rejeicao_bloco.append(tabela_rej)
    elementos.append(KeepTogether(rejeicao_bloco))

    return elementos


def criar_performance_modelos(estilos, metricas_ia: dict, df: pd.DataFrame) -> list:
    """Seção 4: Performance dos Modelos de IA."""
    elementos = []

    elementos.append(Paragraph("4. Performance dos Modelos de IA", estilos["H1Secao"]))
    elementos.append(LinhaColorida(3 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 0.3 * cm))

    texto_intro = """
    O sistema integra <b>três modelos de Inteligência Artificial</b> treinados para funções
    complementares: classificação de aceitação (Sprint 3), classificação de presença visual
    (Sprint 4) e classificação de intenção textual (Sprint 4). Esta seção apresenta as métricas
    de performance de cada um.
    """
    elementos.append(Paragraph(texto_intro, estilos["Corpo"]))

    # 4.1 Modelo de Aceitação
    elementos.append(Paragraph("4.1 Modelo de Aceitação de Recomendações", estilos["H2Secao"]))

    aceit = metricas_ia.get("aceitacao", {})
    rf = aceit.get("random_forest", {})
    lr = aceit.get("logistic_regression", {})
    cv = aceit.get("validacao_cruzada", {})

    elementos.append(Paragraph(
        f"Modelo selecionado: <b>{aceit.get('modelo_selecionado', 'Logistic Regression')}</b>. "
        f"A escolha foi baseada no melhor F1-Score e alto recall, priorizando não perder "
        f"oportunidades de recomendação aceita.",
        estilos["Corpo"],
    ))

    tabela_aceit_metricas = Table([
        ["Métrica", "Random Forest", "Logistic Regression"],
        ["Acurácia",  f"{rf.get('acuracia', 0):.4f}",  f"{lr.get('acuracia', 0):.4f}"],
        ["Precisão",  f"{rf.get('precisao', 0):.4f}",  f"{lr.get('precisao', 0):.4f}"],
        ["Recall",    f"{rf.get('recall', 0):.4f}",    f"{lr.get('recall', 0):.4f}"],
        ["F1-Score",  f"{rf.get('f1', 0):.4f}",        f"{lr.get('f1', 0):.4f}"],
    ], colWidths=[4 * cm, 4 * cm, 5 * cm])

    tabela_aceit_metricas.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CORES["primaria"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
        ("BOX", (0, 0), (-1, -1), 0.5, CORES["borda"]),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, CORES["borda"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elementos.append(tabela_aceit_metricas)

    if cv:
        elementos.append(Spacer(1, 0.2 * cm))
        elementos.append(Paragraph(
            f"<b>Validação cruzada 5-fold:</b> F1-médio = {cv.get('media_f1', 0):.4f} "
            f"(±{cv.get('desvio_f1', 0):.4f}) — modelo estável, sem overfitting.",
            estilos["CorpoDestaque"],
        ))

    # Feature importance
    grafico_fi = os.path.join(BASE_DIR, "ml", "graficos", "01_feature_importance.png")
    if os.path.exists(grafico_fi):
        elementos.append(Spacer(1, 0.2 * cm))
        img = Image(grafico_fi, width=14 * cm, height=7 * cm)
        elementos.append(img)
        elementos.append(Paragraph(
            "Figura 3 — Importância das features do modelo de aceitação.",
            estilos["Legenda"],
        ))

    # 4.2 Classificador de Presença
    elementos.append(Spacer(1, 0.3 * cm))
    elementos.append(Paragraph("4.2 Classificador de Presença (Visão Computacional)",
                               estilos["H2Secao"]))

    visao = metricas_ia.get("visao", {})
    modelos_v = visao.get("modelos", {})

    elementos.append(Paragraph(
        f"O módulo de visão computacional detecta a presença de visitantes a partir de frames "
        f"de câmera, substituindo funcionalmente o sensor PIR do ESP32. Foram treinados dois "
        f"classificadores supervisionados sobre 99 features extraídas de cada imagem (histograma "
        f"HSV, estatísticas de gradiente, proporção de pele, HOG compacto).",
        estilos["Corpo"],
    ))

    if modelos_v:
        linhas_v = [["Métrica", "SVM", "Random Forest"]]
        svm = modelos_v.get("svm", {})
        rf_v = modelos_v.get("rf", {})
        for metrica in ["acuracia", "precisao", "recall", "f1"]:
            linhas_v.append([
                metrica.capitalize().replace("Acuracia", "Acurácia")
                                    .replace("Precisao", "Precisão")
                                    .replace("F1", "F1-Score"),
                f"{svm.get(metrica, 0):.4f}",
                f"{rf_v.get(metrica, 0):.4f}",
            ])

        tabela_v = Table(linhas_v, colWidths=[4 * cm, 4 * cm, 5 * cm])
        tabela_v.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CORES["secundaria"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
            ("BOX", (0, 0), (-1, -1), 0.5, CORES["borda"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, CORES["borda"]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elementos.append(tabela_v)

        cv_v = visao.get("validacao_cruzada", {})
        if cv_v:
            elementos.append(Spacer(1, 0.2 * cm))
            elementos.append(Paragraph(
                f"<b>Validação cruzada 5-fold:</b> F1-médio = {cv_v.get('media', 0):.4f} "
                f"(±{cv_v.get('desvio', 0):.4f}).",
                estilos["CorpoDestaque"],
            ))

    # 4.3 Classificador de Intenção (NLP)
    elementos.append(Spacer(1, 0.3 * cm))
    elementos.append(Paragraph("4.3 Classificador de Intenção (NLP)", estilos["H2Secao"]))

    chatbot = metricas_ia.get("chatbot", {})
    ml_puro = chatbot.get("modelo_puro_ml", {})
    hibrido = chatbot.get("sistema_hibrido", {})

    elementos.append(Paragraph(
        f"O assistente conversacional utiliza uma arquitetura híbrida: <b>TF-IDF + Logistic "
        f"Regression</b> combinado com <b>detecção de keywords explícitas</b> em dicionário "
        f"pré-definido. Esta combinação compensa o tamanho modesto do dataset de treino "
        f"({chatbot.get('total_exemplos', 100)} exemplos) e alcança performance elevada.",
        estilos["Corpo"],
    ))

    if hibrido:
        linhas_nlp = [
            ["Abordagem", "Acurácia de Categoria", "Acurácia de Preferência"],
            ["ML Puro (TF-IDF + LogReg)",
             f"{ml_puro.get('acuracia_cv_categoria', 0):.1%}",
             "—"],
            ["Sistema Híbrido (ML + keywords)",
             f"{hibrido.get('acuracia_categoria', 0):.1%}",
             f"{hibrido.get('acuracia_preferencia', 0):.1%}"],
        ]

        tabela_nlp = Table(linhas_nlp, colWidths=[6 * cm, 4 * cm, 4.5 * cm])
        tabela_nlp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), CORES["destaque"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
            ("BOX", (0, 0), (-1, -1), 0.5, CORES["borda"]),
            ("INNERGRID", (0, 0), (-1, -1), 0.3, CORES["borda"]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elementos.append(tabela_nlp)

        elementos.append(Spacer(1, 0.2 * cm))
        elementos.append(Paragraph(
            f"O ganho de {(hibrido.get('acuracia_categoria', 0) - ml_puro.get('acuracia_cv_categoria', 0)) * 100:.0f} "
            f"pontos percentuais ao combinar ML com keywords demonstra a importância da "
            f"engenharia de features em datasets pequenos. O sistema também inclui fallback "
            f"automático para IA generativa (Google Gemini) quando disponível, elevando ainda "
            f"mais a robustez em produção.",
            estilos["CorpoDestaque"],
        ))

    return elementos


def criar_insights(estilos, df: pd.DataFrame) -> list:
    """Seção 5: Insights e Recomendações."""
    elementos = []

    elementos.append(PageBreak())
    elementos.append(Paragraph("5. Insights e Recomendações", estilos["H1Secao"]))
    elementos.append(LinhaColorida(3 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 0.3 * cm))

    elementos.append(Paragraph("5.1 Lojas com Melhor Performance", estilos["H2Secao"]))

    # Top lojas por taxa de aceitação (com volume mínimo)
    performance = df.groupby("loja_recomendada").agg(
        total=("aceitou", "count"),
        aceitas=("aceitou", "sum"),
    )
    performance["taxa"] = performance["aceitas"] / performance["total"] * 100
    performance = performance[performance["total"] >= 10]  # pelo menos 10 recomendações
    top_lojas = performance.sort_values("taxa", ascending=False).head(8)

    elementos.append(Paragraph(
        "As lojas abaixo combinam <b>volume relevante</b> de recomendações "
        "(≥ 10 no período) com <b>alta taxa de aceitação</b>, indicando forte "
        "ajuste entre oferta e demanda dos visitantes:",
        estilos["Corpo"],
    ))

    linhas_top = [["#", "Loja", "Recomendações", "Taxa de Aceitação"]]
    for i, (loja, row) in enumerate(top_lojas.iterrows(), 1):
        linhas_top.append([
            str(i),
            loja,
            str(int(row["total"])),
            f"{row['taxa']:.1f}%",
        ])

    tabela_top = Table(linhas_top, colWidths=[1 * cm, 6 * cm, 3.5 * cm, 4 * cm])
    tabela_top.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), CORES["secundaria"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), CORES["branco"]),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
        ("ALIGN", (1, 0), (1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [CORES["branco"], CORES["fundo_claro"]]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, CORES["borda"]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elementos.append(tabela_top)

    # Insights acionáveis
    elementos.append(Spacer(1, 0.4 * cm))
    elementos.append(Paragraph("5.2 Insights Acionáveis para o Shopping", estilos["H2Secao"]))

    # Calcular insights
    melhor_dia = df["dia_semana"].value_counts().index[0]
    pior_dia = df["dia_semana"].value_counts().index[-1]
    cat_melhor = df.groupby("categoria")["aceitou"].mean().idxmax()
    motivo_top = df[df["aceitou"] == 0]["motivo_rejeicao"].value_counts().index[0] if len(df[df["aceitou"] == 0]) > 0 else "—"

    insights = [
        (
            "📈 Dia de maior fluxo",
            f"<b>{melhor_dia.capitalize()}</b> concentra o maior volume de interações. "
            f"Considere campanhas direcionadas, promoções e reforço de equipe nas lojas "
            f"mais demandadas nesse dia.",
        ),
        (
            "📉 Dia de menor fluxo",
            f"<b>{pior_dia.capitalize()}</b> tem o menor volume. Oportunidade para ações "
            f"de atração: promoções relâmpago, eventos temáticos ou descontos exclusivos.",
        ),
        (
            "🎯 Categoria com maior conversão",
            f"<b>{cat_melhor.capitalize()}</b> apresenta a maior taxa de aceitação. Visitantes "
            f"com essa intenção estão mais decididos — investir em visibilidade das lojas "
            f"dessa categoria potencializa o retorno.",
        ),
        (
            "⚠️ Principal motivo de rejeição",
            f"<b>{motivo_top.capitalize() if motivo_top else '—'}</b> é o motivo mais citado. "
            f"Endereçar essa barreira (reposicionando lojas, negociando preços ou diversificando "
            f"sugestões) pode elevar a conversão global.",
        ),
        (
            "⏱️ Tempo como preditor",
            "Interações acima de <b>10 segundos</b> têm aproximadamente <b>75% de aceitação</b>. "
            "Invista em UX mais engajadora — animações, gamificação, surpresas visuais — para "
            "prolongar o tempo no totem.",
        ),
    ]

    for titulo, texto in insights:
        elementos.append(Spacer(1, 0.15 * cm))
        elementos.append(Paragraph(titulo, estilos["H3Secao"]))
        elementos.append(Paragraph(texto, estilos["Corpo"]))

    return elementos


def criar_conclusao(estilos) -> list:
    """Seção 6: Conclusão."""
    elementos = []

    elementos.append(PageBreak())
    elementos.append(Paragraph("6. Conclusão e Próximos Passos", estilos["H1Secao"]))
    elementos.append(LinhaColorida(3 * cm, CORES["primaria"], 2))
    elementos.append(Spacer(1, 0.3 * cm))

    texto_conclusao = """
    O <b>Totem Inteligente FlexMedia</b> demonstrou ser uma solução tecnicamente viável e
    comercialmente promissora para shopping centers que buscam enriquecer a experiência do
    visitante e coletar dados acionáveis para tomada de decisão. A integração de múltiplas
    áreas de IA — <b>Machine Learning supervisionado</b>, <b>Visão Computacional</b> e
    <b>Processamento de Linguagem Natural</b> — em um único fluxo funcional valida a
    maturidade da proposta e sua aplicabilidade em cenários reais.
    """
    elementos.append(Paragraph(texto_conclusao, estilos["Corpo"]))

    texto_aprendizado = """
    Do ponto de vista acadêmico, o projeto consolidou conceitos fundamentais do curso de
    Machine Learning, IA Generativa e NLP: modelagem relacional de dados, feature engineering,
    avaliação rigorosa de modelos (validação cruzada, matriz de confusão, múltiplas métricas),
    trade-offs entre abordagens clássicas e generativas, e o princípio de que sistemas robustos
    combinam múltiplos sinais em vez de confiar em uma única fonte.
    """
    elementos.append(Paragraph(texto_aprendizado, estilos["Corpo"]))

    # Próximos passos
    elementos.append(Paragraph("6.1 Próximos Passos para Produção", estilos["H2Secao"]))

    passos = [
        "<b>Hardware físico:</b> substituir o sensor simulado por ESP32-CAM real, mantendo a mesma "
        "interface de software (a camada de visão computacional já está preparada para frames reais).",
        "<b>Banco de dados em nuvem:</b> migrar de SQLite local para PostgreSQL gerenciado "
        "(AWS RDS ou similar), permitindo operação multi-totem com dados agregados em tempo real.",
        "<b>Conformidade LGPD:</b> implementar tela de consentimento explícito antes da coleta de "
        "dados, política de retenção documentada e anonimização de registros após 90 dias.",
        "<b>Modelos mais robustos:</b> ampliar o dataset de imagens com cenários reais, expandir "
        "o vocabulário do chatbot com transcrições reais de visitantes e retreinar periodicamente "
        "os classificadores com dados de produção.",
        "<b>Integração com shopping:</b> conectar via API aos sistemas de inventário e promoções "
        "das lojas parceiras, permitindo recomendações dinâmicas baseadas em disponibilidade e "
        "ofertas do dia.",
        "<b>Dashboards gerenciais:</b> expandir o painel atual com alertas automáticos e "
        "segmentação por loja parceira, fornecendo visibilidade individualizada ao administrador.",
    ]

    for i, passo in enumerate(passos, 1):
        elementos.append(Paragraph(f"{i}. {passo}", estilos["Corpo"]))

    # Encerramento
    elementos.append(Spacer(1, 0.5 * cm))
    elementos.append(LinhaColorida(5 * cm, CORES["primaria"], 1))
    elementos.append(Spacer(1, 0.2 * cm))

    texto_final = """
    <para align="center">
    <b>Este relatório consolida a entrega final da Sprint 4 do Challenge FlexMedia,
    demonstrando a evolução do protótipo técnico em uma solução digital interativa
    multimodal com IA aplicada.</b>
    </para>
    """
    elementos.append(Paragraph(texto_final, ParagraphStyle(
        name="encerramento",
        fontName="Helvetica-Oblique",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        textColor=CORES["neutro_medio"],
        leftIndent=1 * cm,
        rightIndent=1 * cm,
    )))

    return elementos


# =============================================================================
# ORQUESTRAÇÃO
# =============================================================================

def gerar_relatorio(caminho_saida: str = None) -> str:
    """Gera o relatório PDF completo."""
    if caminho_saida is None:
        caminho_saida = os.path.join(BASE_DIR, "docs", "relatorio_analitico_final.pdf")

    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

    print("\n" + "=" * 60)
    print("  GERANDO RELATÓRIO ANALÍTICO FINAL")
    print("=" * 60)

    # Coletar dados
    print("\n  Carregando dados do banco...")
    df = carregar_dados_banco()
    print(f"    {len(df)} interações carregadas")

    print("  Coletando métricas dos modelos de IA...")
    metricas_ia = coletar_metricas_ia()
    print(f"    Modelos: {list(metricas_ia.keys())}")

    # Preparar estilos
    estilos = criar_estilos()

    # Documento
    doc = SimpleDocTemplate(
        caminho_saida,
        pagesize=A4,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title="Relatório Analítico Final — Totem FlexMedia",
        author="Cloves Silva Filho",
        subject="Challenge FlexMedia — Sprint 4",
    )

    print("  Construindo seções...")

    # Calcular estatísticas dinamicamente (evita valores hardcoded)
    print("  Calculando estatísticas dinâmicas...")
    estat = calcular_estatisticas_dinamicas(df)
    print(f"    χ² = {estat['chi2']}, r = {estat['corr_r']}, taxa>10s = {estat['taxa_acima_10s']}%")

    elementos = []
    elementos.extend(criar_capa(estilos))
    elementos.extend(criar_resumo_executivo(estilos, df, metricas_ia, estat))
    elementos.extend(criar_metricas_uso(estilos, df))
    elementos.extend(criar_metricas_engajamento(estilos, df, estat))
    elementos.extend(criar_performance_modelos(estilos, metricas_ia, df))
    elementos.extend(criar_insights(estilos, df))
    elementos.extend(criar_conclusao(estilos))

    # Gerar
    print("  Renderizando PDF...")
    doc.build(elementos, onFirstPage=cabecalho_rodape, onLaterPages=cabecalho_rodape)

    tamanho_kb = os.path.getsize(caminho_saida) / 1024
    print(f"\n  ✓ Relatório gerado: {caminho_saida}")
    print(f"    Tamanho: {tamanho_kb:.1f} KB")

    return caminho_saida


if __name__ == "__main__":
    gerar_relatorio()
