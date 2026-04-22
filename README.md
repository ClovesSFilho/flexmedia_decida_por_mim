# Totem Inteligente "Decida por Mim" — Sprint 4

**Challenge FlexMedia — FIAP**
**Curso:** Machine Learning, IA Generativa e NLP
**Aluno:** Cloves Silva Filho | RA 567250

---

## O que é

Um totem interativo com IA para shopping centers. O visitante chega, o sistema detecta sua presença por visão computacional, entende o que ele quer (por texto ou botões) e recomenda a loja com maior chance de agradar.

## Problema

Visitantes entram no shopping sem saber o que fazer. Administradores não têm dados sobre o que as pessoas realmente procuram.

## Solução

O totem coleta a intenção do visitante, recomenda lojas usando machine learning e gera dados de comportamento para o administrador do shopping.

---

## Como funciona

```
Câmera detecta presença → Visitante informa o que quer → IA recomenda loja → Feedback salvo no banco
                                                                                      ↓
                                                                         Dashboard + Relatório PDF
```

O sistema tem 3 modelos de IA integrados:

1. **Classificador de presença** — OpenCV + SVM, detecta se tem alguém na frente do totem (acurácia 100%, F1 CV = 0.96)
2. **Classificador de intenção** — TF-IDF + keywords, entende texto livre como "to com fome, quero chocolate" (acurácia 98%)
3. **Modelo de recomendação** — Logistic Regression invertida, testa todas as lojas e sugere a melhor para o perfil (F1 = 0.80)

---

## Tecnologias

Python, SQLite, Scikit-Learn, OpenCV, MediaPipe, Streamlit, ReportLab, Matplotlib, Seaborn, SciPy

---

## Estrutura do projeto

```
├── database/create_database.py          # Banco SQLite (3 tabelas)
├── sensors/sensor_simulado.py           # Gerador de dados simulados
├── analysis/analise_estatistica.py      # Análise com testes de hipótese
├── analysis/relatorio_analitico.py      # Gera relatório PDF de 8 páginas
├── ml/modelo_ml.py                      # Treina modelo de aceitação
├── ml/classificador_visao.py            # Treina classificador de presença
├── ml/modelo_recomendacao.py            # Motor de recomendação
├── chatbot/assistente_totem.py          # Chatbot NLP
├── totem/app_totem.py                   # Interface do visitante (Streamlit)
├── dashboard/app_streamlit.py           # Dashboard administrativo
├── vision/detector_presenca.py          # Detecção de presença (OpenCV)
├── config.py                            # Constantes centralizadas
├── docs/relatorio_analitico_final.pdf   # Relatório PDF gerado
└── docs/documentacao_tecnica.md         # Documentação técnica
```

---

## Como executar

```bash
pip install -r requirements.txt

python database/create_database.py       # 1. Criar banco
python sensors/sensor_simulado.py        # 2. Popular dados
python analysis/analise_estatistica.py   # 3. Análise estatística
python ml/modelo_ml.py                   # 4. Treinar modelo de aceitação
python ml/classificador_visao.py         # 5. Treinar classificador de presença
python chatbot/assistente_totem.py       # 6. Treinar chatbot
python ml/modelo_recomendacao.py         # 7. Testar recomendação
python analysis/relatorio_analitico.py   # 8. Gerar relatório PDF

streamlit run totem/app_totem.py         # Interface do visitante
streamlit run dashboard/app_streamlit.py # Dashboard admin
```

---

## Resultados principais

| Modelo | Métrica | Valor |
|--------|---------|-------|
| Aceitação (Logistic Regression) | F1-Score | 0.800 |
| Presença (SVM) | Acurácia | 100% |
| Intenção NLP (híbrido) | Acurácia | 98% |
| Validação cruzada (aceitação) | F1 médio 5-fold | 0.811 |

A variável mais importante para prever aceitação é o **tempo de interação** (31% de importância), confirmado pelo teste chi-quadrado (p < 0.001).

---

## Vídeo de demonstração

**https://youtu.be/tST5qdB795g**

---

## Contexto acadêmico

Projeto do Challenge FlexMedia — Sprint 4 (entrega final). Dados simulados com seed fixa para reprodutibilidade. Detalhes completos na [documentação técnica](docs/documentacao_tecnica.md).
