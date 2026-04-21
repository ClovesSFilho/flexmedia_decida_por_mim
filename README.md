# Totem Inteligente "Decida por Mim" — Challenge FlexMedia

## Descrição do Projeto

Projeto desenvolvido para o **Challenge FlexMedia – FIAP** no curso de **Machine Learning, IA Generativa e NLP**.

O **Totem Inteligente "Decida por Mim"** é um sistema de recomendação interativo com Inteligência Artificial para shopping centers. Ele auxilia visitantes indecisos a decidir o que fazer — comer, comprar, descansar ou curtir uma opção de lazer — e sugere lojas específicas com a maior probabilidade de aceitação para o perfil de cada visitante.

O sistema integra **três módulos de IA** em um pipeline unificado:
- **Visão Computacional** (OpenCV + MediaPipe) para detecção de presença
- **Processamento de Linguagem Natural** (TF-IDF + Logistic Regression + keywords) para interpretar texto livre
- **Aprendizado de Máquina Supervisionado** (Logistic Regression, F1=0.80) para recomendar lojas personalizadas

---

## Sprint 4 — Evolução do Protótipo

A Sprint 4 evoluiu o protótipo técnico da Sprint 3 em uma **solução digital interativa**, adicionando:

| Módulo | Descrição | Tecnologia |
|--------|-----------|-----------|
| Interface do Totem | Experiência completa do visitante em 6 telas | Streamlit |
| Visão Computacional | Detecção de presença humana via câmera/imagem | OpenCV, MediaPipe, SVM |
| Chatbot | Interpretação de texto natural do visitante | TF-IDF, LogReg, Keywords |
| Sistema de Recomendação | Sugere a loja com maior P(aceitação) para o perfil | Logistic Regression invertida |
| Relatório Analítico | PDF executivo de 8 páginas com métricas e insights | ReportLab |

---

## Arquitetura do Sistema (Sprint 4)

```
┌─────────────────────────────────────────────────────────────┐
│                  TOTEM INTELIGENTE FLEXMEDIA                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Câmera/Imagem] → Visão Computacional → Detecta presença   │
│                    (HOG + MediaPipe + SVM)       │           │
│                                                  ▼           │
│  [Visitante] → Interface Totem (Streamlit) → Chatbot IA     │
│                                                  │           │
│                                                  ▼           │
│                                          Motor de            │
│                                          Recomendação        │
│                                          (ML preditivo)      │
│                                                  │           │
│                                                  ▼           │
│                                           Banco SQLite       │
│                                                  │           │
│                                     ┌────────────┴────┐     │
│                                     ▼                 ▼     │
│                               Dashboard          Relatório  │
│                               Analítico          Final PDF  │
│                              (Streamlit)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Estrutura do Projeto

```
flexmedia-totem-ia/
│
├── database/
│   └── create_database.py               # Banco SQLite (3 tabelas, FKs, constraints)
│
├── sensors/
│   └── sensor_simulado.py               # Gerador de dados com regras de correlação
│
├── data/
│   └── interacoes.db                    # Banco (gerado + dados ao vivo do totem)
│
├── vision/                              # ★ SPRINT 4 — Visão Computacional
│   ├── detector_presenca.py             # Detector híbrido HOG + MediaPipe
│   ├── dataset/
│   │   ├── com_presenca/                # 20 imagens com visitante
│   │   └── sem_presenca/                # 15 imagens cenário vazio
│   └── resultados/                      # Imagens anotadas com bounding boxes
│
├── chatbot/                             # ★ SPRINT 4 — NLP / Chatbot
│   ├── assistente_totem.py              # Classificador híbrido (ML + keywords + Gemini)
│   ├── exemplos_intencoes.json          # Dataset 100 frases rotuladas (pt-BR)
│   └── classificador_intencao.joblib    # Modelo NLP treinado
│
├── totem/                               # ★ SPRINT 4 — Interface Interativa
│   └── app_totem.py                     # Experiência do visitante (6 telas)
│
├── ml/
│   ├── modelo_ml.py                     # Treinamento RF + LR (Sprint 3, atualizado)
│   ├── modelo_recomendacao.py           # ★ Motor de recomendação ativa
│   ├── classificador_visao.py           # ★ Classificador de presença (SVM + RF)
│   ├── modelo_treinado.joblib           # LR selecionada (F1=0.800)
│   ├── modelo_completo.joblib           # ★ Modelo + encoders (usado pelo totem)
│   ├── classificador_presenca.joblib    # ★ SVM de presença
│   ├── metricas.json                    # Métricas do modelo de aceitação
│   ├── metricas_visao.json              # ★ Métricas do classificador de presença
│   ├── metricas_chatbot.json            # ★ Métricas do classificador NLP
│   └── graficos/                        # 6 PNGs (3 Sprint 3 + 3 Sprint 4)
│
├── analysis/
│   ├── analise_estatistica.py           # Análise completa (chi-quadrado, correlação)
│   ├── relatorio_analitico.py           # ★ Gerador do relatório PDF final
│   └── graficos/                        # 5 PNGs estatísticos
│
├── dashboard/
│   └── app_streamlit.py                 # Dashboard administrativo
│
├── docs/
│   ├── documentacao_tecnica.md          # Documentação técnica detalhada
│   └── relatorio_analitico_final.pdf    # ★ Relatório executivo (8 páginas)
│
├── diagrams/
│   └── arquitetura.png                  # Diagrama da arquitetura
│
└── README.md
```

Arquivos marcados com ★ são novos na Sprint 4.

---

## Modelos de IA Treinados

### 1. Modelo de Aceitação de Recomendações (Sprint 3)

Prevê se o visitante aceitará a recomendação. Na Sprint 4, é usado de forma invertida pelo Motor de Recomendação para encontrar a loja com maior P(aceitação).

| Métrica | Random Forest | Logistic Regression |
|---------|:---:|:---:|
| Acurácia | 0.688 | **0.694** |
| Precisão | **0.718** | 0.710 |
| Recall | 0.875 | **0.917** |
| F1-Score | 0.789 | **0.800** |

Modelo selecionado: **Logistic Regression** (F1=0.800). Validação cruzada 5-fold: F1 médio=0.811 (±0.023).

### 2. Classificador de Presença Visual (Sprint 4)

Detecta presença humana em imagens usando features extraídas (histograma HSV, gradientes, HOG) + classificador supervisionado.

| Métrica | SVM | Random Forest |
|---------|:---:|:---:|
| Acurácia | **1.000** | 1.000 |
| F1-Score | **1.000** | 1.000 |

Validação cruzada 5-fold: F1 médio=0.960 (±0.080). Dataset: 35 imagens sintéticas (20 com presença, 15 sem).

### 3. Classificador de Intenção NLP (Sprint 4)

Interpreta texto livre do visitante e extrai categoria + preferência. Arquitetura híbrida: TF-IDF + Logistic Regression + dicionário de keywords.

| Abordagem | Acurácia Categoria | Acurácia Preferência |
|-----------|:---:|:---:|
| ML Puro (TF-IDF + LogReg) | 64.0% | — |
| **Sistema Híbrido (ML + keywords)** | **98.0%** | **97.0%** |

Dataset: 100 frases rotuladas em português cobrindo 19 combinações categoria×preferência.

---

## Interface do Totem (Sprint 4)

O visitante interage com o totem através de 8 estados de tela:

1. **Detecção de Presença** — câmera/imagem detecta visitante via visão computacional
2. **Boas-vindas** — visitante informa faixa etária
3. **Seleção de Categoria** — escolhe categoria por botão OU conversa com o chatbot em texto livre
4. **Seleção de Preferência** — refina a escolha (ou já foi detectada pelo chatbot)
5. **Recomendação** — modelo ML sugere a melhor loja com % de compatibilidade e ranking de alternativas
6. **Motivo de Rejeição** — caso recuse, registra o motivo para análise
7. **Agradecimento** — confirma o feedback e permite nova consulta
8. **Encerramento** — resumo da sessão e despedida

---

## Dashboard Interativo

Dashboard administrativo em **Streamlit** com layout wide:

- Sidebar com filtros (categoria, faixa etária, horário, dia, período)
- 5 KPIs em cards
- Gráficos: linha temporal, pizza de categorias, aceitação segmentada, heatmap dia×horário, correlação tempo×aceitação
- Ranking de lojas e motivos de rejeição
- Seção de resultados de ML carregada dinamicamente
- Tabela de dados brutos

---

## Relatório Analítico Final

Relatório executivo em PDF de 8 páginas gerado automaticamente com dados do banco e métricas dos 3 modelos de IA. Conteúdo:

1. Resumo Executivo com 5 KPIs visuais
2. Métricas de Uso (volume, distribuição, padrões temporais)
3. Métricas de Engajamento (aceitação, tempo, rejeição)
4. Performance dos Modelos de IA (aceitação, visão, NLP)
5. Insights e Recomendações acionáveis para o shopping
6. Conclusão e Próximos Passos para produção

---

## Como Executar

### Pré-requisitos

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy streamlit joblib opencv-python mediapipe reportlab pypdfium2
```

### Execução do pipeline completo

```bash
# 1. Criar banco de dados
python database/create_database.py

# 2. Popular com dados simulados
python sensors/sensor_simulado.py

# 3. Rodar análise estatística
python analysis/analise_estatistica.py

# 4. Treinar modelo de ML (aceitação)
python ml/modelo_ml.py

# 5. Treinar classificador de presença (visão)
python ml/classificador_visao.py

# 6. Treinar chatbot (NLP)
python chatbot/assistente_totem.py

# 7. Testar motor de recomendação
python ml/modelo_recomendacao.py

# 8. Gerar relatório PDF
python analysis/relatorio_analitico.py

# 9. Iniciar interface do totem (visitante)
streamlit run totem/app_totem.py

# 10. Iniciar dashboard (admin)
streamlit run dashboard/app_streamlit.py
```

---

## Tecnologias Utilizadas

| Tecnologia | Finalidade |
|-----------|-----------|
| Python | Linguagem principal |
| SQLite | Banco de dados relacional |
| Pandas / NumPy | Manipulação de dados |
| Scikit-Learn | Machine Learning (LR, RF, SVM, TF-IDF) |
| OpenCV | Processamento de imagem |
| MediaPipe | Detecção facial |
| Matplotlib / Seaborn | Visualização de dados |
| SciPy | Testes estatísticos |
| Streamlit | Interface interativa e dashboard |
| ReportLab | Geração de PDF |
| Joblib | Serialização de modelos |
| Google Gemini (opcional) | IA generativa para chatbot |

---

## Segurança e Privacidade

- Múltiplas camadas de validação (CHECK constraints, FK CASCADE, tipagem)
- Dados anonimizados sem vinculação a identidades pessoais
- Considerações de LGPD documentadas para cenário de produção
- Detalhes completos na [documentação técnica](docs/documentacao_tecnica.md)

---

## Vídeo de Demonstração

Link do vídeo (YouTube não listado): *[A INSERIR]*

---

## Contexto Acadêmico

Projeto desenvolvido para o **Challenge FlexMedia – FIAP**, Sprint 4 (entrega final), com foco na aplicação prática de:

- Machine Learning supervisionado e sistema de recomendação
- Visão Computacional (detecção de presença)
- Processamento de Linguagem Natural (classificação de intenção)
- Análise estatística e testes de hipótese
- Modelagem de dados relacional
- Integração de sistemas (sensor → banco → análise → IA → visualização)
- Geração de relatórios executivos automatizados

---

## Aluno Responsável

| Cloves Silva Filho | RA567250 |
