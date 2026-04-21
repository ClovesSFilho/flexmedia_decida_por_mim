# Documentação Técnica – Totem Inteligente FlexMedia

## 1. Visão Geral do Sistema

O Totem Inteligente "Decida por Mim" é um sistema de recomendação interativo projetado para ambientes comerciais como shopping centers. O sistema auxilia visitantes indecisos, coletando intenções por meio de interações rápidas e sugerindo opções relevantes dentro do estabelecimento.

A solução integra quatro camadas funcionais em um pipeline de dados completo: coleta via sensores simulados, armazenamento relacional, análise estatística com testes de hipótese e modelos preditivos de Machine Learning, todos conectados a um dashboard interativo para tomada de decisão.

---

## 2. Arquitetura do Sistema

### 2.1 Pipeline de Dados

O sistema segue um fluxo unidirecional estruturado:

```
Visitante → Totem (sensor) → Banco de Dados SQLite → Análise Estatística → Machine Learning → Dashboard Streamlit
```

Cada etapa processa e enriquece os dados da etapa anterior:

1. **Coleta (sensors/):** O sensor simulado gera interações com regras de correlação que refletem comportamento humano realista em um shopping center.

2. **Armazenamento (database/):** Os dados são persistidos em um banco SQLite com modelagem relacional normalizada, chaves estrangeiras e constraints de validação.

3. **Análise (analysis/):** Os dados são submetidos a análise estatística exploratória completa, incluindo estatísticas descritivas, correlações, análise temporal e teste de hipótese chi-quadrado.

4. **Predição (ml/):** Dois modelos de classificação supervisionada são treinados e comparados para prever a probabilidade de aceitação de recomendações.

5. **Visualização (dashboard/):** Um dashboard interativo em Streamlit consolida todas as métricas, permitindo filtragem em tempo real e acompanhamento de KPIs.

### 2.2 Diagrama da Arquitetura

O diagrama visual da arquitetura pode ser encontrado em:

```
diagrams/arquitetura.png
```

---

## 3. Integração com Hardware (Totem Físico)

### 3.1 Componentes do Totem

O totem físico é composto por um microcontrolador **ESP32 ou ESP32-CAM** conectado a sensores de presença (PIR) e uma tela interativa (touch/display LCD). Esses componentes são responsáveis por detectar a aproximação do visitante e capturar as interações realizadas.

### 3.2 Fluxo de Funcionamento

1. O **sensor PIR** detecta a presença de um visitante próximo ao totem.
2. O **ESP32** inicia uma sessão no sistema ao registrar o evento de presença.
3. O evento é enviado via **Wi-Fi** para a aplicação Python (protocolo HTTP ou MQTT).
4. A aplicação registra a sessão no **banco de dados SQLite**.
5. O visitante interage com a **tela touch**, escolhendo uma intenção (comer, comprar, descansar ou lazer) e informando sua preferência.
6. Os dados de cada interação são armazenados e processados pelo **sistema de análise e Machine Learning**.
7. O **dashboard Streamlit** consolida as métricas e insights de uso em tempo real.

### 3.3 Simulação Acadêmica

Nesta versão acadêmica, os sensores físicos são simulados pelo script `sensors/sensor_simulado.py`, que gera eventos de interação com regras de correlação realistas. Isso permite demonstrar e validar todo o pipeline de dados (sensor → banco → análise → ML → dashboard) sem necessidade de hardware físico.

A estrutura do código foi projetada para que, em uma versão de produção, o script de simulação seja substituído por um módulo de comunicação real com o ESP32 sem alterações nas demais camadas do sistema.

---

## 4. Modelagem de Dados

### 4.1 Estrutura Relacional

O banco de dados utiliza três tabelas normalizadas com relacionamentos bem definidos:

**Tabela `sessoes`** — Representa cada visita de um usuário ao totem.

| Campo | Tipo | Constraint |
|-------|------|-----------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| inicio_sessao | TEXT | NOT NULL |
| fim_sessao | TEXT | NOT NULL |
| faixa_etaria | TEXT | NOT NULL, CHECK IN ('jovem', 'adulto', 'idoso') |
| dia_semana | TEXT | NOT NULL, CHECK IN ('segunda' ... 'domingo') |
| faixa_horaria | TEXT | NOT NULL, CHECK IN ('manha', 'almoco', 'tarde', 'noite') |

**Tabela `interacoes`** — Cada ação do usuário dentro de uma sessão.

| Campo | Tipo | Constraint |
|-------|------|-----------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| sessao_id | INTEGER | NOT NULL, FK → sessoes(id) ON DELETE CASCADE |
| timestamp | TEXT | NOT NULL |
| categoria | TEXT | NOT NULL, CHECK IN ('comer', 'comprar', 'descansar', 'lazer') |
| preferencia | TEXT | NOT NULL |
| tempo_interacao | INTEGER | NOT NULL, CHECK > 0 |

**Tabela `recomendacoes`** — Sugestão feita pelo totem e resposta do usuário.

| Campo | Tipo | Constraint |
|-------|------|-----------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT |
| interacao_id | INTEGER | NOT NULL, FK → interacoes(id) ON DELETE CASCADE |
| loja_recomendada | TEXT | NOT NULL |
| aceitou | INTEGER | NOT NULL, CHECK IN (0, 1) |
| motivo_rejeicao | TEXT | DEFAULT NULL |

### 4.2 Relacionamentos

```
sessoes (1) ──→ (N) interacoes (1) ──→ (1) recomendacoes
```

Uma sessão pode conter múltiplas interações (1 a 3 por sessão). Cada interação gera exatamente uma recomendação. A separação em três tabelas permite análises independentes por sessão, por interação e por performance de recomendação.

### 4.3 Índices

Foram criados cinco índices para otimizar as consultas mais frequentes:

- `idx_interacoes_sessao` — JOINs entre sessões e interações
- `idx_interacoes_categoria` — filtros e agrupamentos por categoria
- `idx_recomendacoes_interacao` — JOINs entre interações e recomendações
- `idx_sessoes_faixa_horaria` — análises temporais por horário
- `idx_sessoes_dia_semana` — análises temporais por dia

### 4.4 Validações e Integridade

- **CHECK constraints** garantem que apenas valores válidos sejam inseridos nos campos enumerados (faixa_etaria, dia_semana, faixa_horaria, categoria) e que tempo_interacao seja positivo.
- **FOREIGN KEYS com CASCADE** garantem que a exclusão de uma sessão remove automaticamente suas interações e recomendações associadas, evitando registros órfãos.
- **PRAGMA foreign_keys = ON** é habilitado explicitamente em todas as conexões, já que o SQLite desabilita FKs por padrão.

---

## 5. Coleta de Dados (Sensor Simulado)

### 5.1 Estratégia de Simulação

O sensor simulado gera dados com regras de correlação que reproduzem padrões de comportamento humano em um shopping center. A simulação não é puramente aleatória — ela incorpora dependências entre variáveis que tornam os dados análogos a dados reais.

### 5.2 Regras de Correlação Implementadas

**Faixa horária → Categoria:** No horário de almoço, a categoria "comer" recebe peso de 55%. À noite, "lazer" recebe 50%. Isso reflete o comportamento natural dos visitantes.

**Faixa etária → Categoria:** Jovens tendem a "lazer" (peso 45%), adultos a "comprar" (40%) e idosos a "descansar" (45%). O peso final de cada categoria é uma combinação ponderada (60% horário, 40% idade).

**Categoria → Preferência:** Cada categoria possui preferências exclusivas. "Comer" gera preferências como "doce" ou "salgado", nunca "eletrônico". Isso garante coerência semântica.

**Categoria + Preferência → Loja:** Cada combinação é mapeada a lojas relevantes. "Comer" + "doce" recomenda Cacau Show ou Starbucks, nunca Nike.

**Tempo de interação → Aceitação:** Esta é a correlação mais forte do sistema. Usuários com tempo > 15s têm probabilidade base de 75% de aceitar. Usuários com tempo < 5s têm apenas 30%. A faixa etária e a categoria ajustam marginalmente essa probabilidade.

**Dia da semana → Volume:** Sábado gera ~25 sessões/dia, quarta-feira apenas ~9, simulando o fluxo real de um shopping.

### 5.3 Volume de Dados

A simulação gera dados distribuídos ao longo de 30 dias, resultando em aproximadamente 471 sessões e 719 interações. A seed fixa (42) garante reprodutibilidade dos resultados.

---

## 6. Análise Estatística

### 6.1 Estatísticas Descritivas

O tempo de interação apresenta média de 12.0 segundos e desvio padrão de 5.5 segundos, com distribuição aproximadamente uniforme entre 2 e 25 segundos. A taxa geral de aceitação é de 66.6%.

### 6.2 Análise de Correlação

A correlação ponto-biserial entre tempo de interação e aceitação é de 0.297 (p < 0.001), indicando correlação positiva moderada e estatisticamente significativa. Usuários que aceitaram recomendações tiveram tempo médio de 13.2s, contra 9.7s dos que rejeitaram.

A taxa de aceitação por faixa de tempo demonstra progressão clara:

| Faixa de Tempo | Taxa de Aceitação | N |
|---------------|:-----------------:|:---:|
| 1-5 segundos | 36.6% | 101 |
| 6-10 segundos | 61.7% | 206 |
| 11-15 segundos | 70.8% | 216 |
| 16-20 segundos | 82.1% | 151 |
| 21+ segundos | 84.4% | 45 |

### 6.3 Teste de Hipótese

Foi aplicado o teste chi-quadrado de independência para validar a associação entre tempo de interação e aceitação.

- **H0:** Não há associação entre tempo de interação e aceitação.
- **H1:** Existe associação entre tempo de interação e aceitação.
- **Resultado:** χ² = 57.80, gl = 1, p = 2.90e-14

**Conclusão:** Rejeita-se H0. Existe associação estatisticamente significativa entre o tempo que o usuário permanece no totem e a probabilidade de aceitar a recomendação.

### 6.4 Análise Temporal

O fim de semana concentra 44% das interações (sábado = 166, domingo = 147). O dia com menor volume é quarta-feira (45 interações). A faixa horária da tarde lidera com 220 interações, seguida pelo horário de almoço com 205.

### 6.5 Performance das Lojas

A Starbucks apresenta a maior taxa de aceitação entre as lojas mais recomendadas (86.5%), seguida por McDonald's (75.0%) e Espaço Cultural (75.0%). O principal motivo de rejeição é "muito longe" (45 ocorrências), sugerindo que a proximidade é um fator relevante para a aceitação.

---

## 7. Machine Learning

### 7.1 Definição do Problema

Classificação binária supervisionada: prever se um usuário aceitará (1) ou rejeitará (0) a recomendação feita pelo totem.

### 7.2 Preparação dos Dados

As variáveis categóricas foram codificadas com LabelEncoder. As features utilizadas são: faixa_etaria, dia_semana, faixa_horaria, categoria, preferencia, tempo_interacao e loja_recomendada. A divisão treino/teste foi feita em 80/20 com estratificação pela variável alvo.

### 7.3 Encoding de Variáveis Categóricas

Foi utilizado **LabelEncoder** para converter variáveis categóricas em valores numéricos inteiros.

Embora variáveis nominais (sem ordem natural) sejam frequentemente codificadas com One-Hot Encoding para evitar que o modelo interprete uma relação ordinal inexistente, optou-se pelo LabelEncoder neste projeto por dois motivos:

1. **Compatibilidade com modelos baseados em árvore:** O Random Forest, um dos modelos treinados, realiza splits binários em cada nó. Isso significa que ele não assume ordenação linear entre os valores — ele testa limiares individuais e consegue isolar categorias específicas mesmo com encoding ordinal. Por essa razão, LabelEncoder não prejudica significativamente a performance de árvores de decisão e seus ensembles.

2. **Redução de dimensionalidade:** O One-Hot Encoding expandiria significativamente o número de features (por exemplo, `loja_recomendada` com 30+ valores únicos geraria 30+ colunas binárias), aumentando a esparsidade dos dados e o custo computacional sem ganho proporcional em um dataset desta escala.

Em um ambiente de produção com modelos sensíveis a ordinalidade (como regressão linear ou SVM), técnicas como **OneHotEncoder** ou **Target Encoding** seriam recomendadas para garantir representação adequada das variáveis categóricas.

### 7.4 Modelos Treinados

**Random Forest Classifier:** Ensemble de 100 árvores com profundidade máxima de 10 e mínimo de 5 amostras por split.

**Logistic Regression:** Modelo linear com máximo de 1000 iterações.

### 7.5 Resultados Comparativos

| Métrica | Random Forest | Logistic Regression |
|---------|:---:|:---:|
| Acurácia | 0.6875 | **0.6944** |
| Precisão | **0.7179** | 0.7097 |
| Recall | 0.8750 | **0.9167** |
| F1-Score | 0.7887 | **0.8000** |

### 7.6 Modelo Selecionado

A Logistic Regression foi selecionada como modelo final com base no F1-Score (0.800), que equilibra precisão e recall. O recall alto (91.7%) é desejável neste contexto — é mais custoso perder uma oportunidade de recomendação aceita do que fazer uma recomendação rejeitada.

### 7.7 Validação Cruzada

A validação cruzada com 5 folds retornou F1-Score médio de 0.811 com desvio padrão de 0.023, confirmando estabilidade do modelo e ausência de overfitting.

### 7.8 Importância das Features

O ranking de importância do Random Forest confirma os achados da análise estatística:

1. `tempo_interacao` — 31.1% (principal preditor)
2. `loja_recomendada` — 17.9%
3. `preferencia` — 13.8%
4. `dia_semana` — 13.4%
5. `faixa_etaria` — 8.8%
6. `faixa_horaria` — 8.5%
7. `categoria` — 6.5%

O tempo de interação domina a predição, corroborando a correlação estatística (r = 0.297, p < 0.001) e o teste chi-quadrado (p = 2.90e-14).

---

## 8. Dashboard Interativo

### 8.1 Tecnologia

O dashboard foi desenvolvido em Streamlit com layout wide e sidebar de filtros. Os gráficos utilizam Matplotlib e Seaborn.

### 8.2 Funcionalidades

- **Filtros interativos:** Categoria, faixa etária, faixa horária, dia da semana e período de datas. Todos os gráficos e métricas atualizam em tempo real ao alterar filtros.
- **KPIs em cards:** Total de interações, sessões únicas, taxa de aceitação, tempo médio e melhor categoria.
- **Gráficos:** Linha temporal, pizza de categorias, barras de aceitação segmentada (3 dimensões), heatmap dia × horário, correlação tempo × aceitação, ranking de lojas e motivos de rejeição.
- **Seção ML:** Métricas carregadas dinamicamente a partir do arquivo `ml/metricas.json`, gerado automaticamente pelo script de treinamento, e gráfico de feature importance.
- **Tabela expansível:** Dados brutos com todas as colunas para exploração manual.

### 8.3 Execução

```bash
streamlit run dashboard/app_streamlit.py
```

---

## 9. Segurança e Privacidade

### 9.1 Validação e Integridade dos Dados

O sistema implementa múltiplas camadas de validação para garantir a integridade e consistência das informações coletadas:

- **CHECK constraints no banco de dados** garantem que apenas valores válidos sejam inseridos nos campos enumerados (`faixa_etaria`, `dia_semana`, `faixa_horaria`, `categoria`), impedindo dados fora do domínio esperado.
- **Restrição de valor positivo** no campo `tempo_interacao` (constraint `> 0`), impedindo registros com tempo inválido ou zerado.
- **Restrição booleana** no campo `aceitou` (constraint `IN (0, 1)`), garantindo que apenas valores válidos de aceitação/rejeição sejam registrados.
- **Integridade referencial** com Foreign Keys configuradas com `ON DELETE CASCADE`, assegurando que a exclusão de uma sessão remove automaticamente suas interações e recomendações associadas, eliminando registros órfãos.
- **Ativação explícita de Foreign Keys** via `PRAGMA foreign_keys = ON` em todas as conexões, dado que o SQLite desabilita esse recurso por padrão.
- **Validação de tipos** antes da inserção: o sensor simulado gera dados já tipados corretamente (inteiros para tempo, strings para categorias), prevenindo erros de tipo no momento da persistência.
- **Tratamento de valores nulos**: o campo `motivo_rejeicao` aceita NULL apenas quando `aceitou = 1`, garantindo que rejeições sempre tenham motivo documentado.

### 9.2 Considerações sobre LGPD

Em um cenário de produção, o sistema deveria implementar:

- **Anonimização:** Os dados não vinculam interações a identidades pessoais. O campo `sessao_id` é um identificador sequencial sem relação com dados pessoais.
- **Consentimento:** O totem deveria exibir um aviso de coleta de dados antes da interação, com opção de recusa.
- **Retenção:** Definir política de retenção de dados com exclusão automática após período determinado.
- **Acesso mínimo:** Apenas o dashboard expõe os dados, sem acesso direto ao banco por usuários finais.

### 9.3 Reprodutibilidade e Rastreabilidade

- A seed fixa (42) na simulação garante reprodutibilidade completa dos resultados.
- Timestamps são registrados em formato ISO 8601 para evitar ambiguidades.
- O banco é recriado do zero a cada execução do pipeline, eliminando risco de dados corrompidos acumulados.

---

## 10. Tecnologias Utilizadas

| Tecnologia | Versão | Finalidade |
|-----------|--------|-----------|
| Python | 3.x | Linguagem principal |
| SQLite | 3.x | Banco de dados relacional |
| Pandas | 2.x | Manipulação e análise de dados |
| NumPy | 1.x | Operações numéricas |
| Scikit-Learn | 1.x | Modelos de Machine Learning |
| Matplotlib | 3.x | Visualização de dados |
| Seaborn | 0.13+ | Visualizações estatísticas |
| SciPy | 1.x | Testes estatísticos |
| Streamlit | 1.x | Dashboard interativo |
| Joblib | 1.x | Serialização de modelos |

---

## 11. Decisões Técnicas

### Por que SQLite e não PostgreSQL?

SQLite é embutido, não requer servidor e é adequado para prototipagem. Em produção, migraria para PostgreSQL ou MySQL com as mesmas tabelas e constraints.

### Por que Logistic Regression e não Random Forest?

Apesar do Random Forest ter melhor precisão (0.718 vs 0.710), a Logistic Regression apresentou melhor F1-Score (0.800 vs 0.789) e recall significativamente maior (0.917 vs 0.875). No contexto de recomendações, é preferível recomendar a mais (falsos positivos) do que perder oportunidades de recomendação aceita (falsos negativos).

### Por que LabelEncoder e não OneHotEncoder?

Conforme detalhado na seção 7.3, a escolha se baseia na compatibilidade do LabelEncoder com modelos baseados em árvore (Random Forest) e na redução de dimensionalidade. Em produção, com modelos lineares, OneHotEncoder ou Target Encoding seriam mais adequados.

### Por que dados simulados?

O enunciado permite dados simulados desde que coerentes e documentados. A simulação com regras de correlação garante padrões detectáveis e permite validar toda a pipeline de ponta a ponta sem necessidade de hardware físico.

---

## 12. Fluxo de Execução

Para reproduzir o projeto do zero:

```bash
# 1. Criar banco de dados
python database/create_database.py

# 2. Popular com dados simulados
python sensors/sensor_simulado.py

# 3. Rodar análise estatística
python analysis/analise_estatistica.py

# 4. Treinar modelos de ML
python ml/modelo_ml.py

# 5. Iniciar dashboard
streamlit run dashboard/app_streamlit.py
```

Cada etapa depende da anterior. O banco deve ser criado antes da simulação, que deve popular os dados antes da análise e do ML.