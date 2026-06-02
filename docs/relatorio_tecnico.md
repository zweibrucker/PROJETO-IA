# Relatório Técnico — Classificação de Acidentes PRF

**Disciplina:** Inteligência Artificial  
**Bloco:** B — Classificação/Predição  
**Sistema-alvo:** PRF — Acidentes em Rodovias Federais  
**Data:** 2026-06-02  
**Repositório:** https://github.com/0x03c1/projeto-ml

---

## 1. Justificativa da Escolha

### 1.1 Por que PRF?

A Polícia Rodoviária Federal registra mais de **65.000 acidentes por ano** nas rodovias federais brasileiras, com impacto direto em vidas humanas e infraestrutura. O dataset público disponibilizado pelo governo federal é rico em atributos categóricos e numéricos que permitem aplicação de técnicas de aprendizado de máquina supervisionado.

A escolha do sistema PRF se justifica por:
- **Dados abertos e atualizados:** publicados anualmente em https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
- **Relevância social:** classificação de gravidade apoia triagem de emergências e políticas de segurança viária
- **Adequação ao Bloco B:** o problema de classificação em três classes (Sem Vítimas, Com Vítimas Feridas, Com Vítimas Fatais) é um caso clássico de classificação multiclasse supervisionada

### 1.2 Por que Classificação?

A variável `classificacao_acidente` é uma variável categórica ordinal com três classes que refletem a severidade crescente do acidente. A tarefa de classificação é a mais adequada porque:
- A classe-alvo é discreta (não contínua)
- As classes têm interpretação direta para triagem operacional
- Métricas como F1-score Macro avaliam equitativamente o desempenho em todas as classes

---

## 2. Descrição do Dataset

### 2.1 Fonte e Acesso

| Atributo | Valor |
|----------|-------|
| Fonte | Portal de Dados Abertos da PRF |
| URL | https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf |
| Arquivo | `acidentes2023_todas_causas_tipos.csv` |
| Data de acesso | 2026-06-02 |
| Licença | Dados Abertos do Governo Federal (dados.gov.br) |
| Formato | CSV com separador `;`, encoding `latin-1` |

### 2.2 Volume

- **Linhas:** 571.052 registros
- **Colunas:** 37 atributos
- **Período:** Janeiro a Dezembro de 2023
- **Tipo de dataset:** "por pessoa" — cada linha representa uma pessoa envolvida em um acidente

### 2.3 Principais Colunas

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `classificacao_acidente` | Categórica (target) | Gravidade: Sem Vítimas / Com Vítimas Feridas / Com Vítimas Fatais |
| `uf` | Categórica | Unidade Federativa do acidente |
| `br` | Numérica | Número da rodovia federal (BR) |
| `causa_acidente` | Categórica | Causa principal registrada pelo agente |
| `tipo_acidente` | Categórica | Tipo (Colisão frontal, Tombamento, etc.) |
| `fase_dia` | Categórica | Pleno dia, Plena Noite, Amanhecer, Anoitecer, Pleno Sol |
| `condicao_metereologica` | Categórica | Chuva, Céu Claro, Nevoeiro, Vento, etc. |
| `tipo_pista` | Categórica | Simples, Dupla, Múltipla |
| `tracado_via` | Categórica | Reta, Curva, Interseção, etc. |
| `uso_solo` | Categórica | Rural ou Urbano |
| `horario` | Texto | Hora do acidente (HH:MM:SS) |
| `data_inversa` | Data | Data do acidente (YYYY-MM-DD) |

### 2.4 Distribuição da Variável-Alvo

| Classe | Quantidade | Proporção |
|--------|-----------|-----------|
| Com Vítimas Feridas | 421.356 | 73,8% |
| Com Vítimas Fatais | 93.332 | 16,3% |
| Sem Vítimas | 56.361 | 9,9% |

O dataset é **desbalanceado**, com a classe "Com Vítimas Feridas" dominando. Isso justifica o uso de `class_weight='balanced'` nos modelos.

---

## 3. Pré-processamento

### 3.1 Carregamento

O CSV foi carregado com `pandas.read_csv(sep=';', encoding='latin-1', low_memory=False)`.

### 3.2 Remoção de Colunas com >50% de Nulos

Colunas com mais de 50% de valores nulos foram removidas automaticamente. Nenhuma das features de interesse (FEATURE_COLUMNS) ultrapassou esse limiar.

### 3.3 Verificação LGPD

Foi implementada verificação automática de colunas com nomes sensíveis (cpf, nome, rg, endereco, email, telefone). Nenhuma coluna desse tipo foi encontrada no dataset PRF 2023.

### 3.4 Seleção de Features

Foram selecionadas 9 features para o modelo:

**Categóricas (8):** `uf`, `causa_acidente`, `tipo_acidente`, `fase_dia`, `condicao_metereologica`, `tipo_pista`, `tracado_via`, `uso_solo`

**Numéricas (1):** `br`

*Nota: as colunas `pessoas` e `veiculos` presentes no dataset "por ocorrência" não existem no dataset "por pessoa" utilizado. A feature `br` (número da rodovia) serve como proxy geoespacial.*

### 3.5 Encoding

**Variável-alvo:** mapeamento ordinal manual:
- `Sem Vítimas` → 0
- `Com Vítimas Feridas` → 1
- `Com Vítimas Fatais` → 2

**Features categóricas:** `OrdinalEncoder` com `handle_unknown='use_encoded_value', unknown_value=-1`, garantindo robustez a valores novos em produção.

**Feature numérica (`br`):** valores não numéricos convertidos a `NaN` e preenchidos com a mediana.

### 3.6 Split Treino/Teste

80/20 com `random_state=42` e `stratify=y` para manter a proporção de classes:

| Conjunto | Tamanho |
|----------|---------|
| Treino | 456.839 linhas |
| Teste | 114.210 linhas |

### 3.7 Persistência

Arrays salvos como `numpy.ndarray` em `data/processed/`:
- `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

---

## 4. Modelo Baseline — Regressão Logística

### 4.1 Descrição

A Regressão Logística é o baseline obrigatório do projeto. Utiliza um `Pipeline` scikit-learn com duas etapas:

1. **StandardScaler:** normaliza cada feature para média 0 e desvio 1, necessário pois a Regressão Logística é sensível à escala.
2. **LogisticRegression:** solver `lbfgs`, multinomial, `max_iter=1000`, `class_weight='balanced'`, `random_state=42`.

O `Pipeline` previne data leakage: o scaler é ajustado apenas nos dados de treino e aplicado identicamente ao teste.

### 4.2 Justificativa

- **Interpretabilidade:** coeficientes revelam a contribuição de cada feature
- **Velocidade:** treina rapidamente em >400k amostras
- **Referência:** se o Random Forest não superar, o problema está nos dados

### 4.3 Métricas (Holdout 20%)

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.4141 |
| Precision Macro | 0.4002 |
| Recall Macro | 0.4469 |
| F1 Macro | 0.3542 |

> **Análise:** O desempenho baixo da Regressão Logística indica que a relação entre as features e a classe-alvo é não-linear. O modelo linear não consegue capturar interações complexas entre `causa_acidente`, `fase_dia` e `condicao_metereologica`. Isso confirma a necessidade do modelo refinado não-linear.

### 4.4 Persistência

Modelo salvo em `models/baseline_logreg.joblib` como bundle `{pipeline, feature_columns, classes, model_type}`.

---

## 5. Modelo Refinado — Random Forest Classifier

### 5.1 Descrição

O Random Forest é um ensemble de árvores de decisão que:
- Reduz overfitting via bagging (bootstrap aggregating)
- Lida com features categóricas após encoding ordinal
- Fornece `feature_importances_` para explicabilidade
- Não requer normalização (diferentemente da Regressão Logística)

### 5.2 Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `n_estimators` | 100 | Padrão robusto, bom custo-benefício computacional |
| `random_state` | 42 | Reprodutibilidade obrigatória |
| `class_weight` | `balanced` | Compensar desbalanceamento da classe "Sem Vítimas" |
| `n_jobs` | -1 | Usar todos os núcleos disponíveis |

### 5.3 Métricas (Holdout 20%)

| Métrica | Valor |
|---------|-------|
| Accuracy | **0.9208** |
| Precision Macro | 0.8516 |
| Recall Macro | 0.8965 |
| F1 Macro | **0.8724** |

> **Análise:** Melhora dramática em relação ao baseline (+50pp em accuracy, +52pp em F1 Macro). O Random Forest captura eficientemente as interações não-lineares entre as features do dataset PRF.

### 5.4 Validação Cruzada (5-Fold)

A validação cruzada estratificada de 5 folds sobre o conjunto de treino fornece estimativa mais confiável do desempenho do modelo do que o holdout único.

| | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média | Desvio |
|-|--------|--------|--------|--------|--------|-------|--------|
| F1 Macro | 0.8614 | 0.8628 | 0.8611 | 0.8595 | 0.8594 | **0.8608** | ±0.0013 |

> O desvio padrão mínimo (±0.0013) indica que o modelo é extremamente estável — não há overfitting a nenhum fold específico.

### 5.5 Feature Importance (Top Features)

Com base nas importâncias medidas (Gini):

| Posição | Feature | Importância |
|---------|---------|-------------|
| 1 | `br` (rodovia) | 18.1% |
| 2 | `tracado_via` | 16.1% |
| 3 | `causa_acidente` | 16.5% |
| 4 | `tipo_acidente` | 14.9% |
| 5 | `uf` | 14.4% |
| 6 | `condicao_metereologica` | 8.7% |
| 7 | `fase_dia` | 5.3% |
| 8 | `tipo_pista` | 3.4% |
| 9 | `uso_solo` | 2.6% |

A rodovia (`br`) é a feature mais importante, indicando que o número da rodovia federal está altamente correlacionado com a gravidade histórica de acidentes naquele trecho.

### 5.6 Persistência

Modelo salvo em `models/random_forest.joblib`.

---

## 6. Integração via API REST

### 6.1 Arquitetura

```
Notebook Cliente (requests)
        │
        ▼ HTTP/JSON
┌─────────────────────────────┐
│  FastAPI (app/main.py)      │  porta 8000
│  ┌──────────────────────┐   │
│  │  BLOCO_ATIVO = "B"   │   │
│  └──────────────────────┘   │
│         │                   │
│         ▼                   │
│  block_b_classifier.py      │
│         │                   │
│    ┌────┴────┐              │
│    │         │              │
│  train()  predict()         │
│    │         │              │
│    ▼         ▼              │
│  app/ml/  models/*.joblib   │
└─────────────────────────────┘
```

### 6.2 Endpoints

#### GET /health
Verifica se o serviço está no ar.

**Resposta:**
```json
{
  "status": "ok",
  "service": "ml_service",
  "version": "0.2.0",
  "bloco_ativo": "B",
  "datasource": "csv"
}
```

#### POST /predict
Classifica um acidente com base nas features informadas.

**Payload:**
```json
{
  "features": {
    "uf": "MG",
    "br": 40,
    "causa_acidente": "Velocidade incompatível",
    "tipo_acidente": "Colisão frontal",
    "fase_dia": "Plena Noite",
    "condicao_metereologica": "Chuva",
    "tipo_pista": "Dupla",
    "tracado_via": "Reta",
    "uso_solo": "Rural"
  }
}
```

**Resposta:**
```json
{
  "prediction": "Com Vítimas Fatais",
  "proba": {
    "Sem Vítimas": 0.05,
    "Com Vítimas Feridas": 0.31,
    "Com Vítimas Fatais": 0.64
  }
}
```

#### POST /train
Dispara o pipeline completo de treino (pré-processamento → Regressão Logística → Random Forest).

**Resposta:**
```json
{
  "baseline_logreg": {
    "model": "LogisticRegression",
    "accuracy": 0.0,
    "f1_macro": 0.0,
    "random_state": 42,
    "data_access_date": "2026-06-02"
  },
  "random_forest": {
    "model": "RandomForestClassifier",
    "accuracy": 0.0,
    "f1_macro": 0.0,
    "cv_f1_macro_mean": 0.0,
    "random_state": 42,
    "data_access_date": "2026-06-02"
  }
}
```

### 6.3 Como Executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar pré-processamento e treino
python app/ml/train.py

# 3. Subir o servidor
uvicorn app.main:app --reload --port 8000

# 4. Testar
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"uf": "SP", "br": 116, "causa_acidente": "Velocidade incompatível", ...}}'
```

---

## 7. Limitações e Trabalhos Futuros

### 7.1 Limitações

1. **Dataset "por pessoa":** o arquivo utilizado tem uma linha por pessoa envolvida, não por acidente. Isso gera redundância nos dados e pode inflacionar métricas de avaliação se o mesmo acidente aparecer múltiplas vezes com a mesma classificação.

2. **Encoder não persistido:** o `OrdinalEncoder` é recriado a cada chamada de `/predict`, o que pode gerar inconsistências se novos valores categóricos aparecerem em produção. Em uma versão de produção, o encoder deve ser serializado junto ao modelo.

3. **Sem features temporais:** o modelo não captura sazonalidade (mês do ano, dia da semana, feriado), que são features relevantes para prever gravidade.

4. **Latência de treino:** o pipeline completo leva vários minutos para processar 500k+ linhas. Em produção, treino deve ser agendado fora do horário de pico.

### 7.2 Trabalhos Futuros

1. **GradientBoostingClassifier (XGBoost/LightGBM):** normalmente supera o Random Forest com tuning adequado e é mais eficiente em datasets grandes.

2. **Feature engineering:** criar features como `hora_do_dia` (numérica), `mes_do_ano`, `fim_de_semana`, `distancia_ao_posto_prf`.

3. **Encoder persistente:** salvar o `OrdinalEncoder` treinado junto ao `joblib` para garantir consistência treino-produção.

4. **Avaliação por UF:** calcular métricas separadas por estado para identificar onde o modelo é mais fraco.

5. **Monitoramento de drift:** implementar detecção de concept drift com `evidently` para acionar retreino automático.

6. **Dataset "por ocorrência":** utilizar o dataset agrupado por ocorrência (que inclui `pessoas` e `veiculos`) para obter features de nível de acidente sem redundância.

---

## 8. Referências

1. **PRF — Dados Abertos:** https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf. Acesso em: 2026-06-02.

2. **scikit-learn:** Pedregosa et al. *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011. https://scikit-learn.org.

3. **FastAPI:** Ramírez, S. *FastAPI*. https://fastapi.tiangolo.com.

4. **Pandas:** McKinney, W. *Data Structures for Statistical Computing in Python*, Proceedings of the 9th Python in Science Conference, 2010.

5. **LGPD:** Lei nº 13.709, de 14 de agosto de 2018. *Lei Geral de Proteção de Dados Pessoais*.

6. **Breiman, L.** *Random Forests*, Machine Learning, 45(1), 5–32, 2001.

7. **Seaborn:** Waskom, M. *seaborn: statistical data visualization*, Journal of Open Source Software, 6(60), 3021, 2021.

8. **Portal de Dados Abertos:** https://dados.gov.br — Licença: Creative Commons Attribution 4.0 International.
