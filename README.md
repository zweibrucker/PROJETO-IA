# Projeto ML — Bloco B: Classificação de Acidentes PRF

**Disciplina:** Inteligência Artificial  
**Bloco:** B — Classificação/Predição  
**Sistema-alvo:** PRF — Acidentes em Rodovias Federais  

## Descrição

Microsserviço de Machine Learning que classifica a **gravidade de acidentes em rodovias federais** com base em características do acidente (UF, causa, tipo, fase do dia, condições meteorológicas, etc.).

Variável-alvo: `classificacao_acidente`
- `Sem Vítimas`
- `Com Vítimas Feridas`
- `Com Vítimas Fatais`

**Fonte dos dados:**  
URL: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf  
Arquivo: `acidentes2023_todas_causas_tipos.csv` | Data de acesso: 2026-06-02 | Licença: Dados Abertos gov.br

## Estrutura de Pastas

```
projeto-ml/
├── app/
│   ├── blocks/
│   │   └── block_b_classifier.py   # Lógica de treino e inferência (Bloco B)
│   ├── datasources/
│   │   └── public_api.py           # Adapter PRF (download e leitura do CSV)
│   ├── ml/
│   │   ├── train.py                # Pipeline de treino (Baseline + RF)
│   │   └── predict.py              # Inferência com o Random Forest
│   ├── config.py                   # Caminhos e constantes globais
│   ├── main.py                     # FastAPI — endpoints /health, /predict, /train
│   ├── preprocessing.py            # Pré-processamento do dataset PRF
│   └── schemas.py                  # Modelos Pydantic de request/response
├── data/
│   ├── raw/
│   │   └── acidentes2023.csv       # CSV original da PRF (206 MB)
│   └── processed/
│       ├── X_train.npy             # Features de treino (numpy)
│       ├── X_test.npy              # Features de teste
│       ├── y_train.npy             # Labels de treino
│       └── y_test.npy              # Labels de teste
├── docs/
│   ├── estudo_viabilidade.md       # Estudo de viabilidade (4-6 páginas)
│   └── relatorio_tecnico.md        # Relatório técnico (8-12 páginas)
├── models/
│   ├── baseline_logreg.joblib      # Regressão Logística (baseline)
│   ├── random_forest.joblib        # Random Forest (modelo refinado)
│   └── metrics.json                # Métricas de ambos os modelos
├── notebooks/
│   ├── 01_eda.ipynb                # Análise Exploratória de Dados
│   ├── 02_baseline.ipynb           # Treino e avaliação dos modelos
│   └── 04_cliente_analitico.ipynb  # Cliente analítico via API REST
├── download_prf.py                 # Script de download dos dados PRF
├── requirements.txt                # Dependências Python
└── train.py                        # Entry point de treino (CLI)
```

## Instalação

```bash
# Clonar o repositório
git clone https://github.com/0x03c1/projeto-ml
cd projeto-ml

# Criar e ativar ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

## Como Baixar os Dados

Os dados são baixados automaticamente ao executar o treino. Para baixar manualmente:

```bash
python download_prf.py
```

O script baixa o arquivo ZIP da PRF 2023 do Google Drive e extrai o CSV em `data/raw/acidentes2023.csv` (206 MB).

**Fonte oficial:** https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf  
**Data de acesso:** 2026-06-02  
**Licença:** Dados Abertos do Governo Federal

## Como Rodar o Treino

```bash
# Opção 1: Script dedicado (recomendado)
python app/ml/train.py

# Opção 2: Entry point geral do projeto
python train.py --bloco B

# Opção 3: Via API REST (após subir o servidor)
curl -X POST http://localhost:8000/train
```

O treino executa em sequência:
1. Pré-processamento do CSV → `data/processed/*.npy`
2. Regressão Logística (baseline) → `models/baseline_logreg.joblib`
3. Random Forest (modelo refinado) → `models/random_forest.joblib`
4. Métricas → `models/metrics.json`

## Como Subir a API

```bash
uvicorn app.main:app --reload --port 8000
```

Documentação interativa: http://localhost:8000/docs  
Health check: http://localhost:8000/health

## Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| GET | `/health` | Verifica se o serviço está ativo |
| POST | `/predict` | Classifica um acidente (recebe features, retorna classe + probabilidades) |
| POST | `/train` | Dispara o treino completo e retorna métricas |

**Exemplo de chamada /predict:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "uf": "MG",
      "br": 40,
      "causa_acidente": "Velocidade incompativel",
      "tipo_acidente": "Colisao frontal",
      "fase_dia": "Plena Noite",
      "condicao_metereologica": "Chuva",
      "tipo_pista": "Dupla",
      "tracado_via": "Reta",
      "uso_solo": "Rural"
    }
  }'
```

**Resposta:**
```json
{
  "prediction": "Com Vítimas Feridas",
  "proba": {
    "Sem Vítimas": 0.08,
    "Com Vítimas Feridas": 0.71,
    "Com Vítimas Fatais": 0.21
  }
}
```

## Como Rodar os Notebooks

```bash
# Iniciar o servidor Jupyter
jupyter notebook notebooks/

# Ordem de execução recomendada:
# 1. notebooks/01_eda.ipynb              → Análise Exploratória
# 2. notebooks/02_baseline.ipynb         → Treino dos modelos
# 3. notebooks/04_cliente_analitico.ipynb → Cliente via API REST
```

> **Importante:** para rodar o notebook `04_cliente_analitico.ipynb`, o servidor FastAPI deve estar no ar (`uvicorn app.main:app --reload`).

## Configurações Importantes

| Parâmetro | Valor | Onde mudar |
|-----------|-------|-----------|
| `random_state` | 42 | `app/config.py` (SEED) |
| `BLOCO_ATIVO` | "B" | `app/main.py` |
| `DATASOURCE_KIND` | "csv" | `app/config.py` |
| Split treino/teste | 80/20 | `app/preprocessing.py` |

## Métricas (após treino)

Verificar `models/metrics.json` para métricas atualizadas após o treino.

## Documentação

- [Estudo de Viabilidade](docs/estudo_viabilidade.md) — público-alvo, ganhos, riscos éticos, manutenção
- [Relatório Técnico](docs/relatorio_tecnico.md) — dataset, pré-processamento, modelos, integração, referências
