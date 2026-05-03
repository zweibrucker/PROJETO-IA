# Microsserviço de ML – Projeto (v2)

Código casca em FastAPI para o componente de Machine Learning do projeto integrador. **Esta versão (v2)** introduz uma camada de fontes de dados (`app/datasources/`) que permite o mesmo código rodar em três cenários:

| Fonte | Quem usa | Como ativar |
|-------|----------|-------------|
| **CSV local** | Qualquer equipe (recomendado para EDA) | `DATASOURCE_KIND = "csv"` (padrão) |
| **Banco de dados** | Grupo 1 (com aplicação web e BD próprios) | `DATASOURCE_KIND = "database"` |
| **API pública** | Grupo 2 (sem app web; consome IBGE/PRF/etc.) | `DATASOURCE_KIND = "api"` |

> **Importante:** este projeto é uma casca intencional. Há `TODO`s em pontos-chave que vocês devem implementar. As partes prontas (estrutura, endpoints, persistência, validação) servem para que vocês foquem no que importa: dados, modelo e métricas.

---

## Blocos disponíveis

| Bloco | Tema | Endpoint principal |
|-------|------|--------------------|
| A | Sistema de Recomendação | `GET /recommend` |
| B | Classificação / Predição | `POST /predict` |
| C | Análise de Sentimento / Texto | `POST /analyze` |
| D | Busca Semântica | `GET /search` |

Escolha **apenas um**.

---

## Como executar

### 1. Pré-requisitos
- Python 3.10 ou superior
- pip atualizado

### 2. Instalação
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Cada bloco tem dependências adicionais comentadas no `requirements.txt`. Descomente apenas as do seu bloco e da sua fonte de dados.

### 3. Configuração
Edite `app/main.py` (linha `BLOCO_ATIVO`) e `app/config.py` (linha `DATASOURCE_KIND`) conforme o que sua equipe escolheu.

### 4. Treino
```bash
python train.py
# ou forçando bloco/fonte:
python train.py --bloco A --datasource csv
```

### 5. Servir
```bash
uvicorn app.main:app --reload --port 8000
```

A documentação interativa fica em `http://localhost:8000/docs`.

---

## Estrutura do projeto

```
ml_service/
├── app/
│   ├── main.py                       # FastAPI: rotas, CORS, healthcheck
│   ├── config.py                     # constantes globais (SEED, paths, fonte)
│   ├── schemas.py                    # contratos Pydantic
│   ├── datasources/                  # camada de fontes de dados
│   │   ├── base.py                   # Protocol + factory
│   │   ├── local_csv.py              # CSVs em data/
│   │   ├── database.py               # SQLAlchemy → BD do projeto
│   │   └── public_api.py             # IBGE, PRF, dados.gov.br, …
│   └── blocks/
│       ├── block_a_recommender.py    # Bloco A — colaborativa item-based
│       ├── block_b_classifier.py     # Bloco B — Logistic Regression
│       ├── block_c_text.py           # Bloco C — léxico → pysentimiento
│       └── block_d_search.py         # Bloco D — TF-IDF → embeddings
├── data/                             # datasets (ignorar arquivos pesados)
├── models/                           # modelos serializados
├── tests/                            # smoke tests
├── train.py                          # entry point de treino
├── requirements.txt
└── README.md
```

---

## Fluxo de trabalho recomendado

1. **Semana 1** — escolher bloco e fonte. Criar `BLOCO_ESCOLHIDO.md` com:
   - bloco escolhido
   - fonte de dados (csv/database/api)
   - justificativa de domínio (2–3 frases)
   - integrantes
2. **Semana 2** — EDA em notebook (`notebooks/`). Inclua gráficos no relatório.
3. **Semana 3** — implementar o **baseline** no bloco escolhido. Rodar `python train.py`.
4. **Semana 4** — refinar o modelo. Registrar métricas em `models/metrics.json`.
5. **Semana 5** — integração:
   - Grupo 1: chamar a API a partir da aplicação web.
   - Grupo 2: produzir um notebook ou dashboard que demonstre o uso real.
6. **Semana 6** — polimento, README final, gravação de demo.

---

## Para o Grupo 2 (sem aplicação web)

A entrega do Grupo 2 substitui a "integração com a app web" por um **notebook de avaliação** que:

1. Roda queries reais contra o microsserviço (via `requests` ou `httpx`).
2. Apresenta métricas e visualizações.
3. Discute pelo menos um caso de uso prático ("se este sistema fosse plugado no portal X, ele permitiria…").

Veja o documento de especificação `Projeto_Integrador_ML_Grupo2_Dados_Publicos.pdf`.

---

## Regras importantes

- Modelo **baseline obrigatório** antes de qualquer modelo sofisticado.
- Semente aleatória **fixa** em todo treino (`SEED = 42` em `app/config.py`).
- Métricas registradas em `models/metrics.json` com data e versão.
- Commits frequentes — média mínima de 1 commit por integrante por semana.
- **NUNCA** committar credenciais. `.env` no `.gitignore`.

---

## Suporte

Dúvidas técnicas: aulas de IA e horário de atendimento.

Bloqueios acima de 48h sem progresso devem ser comunicados ao professor imediatamente.
