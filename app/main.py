"""
Microsserviço de ML — Projeto Integrador
==========================================

Aplicação FastAPI que expõe os 4 blocos de Machine Learning. Cada
equipe implementa APENAS o bloco escolhido (A, B, C ou D).

Arquitetura
-----------
                        ┌─────────────────┐
       HTTP             │  app/main.py    │
   ┌─────────┐  ──────► │  (FastAPI)      │
   │ Cliente │          └────────┬────────┘
   └─────────┘                   │
                          ┌──────┴──────┐
                          │Bloco A/B/C/D│
                          └──────┬──────┘
                                 │
                          ┌──────┴──────┐
                          │ datasources │  ← CSV / DB / API
                          └─────────────┘

Por que essa separação?
- `main.py` cuida APENAS de HTTP (rotas, validação, códigos de erro).
- `blocks/` cuida APENAS de ML (treinar, predizer).
- `datasources/` cuida APENAS de I/O de dados.
Cada camada pode ser testada e trocada de forma independente.

Como rodar
----------
    uvicorn app.main:app --reload --port 8000

Documentação interativa: http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import DATASOURCE_KIND, SERVICE_NAME, SERVICE_VERSION
from app.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RecommendResponse,
    SearchResponse,
    TextAnalysisResponse,
    TextRequest,
)

# Cada bloco é um módulo independente. Vocês implementam o seu.
from app.blocks import (
    block_a_recommender,
    block_b_classifier,
    block_c_text,
    block_d_search,
)

# =============================================================
# Identificação do bloco ativo
# =============================================================
# Altere AQUI para o bloco que sua equipe escolheu: "A", "B", "C" ou "D".
#
# Equipes que tentarem chamar um endpoint de bloco diferente do ativo
# recebem HTTP 404. Isso evita que o Bloco B treinado seja chamado por
# engano via endpoint do Bloco A.
BLOCO_ATIVO: str = "B"   # Bloco B — Classificação PRF

app = FastAPI(
    title="ML Service — Projeto Integrador",
    version=SERVICE_VERSION,
    description=(
        "Microsserviço de Machine Learning para o projeto integrador "
        f"do 6º período. Bloco ativo: {BLOCO_ATIVO}. "
        f"Fonte de dados: {DATASOURCE_KIND}."
    ),
)

# -----------------------------------------------------------------
# CORS
# -----------------------------------------------------------------
# Em desenvolvimento, libera tudo. Em produção, restrinja ao(s)
# domínio(s) da aplicação web da sua equipe (ex: ["https://app.minhaequipe.com"]).
# Sem CORS configurado, navegadores bloqueiam chamadas de outros origens.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # TODO equipe: restringir em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# Health
# =============================================================
@app.get("/health", response_model=HealthResponse, tags=["Sistema"])
def health() -> HealthResponse:
    """Endpoint de healthcheck.

    Útil para a aplicação web verificar se o microsserviço de ML está
    disponível antes de chamar /predict, /recommend, etc. É uma boa
    prática chamar este endpoint na inicialização do frontend e exibir
    um aviso amigável se vier offline.
    """
    return HealthResponse(
        status="ok",
        service=SERVICE_NAME,
        version=SERVICE_VERSION,
        bloco_ativo=BLOCO_ATIVO,
        datasource=DATASOURCE_KIND,
    )


# =============================================================
# Bloco A — Sistema de Recomendação
# =============================================================
@app.get(
    "/recommend",
    response_model=RecommendResponse,
    tags=["Bloco A — Recomendação"],
)
def recommend(user_id: str, k: int = 5) -> RecommendResponse:
    """Retorna os k itens mais recomendados para um usuário.

    Parâmetros:
        user_id: identificador do usuário no domínio da aplicação.
        k:       quantidade de recomendações (1 a 50, padrão 5).

    Códigos de status:
        200 OK             — recomendação retornada.
        400 Bad Request    — k fora do intervalo válido.
        404 Not Found      — Bloco A não está ativo.
        503 Service Unavailable — modelo ainda não foi treinado.
    """
    if BLOCO_ATIVO != "A":
        raise HTTPException(status_code=404, detail="Bloco A não está ativo.")
    if k < 1 or k > 50:
        raise HTTPException(status_code=400, detail="k deve estar entre 1 e 50.")
    try:
        return block_a_recommender.recommend(user_id=user_id, k=k)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelo do Bloco A não encontrado. Rode: python train.py",
        )


# =============================================================
# Bloco B — Classificação / Predição
# =============================================================
@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Bloco B — Classificação"],
)
def predict(req: PredictRequest) -> PredictResponse:
    """Realiza predição a partir das features informadas.

    O conteúdo de `features` depende do modelo treinado pela equipe.
    Veja o exemplo na documentação interativa em /docs.

    Códigos de status:
        200 OK             — predição realizada.
        404 Not Found      — Bloco B não está ativo.
        422 Unprocessable  — feature obrigatória ausente no payload.
        503 Service Unavailable — modelo ainda não foi treinado.
    """
    if BLOCO_ATIVO != "B":
        raise HTTPException(status_code=404, detail="Bloco B não está ativo.")
    try:
        return block_b_classifier.predict(features=req.features)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Modelo do Bloco B não encontrado. Rode: python train.py",
        )
    except KeyError as e:
        raise HTTPException(
            status_code=422,
            detail=f"Feature obrigatória ausente: {e}",
        )


@app.post(
    "/train",
    tags=["Bloco B — Classificação"],
)
def train() -> dict:
    """Dispara o treino completo (baseline + Random Forest) e retorna métricas.

    Treina Regressão Logística e Random Forest nos dados da PRF 2023.
    Salva os modelos em models/ e atualiza models/metrics.json.

    Códigos de status:
        200 OK             — treino concluído, métricas retornadas.
        404 Not Found      — Bloco B não está ativo.
        503 Service Unavailable — erro durante o treino.
    """
    if BLOCO_ATIVO != "B":
        raise HTTPException(status_code=404, detail="Bloco B não está ativo.")
    try:
        return block_b_classifier.train()
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Arquivo não encontrado durante treino: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Erro durante treino: {e}",
        )


# =============================================================
# Bloco C — Texto / Sentimento
# =============================================================
@app.post(
    "/analyze",
    response_model=TextAnalysisResponse,
    tags=["Bloco C — Texto"],
)
def analyze(req: TextRequest) -> TextAnalysisResponse:
    """Analisa o texto e retorna sentimento + score de confiança.

    O baseline (lexical) NÃO precisa de treino prévio, então este
    endpoint nunca devolve 503.
    """
    if BLOCO_ATIVO != "C":
        raise HTTPException(status_code=404, detail="Bloco C não está ativo.")
    return block_c_text.analyze(text=req.text)


# =============================================================
# Bloco D — Busca Semântica
# =============================================================
@app.get(
    "/search",
    response_model=SearchResponse,
    tags=["Bloco D — Busca Semântica"],
)
def search(q: str, k: int = 10) -> SearchResponse:
    """Busca semântica: retorna os k itens mais próximos do significado da query.

    Códigos de status:
        200 OK             — resultado retornado (pode ser lista vazia
                             se nenhum item teve score > 0).
        400 Bad Request    — query vazia ou k fora do intervalo.
        404 Not Found      — Bloco D não está ativo.
        503 Service Unavailable — índice ainda não foi construído.
    """
    if BLOCO_ATIVO != "D":
        raise HTTPException(status_code=404, detail="Bloco D não está ativo.")
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' não pode ser vazia.")
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k deve estar entre 1 e 100.")
    try:
        return block_d_search.search(query=q, k=k)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Índice do Bloco D não encontrado. Rode: python train.py",
        )
