"""
Bloco B — Classificação de Acidentes em Rodovias Federais (PRF).

Sistema-alvo: PRF — Classificação da gravidade de acidentes.

Variável-alvo: classificacao_acidente
    0 = Sem Vítimas
    1 = Com Vítimas Feridas
    2 = Com Vítimas Fatais

Fonte dos dados:
    URL: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
    Arquivo: acidentes2023_todas_causas_tipos.csv
    Data de acesso: 2026-06-02
    Licença: Dados Abertos do Governo Federal (dados.gov.br)

Baseline: Regressão Logística (Pipeline com StandardScaler)
Modelo refinado: Random Forest Classifier (n_estimators=100)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.config import METRICS_PATH, MODELS_DIR, SEED
from app.schemas import PredictResponse

# ---------------------------------------------------------------
# Caminhos dos modelos
# ---------------------------------------------------------------
MODEL_PATH: Path = MODELS_DIR / "random_forest.joblib"
BASELINE_PATH: Path = MODELS_DIR / "baseline_logreg.joblib"

# ---------------------------------------------------------------
# Features e target
# ---------------------------------------------------------------
FEATURES_CATEGORICAS = [
    "uf",
    "causa_acidente",
    "tipo_acidente",
    "fase_dia",
    "condicao_metereologica",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]
FEATURES_NUMERICAS = ["br"]
FEATURE_COLUMNS = FEATURES_CATEGORICAS + FEATURES_NUMERICAS
TARGET_COLUMN = "classificacao_acidente"

CLASS_NAMES = ["Sem Vítimas", "Com Vítimas Feridas", "Com Vítimas Fatais"]


# =============================================================
# Treino
# =============================================================
def train() -> dict:
    """Executa pré-processamento + treino de baseline e RF.

    Delega para app.ml.train.treinar_tudo() que implementa o pipeline
    completo, salva os modelos e atualiza models/metrics.json.
    """
    from app.ml.train import treinar_tudo
    from app.preprocessing import executar_pipeline

    print("[Bloco B] Iniciando pré-processamento...")
    executar_pipeline()

    print("[Bloco B] Iniciando treino dos modelos...")
    metricas = treinar_tudo()

    # Atualiza metrics.json no formato esperado pelo projeto
    METRICS_PATH.write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    print(f"[Bloco B] Métricas salvas em {METRICS_PATH}")
    return metricas


# =============================================================
# Inferência
# =============================================================
_cache: dict | None = None


def _carregar_bundle() -> dict:
    """Carrega o modelo Random Forest com cache em memória."""
    global _cache
    if _cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(MODEL_PATH)
        _cache = joblib.load(MODEL_PATH)
    return _cache


def predict(features: dict[str, Any]) -> PredictResponse:
    """Classifica um acidente com base nas features informadas.

    As features devem corresponder às colunas de FEATURE_COLUMNS.
    Valores ausentes são preenchidos com 'DESCONHECIDO' (categóricas)
    ou 0 (numéricas).
    """
    from sklearn.preprocessing import OrdinalEncoder

    bundle = _carregar_bundle()
    modelo = bundle["pipeline"]
    classes = bundle["classes"]

    # Verificação de features mínimas
    ausentes = [c for c in FEATURE_COLUMNS if c not in features]
    if ausentes:
        raise KeyError(", ".join(ausentes))

    # Montar vetor de features
    row_cat = [[str(features.get(c, "DESCONHECIDO") or "DESCONHECIDO") for c in FEATURES_CATEGORICAS]]
    row_num = [[float(features.get(c, 0) or 0) for c in FEATURES_NUMERICAS]]

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(row_cat)
    row_cat_enc = enc.transform(row_cat)

    X = np.concatenate([row_cat_enc, row_num], axis=1).astype(float)

    pred_idx = int(modelo.predict(X)[0])
    label_inverse = {0: "Sem Vítimas", 1: "Com Vítimas Feridas", 2: "Com Vítimas Fatais"}
    classificacao = label_inverse.get(pred_idx, str(pred_idx))

    proba_dict: dict[str, float] | None = None
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X)[0]
        proba_dict = {label_inverse.get(i, str(i)): round(float(p), 4) for i, p in enumerate(proba)}

    return PredictResponse(prediction=classificacao, proba=proba_dict)
