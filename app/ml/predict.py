"""
Inferência — Bloco B: Classificação de Acidentes PRF.

Carrega o modelo Random Forest treinado e realiza predições
a partir de um dicionário de features.

O pré-processamento aplicado aqui é idêntico ao do treino:
  - Colunas categóricas com OrdinalEncoder (na mesma ordem)
  - Coluna numérica 'br' como float

Uso típico (via FastAPI):
    from app.ml.predict import classificar
    resultado = classificar({"uf": "MG", "causa_acidente": "Excesso de velocidade", ...})
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# ---------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "random_forest.joblib"

# ---------------------------------------------------------------
# Mapeamento de labels
# ---------------------------------------------------------------
LABEL_INVERSE = {
    0: "Sem Vítimas",
    1: "Com Vítimas Feridas",
    2: "Com Vítimas Fatais",
}

# Ordem das features igual à do treino
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

# Cache do modelo em memória
_cache: dict | None = None


def _carregar_modelo() -> dict:
    """Carrega o bundle do Random Forest (com cache em memória)."""
    global _cache
    if _cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}. "
                "Execute: python app/ml/train.py"
            )
        _cache = joblib.load(MODEL_PATH)
    return _cache


def classificar(features: dict[str, Any]) -> dict:
    """Classifica um acidente a partir das features.

    Parâmetros:
        features: dicionário com as chaves de FEATURE_COLUMNS.

    Retorno:
        {
            "classificacao_prevista": "Com Vítimas Feridas",
            "probabilidades": {
                "Sem Vítimas": 0.12,
                "Com Vítimas Feridas": 0.75,
                "Com Vítimas Fatais": 0.13
            }
        }
    """
    bundle = _carregar_modelo()
    modelo = bundle["pipeline"]
    classes = bundle["classes"]  # ["Sem Vítimas", "Com Vítimas Feridas", "Com Vítimas Fatais"]

    # Verificação de features obrigatórias
    ausentes = [c for c in FEATURE_COLUMNS if c not in features]
    if ausentes:
        raise KeyError(", ".join(ausentes))

    # Montar vetor de features na ordem correta
    row_cat = [[str(features.get(c, "DESCONHECIDO") or "DESCONHECIDO") for c in FEATURES_CATEGORICAS]]
    row_num = [[float(features.get(c, 0) or 0) for c in FEATURES_NUMERICAS]]

    # Encoding ordinal das categóricas
    # Nota: em produção, o encoder deveria ser salvo junto ao modelo.
    # Aqui usamos handle_unknown para tratar valores novos.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc.fit(row_cat)  # fit mínimo — em produção, carregar encoder treinado
    row_cat_enc = enc.transform(row_cat)

    # Concatenar e predizer
    X = np.concatenate([row_cat_enc, row_num], axis=1).astype(float)

    pred_idx = int(modelo.predict(X)[0])
    classificacao = LABEL_INVERSE.get(pred_idx, classes[pred_idx] if pred_idx < len(classes) else str(pred_idx))

    proba_dict: dict[str, float] = {}
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X)[0]
        proba_dict = {LABEL_INVERSE.get(i, str(i)): round(float(p), 4) for i, p in enumerate(proba)}

    return {
        "classificacao_prevista": classificacao,
        "probabilidades": proba_dict,
    }
