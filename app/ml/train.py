"""
Treino dos modelos — Bloco B: Classificação de Acidentes PRF.

Fluxo:
    1. Executa pré-processamento (gera data/processed/*.npy)
    2. Treina Regressão Logística (baseline obrigatório)
    3. Treina Random Forest (modelo refinado)
    4. Salva modelos em models/
    5. Atualiza models/metrics.json

Uso:
    python app/ml/train.py

Data de acesso aos dados: 2026-06-02
Fonte: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42

# Nomes das classes na mesma ordem do LabelEncoder
CLASS_NAMES = ["Sem Vítimas", "Com Vítimas Feridas", "Com Vítimas Fatais"]


def carregar_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carrega os arrays pré-processados de data/processed/."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_train = np.load(PROCESSED_DIR / "y_train.npy")
    y_test = np.load(PROCESSED_DIR / "y_test.npy")
    print(f"[TRAIN] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def calcular_metricas(y_true, y_pred) -> dict:
    """Calcula accuracy, precision, recall e f1 (macro)."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }


def treinar_baseline(X_train, X_test, y_train, y_test) -> dict:
    """Treina Regressão Logística (baseline obrigatório).

    Usa Pipeline com StandardScaler para normalizar as features antes
    da regressão logística (que é sensível à escala).
    """
    print("\n[BASELINE] Treinando Regressão Logística...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)
    metricas_resumo = {
        "model": "LogisticRegression",
        "accuracy": round(metricas["accuracy"], 4),
        "precision_macro": round(metricas["precision_macro"], 4),
        "recall_macro": round(metricas["recall_macro"], 4),
        "f1_macro": round(metricas["f1_macro"], 4),
        "random_state": RANDOM_STATE,
        "data_access_date": "2026-06-02",
    }

    print(f"[BASELINE] Accuracy: {metricas_resumo['accuracy']:.4f}")
    print(f"[BASELINE] F1 Macro: {metricas_resumo['f1_macro']:.4f}")
    print(f"[BASELINE] Matriz de confusão:\n{np.array(metricas['confusion_matrix'])}")

    # Salva modelo
    caminho = MODELS_DIR / "baseline_logreg.joblib"
    bundle = {
        "pipeline": pipeline,
        "feature_columns": None,  # definido em block_b_classifier.py
        "classes": CLASS_NAMES,
        "model_type": "logistic_regression_baseline",
    }
    joblib.dump(bundle, caminho)
    print(f"[BASELINE] Modelo salvo em {caminho}")

    return metricas_resumo, metricas


def treinar_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """Treina Random Forest (modelo refinado).

    Usa n_estimators=100, random_state=42, class_weight=balanced.
    Calcula validação cruzada 5-fold no conjunto de treino.
    """
    print("\n[RF] Treinando Random Forest...")

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metricas = calcular_metricas(y_test, y_pred)

    # Validação cruzada 5-fold (F1 macro)
    print("[RF] Executando validação cruzada 5-fold...")
    cv_scores = cross_val_score(
        rf, X_train, y_train, cv=5, scoring="f1_macro", n_jobs=-1
    )
    cv_mean = float(cv_scores.mean())
    cv_std = float(cv_scores.std())
    print(f"[RF] CV F1 Macro: {cv_mean:.4f} ± {cv_std:.4f}")

    # Feature importance
    feature_importances = rf.feature_importances_.tolist()

    metricas_resumo = {
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "accuracy": round(metricas["accuracy"], 4),
        "precision_macro": round(metricas["precision_macro"], 4),
        "recall_macro": round(metricas["recall_macro"], 4),
        "f1_macro": round(metricas["f1_macro"], 4),
        "cv_f1_macro_mean": round(cv_mean, 4),
        "cv_f1_macro_std": round(cv_std, 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "feature_importances": feature_importances,
        "random_state": RANDOM_STATE,
        "data_access_date": "2026-06-02",
    }

    print(f"[RF] Accuracy: {metricas_resumo['accuracy']:.4f}")
    print(f"[RF] F1 Macro: {metricas_resumo['f1_macro']:.4f}")
    print(f"[RF] Matriz de confusão:\n{np.array(metricas['confusion_matrix'])}")

    # Salva modelo
    caminho = MODELS_DIR / "random_forest.joblib"
    bundle = {
        "pipeline": rf,
        "feature_columns": None,  # definido em block_b_classifier.py
        "classes": CLASS_NAMES,
        "model_type": "random_forest",
    }
    joblib.dump(bundle, caminho)
    print(f"[RF] Modelo salvo em {caminho}")

    return metricas_resumo, metricas


def salvar_metrics_json(metricas_baseline: dict, metricas_rf: dict) -> None:
    """Atualiza models/metrics.json com métricas de ambos os modelos."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "baseline_logreg": metricas_baseline,
        "random_forest": metricas_rf,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[TRAIN] Métricas salvas em {METRICS_PATH}")


def treinar_tudo() -> dict:
    """Executa o pipeline completo de treino.

    Executa pré-processamento se necessário, treina baseline e RF,
    salva modelos e métricas. Retorna o dict de métricas completo.
    """
    # Verifica se arrays já foram gerados; se não, executa pré-processamento
    if not (PROCESSED_DIR / "X_train.npy").exists():
        print("[TRAIN] Arrays não encontrados. Executando pré-processamento...")
        from app.preprocessing import executar_pipeline
        executar_pipeline()

    X_train, X_test, y_train, y_test = carregar_arrays()

    metricas_base_resumo, _ = treinar_baseline(X_train, X_test, y_train, y_test)
    metricas_rf_resumo, _ = treinar_random_forest(X_train, X_test, y_train, y_test)

    salvar_metrics_json(metricas_base_resumo, metricas_rf_resumo)

    resultado = {
        "baseline_logreg": metricas_base_resumo,
        "random_forest": metricas_rf_resumo,
    }
    print("\n[TRAIN] Treino concluído com sucesso.")
    return resultado


if __name__ == "__main__":
    treinar_tudo()
