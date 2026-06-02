"""
Pré-processamento dos dados da PRF — Bloco B (Classificação).

Fonte dos dados:
    URL: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
    Arquivo: acidentes2023_todas_causas_tipos.csv
    Data de acesso: 2026-06-02
    Licença: Dados Abertos do Governo Federal (dados.gov.br)

Verificação LGPD:
    O dataset NÃO contém CPF, nome, endereço residencial ou qualquer
    dado pessoal identificável. As colunas de pessoa (idade, sexo) são
    tratadas como atributos agregados do acidente, sem identificação individual.

Uso:
    python app/preprocessing.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ---------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "acidentes2023.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------
# Features selecionadas para o modelo
# ---------------------------------------------------------------
# Nota: o dataset utilizado é o "por pessoa" — portanto não contém
# as colunas 'pessoas' e 'veiculos' do dataset "por ocorrência".
# Usamos features do contexto do acidente que estão disponíveis.
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
FEATURES_NUMERICAS = [
    "br",          # número da rodovia federal
]
FEATURE_COLUMNS = FEATURES_CATEGORICAS + FEATURES_NUMERICAS

TARGET_COLUMN = "classificacao_acidente"

# Mapeamento da target → valor numérico
LABEL_MAP = {
    "Sem Vítimas": 0,
    "Com Vítimas Feridas": 1,
    "Com Vítimas Fatais": 2,
}
LABEL_INVERSE = {v: k for k, v in LABEL_MAP.items()}

RANDOM_STATE = 42


def carregar_csv() -> pd.DataFrame:
    """Carrega o CSV da PRF com encoding latin-1 e separador ';'."""
    print(f"[PRE] Carregando: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV, sep=";", encoding="latin-1", low_memory=False)
    print(f"[PRE] Shape bruto: {df.shape}")
    return df


def remover_colunas_nulas(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Remove colunas com mais de `threshold` de valores nulos."""
    limite = len(df) * threshold
    antes = df.shape[1]
    df = df.dropna(axis=1, thresh=len(df) - int(limite))
    depois = df.shape[1]
    removidas = antes - depois
    print(f"[PRE] Colunas com >50% nulos removidas: {removidas} (restam {depois})")
    return df


def verificar_dados_pessoais(df: pd.DataFrame) -> None:
    """Verifica ausência de dados pessoais identificáveis (LGPD)."""
    colunas_sensiveis = ["cpf", "nome", "rg", "endereco", "email", "telefone"]
    encontradas = [c for c in df.columns if any(s in c.lower() for s in colunas_sensiveis)]
    if encontradas:
        print(f"[AVISO LGPD] Colunas potencialmente pessoais encontradas: {encontradas}")
        df = df.drop(columns=encontradas)
        print("[LGPD] Colunas removidas.")
    else:
        print("[LGPD] Nenhum dado pessoal identificável encontrado. OK.")


def preprocessar(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Executa o pipeline completo de pré-processamento.

    Retorna X_train, X_test, y_train, y_test como arrays numpy.
    """
    # 1. Verificação LGPD
    verificar_dados_pessoais(df)

    # 2. Selecionar apenas features e target
    colunas_necessarias = FEATURE_COLUMNS + [TARGET_COLUMN]
    colunas_disponiveis = [c for c in colunas_necessarias if c in df.columns]
    ausentes = [c for c in colunas_necessarias if c not in df.columns]
    if ausentes:
        print(f"[PRE] Colunas ausentes no dataset: {ausentes} (serão ignoradas)")
    df = df[colunas_disponiveis].copy()

    # 3. Remover linhas sem target
    df = df.dropna(subset=[TARGET_COLUMN])
    print(f"[PRE] Shape após remover nulos no target: {df.shape}")

    # 4. Distribuição da target
    print(f"[PRE] Distribuição target:\n{df[TARGET_COLUMN].value_counts()}")

    # 5. Label encoding da target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(LABEL_MAP)
    df = df.dropna(subset=[TARGET_COLUMN])  # remove labels não mapeados
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # 6. Separar features e target
    features_existentes_cat = [f for f in FEATURES_CATEGORICAS if f in df.columns]
    features_existentes_num = [f for f in FEATURES_NUMERICAS if f in df.columns]
    features_existentes = features_existentes_cat + features_existentes_num

    X = df[features_existentes].copy()
    y = df[TARGET_COLUMN].values

    # 7. Tratar nulos nas features numéricas (preenche com mediana)
    for col in features_existentes_num:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            mediana = X[col].median()
            X[col] = X[col].fillna(mediana)

    # 8. Encoding das features categóricas com OrdinalEncoder
    # Valores nulos são substituídos por "DESCONHECIDO" antes do encoding
    for col in features_existentes_cat:
        if col in X.columns:
            X[col] = X[col].fillna("DESCONHECIDO").astype(str)

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[features_existentes_cat] = enc.fit_transform(X[features_existentes_cat])

    X_arr = X.values.astype(float)
    print(f"[PRE] Shape final X: {X_arr.shape}")

    # 9. Split 80/20 com random_state=42 e estratificação
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[PRE] X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def salvar_arrays(X_train, X_test, y_train, y_test) -> None:
    """Salva os arrays processados em data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)
    print(f"[PRE] Arrays salvos em {PROCESSED_DIR}")


def executar_pipeline() -> tuple:
    """Executa o pipeline completo e retorna os arrays."""
    df = carregar_csv()
    df = remover_colunas_nulas(df)
    X_train, X_test, y_train, y_test = preprocessar(df)
    salvar_arrays(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    executar_pipeline()
