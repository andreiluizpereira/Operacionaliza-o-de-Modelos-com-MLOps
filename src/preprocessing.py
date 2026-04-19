from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.utils import ensure_dir, get_logger, load_project_config, project_root


# Antes de montar as features, esta funcao cria um encoder que entende categorias.
# Ela roda antes do pre-processador porque o pipeline de colunas usa esse bloco.
def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)


# Antes de qualquer transformacao, esta funcao le CSV ou Parquet.
# Ela roda cedo porque a etapa de preprocessing precisa de uma tabela em memoria.
def load_dataset(data_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(data_path) if str(data_path).endswith(".parquet") else pd.read_csv(data_path)


# Antes de separar treino e alvo, esta funcao tira colunas que nao ajudam.
# Ela roda depois do load porque a limpeza vem logo no inicio do preprocessing.
def drop_unused_features(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    return df.drop(columns=columns_to_drop, errors="ignore")


# Antes do treino, esta funcao separa as colunas de entrada da coluna resposta.
# Ela roda depois da limpeza para evitar tentar treinar com a coluna alvo misturada.
def prepare_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in df.columns:
        if f"{target_column}." in df.columns:
            target_column = f"{target_column}."
        else:
            raise ValueError(f"Target column '{target_column}' not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y


# Antes do modelo ver os dados, esta funcao coloca os numeros numa escala mais calma.
# Ela roda depois do split porque o ajuste do scaler deve olhar so para o treino.
def build_numeric_transformer() -> Pipeline:
    return Pipeline([("scaler", RobustScaler())])


# Antes de juntar tudo, esta funcao trata as categorias faltantes e cria colunas novas.
# Ela roda no mesmo lugar do numeric transformer porque ambos alimentam o mesmo pre-processador.
def build_categorical_transformer() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )


# Antes de criar a reducao, esta funcao arruma o bloco de configuracao.
# Ela roda primeiro para dizer ao pipeline se a reducao vai existir ou nao.
def _normalize_reduction_config(reduction_config: dict[str, Any] | None) -> dict[str, Any]:
    if reduction_config is None:
        return {"enabled": False, "compare_with_baseline": False}
    cfg = dict(reduction_config)
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["compare_with_baseline"] = bool(cfg.get("compare_with_baseline", False))
    cfg["method"] = str(cfg.get("method")).lower()
    cfg["n_components"] = cfg.get("n_components")
    if cfg.get("random_state") is not None:
        cfg["random_state"] = int(cfg["random_state"])
    if cfg["enabled"]:
        if not cfg.get("method"):
            raise ValueError("Missing preprocessing.dimensionality_reduction.method in configs/preprocessing.yaml")
        if cfg["n_components"] is None:
            raise ValueError("Missing preprocessing.dimensionality_reduction.n_components in configs/preprocessing.yaml")
    return cfg


# Antes de encaixar a reducao no pipeline, esta funcao escolhe PCA ou TruncatedSVD.
# Ela roda depois da configuracao porque precisa saber qual metodo o usuario pediu.
def _make_dimensionality_reducer(reduction_config: dict[str, Any]) -> PCA | TruncatedSVD:
    method = reduction_config["method"]
    n_components = reduction_config["n_components"]
    random_state = reduction_config.get("random_state")

    if method == "pca":
        return PCA(n_components=n_components, random_state=random_state)
    if method in {"truncated_svd", "svd"}:
        if isinstance(n_components, float):
            raise ValueError("TruncatedSVD requires an integer n_components value.")
        return TruncatedSVD(n_components=int(n_components), random_state=random_state)
    raise ValueError(f"Unsupported dimensionality reduction method: {method}")


# Antes do treino, esta funcao monta o bloco de features que vira entrada do modelo.
# Ela roda depois dos transformadores numerico e categorico porque junta tudo em um fluxo so.
def build_preprocessor(
    numeric_columns: list[str] | None = None,
    categorical_columns: list[str] | None = None,
    reduction_config: dict[str, Any] | None = None,
) -> ColumnTransformer | Pipeline:
    numeric_cols = numeric_columns or []
    categorical_cols = categorical_columns or []
    feature_transformer = ColumnTransformer(
        transformers=[
            ("numeric", build_numeric_transformer(), numeric_cols),
            ("categorical", build_categorical_transformer(), categorical_cols),
        ]
    )

    normalized_reduction = _normalize_reduction_config(reduction_config)
    if not normalized_reduction["enabled"]:
        return feature_transformer

    reducer = _make_dimensionality_reducer(normalized_reduction)
    return Pipeline(
        [
            ("features", feature_transformer),
            ("reducer", reducer),
        ]
    )


# Antes do treino e depois da ingestao, esta funcao prepara os dados e salva a versao limpa.
# Ela roda no meio do fluxo porque gera as tabelas que as etapas seguintes vao usar.
def run(df: pd.DataFrame | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_project_config()
    logger = get_logger(__name__, cfg.get("logging"))

    data_cfg = cfg.get("data", {})
    prep_cfg = cfg.get("preprocessing", {})
    paths_cfg = cfg.get("paths", {})

    if df is None:
        ingested_path = (
            project_root()
            / paths_cfg.get("interim_data_dir", "data/interim")
            / data_cfg.get("ingested_filename", "adult_income_ingested.parquet")
        )
        df = load_dataset(ingested_path)

    df = df.copy()
    if "sex" in df.columns and "gender" not in df.columns:
        df = df.rename(columns={"sex": "gender"})
    if "income_>50K." in df.columns and "income_>50K" not in df.columns:
        df = df.rename(columns={"income_>50K.": "income_>50K"})

    columns_to_drop = data_cfg.get("columns_to_drop")
    if not columns_to_drop:
        raise ValueError("Missing data.columns_to_drop in configs/data.yaml")
    cleaned = drop_unused_features(df, columns_to_drop)

    numeric_cols = prep_cfg.get("numeric_columns")
    categorical_cols = prep_cfg.get("categorical_columns")
    if not numeric_cols:
        raise ValueError("Missing preprocessing.numeric_columns in configs/preprocessing.yaml")
    if not categorical_cols:
        raise ValueError("Missing preprocessing.categorical_columns in configs/preprocessing.yaml")
    reduction_cfg = _normalize_reduction_config(prep_cfg.get("dimensionality_reduction"))
    processed_dir = ensure_dir(project_root() / paths_cfg.get("processed_data_dir", "data/processed"))
    output_path = processed_dir / data_cfg.get("processed_filename", "adult_income_clean.parquet")
    cleaned.to_parquet(output_path, index=False)

    target_column = data_cfg.get("target_column")
    if not target_column:
        raise ValueError("Missing data.target_column in configs/data.yaml")
    X, y = prepare_features_and_target(cleaned, target_column)
    split_cfg = cfg.get("modeling", {}).get("holdout", {})
    test_size = float(split_cfg.get("test_size", 0.3))
    random_state = int(split_cfg.get("random_state", 14))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info("Preprocessed data saved to %s", output_path)
    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    return {
        "cleaned_data": cleaned,
        "cleaned_path": output_path,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "dimensionality_reduction": reduction_cfg,
    }
