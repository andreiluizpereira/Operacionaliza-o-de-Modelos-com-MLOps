from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import build_preprocessor
from src.utils import ensure_dir, get_logger, load_project_config, project_root, save_json


# Antes de comparar modelos, esta funcao transforma acertos e erros em numeros simples.
# Ela roda logo depois da predicao para que as variantes sejam comparadas do mesmo jeito.
def _compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _mlflow_param_value(value: Any) -> Any:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _init_mlflow(modeling_cfg: dict[str, Any], root: Path, logger) -> dict[str, Any] | None:
    mlflow_cfg = modeling_cfg.get("mlflow", {})
    enabled = bool(mlflow_cfg.get("enabled", False))
    if not enabled:
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "MLflow is enabled in configs/modeling.yaml, but the package is not installed. Install with `pip install mlflow`."
        ) from exc

    tracking_cfg = str(mlflow_cfg.get("tracking_uri", modeling_cfg.get("tracking_uri", "mlruns")))
    experiment_name = str(
        mlflow_cfg.get("experiment_name", modeling_cfg.get("experiment_name", "adult-income-experiments"))
    )
    run_name_prefix = str(mlflow_cfg.get("run_name_prefix", "baseline"))
    log_model = bool(mlflow_cfg.get("log_model", False))

    if "://" in tracking_cfg:
        tracking_uri = tracking_cfg
    else:
        # as_uri() avoids Windows path parsing issues (e.g. drive letters seen as URI schemes)
        tracking_uri = (root / tracking_cfg).resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow enabled | experiment=%s | tracking_uri=%s", experiment_name, tracking_uri)

    return {
        "client": mlflow,
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "run_name_prefix": run_name_prefix,
        "log_model": log_model,
    }


# Antes de treinar cada modelo, esta funcao cria um nome curto para a variante das features.
# Ela roda junto com a configuracao para separar baseline e reducao sem confusao.
def _format_reduction_label(reduction_cfg: dict[str, Any]) -> str:
    method = str(reduction_cfg.get("method", "pca")).lower()
    n_components = reduction_cfg.get("n_components", 0.95)
    if isinstance(n_components, float):
        if 0 < n_components <= 1:
            component_label = f"{int(round(n_components * 100))}pct"
        else:
            component_label = str(n_components).replace(".", "p")
    else:
        component_label = f"{int(n_components)}c"
    return f"{method}_{component_label}"


# Antes de chamar os modelos, esta funcao decide quais variantes de features vao entrar.
# Ela roda antes do treino porque organiza se teremos baseline, reducao, ou os dois.
def _feature_variants(reduction_cfg: dict[str, Any] | None) -> list[dict[str, Any]]:
    cfg = dict(reduction_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    compare_with_baseline = bool(cfg.get("compare_with_baseline", False))

    baseline = {
        "feature_variant": "baseline",
        "reduction_config": None,
        "uses_dimensionality_reduction": False,
        "reduction_method": None,
        "reduction_n_components": None,
    }

    if not enabled:
        return [baseline]

    reduced = {
        "feature_variant": _format_reduction_label(cfg),
        "reduction_config": cfg,
        "uses_dimensionality_reduction": True,
        "reduction_method": str(cfg.get("method", "pca")).lower(),
        "reduction_n_components": cfg.get("n_components", 0.95),
    }

    if compare_with_baseline:
        return [baseline, reduced]
    return [reduced]


# Antes do treino, esta funcao junta pre-processamento e modelo num unico pacote.
# Ela roda depois da preparacao dos dados porque o modelo precisa receber tudo pronto.
def _build_pipeline(
    estimator,
    numeric_columns: list[str],
    categorical_columns: list[str],
    reduction_config: dict[str, Any] | None = None,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    numeric_columns=numeric_columns,
                    categorical_columns=categorical_columns,
                    reduction_config=reduction_config,
                ),
            ),
            ("model", estimator),
        ]
    )


# Antes de escolher o melhor modelo, esta funcao mede o desempenho medio do treino.
# Ela roda durante a validacao cruzada para evitar que uma divisao unica engane a gente.
def _cv_f1_scores(pipe: Pipeline, X_train, y_train, cv: StratifiedKFold) -> list[float]:
    scores = cross_val_score(pipe, X_train, y_train, scoring="f1_macro", cv=cv, n_jobs=1)
    return [float(score) for score in scores]


# Antes de salvar o resultado, esta funcao conta quantas features sobraram e o que a reducao fez.
# Ela roda depois do treino para comparar a versao original com a versao comprimida.
def _extract_feature_metadata(model: Pipeline, X_train) -> dict[str, Any]:
    sample = X_train.iloc[: min(20, len(X_train))]
    preprocessor = model.named_steps["preprocessor"]

    transformed = preprocessor.transform(sample)
    feature_count_after = int(transformed.shape[1])

    metadata: dict[str, Any] = {
        "feature_count_after_reduction": feature_count_after,
        "feature_count_before_reduction": feature_count_after,
        "reduction_effective_components": None,
        "reduction_explained_variance_ratio": None,
    }

    if isinstance(preprocessor, Pipeline) and "reducer" in preprocessor.named_steps:
        base_features = preprocessor.named_steps["features"].transform(sample)
        feature_count_before = int(base_features.shape[1])
        reducer = preprocessor.named_steps["reducer"]
        explained_variance = getattr(reducer, "explained_variance_ratio_", None)
        explained_sum = None
        if explained_variance is not None:
            explained_sum = float(explained_variance.sum())

        metadata.update(
            {
                "feature_count_before_reduction": feature_count_before,
                "feature_count_after_reduction": feature_count_after,
                "reduction_effective_components": int(
                    getattr(reducer, "n_components_", getattr(reducer, "n_components", feature_count_after))
                ),
                "reduction_explained_variance_ratio": explained_sum,
            }
        )

    return metadata


# Antes de guardar um dicionario em JSON, esta funcao troca valores estranhos por None.
# Ela roda antes do relatorio porque o JSON precisa ficar limpo e facil de ler.
def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


# Antes de exportar a comparacao, esta funcao limpa cada linha para caber bem no JSON.
# Ela roda logo antes do arquivo final para evitar valores quebrados no relatorio.
def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [_json_safe(record) for record in df.to_dict(orient="records")]


# Antes de escolher o vencedor, esta funcao junta modelo, metricas e explicacao em um so bloco.
# Ela roda depois do treino de cada algoritmo porque cada rodada precisa virar um registro padrao.
def _model_record(
    base_model_name: str,
    feature_variant: str,
    trained_model: Pipeline,
    cv_scores: list[float],
    best_params: dict[str, Any],
    holdout_metrics: dict[str, float],
    reduction_cfg: dict[str, Any] | None,
    X_train,
    training_time_s: float,
) -> dict[str, Any]:
    cv_series = pd.Series(cv_scores, dtype=float)
    cv_mean = float(cv_series.mean())
    cv_std = float(cv_series.std(ddof=0)) if len(cv_scores) > 1 else 0.0

    record = {
        "model": base_model_name,
        "model_id": f"{base_model_name}__{feature_variant}",
        "feature_variant": feature_variant,
        "uses_dimensionality_reduction": bool(reduction_cfg and reduction_cfg.get("enabled", False)),
        "reduction_method": None if reduction_cfg is None else reduction_cfg.get("method"),
        "reduction_n_components": None if reduction_cfg is None else reduction_cfg.get("n_components"),
        "cv_f1_macro": cv_mean,
        "cv_f1_macro_std": cv_std,
        "cv_fold_f1_macro": cv_scores,
        "best_params": best_params,
        "holdout_metrics": holdout_metrics,
        "training_time_s": training_time_s,
    }
    record.update(_extract_feature_metadata(trained_model, X_train))

    if record["uses_dimensionality_reduction"]:
        record["interpretability_note"] = (
            "Principal components reduce direct feature interpretability compared with the baseline representation."
        )
    else:
        record["interpretability_note"] = "Direct feature names are preserved in the baseline representation."

    return record | {"model_obj": trained_model}


def _log_mlflow_run(mlflow_state: dict[str, Any], record: dict[str, Any], logger) -> None:
    mlflow = mlflow_state["client"]
    run_name = f"{mlflow_state['run_name_prefix']}_{record['model_id']}"
    reduction_method = record.get("reduction_method") or "none"

    with mlflow.start_run(
        run_name=run_name,
        tags={
            "stage": "baseline",
            "model": record["model"],
            "feature_variant": record["feature_variant"],
        },
    ):
        params = {
            **record.get("best_params", {}),
            "model_id": record["model_id"],
            "feature_variant": record["feature_variant"],
            "uses_dimensionality_reduction": record["uses_dimensionality_reduction"],
            "reduction_method": reduction_method,
            "reduction_n_components": record.get("reduction_n_components"),
        }
        mlflow.log_params({str(k): _mlflow_param_value(v) for k, v in params.items()})

        model_obj = record.get("model_obj")
        estimator = None
        if model_obj is not None and hasattr(model_obj, "named_steps"):
            estimator = model_obj.named_steps.get("model")

        if estimator is not None:
            mlflow.set_tag("model_class", f"{estimator.__class__.__module__}.{estimator.__class__.__name__}")
        mlflow.set_tag("reducer_method", str(reduction_method))

        for step, score in enumerate(record.get("cv_fold_f1_macro", []), start=1):
            mlflow.log_metric("fold_f1_macro", float(score), step=step)

        mlflow.log_metric("cv_f1_macro", float(record.get("cv_f1_macro", 0.0)))
        mlflow.log_metric("cv_f1_macro_std", float(record.get("cv_f1_macro_std", 0.0)))
        mlflow.log_metric("training_time_s", float(record.get("training_time_s", 0.0)))

        holdout_metrics = record.get("holdout_metrics", {})
        for metric_name, metric_value in holdout_metrics.items():
            mlflow.log_metric(f"holdout_{metric_name}", float(metric_value))

        for key in (
            "feature_count_before_reduction",
            "feature_count_after_reduction",
            "reduction_effective_components",
            "reduction_explained_variance_ratio",
        ):
            value = record.get(key)
            if value is None:
                continue
            if isinstance(value, float) and pd.isna(value):
                continue
            mlflow.log_metric(key, float(value))

        if mlflow_state.get("log_model", False) and model_obj is not None:
            try:
                import mlflow.sklearn

                mlflow.sklearn.log_model(model_obj, artifact_path="model")
            except Exception as exc:  # pragma: no cover - best effort artifact logging
                logger.warning("MLflow model artifact logging failed for %s: %s", record["model_id"], exc)


# Antes de comparar as variantes, esta funcao treina o Perceptron e mede o resultado.
# Ela roda dentro do experimento para servir como baseline simples e rapido.
def _train_perceptron(
    seed: int,
    max_iter: int,
    X_train,
    y_train,
    X_test,
    y_test,
    cv: StratifiedKFold,
    numeric_columns: list[str],
    categorical_columns: list[str],
    feature_variant: str,
    reduction_config: dict[str, Any] | None,
) -> dict[str, Any]:
    pipe = _build_pipeline(
        Perceptron(random_state=seed, max_iter=max_iter),
        numeric_columns,
        categorical_columns,
        reduction_config=reduction_config,
    )
    t0 = time.time()
    cv_scores = _cv_f1_scores(pipe, X_train, y_train, cv)
    model = pipe.fit(X_train, y_train)
    preds = model.predict(X_test)
    training_time_s = float(time.time() - t0)
    return _model_record(
        base_model_name="perceptron",
        feature_variant=feature_variant,
        trained_model=model,
        cv_scores=cv_scores,
        best_params={"random_state": seed, "max_iter": max_iter},
        holdout_metrics=_compute_metrics(y_test, preds),
        reduction_cfg=reduction_config,
        X_train=X_train,
        training_time_s=training_time_s,
    )


# Antes de comparar as variantes, esta funcao treina a arvore e procura bons hiperparametros.
# Ela roda no meio do experimento porque e o modelo com ajuste mais cuidadoso.
def _train_decision_tree(
    seed: int,
    cfg: dict[str, Any],
    X_train,
    y_train,
    X_test,
    y_test,
    cv: StratifiedKFold,
    numeric_columns: list[str],
    categorical_columns: list[str],
    feature_variant: str,
    reduction_config: dict[str, Any] | None,
) -> dict[str, Any]:
    pipe = _build_pipeline(
        DecisionTreeClassifier(random_state=seed),
        numeric_columns,
        categorical_columns,
        reduction_config=reduction_config,
    )

    param_distributions = {
        "model__criterion": cfg.get("criterion_choices", ["gini", "entropy"]),
        "model__max_depth": randint(int(cfg.get("max_depth_low", 2)), int(cfg.get("max_depth_high", 30)) + 1),
        "model__min_samples_leaf": randint(
            int(cfg.get("min_samples_leaf_low", 1)),
            int(cfg.get("min_samples_leaf_high", 50)) + 1,
        ),
        "model__class_weight": cfg.get("class_weight_choices", [None, "balanced"]),
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=int(cfg.get("n_iter", 20)),
        scoring="f1_macro",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=1,
        error_score=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    preds = search.best_estimator_.predict(X_test)
    cv_scores: list[float] = []
    best_index = int(search.best_index_)
    for split_idx in range(cv.get_n_splits()):
        split_key = f"split{split_idx}_test_score"
        if split_key in search.cv_results_:
            cv_scores.append(float(search.cv_results_[split_key][best_index]))
    if not cv_scores:
        cv_scores = [float(search.best_score_)]
    training_time_s = float(time.time() - t0)
    return _model_record(
        base_model_name="decision_tree",
        feature_variant=feature_variant,
        trained_model=search.best_estimator_,
        cv_scores=cv_scores,
        best_params=search.best_params_,
        holdout_metrics=_compute_metrics(y_test, preds),
        reduction_cfg=reduction_config,
        X_train=X_train,
        training_time_s=training_time_s,
    )


# Antes de comparar as variantes, esta funcao treina o SVM linear e mede o resultado.
# Ela roda junto das outras para servir como baseline linear mais forte.
def _train_linear_svm(
    seed: int,
    max_iter: int,
    X_train,
    y_train,
    X_test,
    y_test,
    cv: StratifiedKFold,
    numeric_columns: list[str],
    categorical_columns: list[str],
    feature_variant: str,
    reduction_config: dict[str, Any] | None,
) -> dict[str, Any]:
    pipe = _build_pipeline(
        LinearSVC(random_state=seed, max_iter=max_iter),
        numeric_columns,
        categorical_columns,
        reduction_config=reduction_config,
    )
    t0 = time.time()
    cv_scores = _cv_f1_scores(pipe, X_train, y_train, cv)
    model = pipe.fit(X_train, y_train)
    preds = model.predict(X_test)
    training_time_s = float(time.time() - t0)
    return _model_record(
        base_model_name="linear_svm",
        feature_variant=feature_variant,
        trained_model=model,
        cv_scores=cv_scores,
        best_params={"random_state": seed, "max_iter": max_iter},
        holdout_metrics=_compute_metrics(y_test, preds),
        reduction_cfg=reduction_config,
        X_train=X_train,
        training_time_s=training_time_s,
    )


# Depois do preprocessing e antes da avaliacao final, esta funcao compara os modelos e escolhe um.
# Ela roda no centro do pipeline porque transforma treino em artefato final.
def run(prep_result: dict[str, Any] | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_project_config()
    logger = get_logger(__name__, cfg.get("logging"))

    if prep_result is None:
        raise ValueError("prep_result is required. Run preprocessing first.")

    modeling_cfg = cfg.get("modeling", {})
    holdout_cfg = modeling_cfg.get("holdout", {})
    cv_cfg = modeling_cfg.get("cv", {})
    reduction_cfg = prep_result.get("dimensionality_reduction", {})
    feature_variants = _feature_variants(reduction_cfg)

    if bool(reduction_cfg.get("compare_with_baseline", False)) and not bool(reduction_cfg.get("enabled", False)):
        logger.warning("compare_with_baseline is true, but dimensionality reduction is disabled. Running baseline only.")

    mlflow_state = _init_mlflow(modeling_cfg, project_root(), logger)

    X_train = prep_result["X_train"]
    y_train = prep_result["y_train"]
    X_test = prep_result["X_test"]
    y_test = prep_result["y_test"]
    numeric_columns = prep_result.get("numeric_columns", [])
    categorical_columns = prep_result.get("categorical_columns", [])

    seed = int(modeling_cfg.get("random_state", 14))
    cv = StratifiedKFold(
        n_splits=int(cv_cfg.get("n_splits", 10)),
        shuffle=bool(cv_cfg.get("shuffle", True)),
        random_state=seed,
    )

    results: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    # treina cada modelo para variante baseline das features, depois treina cada modelo para variante reduzida com PCA (se habilitada), e guarda os resultados padronizados.
    for variant in feature_variants:
        feature_variant = variant["feature_variant"]
        variant_reduction_cfg = variant["reduction_config"]

        # treinamento do Perceptron
        if modeling_cfg.get("perceptron", {}).get("enabled", True):
            logger.info("Training perceptron [%s]", feature_variant)
            out = _train_perceptron(
                seed=seed,
                max_iter=int(modeling_cfg.get("perceptron", {}).get("max_iter", 1000)),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                feature_variant=feature_variant,
                reduction_config=variant_reduction_cfg,
            )
            if mlflow_state is not None:
                _log_mlflow_run(mlflow_state, out, logger)
            results.append({k: v for k, v in out.items() if k != "model_obj"})
            trained_models[out["model_id"]] = out["model_obj"]

        # treinamento da Decision Tree
        if modeling_cfg.get("decision_tree", {}).get("enabled", True):
            logger.info("Training decision_tree [%s]", feature_variant)
            out = _train_decision_tree(
                seed=seed,
                cfg=modeling_cfg.get("decision_tree", {}),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                feature_variant=feature_variant,
                reduction_config=variant_reduction_cfg,
            )
            if mlflow_state is not None:
                _log_mlflow_run(mlflow_state, out, logger)
            results.append({k: v for k, v in out.items() if k != "model_obj"})
            trained_models[out["model_id"]] = out["model_obj"]

        # treinamento do SVM linear
        if modeling_cfg.get("linear_svm", {}).get("enabled", True):
            logger.info("Training linear_svm [%s]", feature_variant)
            out = _train_linear_svm(
                seed=seed,
                max_iter=int(modeling_cfg.get("linear_svm", {}).get("max_iter", 5000)),
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                cv=cv,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                feature_variant=feature_variant,
                reduction_config=variant_reduction_cfg,
            )
            if mlflow_state is not None:
                _log_mlflow_run(mlflow_state, out, logger)
            results.append({k: v for k, v in out.items() if k != "model_obj"})
            trained_models[out["model_id"]] = out["model_obj"]

    comparison = pd.DataFrame(results).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    if comparison.empty:
        raise RuntimeError("No models were trained. Check modeling.yaml.")

    best_model_id = str(comparison.iloc[0]["model_id"])
    best_model_base_name = str(comparison.iloc[0]["model"])
    best_model = trained_models[best_model_id]
    best_model_metadata = _json_safe(comparison.iloc[0].to_dict())
    comparison_records = _json_safe_records(comparison)

    artifacts_dir = ensure_dir(project_root() / cfg.get("paths", {}).get("artifacts_dir", "artifacts"))
    model_path = artifacts_dir / modeling_cfg.get("best_model_filename", "best_model.joblib")
    joblib.dump(best_model, model_path)

    reports_dir = ensure_dir(project_root() / cfg.get("paths", {}).get("reports_dir", "outputs/reports"))
    comparison_path = reports_dir / modeling_cfg.get("comparison_filename", "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "comparison": comparison_records,
        "best_model_id": best_model_id,
        "best_model_name": best_model_id,
        "best_model_base_name": best_model_base_name,
        "best_model_metadata": best_model_metadata,
        "best_model_path": str(model_path),
        "comparison_path": str(comparison_path),
        "splits": {
            "test_size": float(holdout_cfg.get("test_size", 0.3)),
            "random_state": seed,
        },
        "feature_variants": [variant["feature_variant"] for variant in feature_variants],
        "mlflow": {
            "enabled": mlflow_state is not None,
            "tracking_uri": None if mlflow_state is None else mlflow_state["tracking_uri"],
            "experiment_name": None if mlflow_state is None else mlflow_state["experiment_name"],
        },
        "trained_models": trained_models,
        "data": {
            "X_test": X_test,
            "y_test": y_test,
        },
    }

    save_json(
        {
            "best_model_id": best_model_id,
            "best_model_name": best_model_id,
            "best_model_base_name": best_model_base_name,
            "best_model_path": str(model_path),
            "comparison_path": str(comparison_path),
            "best_model_metadata": best_model_metadata,
            "comparison": comparison_records,
            "mlflow": {
                "enabled": mlflow_state is not None,
                "tracking_uri": None if mlflow_state is None else mlflow_state["tracking_uri"],
                "experiment_name": None if mlflow_state is None else mlflow_state["experiment_name"],
            },
        },
        reports_dir / modeling_cfg.get("training_summary_filename", "training_summary.json"),
    )

    logger.info("Best model: %s", best_model_id)
    logger.info("Model saved to %s", model_path)

    return summary
