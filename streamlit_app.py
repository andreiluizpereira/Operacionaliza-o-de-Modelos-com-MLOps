from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.utils import load_project_config, project_root


CLASS_LABELS = {
    0: "<= 50K",
    1: "> 50K",
    "0": "<= 50K",
    "1": "> 50K",
}


def _format_class_label(value: Any) -> str:
    return CLASS_LABELS.get(value, str(value))


def _format_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value):.{digits}f}"


@st.cache_data(show_spinner=False)
def _load_config() -> dict[str, Any]:
    return load_project_config()


@st.cache_data(show_spinner=False)
def _load_json(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def _load_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    if file_path.suffix.lower() == ".parquet":
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


@st.cache_resource(show_spinner=False)
def _load_model(path: str):
    return joblib.load(path)


def _artifact_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    paths_cfg = config.get("paths", {})
    data_cfg = config.get("data", {})
    modeling_cfg = config.get("modeling", {})
    quality_cfg = config.get("quality", {})

    reports_dir = root / paths_cfg.get("reports_dir", "outputs/reports")
    quality_dir = root / paths_cfg.get("quality_dir", "outputs/quality")
    processed_dir = root / paths_cfg.get("processed_data_dir", "data/processed")
    artifacts_dir = root / paths_cfg.get("artifacts_dir", "artifacts")

    return {
        "model": artifacts_dir / modeling_cfg.get("best_model_filename", "best_model.joblib"),
        "processed_data": processed_dir / data_cfg.get("processed_filename", "adult_income_clean.parquet"),
        "evaluation": reports_dir / modeling_cfg.get("evaluation_filename", "evaluation_report.json"),
        "training_summary": reports_dir / modeling_cfg.get("training_summary_filename", "training_summary.json"),
        "comparison": reports_dir / modeling_cfg.get("comparison_filename", "model_comparison.csv"),
        "quality": quality_dir / quality_cfg.get("report_filename", "quality_report.json"),
    }


def _feature_columns(config: dict[str, Any]) -> tuple[list[str], list[str], str]:
    target = str(config.get("data", {}).get("target_column", "income_>50K"))
    prep_cfg = config.get("preprocessing", {})
    numeric = [col for col in prep_cfg.get("numeric_columns", []) if col != target]
    categorical = [col for col in prep_cfg.get("categorical_columns", []) if col != target]
    return numeric, categorical, target


def _mode_or_first(series: pd.Series, options: list[str]) -> str:
    if not options:
        return ""
    mode = series.dropna().astype(str).mode()
    if not mode.empty and mode.iloc[0] in options:
        return str(mode.iloc[0])
    return options[0]


def _numeric_input(column: str, data: pd.DataFrame, container) -> int | float:
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if values.empty:
        return container.number_input(column, value=0.0)

    is_integer = pd.api.types.is_integer_dtype(data[column])
    minimum = values.min()
    maximum = values.max()
    median = values.median()

    if is_integer:
        return int(
            container.number_input(
                column,
                min_value=int(minimum),
                max_value=int(maximum),
                value=int(round(median)),
                step=1,
            )
        )

    return float(
        container.number_input(
            column,
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(median),
        )
    )


def _categorical_input(column: str, data: pd.DataFrame, container) -> str:
    options = sorted(data[column].dropna().astype(str).unique().tolist())
    default = _mode_or_first(data[column], options)
    index = options.index(default) if default in options else 0
    return str(container.selectbox(column, options=options, index=index))


def _build_prediction_row(
    config: dict[str, Any],
    data: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    numeric_columns, categorical_columns, _ = _feature_columns(config)
    values: dict[str, Any] = {}

    with st.form("prediction_form"):
        st.subheader("Dados do individuo")
        left, right = st.columns(2)
        columns = [left, right]

        for idx, column in enumerate(numeric_columns):
            values[column] = _numeric_input(column, data, columns[idx % 2])

        for idx, column in enumerate(categorical_columns):
            values[column] = _categorical_input(column, data, columns[idx % 2])

        submitted = st.form_submit_button("Prever renda")

    feature_order = numeric_columns + categorical_columns
    return submitted, pd.DataFrame([{column: values[column] for column in feature_order}])


def _prediction_probabilities(model, row: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()

    probabilities = model.predict_proba(row)[0]
    classes = getattr(model, "classes_", list(range(len(probabilities))))
    return pd.DataFrame(
        {
            "classe": [_format_class_label(value) for value in classes],
            "probabilidade": [float(value) for value in probabilities],
        }
    )


def _comparison_table(evaluation_report: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in evaluation_report.get("comparison", []):
        holdout = item.get("holdout_metrics", {})
        rows.append(
            {
                "modelo": item.get("model_id"),
                "variante": item.get("feature_variant"),
                "reducao": bool(item.get("uses_dimensionality_reduction")),
                "cv_f1_macro": item.get("cv_f1_macro"),
                "holdout_f1_macro": holdout.get("f1_macro"),
                "holdout_accuracy": holdout.get("accuracy"),
                "tempo_treino_s": item.get("training_time_s"),
                "features_final": item.get("feature_count_after_reduction"),
            }
        )
    return pd.DataFrame(rows)


def _render_sidebar(paths: dict[str, Path]) -> None:
    with st.sidebar:
        st.header("Execucao")
        for label, path in (
            ("Modelo", paths["model"]),
            ("Base limpa", paths["processed_data"]),
            ("Avaliacao", paths["evaluation"]),
            ("Qualidade", paths["quality"]),
        ):
            status = "OK" if path.exists() else "Pendente"
            st.write(f"{label}: {status}")

        st.divider()
        st.caption("Comandos")
        st.code("python main.py\nstreamlit run streamlit_app.py", language="bash")


def _render_prediction_tab(config: dict[str, Any], data: pd.DataFrame, paths: dict[str, Path]) -> None:
    if data.empty:
        st.warning("Base processada nao encontrada. Rode `python main.py` antes de abrir a predicao.")
        return
    if not paths["model"].exists():
        st.warning("Modelo treinado nao encontrado. Rode `python main.py` para gerar o artefato.")
        return

    try:
        model = _load_model(str(paths["model"]))
    except Exception as exc:
        st.error(f"Nao foi possivel carregar o modelo: {exc}")
        return

    submitted, row = _build_prediction_row(config, data)
    if not submitted:
        return

    prediction = model.predict(row)[0]
    label = _format_class_label(prediction)

    st.subheader("Resultado")
    st.metric("Classe prevista", label)
    if label == "> 50K":
        st.success("Renda anual prevista acima de 50K.")
    else:
        st.info("Renda anual prevista ate 50K.")

    probability_table = _prediction_probabilities(model, row)
    if not probability_table.empty:
        probability_chart = probability_table.set_index("classe")
        st.bar_chart(probability_chart)
        st.dataframe(
            probability_table.assign(probabilidade=probability_table["probabilidade"].map(_format_pct)),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("Entrada usada na predicao"):
        st.dataframe(row, hide_index=True, use_container_width=True)


def _render_metrics_tab(evaluation_report: dict[str, Any]) -> None:
    if not evaluation_report:
        st.warning("Relatorio de avaliacao nao encontrado. Rode `python main.py`.")
        return

    best_model = evaluation_report.get("best_model_name", "n/d")
    metrics = evaluation_report.get("metrics", {})

    st.subheader("Modelo selecionado")
    st.write(best_model)

    metric_columns = st.columns(4)
    for container, key, label in zip(
        metric_columns,
        ("accuracy", "precision_macro", "recall_macro", "f1_macro"),
        ("Accuracy", "Precision macro", "Recall macro", "F1 macro"),
    ):
        container.metric(label, _format_pct(metrics.get(key)))

    comparison = _comparison_table(evaluation_report)
    if not comparison.empty:
        st.subheader("Comparacao dos modelos")
        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
            column_config={
                "cv_f1_macro": st.column_config.NumberColumn(format="%.4f"),
                "holdout_f1_macro": st.column_config.NumberColumn(format="%.4f"),
                "holdout_accuracy": st.column_config.NumberColumn(format="%.4f"),
                "tempo_treino_s": st.column_config.NumberColumn(format="%.2f"),
            },
        )

    best_metadata = evaluation_report.get("best_model_metadata", {})
    fold_scores = best_metadata.get("cv_fold_f1_macro", [])
    if fold_scores:
        st.subheader("F1 macro por fold")
        fold_df = pd.DataFrame({"fold": range(1, len(fold_scores) + 1), "f1_macro": fold_scores}).set_index("fold")
        st.line_chart(fold_df)

    confusion_matrix = evaluation_report.get("confusion_matrix", [])
    if len(confusion_matrix) == 2:
        st.subheader("Matriz de confusao")
        cm_df = pd.DataFrame(
            confusion_matrix,
            index=["Real <= 50K", "Real > 50K"],
            columns=["Previsto <= 50K", "Previsto > 50K"],
        )
        st.dataframe(cm_df, use_container_width=True)


def _render_data_tab(
    config: dict[str, Any],
    data: pd.DataFrame,
    quality_report: dict[str, Any],
) -> None:
    if data.empty:
        st.warning("Base processada nao encontrada.")
        return

    numeric_columns, categorical_columns, target = _feature_columns(config)
    left, right, third = st.columns(3)
    left.metric("Linhas", f"{len(data):,}".replace(",", "."))
    right.metric("Colunas", str(data.shape[1]))
    third.metric("Variaveis de entrada", str(len(numeric_columns) + len(categorical_columns)))

    if target in data.columns:
        st.subheader("Distribuicao da classe alvo")
        target_dist = data[target].value_counts(normalize=True).sort_index()
        target_df = pd.DataFrame(
            {
                "classe": [_format_class_label(value) for value in target_dist.index],
                "proporcao": target_dist.values,
            }
        ).set_index("classe")
        st.bar_chart(target_df)

    st.subheader("Amostra da base limpa")
    st.dataframe(data.head(50), hide_index=True, use_container_width=True)

    st.subheader("Resumo numerico")
    st.dataframe(data[numeric_columns].describe().T, use_container_width=True)

    if quality_report:
        st.subheader("Qualidade dos dados")
        passed = quality_report.get("passed")
        if passed:
            st.success("Sem problemas criticos no relatorio de qualidade.")
        else:
            st.warning("Existem pontos criticos no relatorio de qualidade.")

        missing = quality_report.get("missing_values", {})
        if missing:
            missing_rows = [
                {"coluna": column, "faltantes": values.get("count"), "pct": values.get("pct")}
                for column, values in missing.items()
            ]
            st.dataframe(pd.DataFrame(missing_rows), hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Adult Income MLOps", layout="wide")

    config = _load_config()
    paths = _artifact_paths(config)
    data = _load_table(str(paths["processed_data"]))
    evaluation_report = _load_json(str(paths["evaluation"]))
    quality_report = _load_json(str(paths["quality"]))

    st.title("Adult Income MLOps")
    st.caption("Predicao e acompanhamento do pipeline de classificacao.")

    _render_sidebar(paths)

    prediction_tab, metrics_tab, data_tab = st.tabs(["Predicao", "Desempenho", "Dados"])
    with prediction_tab:
        _render_prediction_tab(config, data, paths)
    with metrics_tab:
        _render_metrics_tab(evaluation_report)
    with data_tab:
        _render_data_tab(config, data, quality_report)


if __name__ == "__main__":
    main()
