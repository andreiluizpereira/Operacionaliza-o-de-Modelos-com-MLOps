from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_dir, get_logger, load_project_config, project_root, save_json


# Antes do relatorio final, esta funcao olha os numeros e marca valores muito estranhos.
# Ela roda primeiro porque ajuda a separar o que parece normal do que merece alerta.
def _iqr_outliers(series: pd.Series) -> dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0, "pct": 0.0, "lower": None, "upper": None}

    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((clean < lower) | (clean > upper)).sum())
    return {
        "count": count,
        "pct": round(count / len(clean) * 100, 4),
        "lower": float(lower),
        "upper": float(upper),
    }


# Depois da ingestao, esta funcao revisa o dataset e guarda um relatorio de qualidade.
# Ela roda antes do treino para impedir que dados ruins avancem pelo pipeline.
def run(df: pd.DataFrame | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_project_config()
    logger = get_logger(__name__, cfg.get("logging"))

    if df is None:
        paths_cfg = cfg.get("paths", {})
        data_cfg = cfg.get("data", {})
        ingested_path = project_root() / paths_cfg.get("interim_data_dir", "data/interim") / data_cfg.get("ingested_filename", "adult_income_ingested.parquet")
        if not ingested_path.exists():
            raise FileNotFoundError(f"Ingested file not found: {ingested_path}")
        df = pd.read_parquet(ingested_path)

    data_cfg = cfg.get("data", {})
    quality_cfg = cfg.get("quality", {})
    thresholds = quality_cfg.get("thresholds", {})

    target_col = data_cfg.get("target_column", "income_>50K")
    expected_columns = data_cfg.get("expected_columns", [])
    report_dir = ensure_dir(project_root() / cfg.get("paths", {}).get("quality_dir", "outputs/quality"))
    report_path = report_dir / quality_cfg.get("report_filename", "quality_report.json")

    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100).round(4)
    duplicate_count = int(df.duplicated().sum())

    placeholder_counts = {}
    for col in df.select_dtypes(include="object").columns:
        placeholder_counts[col] = int((df[col] == "?").sum())

    numeric_checks = {}
    for col in ("capital-gain", "capital-loss"):
        if col in df.columns:
            numeric_checks[col] = _iqr_outliers(df[col])

    class_distribution = {}
    if target_col in df.columns:
        class_distribution = (
            df[target_col].value_counts(normalize=True).sort_index().round(4).to_dict()
        )

    missing_required = [col for col in expected_columns if col not in df.columns]
    max_missing_pct = float(thresholds.get("max_missing_pct", 5.0))
    max_duplicate_pct = float(thresholds.get("max_duplicate_pct", 0.0))
    min_positive_pct = float(thresholds.get("min_positive_pct", 20.0))

    if target_col in df.columns and 1 in class_distribution:
        positive_pct = float(class_distribution[1] * 100)
    else:
        positive_pct = None

    critical_issues = []
    if missing_required:
        critical_issues.append(f"missing_columns={missing_required}")
    if duplicate_count / len(df) * 100 > max_duplicate_pct:
        critical_issues.append(f"duplicates={duplicate_count}")
    if positive_pct is not None and positive_pct < min_positive_pct:
        critical_issues.append(f"positive_class_pct={positive_pct:.2f}")

    report = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target_column": target_col,
        "missing_required_columns": missing_required,
        "missing_values": {
            col: {
                "count": int(missing_counts[col]),
                "pct": float(missing_pct[col]),
            }
            for col in df.columns
            if int(missing_counts[col]) > 0
        },
        "placeholder_counts": placeholder_counts,
        "duplicate_count": duplicate_count,
        "duplicate_pct": round(duplicate_count / len(df) * 100, 4),
        "numeric_outliers": numeric_checks,
        "class_distribution": class_distribution,
        "thresholds": {
            "max_missing_pct": max_missing_pct,
            "max_duplicate_pct": max_duplicate_pct,
            "min_positive_pct": min_positive_pct,
        },
        "critical_issues": critical_issues,
        "passed": len(critical_issues) == 0,
    }

    save_json(report, report_path)
    logger.info("Quality report saved to %s", report_path)

    if quality_cfg.get("fail_pipeline_on_error", False) and critical_issues:
        raise RuntimeError(f"Quality checks failed: {critical_issues}")

    return {
        "report": report,
        "report_path": report_path,
    }
