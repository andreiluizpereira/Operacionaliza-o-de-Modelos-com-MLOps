from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import ensure_dir, get_logger, load_project_config, project_root


def run(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_project_config()
    logger = get_logger(__name__, cfg.get("logging"))

    pipeline_cfg = cfg.get("pipeline", {})
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})

    raw_dir = project_root() / paths_cfg.get("raw_data_dir", "data/raw")
    interim_dir = ensure_dir(project_root() / paths_cfg.get("interim_data_dir", "data/interim"))

    input_filename = data_cfg.get("raw_filename", "train.csv")
    input_path = raw_dir / input_filename
    output_filename = data_cfg.get("ingested_filename", "adult_income_ingested.parquet")
    output_path = interim_dir / output_filename

    if not input_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    logger.info("Reading raw data from %s", input_path)
    df = pd.read_csv(
        input_path,
        na_values=["?"],
        skipinitialspace=True,
    )

    if "sex" in df.columns and "gender" not in df.columns:
        df = df.rename(columns={"sex": "gender"})

    if "income_>50K." in df.columns and "income_>50K" not in df.columns:
        df = df.rename(columns={"income_>50K.": "income_>50K"})

    df.to_parquet(output_path, index=False)
    logger.info("Ingested data saved to %s", output_path)
    logger.info("Shape: %s", df.shape)

    return {
        "data": df,
        "input_path": input_path,
        "output_path": output_path,
        "pipeline": pipeline_cfg,
        "data_config": data_cfg,
    }
