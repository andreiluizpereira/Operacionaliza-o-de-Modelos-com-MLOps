from __future__ import annotations

from src import evaluate, ingestion, preprocessing, train, validation
from src.utils import get_logger, load_project_config


def main() -> None:
    config = load_project_config()
    logger = get_logger("pipeline", config.get("logging"))

    logger.info("Starting pipeline")
    ingest_result = ingestion.run(config)
    validation.run(ingest_result["data"], config)
    prep_result = preprocessing.run(ingest_result["data"], config)
    train_result = train.run(prep_result, config)
    evaluate.run(train_result, config)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()
