"""Pipeline entrypoint."""

from __future__ import annotations

from src import evaluate, ingestion, preprocessing, train, validation
from src.utils import get_logger, load_project_config


# Antes de qualquer etapa, esta funcao carrega a configuracao e liga as partes do projeto.
# Ela roda primeiro porque organiza a sequencia do pipeline e evita passos soltos.
def main() -> None:
    """Execute the full pipeline in a simple, readable order."""
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
