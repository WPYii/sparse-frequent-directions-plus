import logging
from pathlib import Path

log_dir = Path("resources/logs")
log_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("RBDA")
logger.setLevel(logging.INFO)

if not logger.handlers:

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_dir / "rbda.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)