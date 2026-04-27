from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))
ARTIFACT_ROOT = Path(os.getenv("ARTIFACT_ROOT", PROJECT_ROOT / "artifacts"))


class Settings:
    FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
    FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

    DATA_ROOT = DATA_ROOT
    ARTIFACT_ROOT = ARTIFACT_ROOT
    DB_PATH = DATA_ROOT / "app.db"

