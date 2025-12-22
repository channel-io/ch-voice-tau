from decouple import config as _config
from pathlib import Path

class VoiceTauConfig:
    LOG_LEVEL: str = _config("LOG_LEVEL", default="INFO")
    OPENAI_API_KEY: str = _config("OPENAI_API_KEY", None)
    GOOGLE_API_KEY: str = _config("GOOGLE_API_KEY", default="")
    DATA_DIR: Path = _config("DATA_DIR", default=Path("data"))