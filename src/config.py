from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILEPATH = Path(__file__).parent / ".env"

class Config(BaseSettings):
    """
    Configurations for the application
    """
    model_config = SettingsConfigDict(
        env_file= ENV_FILEPATH,    
        env_file_encoding="utf-8",
        extra="ignore",
    )

    NAME: str
    SEED: int
    TRAIN_TEST_SPLIT: float
    TRAIN_VALIDATION_SPLIT: float
    HOST: str
    PORT: int
    RUN_WITH_RELOAD: bool
    API_DESCRIPTION: str
    API_VERSION: str
    KAGGLE_API_TOKEN: str

load_dotenv(ENV_FILEPATH)
config = Config()