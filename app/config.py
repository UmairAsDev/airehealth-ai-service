"""
Application settings loaded from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from urllib.parse import quote_plus


class Settings(BaseSettings):
    # Database
    DB_LOCAL_USERNAME: str = "root"
    DB_LOCAL_PASSWORD: str = ""
    DB_LOCAL_NAME: str = ""
    DB_LOCAL_HOST: str = "localhost"
    DB_LOCAL_PORT: int = 3306

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"
    environment: str = "development"

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{quote_plus(self.DB_LOCAL_USERNAME)}:{quote_plus(self.DB_LOCAL_PASSWORD)}"
            f"@{self.DB_LOCAL_HOST}:{self.DB_LOCAL_PORT}/{self.DB_LOCAL_NAME}"
        )

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}






@lru_cache()
def get_settings() -> Settings:
    return Settings()
