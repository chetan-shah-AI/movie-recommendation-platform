from pydantic_settings import BaseSettings

import os
from dotenv import load_dotenv

load_dotenv() 


class Settings(BaseSettings):

    APP_NAME: str = "Movie Recommendation Platform"

    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    LLM_MODEL: str = os.getenv('LLM_MODEL')

    LANGFUSE_PUBLIC_KEY: str = os.getenv('LANGFUSE_PUBLIC_KEY')
    LANGFUSE_SECRET_KEY: str = os.getenv('LANGFUSE_SECRET_KEY')

    # IMPORTANT FIX
    LANGFUSE_BASE_URL: str = os.getenv('LANGFUSE_BASE_URL')

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()