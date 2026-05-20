from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str = "Movie Recommendation Platform"

    LANGFUSE_PUBLIC_KEY: str = "pk-lf-3fc41186-a399-420f-a9e9-2d6c7ab44650"
    LANGFUSE_SECRET_KEY: str = "sk-lf-123c6142-77da-46f5-b7ba-3d4511556821"

    # IMPORTANT FIX
    LANGFUSE_BASE_URL: str = (
        "https://cloud.langfuse.com"
    )

    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()