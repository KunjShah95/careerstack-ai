from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    # JSearch via RapidAPI. Optional: without it the JSearch source is
    # disabled and discovery runs on Adzuna + Greenhouse (see jobs/jsearch.py).
    rapidapi_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    demo_mode: bool = False

    # No default: the app must not start on a guessable secret. Generate
    # one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 10080  # 7 days


settings = Settings()
