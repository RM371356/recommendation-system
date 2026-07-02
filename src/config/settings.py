from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "recommendation-system"

    model_name: str = "mlp"

    epochs: int = 10
    batch_size: int = 1024
    learning_rate: float = 0.001
    embedding_dim: int = 64

    seed: int = 42

    sample_data: bool = True
    sample_size: int = 100000

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()