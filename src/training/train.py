import json
import sys
import io
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Força encoding UTF-8 no stdout/stderr para evitar erros no Windows (CP1252)
# com emojis que o MLflow imprime internamente
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.config.settings import settings
from src.data.dataset import RatingsDataset
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.seed import set_seed
from src.utils.logger import get_logger

logger = get_logger(__name__)

def create_dataloader(
    csv_path: str,
) -> DataLoader:
    """
    Create DataLoader from feature dataset.
    """

    dataframe = pd.read_csv(csv_path)

    dataset = RatingsDataset(
        dataframe=dataframe,
    )
    logger.info(
    f"Dataset shape: {dataframe.shape}"
    )

    logger.info(
        f"Users: {dataframe['user_idx'].nunique():,}"
    )

    logger.info(
        f"Movies: {dataframe['movie_idx'].nunique():,}"
    )

    logger.info(
        f"Interactions: {len(dataframe):,}"
    )
    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=True,
    )


def create_model(
    dataframe: pd.DataFrame,
):
    """
    Create recommendation model.
    """

    return ModelFactory.create(
        model_name=settings.model_name,
        n_users=dataframe["user_idx"].nunique(),
        n_items=dataframe["movie_idx"].nunique(),
        embedding_dim=settings.embedding_dim,
    )


def configure_mlflow() -> None:
    """
    Configure MLflow tracking.
    """

    try:
        # Testa a conectividade com o servidor do MLflow
        import urllib.request
        urllib.request.urlopen(settings.mlflow_tracking_uri, timeout=1.5)
        mlflow.set_tracking_uri(
            settings.mlflow_tracking_uri,
        )
        logger.info(f"Using MLflow server tracking URI: {settings.mlflow_tracking_uri}")
    except Exception as exc:
        logger.warning(
            f"Could not connect to MLflow server at {settings.mlflow_tracking_uri}: {exc}. "
            "Falling back to local tracking directory (file:./mlruns)."
        )
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(
        settings.mlflow_experiment_name,
    )


def save_model(
    model: torch.nn.Module,
) -> None:
    """
    Persist model locally in DVC output directory and API artifacts directory.
    """

    for dir_path in ["models", "artifacts/model"]:
        path = Path(dir_path)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            path / "best_model.pth",
        )


def train_pipeline() -> None:
    """
    Execute training pipeline.
    """

    set_seed(
        settings.seed,
    )

    configure_mlflow()

    feature_file = (
        "data/features/features.csv"
    )
    
    train_file = "data/splits/train.csv"
    val_file = "data/splits/val.csv"

    if not Path(feature_file).exists():
        raise FileNotFoundError(
            f"Feature file not found: {feature_file}"
        )

    if not Path(train_file).exists() or not Path(val_file).exists():
        raise FileNotFoundError(
            "Split files not found. Ensure split_data script has run."
        )

    dataframe = pd.read_csv(
        feature_file,
    )

    train_loader = create_dataloader(
        train_file,
    )
    val_loader = create_dataloader(
        val_file,
    )

    # Cria o modelo baseado no dataframe original (contagem total de users/items)
    model = create_model(
        dataframe,
    )

    trainer = Trainer(
        model=model,
        learning_rate=settings.learning_rate,
    )

    with mlflow.start_run() as run:

        mlflow.log_params(
            {
                "model_name": settings.model_name,
                "epochs": settings.epochs,
                "batch_size": settings.batch_size,
                "learning_rate": settings.learning_rate,
                "embedding_dim": settings.embedding_dim,
                "n_users": dataframe["user_idx"].nunique(),
                "n_items": dataframe["movie_idx"].nunique(),
            }
        )

        trainer.train(
            train_loader=train_loader,
            epochs=settings.epochs,
            val_loader=val_loader,
        )

        # Recupera os pesos do melhor modelo salvos pelo early stopping
        best_model_path = Path("models/best_model.pth")
        if best_model_path.exists():
            logger.info("Loading best model weights from early stopping...")
            model.load_state_dict(torch.load(best_model_path, map_location="cpu"))

        save_model(
            model,
        )

        # Salva o arquivo de configuração para recriação do modelo na inferência
        config = {
            "model_name": settings.model_name,
            "n_users": int(dataframe["user_idx"].nunique()),
            "n_items": int(dataframe["movie_idx"].nunique()),
            "embedding_dim": int(settings.embedding_dim),
        }
        
        for dir_path in ["models", "artifacts/model"]:
            config_path = Path(dir_path) / "config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "w", encoding="utf8") as file:
                json.dump(config, file, indent=4)

        #mlflow.pytorch.log_model(
        #    pytorch_model=model,
        #    artifact_path="model",
        #)

        #model_uri = (
        #    f"runs:/{run.info.run_id}/model"
        #)

        #try:
        #    mlflow.register_model(
        #        model_uri=model_uri,
        #        name="MovieRecommender",
        #    )
        #except Exception as exc:
        #    print(
        #        f"Registry warning: {exc}"
        #    )

    print(
        "Training completed successfully."
    )


if __name__ == "__main__":
    train_pipeline()