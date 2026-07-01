import json
from pathlib import Path

import mlflow
import pandas as pd
import torch

from src.config.settings import settings
from src.models.model_factory import ModelFactory
from src.evaluation.metrics import RecommendationMetrics


def evaluate_model() -> None:
    """
    A função evaluate_model é responsável por avaliar o desempenho do modelo de recomendação usando o conjunto de testes.
    Ela carrega o modelo treinado, infere previsões sobre o split de teste e calcula métricas de desempenho reais.
    """
    # Verifica caminhos de dependências
    test_path = Path("data/splits/test.csv")
    model_path = Path("artifacts/model/best_model.pth")
    config_path = Path("artifacts/model/config.json")

    if not test_path.exists() or not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Evaluation artifacts not found. Ensure training stage has successfully executed."
        )

    # Carrega as configurações de arquitetura do modelo
    with open(config_path, "r", encoding="utf8") as file:
        config = json.load(file)

    # Reconstrói a estrutura do modelo e carrega os pesos treinados
    model = ModelFactory.create(
        model_name=config["model_name"],
        n_users=config["n_users"],
        n_items=config["n_items"],
        embedding_dim=config["embedding_dim"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Carrega o conjunto de testes
    test_df = pd.read_csv(test_path)
    users = torch.tensor(test_df["user_idx"].values, dtype=torch.long)
    items = torch.tensor(test_df["movie_idx"].values, dtype=torch.long)
    y_true = test_df["label"].values

    # Realiza as previsões do modelo
    with torch.no_grad():
        predictions = model(users, items).squeeze()
        probs = torch.sigmoid(predictions).numpy()
        # Garante vetor 1D caso haja apenas uma amostra de teste
        if probs.ndim == 0:
            probs = probs.reshape(-1)
        y_pred = (probs >= 0.5).astype(int)

    # Calcula as métricas reais do modelo
    metrics = RecommendationMetrics()
    accuracy = metrics.accuracy(y_true, y_pred)
    precision = metrics.precision(y_true, y_pred)
    recall = metrics.recall(y_true, y_pred)
    f1 = metrics.f1(y_true, y_pred)

    print(f"Test Metrics -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Configuração de fallback do MLflow
    try:
        import urllib.request
        urllib.request.urlopen(settings.mlflow_tracking_uri, timeout=1.5)
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    except Exception:
        mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment(settings.mlflow_experiment_name)

    # Registra as métricas reais calculadas no MLflow
    with mlflow.start_run():
        mlflow.log_params({
            "test_samples": len(test_df),
        })
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)


if __name__ == "__main__":
    evaluate_model()