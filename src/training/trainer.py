import time

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:

    def __init__(
        self,
        model,
        learning_rate: float,
    ) -> None:

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        logger.info(
            f"Device selected: {self.device}"
        )

        self.model = model.to(
            self.device
        )

        self.optimizer = (
            torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
            )
        )

        self.loss_function = (
            torch.nn.BCEWithLogitsLoss()
        )

        self.early_stopping = (
            EarlyStopping(
                patience=3
            )
        )
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> float:

        self.model.eval()

        total_loss = 0.0

        with torch.no_grad():

            for users, items, labels in val_loader:

                users = users.to(self.device)
                items = items.to(self.device)
                labels = labels.to(self.device)

                predictions = self.model(
                    users,
                    items,
                )

                loss = self.loss_function(
                    predictions.squeeze(),
                    labels.float(),
                )

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> None:

        logger.info(
            f"Starting training for {epochs} epochs"
        )

        for epoch in range(epochs):

            start_time = time.time()

            self.model.train()

            epoch_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                leave=True,
            )

            for batch_idx, (
                users,
                items,
                labels,
            ) in enumerate(progress_bar):

                users = users.to(
                    self.device
                )

                items = items.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                predictions = (
                    self.model(
                        users,
                        items,
                    )
                )

                loss = self.loss_function(
                    predictions.squeeze(),
                    labels.float(),
                )

                self.optimizer.zero_grad()

                loss.backward()

                self.optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

                if batch_idx % 100 == 0:

                    logger.info(
                        f"Epoch={epoch+1} "
                        f"Batch={batch_idx} "
                        f"Loss={loss.item():.4f}"
                    )

            avg_loss = (
                epoch_loss
                / len(train_loader)
            )
 
            val_loss = self.validate(
                val_loader,
            )

            elapsed = (
                time.time()
                - start_time
            )

            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train={avg_loss:.4f} | "
                f"Validation={val_loss:.4f}"
                f" | Time={elapsed:.2f}s"
            )

            mlflow.log_metric(
                "train_loss",
                avg_loss,
                step=epoch,
            )

            mlflow.log_metric(
                "val_loss",
                val_loss,
                step=epoch,
            )

            stop_training = (
                self.early_stopping.step(
                    val_loss,
                    self.model,
                )
            )

            if stop_training:

                logger.info(
                    "Early stopping triggered."
                )

                break

        logger.info(
            "Training finished."
        )