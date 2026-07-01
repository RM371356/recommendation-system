from pathlib import Path

import torch


class EarlyStopping:

    def __init__(
        self,
        patience: int = 3,
    ) -> None:

        self.patience = patience
        self.counter = 0
        self.best_loss = float("inf")

    def step(
        self,
        validation_loss: float,
        model,
    ) -> bool:

        if validation_loss < self.best_loss:

            self.best_loss = validation_loss
            self.counter = 0

            Path("models").mkdir(
                exist_ok=True,
                parents=True,
            )

            torch.save(
                model.state_dict(),
                "models/best_model.pth",
            )

            return False

        self.counter += 1

        return self.counter >= self.patience