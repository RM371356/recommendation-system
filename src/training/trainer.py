import mlflow
import torch
from torch.utils.data import DataLoader
from src.training.early_stopping import EarlyStopping

class Trainer:
    """
        A classe Trainer é responsável por gerenciar o processo de treinamento do modelo de recomendação. 
        Ela recebe um modelo, uma taxa de aprendizado e um conjunto de dados de treinamento, e executa o loop de treinamento por um número especificado de épocas, calculando a perda e atualizando os pesos do modelo usando o otimizador Adam. 
        Durante o treinamento, as métricas de perda são registradas no MLflow para monitoramento e análise posterior.
    """
    # Construtor da classe Trainer, que inicializa o modelo a ser treinado, o otimizador Adam para atualizar os pesos do modelo durante o treinamento, e a função de perda Binary Cross-Entropy com Logits, que é adequada para tarefas de classificação binária.
    def __init__(
        self,
        model,
        learning_rate: float,
    ):
        # Inicializa o Trainer com o modelo a ser treinado e a taxa de aprendizado para o otimizador.
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )

        # Define a função de perda como Binary Cross-Entropy com Logits, que é adequada para tarefas de classificação binária, onde o modelo prevê a probabilidade de interação entre usuários e itens.
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    # Método de treinamento que executa o processo de treinamento do modelo usando o conjunto de dados de treinamento fornecido e o número especificado de épocas.
    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: DataLoader | None = None,
    ) -> None:
        early_stopping = EarlyStopping(patience=3)

        # Loop de treinamento que itera por um número especificado de épocas, permitindo que o modelo aprenda a partir dos dados de treinamento ao longo do tempo.
        for epoch in range(epochs): 
            self.model.train()
            epoch_loss = 0

            # Loop interno que itera sobre os lotes de dados fornecidos pelo DataLoader, onde cada lote contém os índices dos usuários, índices dos itens e os rótulos (avaliações) correspondentes para o treinamento do modelo de recomendação.
            for users, items, labels in train_loader: 
                predictions = self.model(users, items)

                # Calcula a perda entre as previsões do modelo e os rótulos reais usando a função de perda definida, que mede a discrepância entre as previsões do modelo e as avaliações reais dos usuários para os itens, permitindo que o modelo ajuste seus pesos para melhorar a precisão das previsões ao longo do tempo.
                loss = self.loss_function(
                    predictions.squeeze(),
                    labels.float(),
                )

                # Zera os gradientes do otimizador antes de calcular os novos gradientes.
                self.optimizer.zero_grad()
                # Calcula os gradientes da perda em relação aos parâmetros do modelo.
                loss.backward()
                # Atualiza os parâmetros do modelo usando os gradientes calculados.
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)

            # Loga a perda total para a época atual no MLflow, permitindo que o progresso do treinamento seja monitorado ao longo do tempo, e associando a métrica de perda à época correspondente para análise posterior.
            mlflow.log_metric(
                "train_loss",
                avg_train_loss,
                step=epoch,
            )

            # Avaliação no conjunto de validação se fornecido
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for users, items, labels in val_loader:
                        predictions = self.model(users, items)
                        loss = self.loss_function(
                            predictions.squeeze(),
                            labels.float(),
                        )
                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                # Executa o passo de early stopping salvando o melhor modelo se a perda de validação diminuir
                if early_stopping.step(avg_val_loss, self.model):
                    print("Early stopping triggered. Training stopped.")
                    break
            else:
                print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f}")