import torch
import torch.nn as nn

from src.models.base_model import BaseModel


class MLPRecommender(BaseModel):
    """
        MLPRecommender é uma classe que implementa um modelo de recomendação baseado em uma rede neural de múltiplas camadas (MLP). 
        Ela herda da classe BaseModel e define a arquitetura do modelo, que inclui camadas de embedding para usuários e itens, seguidas por camadas totalmente conectadas com funções de ativação ReLU e dropout para evitar overfitting. 
        O método "forward" implementa a passagem direta do modelo, concatenando as embeddings dos usuários e itens e passando-as pelas camadas totalmente conectadas para gerar a previsão de interação entre eles.
    """
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
    ):
        super().__init__()

        # Define a camada de embedding para os usuários, que converte os índices dos usuários em representações vetoriais densas de dimensão "embedding_dim", 
        # permitindo que o modelo capture as características latentes dos usuários durante o treinamento do modelo de recomendação.
        self.user_embedding = nn.Embedding(
            n_users,
            embedding_dim,
        )

        # Define a camada de embedding para os itens, que converte os índices dos itens em representações vetoriais densas de dimensão "embedding_dim", 
        # permitindo que o modelo capture as características latentes dos itens durante o treinamento do modelo de recomendação.
        self.item_embedding = nn.Embedding(
            n_items,
            embedding_dim,
        )

        # Define as camadas totalmente conectadas para a arquitetura do modelo, que consistem em uma sequência de camadas lineares intercaladas 
        # com funções de ativação ReLU e dropout para melhorar a capacidade de generalização do modelo durante o treinamento.
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, user, item):
        """
            Implementa a passagem direta do modelo, recebendo os índices do usuário e do item, obtendo suas embeddings correspondentes, concatenando-as e passando-as pelas camadas totalmente conectadas para gerar a previsão de interação entre eles.
            Args:
                user: Tensor contendo os índices dos usuários.
                item: Tensor contendo os índices dos itens.
            Returns:
                Tensor contendo a previsão de interação entre o usuário e o item.
        """
        # Obtém as embeddings dos usuários e itens usando as camadas de embedding definidas na arquitetura do modelo, 
        # convertendo os índices dos usuários e itens em representações vetoriais densas que capturam suas características latentes.
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)

        # Concatena as embeddings dos usuários e itens ao longo da dimensão 1 (dim=1) para criar uma representação combinada que será usada como entrada para as camadas totalmente conectadas, 
        # permitindo que o modelo aprenda a interação entre os usuários e itens com base em suas características latentes.
        concatenated = torch.cat(
            [user_embedding, item_embedding],
            dim=1,
        )

        # Passa a representação concatenada pelas camadas totalmente conectadas definidas na arquitetura do modelo para gerar a previsão de interação entre o usuário e o item, 
        # aplicando as funções de ativação ReLU e dropout conforme definido na sequência de camadas.
        return self.layers(concatenated)