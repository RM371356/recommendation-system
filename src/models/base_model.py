from abc import ABC
from abc import abstractmethod

import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """
        BaseModel é uma classe abstrata que serve como base para os modelos de recomendação. 
        Ela herda de nn.Module do PyTorch e ABC (Abstract Base Class) para definir a estrutura básica que os modelos de recomendação devem seguir. 
        A classe exige que as subclasses implementem o método "forward", que define a passagem direta do modelo, recebendo os índices do usuário e do item e retornando a previsão de interação entre eles.
    """
    @abstractmethod
    def forward(self, user, item):
        """
            Método abstrato que define a passagem direta do modelo. 
            Ele deve ser implementado pelas subclasses para receber os índices do usuário e do item e retornar a previsão de interação entre eles.
            Args:
                user: Tensor contendo os índices dos usuários.
                item: Tensor contendo os índices dos itens.
            Returns:
                Tensor contendo a previsão de interação entre o usuário e o item.
        """
        pass