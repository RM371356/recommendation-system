import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
        Configura a semente para os geradores de números aleatórios em Python, NumPy e PyTorch, garantindo que os resultados do treinamento sejam reproduzíveis.
        Args:
            seed: Um valor inteiro usado para inicializar os geradores de números aleatórios.
    """
    # A função set_seed é usada para configurar a semente para os geradores de números aleatórios em Python, NumPy e PyTorch, garantindo que os resultados do treinamento sejam reproduzíveis.
    random.seed(seed)
    np.random.seed(seed)
    
    # Configura a semente para o gerador de números aleatórios do PyTorch, garantindo que as operações aleatórias no PyTorch sejam reproduzíveis.
    torch.manual_seed(seed)
    
    # Se uma GPU estiver disponível, configura a semente para o gerador de números aleatórios do CUDA, garantindo que as operações aleatórias na GPU também sejam reproduzíveis.
    torch.cuda.manual_seed_all(seed)