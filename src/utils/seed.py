import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
        Configura a semente para os geradores de números aleatórios em Python, NumPy
          e PyTorch, garantindo que os resultados do treinamento sejam reproduzíveis.
        Args:
            seed: Um valor inteiro usado para inicializar os geradores de números
              aleatórios.
    """
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed_all(seed)