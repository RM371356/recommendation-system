from typing import Any
from src.models.mlp_model import MLPRecommender


class ModelFactory:
    """
        ModelFactory é uma classe de fábrica que fornece uma interface para criar instâncias de modelos de recomendação com base em um nome de modelo especificado.
        Ele suporta a criação de diferentes tipos de modelos de recomendação, como o MLPRecommender, e pode ser facilmente estendido para incluir novos modelos no futuro.
    """
    @staticmethod
    
    def create(model_name: str, **kwargs: Any):
        """
            cria uma instância do modelo de recomendação especificado.
            Args:
                model_name: O nome do modelo a ser criado (por exemplo, "mlp").
                **kwargs: Argumentos adicionais necessários para a criação do modelo, como número de usuários, número de itens, etc.
            Returns:
                Uma instância do modelo de recomendação solicitado. 
        """
        if model_name == "mlp":
            return MLPRecommender(**kwargs)
        raise ValueError(f"Model {model_name} not supported")