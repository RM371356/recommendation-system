import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


class RecommendationMetrics:
    """
        A classe RecommendationMetrics é responsável por calcular as métricas de avaliação para o modelo de recomendação, incluindo acurácia, precisão, recall e F1-score. 
        Ela fornece métodos estáticos para calcular cada uma dessas métricas com base nos rótulos verdadeiros e nas previsões do modelo.
    """
    # Métodos estáticos para calcular as métricas de avaliação, permitindo que sejam chamadas diretamente na classe sem a necessidade de instanciar um objeto.
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    # Método estático para calcular a precisão, que mede a proporção de previsões positivas corretas em relação ao total de previsões positivas feitas pelo modelo.
    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(y_true, y_pred)

    # Método estático para calcular o recall, que mede a proporção de previsões positivas corretas em relação ao total de rótulos positivos reais, 
    # indicando a capacidade do modelo de identificar corretamente as interações positivas entre usuários e itens.
    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    # Método estático para calcular o F1-score, que é a média harmônica entre precisão e recall, 
    # fornecendo uma medida equilibrada da performance do modelo de recomendação, especialmente em casos de classes desbalanceadas.
    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(y_true, y_pred)