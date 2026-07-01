from sklearn.dummy import DummyClassifier


class BaselineModel:
    """
        Um modelo de baseline que utiliza um classificador dummy para fazer previsões.
    """
    def __init__(self):
        """
            Inicializa o modelo de baseline com um classificador dummy que utiliza a estratégia "most_frequent", ou seja, sempre prevê a classe mais frequente presente nos dados de treinamento.
        """
        self.model = DummyClassifier(strategy="most_frequent")

    def fit(self, x_train, y_train):
        """
            Treina o modelo de baseline usando os dados de treinamento fornecidos.
            Args:
                x_train: Os dados de treinamento (características).
                y_train: Os rótulos de treinamento (classes).
            Returns:
                None
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        """
            Faz previsões usando o modelo treinado.
            Args:
                x_test: Os dados de teste (características).
            Returns:
                Um array com as previsões do modelo.
        """ 
        return self.model.predict(x_test)