from abc import ABC, abstractmethod

import pandas as pd

from src.config.settings import settings


class PreprocessStrategy(ABC):
    """
        PreprocessStrategy é uma classe abstrata que define a interface para estratégias
          de pré-processamento de dados. 
        Ela exige que as subclasses implementem o método "process", que recebe um
          DataFrame do pandas e retorna um DataFrame processado. 
        Essa estrutura permite a implementação de diferentes estratégias de
          pré-processamento, como binarização de avaliações, normalização, etc.,
            que podem ser facilmente integradas ao pipeline de recomendação.
    """ 
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class RatingBinarizer(PreprocessStrategy):
    """
        RatingBinarizer é uma implementação concreta da PreprocessStrategy que converte
          as avaliações numéricas em rótulos binários. 
        Ele classifica as avaliações como positivas (1) para avaliações maiores ou
          iguais a 4 e negativas (0) para avaliações menores que 4. 
        Essa abordagem é comumente usada em sistemas de recomendação para simplificar
         o problema de previsão, transformando-o em uma tarefa de classificação binária.
    """
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()
        processed_df["label"] = (
            processed_df["rating"] >= 4
        ).astype(int)
        return processed_df


def run_preprocessing() -> None:
    """
        run_preprocessing é a função principal que executa o processo de
          pré-processamento dos dados de avaliações.
            Ela lê os dados de avaliações a partir do arquivo "ratings.csv", aplica a
            estratégia de binarização de avaliações usando a classe RatingBinarizer,
            e salva o DataFrame processado em um novo arquivo CSV chamado
            "ratings_processed.csv".
        Args:
            None: Esta função não recebe argumentos, pois os caminhos dos arquivos 
            de entrada e saída são fixos.
        Returns:
            None: O resultado do pré-processamento é salvo em um arquivo CSV, e a função
              não retorna nenhum valor.
    """
    # Carrega os dados de avaliações a partir do arquivo "ratings.csv" usando o pandas,
    #  criando um DataFrame que contém as informações de avaliações dos usuários
    #  para os filmes.
    ratings = pd.read_csv("data/raw/ratings.csv")

    if settings.sample_data and len(ratings) > settings.sample_size:
        ratings = ratings.sample(
            n=settings.sample_size,
            random_state=settings.seed,
        ).reset_index(drop=True)

    # Aplica a estratégia de pré-processamento de binarização de avaliações,
    #  criando uma instância da classe RatingBinarizer e chamando seu método
    #  "process" para transformar as avaliações numéricas em rótulos binários.
    strategy = RatingBinarizer()
    processed = strategy.process(ratings)

    # Salva o DataFrame processado em um novo arquivo CSV chamado
    # "ratings_processed.csv", sem incluir o índice do DataFrame no arquivo de saída.
    processed.to_csv(
        "data/processed/ratings_processed.csv",
        index=False,
    )


if __name__ == "__main__":
    run_preprocessing()