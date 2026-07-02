from pathlib import Path

import pandas as pd


class DataLoader:
    """
        DataLoader é uma classe responsável por carregar os dados de avaliações e filmes
          a partir de arquivos CSV localizados em um diretório especificado. 
        Ele fornece métodos para ler os arquivos "ratings.csv" e "movies.csv" e retornar
          os dados como DataFrames do pandas, facilitando o acesso e a manipulação dos
            dados para as etapas subsequentes do processo de recomendação.
    """
    def __init__(self, data_dir: str):
        """
            Inicializa o DataLoader com o diretório onde os arquivos de dados estão
              localizados.
            Args:
                data_dir: O caminho para o diretório que contém os arquivos
                  "ratings.csv" e "movies.csv".
        """
        self.data_dir = Path(data_dir)

    def load_ratings(self) -> pd.DataFrame:
        """Carrega os dados de avaliações a partir do arquivo "ratings.csv" e retorna
          um DataFrame do pandas contendo as informações de avaliações dos usuários
            para os filmes.
            Returns:
                Um DataFrame do pandas contendo os dados de avaliações dos usuários
                  para os filmes, com colunas como "userId", "movieId", "rating"
                    e "timestamp".
        """
        return pd.read_csv(self.data_dir / "ratings.csv")

    def load_movies(self) -> pd.DataFrame:
        """Carrega os dados de filmes a partir do arquivo "movies.csv" e retorna um
          DataFrame do pandas contendo as informações dos filmes.
            Returns:
                Um DataFrame do pandas contendo os dados dos filmes, com colunas como
                  "movieId", "title" e "genres".
        """
        return pd.read_csv(self.data_dir / "movies.csv")