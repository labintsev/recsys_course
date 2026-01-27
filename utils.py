"""Утилиты для загрузки данных и других вспомогательных функций."""

from pathlib import Path

import pandas as pd


def load_data(
    data_dir: str = "data/ml-latest-small",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Загружает данные о рейтингах и фильмах.

    Args:
        data_dir: Путь к директории с данными

    Returns:
        Кортеж (ratings_df, movies_df)
    """
    ratings_df = pd.read_csv(Path(data_dir) / "ratings.csv")
    movies_df = pd.read_csv(Path(data_dir) / "movies.csv")
    return ratings_df, movies_df
