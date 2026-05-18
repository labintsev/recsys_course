"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(tokenizer=lambda s: s.split("|"), lowercase=False)
        genre_vectors = vectorizer.fit_transform(self.movies_df["genres"]).toarray()

        max_movie_id = int(self.movies_df["movieId"].max())
        embeddings = np.zeros((max_movie_id + 1, genre_vectors.shape[1]))

        for row_index, movie_id in enumerate(self.movies_df["movieId"].astype(int)):
            embeddings[movie_id] = genre_vectors[row_index]

        self.embeddings = embeddings

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        if item_id <= 0 or item_id >= self.embeddings.shape[0]:
            return 0.0

        target_vec = self.embeddings[item_id]
        target_norm = np.linalg.norm(target_vec)
        if target_norm == 0:
            return 0.0

        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return 0.0

        rated_embeddings = self.embeddings[rated_items]
        rated_norms = np.linalg.norm(rated_embeddings, axis=1)

        valid_mask = rated_norms > 0
        if not np.any(valid_mask):
            return 0.0

        valid_rated_items = rated_items[valid_mask]
        valid_embeddings = rated_embeddings[valid_mask]
        valid_norms = rated_norms[valid_mask]

        similarities = (valid_embeddings @ target_vec) / (valid_norms * target_norm)

        if len(similarities) == 0:
            return 0.0

        top_indices = np.argsort(similarities)[::-1][:k]
        top_similarities = similarities[top_indices]
        top_ratings = user_ratings[valid_rated_items[top_indices]]

        similarity_sum = top_similarities.sum()
        if similarity_sum == 0:
            return 0.0

        predicted_rating = np.dot(top_similarities, top_ratings) / similarity_sum
        return float(np.clip(predicted_rating, 0.0, 5.0))

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        if len(rated_items) == 0:
            return []

        rated_embeddings = self.embeddings[rated_items]
        rated_ratings = user_ratings[rated_items]

        valid_mask = np.linalg.norm(rated_embeddings, axis=1) > 0
        if not np.any(valid_mask):
            return []

        rated_embeddings = rated_embeddings[valid_mask]
        rated_ratings = rated_ratings[valid_mask]

        user_profile = np.average(rated_embeddings, axis=0, weights=rated_ratings)
        profile_norm = np.linalg.norm(user_profile)
        if profile_norm == 0:
            return []

        unrated_items = np.where(user_ratings == 0)[0]
        unrated_items = unrated_items[unrated_items > 0]
        unrated_items = unrated_items[unrated_items < self.embeddings.shape[0]]

        candidate_embeddings = self.embeddings[unrated_items]
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)

        valid_candidates_mask = candidate_norms > 0
        valid_items = unrated_items[valid_candidates_mask]
        valid_embeddings = candidate_embeddings[valid_candidates_mask]
        valid_norms = candidate_norms[valid_candidates_mask]

        similarities = (valid_embeddings @ user_profile) / (valid_norms * profile_norm)
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]

        return [int(item_id) for item_id in valid_items[top_indices]]


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
