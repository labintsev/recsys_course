import unittest

from sem1 import evaluate_rec_systems, random_recommend, top_n_recommend
from utils import load_data


class TestSeminar1(unittest.TestCase):
    def setUp(self):
        self.score = 0
        return super().setUp()

    def test1(self):
        # Тест случайных рекомендаций
        recs = random_recommend(n_recommendations=10)
        self.assertEqual(len(recs), 10)
        self.assertTrue(all(isinstance(r, int) for r in recs))
        ratings_df, _ = load_data()
        all_movie_ids = ratings_df["movieId"].unique()
        self.assertTrue(all(r in all_movie_ids for r in recs))
        self.score += 3

    def test2(self):
        # Тест популярных фильмов
        recs_50 = top_n_recommend(n_recommendations=10, min_ratings=50)
        self.assertEqual(len(recs_50), 10)
        top_10_ids_50 = (318, 858, 2959, 1276, 750, 904, 1221, 48516, 1213, 912)
        self.assertTrue(all(rec[0] in top_10_ids_50 for rec in recs_50))
        recs_10 = top_n_recommend(n_recommendations=10)
        self.assertEqual(len(recs_10), 10)
        top_10_ids_10 = (1041, 3451, 1178, 1104, 2360, 1217, 318, 951, 1927, 922)
        self.assertTrue(all(rec[0] in top_10_ids_10 for rec in recs_10))
        self.score += 3

    def test3(self):
        # Тест оценки системы
        metrics = evaluate_rec_systems()
        self.assertAlmostEqual(metrics["random_accuracy"], 0.1, places=2)
        self.assertAlmostEqual(metrics["popular_accuracy"], 0.1, places=2)
        metrics = evaluate_rec_systems(user_id=609)
        self.assertAlmostEqual(metrics["random_accuracy"], 0.0, places=2)
        self.assertAlmostEqual(metrics["popular_accuracy"], 0.1, places=2)
        metrics = evaluate_rec_systems(user_id=608)
        self.assertAlmostEqual(metrics["random_accuracy"], 0.2, places=2)
        self.assertAlmostEqual(metrics["popular_accuracy"], 0.1, places=2)
        self.score += 4

    def tearDown(self):
        print("============")
        print(f"Seminar 1 score = {self.score}")
        return super().tearDown()


if __name__ == "__main__":
    unittest.main()
