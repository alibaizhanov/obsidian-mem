"""
Embedder — генерация векторных embeddings.

Использует sentence-transformers с моделью all-MiniLM-L6-v2:
- 80MB размер модели
- 384-мерные вектора
- На Mac M1 использует Metal GPU автоматически
- Полностью локально, ничего не отправляется в облако
"""

from sentence_transformers import SentenceTransformer
from typing import Optional
import numpy as np


class Embedder:
    """Локальный генератор embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy loading — модель загружается при первом использовании"""
        if self._model is None:
            print(f"🧠 Загружаю модель {self.model_name}...")
            self._model = SentenceTransformer(self.model_name)
            print(f"✅ Модель загружена ({self.dimensions}D)")
        return self._model

    @property
    def dimensions(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        """Embed одного текста → вектор"""
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed нескольких текстов → матрица векторов"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity между двумя векторами"""
        return float(np.dot(vec_a, vec_b))

    def search(self, query_vec: np.ndarray, corpus_vecs: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
        """
        Поиск top-K ближайших векторов.
        Возвращает [(index, score), ...]
        """
        scores = np.dot(corpus_vecs, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


if __name__ == "__main__":
    embedder = Embedder()

    # Тест
    texts = [
        "PostgreSQL connection pool exhaustion",
        "Проблемы с базой данных при высокой нагрузке",
        "React компонент для дашборда",
        "Kafka consumer lag issues",
    ]

    vectors = embedder.embed_batch(texts)
    print(f"\n📐 Vectors shape: {vectors.shape}")

    query = embedder.embed("проблема с производительностью базы данных")
    results = embedder.search(query, vectors, top_k=3)

    print(f"\n🔍 Query: 'проблема с производительностью базы данных'")
    for idx, score in results:
        print(f"   {score:.3f}  {texts[idx]}")
