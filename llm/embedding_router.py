import os

import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from prompts.route_examples import ROUTE_EXAMPLES


class EmbeddingRouter:
    def __init__(
        self,
        examples: dict[str, list[str]] = ROUTE_EXAMPLES,
        model: str = "text-embedding-3-small",
        confidence_threshold: float = 0.30,
        margin_threshold: float = 0.03,
    ):
        load_dotenv()
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"), model=model)
        self._route_centroids = self._build_centroids(examples)

    def _build_centroids(self, examples: dict[str, list[str]]) -> dict[str, np.ndarray]:
        routes = list(examples.keys())
        flat_texts = [text for route in routes for text in examples[route]]
        flat_vectors = np.array(self.embeddings.embed_documents(flat_texts))

        centroids = {}
        offset = 0
        for route in routes:
            count = len(examples[route])
            centroids[route] = flat_vectors[offset:offset + count].mean(axis=0)
            offset += count
        return centroids

    def classify(self, query: str) -> tuple[str | None, float]:
        query_vector = np.array(self.embeddings.embed_query(query))
        scores = {
            route: self._cosine_similarity(query_vector, centroid)
            for route, centroid in self._route_centroids.items()
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_route, best_score = ranked[0]
        second_best_score = ranked[1][1]

        if best_score >= self.confidence_threshold and (best_score - second_best_score) >= self.margin_threshold:
            return best_route, best_score
        return None, best_score

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


_router_instance: EmbeddingRouter | None = None


def get_embedding_router() -> EmbeddingRouter:
    global _router_instance
    if _router_instance is None:
        _router_instance = EmbeddingRouter()
    return _router_instance
