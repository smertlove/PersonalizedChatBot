from datetime import datetime
from sentence_transformers import SentenceTransformer

import numpy as np
from numpy.linalg import norm

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from .config import DBRetrieverConfig

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class Vectorizer:

    def __init__(self):
        self.model = SentenceTransformer(DBRetrieverConfig.MODEL_NAME).to(DBRetrieverConfig.DEVICE)

    def vectorize(self, string: str):
        return self.model.encode(string, show_progress_bar=False)


class Database:

    time_tmpl = '%Y-%m-%d %H:%M:%S'

    def __init__(
        self,
        vectorizer:Vectorizer,
        populate:None|str = DBRetrieverConfig.POPULATE
    ) -> None:

        self.vectorizer = vectorizer

        dates = list()
        raw_facts = list()

        if populate is not None:

            with open(populate, "r", encoding="utf-8") as file:
                for fact_entry in file.read().strip().split("\n"):
                    date, fact = fact_entry.split("\t", maxsplit=1)
                    raw_facts.append(fact.strip())
                    dates.append(date.strip())

        embeddings = [
            self.vectorizer.vectorize(fact)
            for fact
            in raw_facts
        ]

        self.dim = self.vectorizer.vectorize("Это тестовое предложение").shape[0]

        self.client = QdrantClient(":memory:")
        self.collection_name = "facts"
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
        )

        points = []
        for embedding, date, fact in zip(embeddings, dates, raw_facts):
            point = PointStruct(
                id=uuid.uuid4(),
                vector=embedding,
                payload={
                    "fact":fact,
                    "date":date,
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def make_point(self, fact, date):
        return PointStruct(
            id=uuid.uuid4(),
            vector=self.vectorizer.vectorize(fact),
            payload={
                "fact":fact,
                "date":date,
            }
        )

    def append(self, fact: str):

        date = datetime.strftime(datetime.now(), self.time_tmpl)
        point = self.make_point(fact, date)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def delete(self, ids_) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids_
        )

    def get_top_n_closest_embeddings(
        self,
        string: str,
        n=DBRetrieverConfig.FACTS_PER_TRIPLET,
        threshold=DBRetrieverConfig.RANKING_THRESHOLD
    ):
        text_emb = self.vectorizer.vectorize(string)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=text_emb,
            limit=n,
            score_threshold=threshold,
            with_vectors=True,
        ).points

        return results


class CollisionResolver:

    def get_model(self):
        return AgglomerativeClustering(
        n_clusters = None,
        linkage    = 'single',
        metric     = 'cosine',
        distance_threshold  = DBRetrieverConfig.CLUSTERING_THRESHOLD
    )

    def del_collisions(self, points, times=DBRetrieverConfig.DISAMBIG_REPS):

        if times > 1:
            points = self.del_collisions(points, times=times-1)

        sorted_points = sorted(points, key=lambda p: p.payload["date"])

        distance_matrix = pairwise_distances([p.vector for p in sorted_points], metric='cosine')

        # Моделька нужна каждый раз новая
        cluster_model = self.get_model()

        clusters = cluster_model.fit_predict(distance_matrix)

        resolved_points = []
        for cluster_id in set(clusters):

            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_points = [sorted_points[i] for i in cluster_indices]
            cluster_points.sort(reverse=True, key=lambda p: p.payload["date"])

            resolved_points.append(cluster_points[0])

        return resolved_points

