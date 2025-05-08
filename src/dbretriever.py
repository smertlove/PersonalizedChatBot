from datetime import datetime
from sentence_transformers import SentenceTransformer

import numpy as np
from numpy.linalg import norm

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from .config import DBRetrieverConfig


class Vectorizer:

    model = SentenceTransformer(DBRetrieverConfig.MODEL_NAME).to(DBRetrieverConfig.DEVICE)

    @classmethod
    def vectorize(cls, string: str):
        return cls.model.encode(string)


class Database:

    time_tmpl = '%Y-%m-%d %H:%M:%S'

    def __init__(self, vectorizer:Vectorizer, populate:None|str = DBRetrieverConfig.POPULATE) -> None:

        self.vectorizer = vectorizer

        self.dates = list()
        self.raw_facts = list()

        if populate is not None:

            with open(populate, "r", encoding="utf-8") as file:
                for fact_entry in file.read().strip().split("\n"):
                    date, fact = fact_entry.split("\t", maxsplit=1)
                    self.raw_facts.append(fact.strip())
                    self.dates.append(date.strip())

        self.embeddings = [
            self.vectorizer.vectorize(fact)
            for fact
            in self.raw_facts
        ]

        self.ptr = 0

    def __getitem__(self, i: int) -> tuple[int, datetime, str, list[float]]:
        return (
            i,
            self.dates[i],
            self.raw_facts[i],
            self.embeddings[i]
        )

    def append(self, fact: str):
        embedding = self.vectorizer.vectorize(fact)
        date = datetime.strftime(datetime.now(), self.time_tmpl)

        self.embeddings.append(embedding)
        self.dates.append(date)
        self.raw_facts.append(fact)

    def delete(self, i: int) -> None:
        self.raw_facts.pop(i)
        self.embeddings.pop(i)
        self.dates.pop(i)

    def __next__(self):
        if self.ptr > len(self.raw_facts):
            self.ptr = 0
            raise StopIteration
        result = self[self.ptr]
        self.ptr += 1
        return result


def get_top_n_closest_embeddings(
        string: str,
        database: Database,
        n=DBRetrieverConfig.FACTS_PER_TRIPLET,
        thrashold=DBRetrieverConfig.RANKING_THRESHOLD
    ) -> list[tuple[int, str]]:

    text_emb = database.vectorizer.vectorize(string)

    similarities = []
    for i, date, fact, emb in database:
        cos_sim = np.dot(text_emb.reshape(-1), emb) / (norm(text_emb) * norm(emb))
        if float(cos_sim) >= thrashold:
            similarities.append((i, cos_sim))

    similarities.sort(key=lambda x: x[1], reverse=True)

    top_n = [
        database[i]
        for i, similarity
        in similarities[:n]
    ]
    return top_n


class CollisionResolver:

    def get_model(self):
        return AgglomerativeClustering(
        n_clusters = None,
        linkage    = 'single',
        metric     = 'cosine',
        distance_threshold  = DBRetrieverConfig.CLUSTERING_THRESHOLD
    )

    def del_collisions(self, facts, times=DBRetrieverConfig.DISAMBIG_REPS):

        if times > 1:
            facts = self.del_collisions(facts, times=times-1)

        facts = sorted(facts, key=lambda t: t[1])

        distance_matrix = pairwise_distances([c[-1] for c in facts], metric='cosine')

        # Моделька нужна каждый раз новая
        cluster_model = self.get_model()

        clusters = cluster_model.fit_predict(distance_matrix)

        resolved_facts = []
        for cluster_id in set(clusters):

            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_facts = [facts[i][:-1] for i in cluster_indices]
            cluster_facts.sort(reverse=True, key=lambda x: x[1])

            resolved_facts.append(cluster_facts[0])

        return resolved_facts

