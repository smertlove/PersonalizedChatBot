from .dbretriever import Database, Vectorizer, get_top_n_closest_embeddings, CollisionResolver
from .generator import ResponseGenerator

from pprint import pprint

class ChatBot:

    def __init__(self):

        # агент Сани
        self.cold_starter = ...

        # агент Лизы
        self.extractor = ...

        # Ассоциативная память
        self.vectorizer = Vectorizer()
        self.database = Database(self.vectorizer)
        self.resolver = CollisionResolver()

        # Генератор ответов
        self.generator = ResponseGenerator()

    def response(self, request: str) -> str:

        # Допустим тут отработал агент Лизы
        extracted_thriplets = [
            "I fear future",
            "I am unemployed",
            "I love Genshin Impact",
            "I live with my parents"
        ]

        # Для каждого триплета достали похожие факты
        associative_facts = []

        for i, fact in enumerate(extracted_thriplets):
            associative_facts.extend(
                get_top_n_closest_embeddings(
                    fact,
                    self.database
                ) + [
                        [
                            len(self.database.raw_facts) + i,
                            "9999-99-99 99:99:99",
                            fact,
                            self.database.vectorizer.vectorize(fact)
                        ]
                ]
            )

        # Удалили коллизии из полученных из базы фактов
        facts_to_RAG = [
            row
            for row
            in self.resolver.del_collisions(associative_facts)
            if row[1][:4] != "9999"
        ]

        # Удалили из ассоциативных фактов новые факты
        associative_facts = [
            row
            for row
            in associative_facts
            if row[1][:4] != "9999"
        ]

        # И из самой базы их тоже удалили
        l_facts_ids_set = {fact[0] for fact in associative_facts}
        rag_facts_ids_set = {fact[0] for fact in facts_to_RAG}
        ids2del = l_facts_ids_set - rag_facts_ids_set

        for id_ in sorted(ids2del, reverse=True):
            self.database.delete(id_)

        # Добавили в базу новые факты
        for fact in extracted_thriplets:
            self.database.append(fact)

        # Перекинули факты в формат, который есть генератор
        facts_to_RAG = [
            f"{fact[1]} - {fact[2]}"
            for fact
            in facts_to_RAG
        ]

        print("Эти факты будут подсунуты в промпт:")
        pprint(facts_to_RAG)

        print("Ответ модели:")
        # Отдали ответ
        return self.generator.gen_response(request, facts_to_RAG)
