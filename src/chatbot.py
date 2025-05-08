from .cold_start import PersonaModel
from .dbretriever import Database, Vectorizer, get_top_n_closest_embeddings, CollisionResolver
from .generator import ResponseGenerator
from .extractor import FactExtractorAgent

from .config import ColdStartConfig, DBRetrieverConfig


class ChatBot:

    def __init__(self):

        self.new_user = True

        # Холодный старт
        print("Init cold start...")
        self.cold_starter = PersonaModel()

        print("Init extractor...")
        # Экстрактор триплетов
        self.extractor = FactExtractorAgent()

        print("Init db retriever...")
        # Ассоциативная память
        self.vectorizer = Vectorizer()
        self.database = Database(self.vectorizer)
        self.resolver = CollisionResolver()

        print("Init generator...")
        # Генератор ответов
        self.generator = ResponseGenerator()

        print("Ready!!")

    def __gen_similar_facts(self, request, thriplets: list[str]):
        user_embedding = self.cold_starter.model.encode(" ".join(thriplets))
        similar_to_user = self.cold_starter.find_similar_users(user_embedding)

        associative_facts = []
        for i, user in enumerate(similar_to_user):
            persona = self.cold_starter.data['persona'][user[0]]
            for j, fact in enumerate(persona.split(ColdStartConfig.FACTS_SEP)):
                associative_facts.append(
                    [
                        i*100 + j*10,
                        "1986-88-60 00:25:24",
                        fact,
                        self.database.vectorizer.vectorize(fact)
                    ]
                )
        return associative_facts

    def __get_facts_from_database(self, request, thriplets: list[str]):
        associative_facts = []

        if thriplets:
            for i, fact in enumerate(thriplets):
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
        else:
            associative_facts.extend(
                get_top_n_closest_embeddings(
                    request,
                    self.database,
                    n=DBRetrieverConfig.FACTS_PER_TRIPLET * 3
                )
            )
        return associative_facts

    def response(self, request: str,) -> str:

        # Допустим тут отработал агент Лизы
        extracted_thriplets = self.extractor.process_dialogue(request)
        extracted_thriplets = [
            " ".join(list(thriplet.values()))
            for thriplet
            in extracted_thriplets
        ]

        if self.new_user:
            associative_facts = self.__gen_similar_facts(request, extracted_thriplets)

        else:
            # Для каждого триплета достали похожие факты
            associative_facts = self.__get_facts_from_database(request, extracted_thriplets)

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

        if not self.new_user:
            # И из самой базы их тоже удалили
            l_facts_ids_set = {fact[0] for fact in associative_facts}
            rag_facts_ids_set = {fact[0] for fact in facts_to_RAG}
            ids2del = l_facts_ids_set - rag_facts_ids_set

            for id_ in sorted(ids2del, reverse=True):
                print(f"Deleting {id_}) {self.database[id_][2]}")
                self.database.delete(id_)
        else:
            self.new_user = False

        # Добавили в базу новые факты
        for fact in extracted_thriplets:
            print(f"Adding: {fact}")
            self.database.append(fact)

        # Перекинули факты в формат, который есть генератор
        facts_to_RAG = [
            f"{fact[1]} - {fact[2]}"
            for fact
            in facts_to_RAG
        ]

        print("Эти факты будут подсунуты в промпт:")
        from pprint import pprint
        pprint(facts_to_RAG)

        return self.generator.gen_response(request, facts_to_RAG)
