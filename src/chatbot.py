from .cold_start import PersonaModel
from .dbretriever import Database, Vectorizer, CollisionResolver
from .generator import ResponseGenerator
from .extractor import FactExtractorAgent

from .config import ColdStartConfig, DBRetrieverConfig


def filter2del(points, resolved_points):

    p_set = set([p.id for p in points])
    rp_set = set([p.id for p in resolved_points])
    p2del = p_set - rp_set

    return [p for p in points if p.id in p2del]


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
        for user in similar_to_user:
            persona = self.cold_starter.data['persona'][user[0]]
            for fact in persona.split(ColdStartConfig.FACTS_SEP):
                associative_facts.append(
                    self.database.make_point(fact, "1986-88-60 00:25:24"))
        return associative_facts

    def __get_facts_from_database(self, request, thriplets: list[str]):
        associative_facts = []

        if thriplets:
            for fact in thriplets:
                facts_from_db = self.database.get_top_n_closest_embeddings(fact)
                facts_from_db.append(
                    self.database.make_point(fact, "9999-99-99 99:99:99")
                )
                associative_facts.extend(facts_from_db)
        else:
            facts_from_db = self.database.get_top_n_closest_embeddings(
                request,
                n=DBRetrieverConfig.FACTS_PER_TRIPLET * 3
            )
            associative_facts.extend(facts_from_db)
        return associative_facts

    def response(self, request: str,) -> str:

        # Выделили триплеты
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
        if len(associative_facts) > 2:
            facts_to_RAG = [
                p
                for p
                in self.resolver.del_collisions(associative_facts)
                if p.payload["date"][:4] != "9999"
            ]
        else:
            facts_to_RAG = associative_facts.copy()

        # Удалили из ассоциативных фактов новые факты
        old_associative_facts = [
            p
            for p
            in associative_facts
            if p.payload["date"][:4] != "9999"
        ]

        if not self.new_user:
            # И из самой базы их тоже удалили
            ids2del = filter2del(old_associative_facts, facts_to_RAG)
            self.database.delete([p.id for p in ids2del])
        else:
            self.new_user = False

        # Добавили в базу новые факты
        for fact in extracted_thriplets:
            self.database.append(fact)

        # Перекинули факты в формат, который ест генератор
        facts_to_RAG = [
            f"{fact.payload['date']} - {fact.payload['fact']}"
            for fact
            in facts_to_RAG
        ]

        return self.generator.gen_response(request, facts_to_RAG)
