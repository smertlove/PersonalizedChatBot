class ColdStartConfig:
    # MODEL_NAME = r"sch-allie/bert_persona"
    # MODEL_NAME = r"sch-allie/bert_another_persona"
    MODEL_NAME = r"sch-allie/bert-persona-model"
    PERSONAS = "https://raw.githubusercontent.com/smertlove/PersonalizedChatBot/refs/heads/main/src/data/personas_forcoldstart.csv"
    CSV_SEP = ","
    FACTS_SEP = "\n"
    DEVICE = r"cuda"
    TOP_N_PERSONAS = 1


class DBRetrieverConfig:
    MODEL_NAME = r"sentence-transformers/all-MiniLM-L6-v2"
    DEVICE = r"cuda"

    # Максимальное кол-во фактов, которые ассоциативно подтягиваются из базы
    FACTS_PER_TRIPLET = 5

    # Минимальный порог косинусной близости для того, чтобы взять факт в выборку для ранжирования
    RANKING_THRESHOLD = 0.1

    # Кол-во применений алгоритма разрешения коллизий
    DISAMBIG_REPS = 1

    # Минимальный порог расстояния между объектами для их попадания в один кластер
    CLUSTERING_THRESHOLD = 0.01
    # CLUSTERING_THRESHOLD = 0.1

    # POPULATE = Path(__file__).parent / "data" / "syntetic_facts.txt"
    POPULATE = None


class GeneratorConfig:
    MODEL_NAME = r"google/gemma-3-1b-it"
    DEVICE = r"cuda"
    SYSTEMPROMPT = (
r"""
I need your help in the generation task. I will show you some facts about my persona (user).
You are an assistant. Generate an answer only to the last user's message/query.
Consider the previous context (messages) and facts.
You should respond only in 2-3 sentences.
"""
)
