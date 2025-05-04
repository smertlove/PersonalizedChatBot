import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .config import ColdStartConfig


class PersonaModel:
    def __init__(self, model_path=ColdStartConfig.MODEL_NAME, data_path=ColdStartConfig.PERSONAS):
        self.model = SentenceTransformer(model_path).to(ColdStartConfig.DEVICE)

        self.data = pd.read_csv(data_path, sep="\t", index_col=0)
        self.data["persona_embeddings"] = self.data["facts"].map(lambda fact: self.model.encode(fact))

        self.embeddings = self.data["persona_embeddings"]

    def find_similar_users(self, target_user_embed, top_n=ColdStartConfig.TOP_N_PERSONAS):
        similarities = {}
        for user_id, embedding in self.embeddings.items():
            sim_score = cosine_similarity([target_user_embed], [embedding])[0][0]
            similarities[user_id] = sim_score

        similar_users = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return similar_users

    def get_unique_facts(self, similar_users):
        new_facts = []
        for user in similar_users:
            idx = user[0]
            facts = self.data['facts'][idx].split('.')
            for fact in facts:
                fact = fact.lower().strip('\n')
                if fact not in new_facts and fact != '':
                    new_facts.append(fact)
        return new_facts


def main():
    model_path = ColdStartConfig.MODEL_NAME
    data_path = ColdStartConfig.PERSONAS
    
    persona_model = PersonaModel(model_path, data_path)

    user_input = input("Please enter the persona ")
    user_embedding = persona_model.model.encode(user_input)

    similar_to_user = persona_model.find_similar_users(user_embedding)

    for user in similar_to_user:
        print(persona_model.data['facts'][user[0]]) #similar personas

    unique_facts = persona_model.get_unique_facts(similar_to_user) #facts to use in dialogue

    print("------\nUnique Facts:\n")
    for fact in unique_facts:
        print(fact)


if __name__ == "__main__":
    main()