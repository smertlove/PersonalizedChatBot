import spacy
# import coreferee
from typing import List, Dict


class FactExtractorAgent:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        # self.nlp = spacy.load("en_core_web_trf")
        # if "coreferee" not in self.nlp.pipe_names:
        #     self.nlp.add_pipe("coreferee")

    # Разрешение кореференции
    # def resolve_coreference(self, text: str) -> str:
    #     doc = self.nlp(text)
    #     resolved_tokens = []
    #     for token in doc:
    #         resolved = doc._.coref_chains.resolve(token)
    #         if resolved:
    #             resolved_tokens.append(resolved[0].text if isinstance(resolved, list) else resolved.text)
    #         else:
    #             resolved_tokens.append(token.text)
    #     return " ".join(resolved_tokens).replace(" .", ".").replace(" ,", ",")

    # Полная именная группа
    def get_full_np(self, token) -> str:

        span = token.subtree
        return " ".join([t.text for t in span]).strip()

    # Притяжательная форма объекта
    def get_possessive_form(self, token) -> str:

        if token.dep_ == "poss":
            possessor = token.head
            return f"{possessor.text}'s {token.text}"
        return token.text
    # Извлечение триплетов синтаксическим деревом
    def extract_triplets_spacy(self, text: str) -> List[Dict[str, str]]:
        text = text.lower()
        doc = self.nlp(text)
        triplets = []
        seen_triplets = set()

        for sent in doc.sents:
            for token in sent:
            # Пассивный залог
                if token.dep_ == "ROOT" and token.tag_ in {"VBN", "VBD"}:
                    subj = [w for w in token.children if w.dep_ in {"nsubjpass", "nsubj"}]
                    agents = [w for w in token.children if w.dep_ == "agent"]
                    if subj and agents:
                        agent_obj = next((w for w in agents[0].children if w.dep_ == "pobj"), None)
                        if agent_obj:
                            triplet = {
                                "subject": self.get_full_np(agent_obj),
                                "predicate": token.lemma_,
                                "object": self.get_full_np(subj[0])
                            }
                            triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                            if triplet_str not in seen_triplets:
                                triplets.append(triplet)
                                seen_triplets.add(triplet_str)

                # Герундий
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    subj = [w for w in token.children if w.dep_ in {"nsubj", "nsubjpass"}]
                    if subj and subj[0].tag_ == "VBG":  # герундий
                        triplet = {
                            "subject": self.get_full_np(subj[0]),
                            "predicate": token.lemma_,
                            "object": next((self.get_full_np(w) for w in token.children if w.dep_ in {"dobj", "obj"}), "")
                        }
                        triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                        if triplet_str not in seen_triplets:
                            triplets.append(triplet)
                            seen_triplets.add(triplet_str)

              # Вложенные высказывания
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    complements = [w for w in token.children if w.dep_ == "ccomp"]
                    subj = next((w for w in token.children if w.dep_ == "nsubj"), None)
                    for comp in complements:
                        comp_subj = next((w for w in comp.children if w.dep_ == "nsubj"), None)
                        comp_verb = next((w for w in comp.children if w.pos_ == "VERB" and w.dep_ in {"ROOT", "xcomp", "ccomp"}), comp)
                        comp_obj = next((w for w in comp_verb.children if w.dep_ in {"dobj", "obj", "attr"}), None)
                        if comp_subj and comp_obj:
                            triplet = {
                                "subject": self.get_full_np(comp_subj),
                                "predicate": comp_verb.lemma_,
                                "object": self.get_full_np(comp_obj)
                            }
                            triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                            if triplet_str not in seen_triplets:
                                triplets.append(triplet)
                                seen_triplets.add(triplet_str)
            # consider, call, name
                if token.dep_ == "ROOT" and token.lemma_ in {"consider", "call", "name"}:
                    subj = [w for w in token.children if w.dep_ == "nsubj"]
                    dobj = [w for w in token.children if w.dep_ in {"dobj", "obj"}]
                    objcomp = [w for w in token.children if w.dep_ in {"oprd", "attr", "acomp"}]
                    if subj and dobj and objcomp:
                        triplet = {
                            "subject": self.get_full_np(subj[0]),
                            "predicate": token.lemma_,
                            "object": self.get_full_np(dobj[0]) + " is " + self.get_full_np(objcomp[0])
                        }
                        triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                        if triplet_str not in seen_triplets:
                            triplets.append(triplet)
                            seen_triplets.add(triplet_str)

                # SVO
                if token.dep_ == "ROOT" and token.pos_ in {"VERB", "AUX"}:
                    subj = [w for w in token.children if w.dep_ in {"nsubj", "nsubjpass"}]
                    dobj = [w for w in token.children if w.dep_ in {"dobj", "attr", "oprd"}]
                    iobj = [w for w in token.children if w.dep_ == "dative"]

                    # xcomp
                    xcomp = [w for w in token.children if w.dep_ == "xcomp"]
                    for xc in xcomp:
                        xcomp_obj = [c for c in xc.children if c.dep_ in {"dobj", "attr", "oprd"}]
                        if subj and xcomp_obj:
                            triplet = {
                                "subject": self.get_full_np(subj[0]),
                                "predicate": f"{token.lemma_} to {xc.lemma_}",
                                "object": self.get_full_np(xcomp_obj[0])
                            }
                            triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                            if triplet_str not in seen_triplets:
                                triplets.append(triplet)
                                seen_triplets.add(triplet_str)

                    # Двойной объект кому что
                    if subj and dobj and iobj:
                        triplet = {
                            "subject": self.get_full_np(subj[0]),
                            "predicate": token.lemma_ + " to",
                            "object": self.get_full_np(iobj[0]),
                            "target": self.get_full_np(dobj[0])
                        }
                        triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}:{triplet['target']}"
                        if triplet_str not in seen_triplets:
                            triplets.append(triplet)
                            seen_triplets.add(triplet_str)
                        continue

                    # Простое SVO или с предлогом
                    for s in subj:
                        if dobj:
                            for o in dobj:
                                triplet = {
                                    "subject": self.get_full_np(s),
                                    "predicate": token.lemma_,
                                    "object": self.get_full_np(o)
                                }
                                triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                                if triplet_str not in seen_triplets:
                                    triplets.append(triplet)
                                    seen_triplets.add(triplet_str)

                        # Предлог и объект
                        for prep in [w for w in token.children if w.dep_ == "prep"]:
                            pobj = [w for w in prep.children if w.dep_ == "pobj"]
                            if pobj:
                                prep_phrase = f"{token.lemma_} {prep.text}"
                                triplet = {
                                    "subject": self.get_full_np(s),
                                    "predicate": prep_phrase,
                                    "object": self.get_full_np(pobj[0])
                                }
                                triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                                if triplet_str not in seen_triplets:
                                    triplets.append(triplet)
                                    seen_triplets.add(triplet_str)

                    # Отрицание
                    negation = [w for w in token.children if w.dep_ == "neg"]
                    if negation:
                        for triplet in triplets:
                            if triplet["predicate"] == token.lemma_:
                                triplet["predicate"] = "not " + triplet["predicate"]

                # Сравнительные конструкции
                if token.dep_ == "ROOT" and any(w.dep_ == "advmod" and w.text.lower() in {"better", "worse", "more", "less"} for w in token.children):
                    subj = [w for w in token.children if w.dep_ == "nsubj"]
                    than = [w for w in token.children if w.dep_ == "prep" and w.text.lower() == "than"]
                    if subj and than:
                        pobj = [w for w in than[0].children if w.dep_ == "pobj"]
                        if pobj:
                            triplet = {
                                "subject": self.get_full_np(subj[0]),
                                "predicate": f"{token.lemma_} better than",
                                "object": self.get_full_np(pobj[0])
                            }
                            triplet_str = f"{triplet['subject']}:{triplet['predicate']}:{triplet['object']}"
                            if triplet_str not in seen_triplets:
                                triplets.append(triplet)
                                seen_triplets.add(triplet_str)

        return triplets

    def process_dialogue(self, dialogue: str) -> List[Dict[str, str]]:
        # resolved_text = self.resolve_coreference(dialogue)
        # return self.extract_triplets_spacy(resolved_text)
        return self.extract_triplets_spacy(dialogue)
