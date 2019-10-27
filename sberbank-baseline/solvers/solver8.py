import random
import re
import regex
import time
import joblib
import pymorphy2
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from solvers.utils import BertEmbedder


class Solver(BertEmbedder):

    def __init__(self, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.bert_classifier = LogisticRegression(verbose=10)
        self.pos_classifier = XGBClassifier(n_estimators=10)
        self.pos_vectorizer = CountVectorizer()
        self.label_encoder = LabelEncoder()
        self.morph = pymorphy2.MorphAnalyzer()

    def init_seed(self):
        return random.seed(self.seed)

    def get_sentence_pos(self, sent):
        split = regex.findall(r"\w+|[^\w\s]", sent.lower())
        primary_pos_choices = []
        for word in split:
            morph_word = self.morph.parse(word)
            if str(morph_word[0].tag) == "PNCT":
                primary_pos_choices.append("PNCT")
            else:
                primary_pos_choices.append(str(morph_word[0].tag.POS))
        return " ".join(primary_pos_choices)

    def predict_from_model(self, task):
        decisions, questions = dict(), [re.sub("^[^а-яА-ЯёЁ]*", "", choice["text"])
                                        for choice in task["question"]["choices"]]
        used_answers, answers = set(), [self.unify_type(ans["text"]) for ans in task["question"]["left"]]
        poses = [self.get_sentence_pos(sent) for sent in questions]
        embeddings = np.vstack(self.sentence_embedding(questions))
        probas_bert = self.bert_classifier.predict_proba(embeddings)
        probas_pos = self.pos_classifier.predict_proba(self.pos_vectorizer.transform(poses).toarray())
        probas = probas_bert + probas_pos
        probas = probas[:, self.label_encoder.transform(answers)]
        letters = "ABCDE"
        current_letter = np.argmax(np.max(probas, axis=0))
        for num in range(5):
            letter = letters[current_letter]
            options = np.argsort(probas[:, current_letter])[::-1]
            letters = letters[:current_letter] + letters[current_letter + 1:]
            probas = np.concatenate([probas[:, :current_letter], probas[:, current_letter + 1:]], axis=1)
            if num < 4:
                current_letter = np.argmax(np.max(probas, axis=0))
            try:
                answer = next(option for option in options if option not in used_answers)
            except StopIteration:
                print(letter)
                print("OOOOOPS!!")
                print()
                decisions[letter] = "1"
                continue
            used_answers.add(answer)
            answer_id = str(answer + 1)
            decisions[letter] = answer_id
        return decisions

    def unify_type(self, type_):
        if regex.search("деепричаст", type_):
            return "ошибка в построении предложения с деепричастным оборотом"
        elif regex.search(" причаст", type_):
            return "ошибка в построении предложения с причастным оборотом"
        elif regex.search("сказуем", type_) and regex.search("подлежащ", type_):
            return "ошибка связи между подлежащим и сказуемым"
        elif regex.search("предло", type_) and regex.search("падеж", type_):
            return "неправильное употребление падежной формы существительного с предлогом"
        elif regex.search("косвен", type_) and regex.search("реч", type_):
            return "ошибка в построении предложения с косвенной речью"
        elif regex.search("несогласован", type_) and regex.search("приложени", type_):
            return "ошибка в построении предложения с несогласованным приложением"
        elif regex.search("однородн", type_) and regex.search("член", type_):
            return "ошибка в построении предложения с однородными членами"
        elif regex.search("сложного", type_) or regex.search("сложное", type_):
            return "ошибка в построении сложного предложения"
        elif regex.search("сложноподчин", type_):
            return "ошибка в построении сложного предложения"
            # return "ошибка в построении сложноподчинённого предложения"
        elif regex.search("числительн", type_):
            return "ошибка в употреблении имени числительного"
        elif regex.search("глагол", type_) and regex.search("врем", type_):
            return "ошибка видовременной соотнесённости глагольных форм"
        else:
            return "другое"

    def fit(self, tasks):
        self.corpus, self.types = list(), list()
        for task in tasks:
            for key in "ABCDE":
                try:
                    answer = next(
                        ans["text"] for ans in task["question"]["left"] if
                        str(ans["id"]) == str(key))
                    question_number = task["solution"]["correct"][key]
                    question = next(
                        quest["text"] for quest in task["question"]["choices"] if
                        str(quest["id"]) == str(question_number))
                    question = re.sub("^[^а-яА-ЯёЁ]*", "", question)
                    answer = self.unify_type(answer)
                    self.corpus.append(question)
                    self.types.append(answer)
                except:
                    print(task)
                    print()
        start = time.time()

    #         print("Encoding sentences with bert...")
    #         X = np.vstack(self.sentence_embedding(self.corpus))
    #         print("Encoding finished. This took {} seconds".format(time.time() - start))
    #         y = self.label_encoder.fit_transform(self.types)
    #         self.classifier.fit(X, y)

    def load(self, path="data/models/solver8.pkl"):
        model = joblib.load(path)
        self.bert_classifier = model["classifier"]
        self.label_encoder = model["label_encoder"]
        self.pos_classifier = model["pos_classifier"]
        self.pos_vectorizer = model["pos_vectorizer"]

    def save(self, path="data/models/solver8.pkl"):
        model = {
            "classifier": self.bert_classifier,
            "label_encoder": self.label_encoder,
            "pos_classifier": self.pos_classifier,
            "pos_vectorizer": self.pos_vectorizer
        }
        joblib.dump(model, path)


# class Solver(BertEmbedder):
#
#     def __init__(self, seed=42):
#         super(Solver, self).__init__()
#         self.seed = seed
#         self.init_seed()
#         self.classifier = LogisticRegression(verbose=10)
#         self.label_encoder = LabelEncoder()
#
#     def init_seed(self):
#         return random.seed(self.seed)
#
#     def predict_from_model(self, task):
#         decisions, questions = dict(), [re.sub("^[^а-яА-ЯёЁ]*", "", choice["text"])
#                                       for choice in task["question"]["choices"]]
#         used_answers, answers = set(), [self.unify_type(ans["text"]) for ans in task["question"]["left"]]
#         embeddings = np.vstack(self.sentence_embedding(questions))
#         probas = self.classifier.predict_proba(embeddings)
#         probas = probas[:, self.label_encoder.transform(answers)]
#         letters = "ABCDE"
#         current_letter = np.argmax(np.max(probas, axis=0))
#         for num in range(5):
#             letter = letters[current_letter]
#             options = np.argsort(probas[:, current_letter])[::-1]
#             letters = letters[:current_letter] + letters[current_letter + 1:]
#             probas = np.concatenate([probas[:, :current_letter], probas[:, current_letter + 1:]], axis=1)
#             if num < 4:
#                 current_letter = np.argmax(np.max(probas, axis=0))
#             try:
#                 answer = next(option for option in options if option not in used_answers)
#             except StopIteration:
#                 decisions[letter] = "1"
#                 continue
#             used_answers.add(answer)
#             answer_id = str(answer + 1)
#             decisions[letter] = answer_id
#         return decisions
#
#     def unify_type(self, type_):
#         if regex.search("деепричаст", type_):
#             return "ошибка в построении предложения с деепричастным оборотом"
#         elif regex.search(" причаст", type_):
#             return "ошибка в построении предложения с причастным оборотом"
#         elif regex.search("сказуем", type_) and regex.search("подлежащ", type_):
#             return "ошибка связи между подлежащим и сказуемым"
#         elif regex.search("предло", type_) and regex.search("падеж", type_):
#             return "неправильное употребление падежной формы существительного с предлогом"
#         elif regex.search("косвен", type_) and regex.search("реч", type_):
#             return "ошибка в построении предложения с косвенной речью"
#         elif regex.search("несогласован", type_) and regex.search("приложени", type_):
#             return "ошибка в построении предложения с несогласованным приложением"
#         elif regex.search("однородн", type_) and regex.search("член", type_):
#             return "ошибка в построении предложения с однородными членами"
#         elif regex.search("сложного", type_) or regex.search("сложное", type_):
#             return "ошибка в построении сложного предложения"
#         elif regex.search("сложноподчин", type_):
#             return "ошибка в построении сложноподчинённого предложения"
#         elif regex.search("числительн", type_):
#             return "ошибка в употреблении имени числительного"
#         elif regex.search("глагол", type_) and regex.search("врем", type_):
#             return "ошибка видовременной соотнесённости глагольных форм"
#         else:
#             return "другое"
#
#     def fit(self, tasks):
#         self.corpus, self.types = list(), list()
#         for task in tasks:
#             for key in "ABCDE":
#                 try:
#                     answer = next(
#                         ans["text"] for ans in task["question"]["left"] if
#                         str(ans["id"]) == str(key))
#                     question_number = task["solution"]["correct"][key]
#                     question = next(
#                         quest["text"] for quest in task["question"]["choices"] if
#                         str(quest["id"]) == str(question_number))
#                     question = re.sub("^[^а-яА-ЯёЁ]*", "", question)
#                     answer = self.unify_type(answer)
#                     self.corpus.append(question)
#                     self.types.append(answer)
#                 except:
#                     print(task)
#                     print()
#         start = time.time()
#         print("Encoding sentences with bert...")
#         X = np.vstack(self.sentence_embedding(self.corpus))
#         print("Encoding finished. This took {} seconds".format(time.time() - start))
#         y = self.label_encoder.fit_transform(self.types)
#         self.classifier.fit(X, y)
#
#     def load(self, path="data/models/solver8.pkl"):
#         model = joblib.load(path)
#         self.classifier = model["classifier"]
#         self.label_encoder = model["label_encoder"]
#
#     def save(self, path="data/models/solver8.pkl"):
#         model = {"classifier": self.classifier,
#                  "label_encoder": self.label_encoder}
#         joblib.dump(model, path)
