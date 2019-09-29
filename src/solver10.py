import re
import random
import numpy as np

from solvers_utils import standardize_task

class Solver(object):
    """
    Solver for tasks 10, 11, 12
    """

    def __init__(self, vocabulary=None, seed=42):
        super(Solver, self).__init__()
        self.seed = seed
        self.init_seed()
        self.vocabulary = vocabulary
        # self.morph = pymorphy2.MorphAnalyzer()
        self.word_is_known = lambda word: word in self.vocabulary

    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        result_ids, result_parts, task = dict(), [], standardize_task(task)

        # parse task to find the list of letters to check:
        match = re.search(r'буква ([ЭОУАЫЕЁЮЯИэоуаыеёюяи])', task["text"])
        if match:
            letters = [match.group(1).lower()]
        elif "же букв" in task["text"]:
            letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        else:
            letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

        # find a set of possible answers:
        for vowel in letters:
            result_ids_, result_parts_ = self.get_answer_by_vowel(task["choices"], vowel)
            if len(result_ids_):
                result_ids[vowel] = result_ids_
                result_parts.extend(result_parts_)

        if task["question"]["type"] == "multiple_choice":
            return sorted(list(set(np.concatenate(list(result_ids.values())))))
        else:
            return "".join(self.sort_parts_by_confidence(result_parts)[0])

    def get_answer_by_vowel(self, choices, vowel):
        result_ids = list()
        result_words = list()
        for choice in choices:
            parts = [re.sub(r"^\d\) ?| ?\(.*?\) ?", "", x) for x in choice["parts"]]
            parts = [x.replace("..", vowel) for x in parts]
            if all(self.word_is_known(word) for word in parts):
                result_ids.append(choice["id"])
                result_words.append(parts)
        return result_ids, result_words


    def sort_parts_by_confidence(self, parts_list):
        # TODO: using dicts or something else
        return np.random.permutation(parts_list)

    def set_vocabulary(self, vocabulary):
        self.vocabulary = vocabulary

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass

    def __call__(self, task):
        self.predict_from_model(task)
