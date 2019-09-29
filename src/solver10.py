import re
import random
import pymorphy2
import numpy as np
from solvers_utils import standardize_task


class Solver(object):
    """
    Solver for tasks 10, 11, 12
    """

    def __init__(self, vocabulary=None, morph=None, seed=42):
        super(Solver, self).__init__()
        self.exceptions = ["проигровать", "проигровать", "ехай", "едь", "правевший"]
        self.prefixes = ["супер", "ультра", "экстра", "гипер", "сверх"]
        self.seed = seed
        self.init_seed()
        self.vocab = vocabulary
        self.morph = morph if morph else pymorphy2.MorphAnalyzer()
        self.remove_prefix = lambda text, prefix: text[len(prefix):] if text.startswith(prefix) else text
        self.word_is_known = lambda word: (any(self.remove_prefix(word, p) in self.vocab for p in self.prefixes) or
                                           word in self.vocab) and word not in self.exceptions

    def init_seed(self):
        return random.seed(self.seed)

    def predict_from_model(self, task):
        result_ids, result_parts, task = [], [], standardize_task(task)

        # parse task to find the list of letters to check:
        match = re.search(r'буква ([ЭОУАЫЕЁЮЯИэоуаыеёюяи])', task["text"])
        if match:
            letters = [match.group(1).lower()]
        elif "же букв" in task["text"]:
            letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
        else:
            letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

        # find a set of possible answers:
        for choice_ in task["choices"]:
            result_ids_, result_parts_ = self.get_answer_by_choice(choice_, letters)
            if len(result_ids_):
                result_ids.extend(result_ids_)
                result_parts.extend(result_parts_)
        print(result_parts)
        if task["question"]["type"] == "multiple_choice":
            return sorted(list(set(result_ids)))
        else:
            return "".join(self.sort_parts_by_confidence(result_parts)[0])

    def get_answer_by_choice(self, choice, letters):

        def check_parts_agreed(parts):
            all_agreed = np.mean([check_agreed(part_) for part_ in parts]) == 1
            return all_agreed

        def check_agreed(part):
            # IMPLEMENTED ONLY FOR VERBS + NOUNS:
            # initial word parsing
            initial_word = re.sub(r"^\d\) ?| ?\(.*?\) ?", "", part)
            initial = self.morph.parse(initial_word)[0]
            # context word parsing
            context_words = re.findall("\([\w\ ]+\)", part)
            if not len(context_words):
                return True

            context_word = context_words[0].strip("(").strip(")").strip()
            context = self.morph.parse(context_word)[0]

            if initial.tag.POS == "VERB" and len(context_word.split()) == 1:
                if context.tag.POS == "NOUN":
                    ans_formed = initial.normalized.inflect({context.tag.number,
                                                             initial.tag.person,
                                                             initial.tag.tense} - {None})
                    return initial.word.replace("ё", "е") == ans_formed.word.replace("ё", "е")

            return True

        result_ids = list()
        result_parts = list()
        result_raw_parts = list()

        for letter in letters:
            parts = [re.sub(r"^\d\) ?| ?\(.*?\) ?", "", x) for x in choice["parts"]]
            parts = [part.replace("..", letter) for part in parts]
            raw_parts = [part.replace("..", letter) for part in choice["parts"]]
            if all(self.word_is_known(part_) and check_agreed(raw_part_)
                   for part_, raw_part_ in zip(parts, raw_parts)):
                result_ids.append(choice["id"])
                result_parts.append(parts)
                result_raw_parts.append(raw_parts)

        if len(result_ids) > 1:
            # TODO: check that both words agreed with context (if it exists):
            result_agreed = list(map(check_parts_agreed, result_raw_parts))
            result_parts = np.array(result_parts)[np.array(result_agreed)]
            result_ids = np.array(result_ids)[np.array(result_agreed)]
            chosen_id = np.random.choice(list(range(len(result_parts))))
            result_parts = result_parts[chosen_id: chosen_id + 1]
            result_ids = result_ids[chosen_id: chosen_id + 1]
        return result_ids, result_parts


    def sort_parts_by_confidence(self, parts_list):
        # TODO: using dicts or something else
        return np.random.permutation(parts_list)

    def set_vocabulary(self, vocabulary):
        self.vocab = vocabulary

    def load(self, path=""):
        pass

    def save(self, path=""):
        pass

    def fit(self, path=""):
        pass