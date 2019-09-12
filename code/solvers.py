import re
import random

import numpy as np
import pandas as pd

from solvers_utils import remove_additional, check_pair


df_dict_full = pd.read_csv("../models/data/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
df_dict_full.columns = ["Lemma"]
big_words_set = frozenset(df_dict_full["Lemma"].values)

df_dict = pd.read_table("../models/data/dictionaries/freqrnc2011.csv")
small_words_dict = df_dict.set_index("Lemma")[["Freq(ipm)"]].to_dict()["Freq(ipm)"]


def solver_11_12(task):
    if "Выпишите слово" in task["text"]:
        letter_list = re.findall("буква [А-Яа-я]", task["text"])

        if len(letter_list):
            letter = letter_list[-1][-1].lower()
            words_list = task["text"].replace("..", letter).split("\n")[1:]
            conf = 0.
            most_conf_word = None
            for word_ in words_list:
                word_ = remove_additional(word_)
                temp_conf = small_words_dict.get(word_, -1.0)
                if temp_conf > conf:
                    conf = temp_conf
                    most_conf_word = word_
            if conf == 0.:
                for word_ in words_list:
                    word_ = remove_additional(word_)
                    words_ = big_words_set.intersection({word_})
                    if len(words_):
                        most_conf_word = word_
                        break

            return most_conf_word

    else:
        ids = []
        for answer_ in task["question"]["choices"]:
            if len(check_pair(*answer_["text"].split(", "), big_words_set)):
                ids.append(answer_['id'])

        return ids


df_dict_orfoepicheskiy = pd.read_csv("../models/data/dictionaries/orfoepicheckiy_ege2019.txt",
                                     header=None,
                                     names=["word"])
df_dict_orfoepicheskiy.drop_duplicates(inplace=True)


def solver_4(task):
    text = task["text"]
    needle = "\n"
    start_index = text.find(needle)
    if start_index == -1:
        # pick a random word from the text
        words = [word for word in text.split() if len(word) > 1]
        answer = random.choice(words).lower()
    else:
        words = np.array(text[start_index + len(needle):].split("\n")[:5])
        is_met = np.isin(words, df_dict_orfoepicheskiy)
        if len(is_met):
            answer = random.choice(words[~is_met]).split()[0].lower()
        else:
            # pick a random word from the text
            words = [word for word in text.split() if len(word) > 1]
            answer = random.choice(words).lower()
    return answer
