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
    #     if "Выпишите слово" in task["text"]:
    if not task["question"]["type"] == "multiple_choice":
        # find letter, that we need to insert into words
        letter_list = re.findall("буква [А-Яа-я]", task["text"])
        # if no letter found:
        if len(letter_list):
            letter_list = [letter_list[-1][-1].lower()]
        else:
            letter_list = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

        conf = 0
        answer = None
        for letter in letter_list:
            words_list = task["text"].replace("..", letter).split("\n")[1:]
            # check first dictionary and find the most confident word from the list
            for word_ in words_list:
                word_ = remove_additional(word_)
                temp_conf = small_words_dict.get(word_, -1.0)
                if temp_conf > conf:
                    conf = temp_conf
                    answer = word_
            # if no such form of word in small dictionary, check it in big one:
            if conf == 0.:
                for word_ in words_list:
                    word_ = remove_additional(word_)
                    if word_ in big_words_set:
                        conf = 1.
                        answer = word_
                        break
        # if no words found in dictionaries, choose random word from list:
        if not answer:
            letter_to_insert = letter_list[0] if len(letter_list) == 1 else "и"
            answer = random.choice(task["text"].replace("..", letter_to_insert).split("\n")[1:])

    else:
        answer = []
        for choices_ in task["question"]["choices"]:
            if len(check_pair(*choices_["text"].split(", "), big_words_set)):
                answer.append(choices_['id'])

    return answer


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
