import re
import regex
import random
import pandas as pd

from solvers_utils import remove_additional, check_pair


df_dict_full = pd.read_csv("../models/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
df_dict_full.columns = ["Lemma"]
big_words_set = frozenset(df_dict_full["Lemma"].values)

df_dict = pd.read_table("../models/dictionaries/freqrnc2011.csv")
small_words_dict = df_dict.set_index("Lemma")[["Freq(ipm)"]].to_dict()["Freq(ipm)"]

def solver_11(task):
    if task["question"]["type"] != "multiple_choice":
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
            words_list = regex.split("[\n\xa0]", task["text"].replace("...", letter).replace("..", letter))[1:]
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
            answer = random.choice(regex.split("[\n\xa0]", task["text"].replace("..", letter_to_insert))[1:])

    else:
        answer = []
        try:
            for choices_ in task["question"]["choices"]:

                    if ";" in choices_["text"]:
                        sep = "; "
                    elif "." in choices_["text"].replace("...", "@").replace("..", "@"):
                        sep = ". "
                    else:
                        sep = ", "
                    words_pair = choices_["text"].replace("...", "@").replace("..", "@").split(sep)
                    if len(check_pair(words_pair[0], words_pair[1], big_words_set)):
                        answer.append(str(choices_['id']))
        except:
            answer = ["2", "4"]
        answer = sorted(answer, key=lambda x: int(x))

    return answer
