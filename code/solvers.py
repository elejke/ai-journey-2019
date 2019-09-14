import re
import random
import string

import numpy as np
import pandas as pd

from solvers_utils import remove_additional, check_pair, repair_words


df_dict_full = pd.read_csv("../models/data/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
df_dict_full.columns = ["Lemma"]
big_words_set = frozenset(df_dict_full["Lemma"].values)

df_dict = pd.read_table("../models/data/dictionaries/freqrnc2011.csv")
small_words_dict = df_dict.set_index("Lemma")[["Freq(ipm)"]].to_dict()["Freq(ipm)"]


def solver_10(task):
    #     if "Выпишите слово" in task["text"]:

    if task["question"]["type"] == "multiple_choice":

        answers = []
        for choice_ in task["question"]["choices"]:
            if ";" in choice_["text"]:
                sep = "; "
            elif "." in choice_["text"].replace("..", "@").replace("...", "@").replace(".. ", "@"):
                sep = ". "
            else:
                sep = ", "

            if len(repair_words(choice_["text"].replace("..", "@").replace(".. ", "@").replace("...", "@").split(sep),
                                big_words_set, False)):
                answers.append(choice_["id"])

        #         print(answers)
        return answers
    else:
        for choice_ in task["text"].split("\n")[1:]:
            if ";" in choice_:
                sep = "; "
            elif "." in choice_.replace("..", "@").replace("...", "@").replace(".. ", "@"):
                sep = ". "
            else:
                sep = ", "

            letters = repair_words(choice_.replace("..", "@").replace("...", "@").replace(".. ", "@").split(sep),
                                   big_words_set, False)
            #             print(letters)
            if len(letters):
                letter_ = random.choice(letters)
                words = choice_.replace("..", letter_).replace("...", letter_).split(sep)
                words = [remove_additional(word_) for word_ in words]
                answer = "".join(words)
                return answer

        letter_ = random.choice(list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"))
        answer = "".join(choice_.replace("..", letter_).replace("...", letter_).replace(".. ", letter_).split(sep))
        return answer


def solver_11_12(task):
    #     if "Выпишите слово" in task["text"]:
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
            words_list = task["text"].replace("..", letter).replace("...", letter).split("\n")[1:]
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
            if ";" in choices_["text"]:
                sep = "; "
            elif "." in choices_["text"].replace("..", "@").replace("...", "@"):
                sep = ". "
            else:
                sep = ", "

            if len(check_pair(*choices_["text"].replace("..", "@").replace("...", "@").split(sep), big_words_set)):
                answer.append(choices_['id'])

    return answer


def solver_15(task):
    text = task["text"]
    questions, options = text.split("\n")
    if re.match("^.*\sН[^нН]*$", questions):
        missed_str = "н"
    else:
        missed_str = "нн"

    options = options.translate(str.maketrans('', '', string.punctuation[:7] + string.punctuation[9:])).split()

    possible_answers = {}
    for option in options:
        if re.match("^[а-яА-ЯёЁ]*\(\d+\)[а-яА-ЯёЁ]*$", option):
            number = option.split("(")[1].split(")")[0]
            variants = (option.split("(")[0] + missed_str + option.split(")")[1]).lower()
            possible_answers[number] = variants

    answers = []
    for k in possible_answers:
        if possible_answers[k] in big_words_set:
            answers.append(k)

    return answers


df_dict_orfoepicheskiy = pd.concat([
    pd.read_csv("../models/data/dictionaries/orfoepicheckiy_ege2019.txt",
                header=None,
                names=["word"]),
    pd.read_csv("../models/data/dictionaries/orfoepicheckiy_automatic_povtoru.txt",
                header=None,
                names=["word"])
], ignore_index=True)
df_dict_orfoepicheskiy.drop_duplicates(inplace=True)
df_dict_orfoepicheskiy["lowercase"] = df_dict_orfoepicheskiy["word"].str.lower()


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
        for i in range(len(words)):
            subwords = words[i].split()
            subwords = list(filter(lambda x: x.lower() != x, subwords))
            words[i] = subwords[0]
        is_met = np.isin(words, df_dict_orfoepicheskiy["word"])
        # if every word is met
        if sum(~is_met) == 0:
            # pick a random word from answers
            answer = random.choice(words).split()[0].lower()
        # if one word is not met
        elif sum(~is_met) == 1:
            answer = words[~is_met][0].lower()
        # if more words are not met
        else:
            is_met_in_lowercase = np.isin(list(map(str.lower, words)), df_dict_orfoepicheskiy["lowercase"])
            wrong_stress = np.logical_xor(is_met, is_met_in_lowercase)
            # if there are words that are not met in original dict but met in lowercase dict
            # then it means that these words have wrong stress
            if sum(wrong_stress) != 0:
                answer = random.choice(words[wrong_stress]).lower()
            else:
                answer = random.choice(words[~is_met]).lower()
    return answer
