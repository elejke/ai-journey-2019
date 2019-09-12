import re
import pandas as pd

from solvers_utils import remove_additional, check_pair

df_dict_full = pd.read_csv("../data/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
df_dict_full.columns = ["Lemma"]
big_words_set = frozenset(df_dict_full["Lemma"].values)

df_dict = pd.read_table("../data/dictionaries/freqrnc2011.csv")
small_words_dict = df_dict.set_index("Lemma")[["Freq(ipm)"]].to_dict()["Freq(ipm)"]

def solver_11_12(task):
    if "Выпишите слово" in task["text"]:
        letter_list = re.findall("буква [А-Яа-я]", task["text"])

        if len(letter_list):
            bukva = letter_list[-1][-1].lower()
            words_list = task["text"].replace("..", bukva).split("\n")[1:]
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
                    words_ = small_words_dict.intersection({word_})
                    if len(words_):
                        most_conf_word = word_
                        break

            # key_ = list(task["solution"].keys())[0]

            # print("Our answer: " + str(most_conf_word) + ", " + key_ + ": " + str(task["solution"][key_]))

            # score = task["score"] * (task["solution"][key_] == most_conf_word)

            # print("Score: " + str(score))

            # return score

            return most_conf_word

    elif "пропущена одна и та же буква":
        ids = []
        for answer_ in task["question"]["choices"]:
            if len(check_pair(*answer_["text"].split(", "))):
                ids.append(answer_['id'])

        # key_ = list(task["solution"].keys())[0]

        # print("Our answer: " + str(ids) + ", " + key_ + ": " + str(task["solution"][key_]))

        # if key_ == "correct":
        #     score = task["score"] * (task["solution"][key_] == ids)
        # else:
        #     score = task["score"] * (ids in task["solution"][key_])

        # print("Score: " + str(score))

        return ids
    else:
        print("Unknown question type")
        return None