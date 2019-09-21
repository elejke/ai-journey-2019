import os
import re
import json
import copy
import regex
import pickle
import random
import string

import nltk.corpus

import numpy as np
import pandas as pd

import fasttext

import pymorphy2

from solvers_utils import remove_additional, check_pair, repair_words


df_dict_full = pd.read_csv("../models/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
df_dict_full.columns = ["Lemma"]
big_words_set = frozenset(df_dict_full["Lemma"].values)

df_dict = pd.read_table("../models/dictionaries/freqrnc2011.csv")
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
    _splits = text.split("\n")
    questions, options = _splits[0], _splits[1]
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
    if len(answers) == 0:
        answers.append(random.choice(list(possible_answers.keys())))
    return answers


df_dict_orfoepicheskiy = pd.concat([
    pd.read_csv("../models/dictionaries/orfoepicheckiy_ege2019.txt",
                header=None,
                names=["word"]),
    pd.read_csv("../models/dictionaries/orfoepicheckiy_automatic_povtoru.txt",
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


def solver_25(task):
    text = task["text"]
    text_lowered = text.lower()
    boundaries = regex.search("\s\d+[\p{Pd}−]\d+", text)
    if boundaries:
        boundaries = re.split("\D", boundaries.group().strip())
        start_sentence_num = int(boundaries[0])
        end_sentence_num = int(boundaries[1])

        sentences = []
        numbers = []
        r = re.compile(r"[^а-яА-ЯёЁ\s]")
        for sentence in re.split("[(.]", text):
            if ")" in sentence:
                sentence_split = sentence.split(")")
                if sentence_split[0].isdigit():
                    if (int(sentence_split[0]) >= start_sentence_num) and \
                            (int(sentence_split[0]) <= end_sentence_num):
                        sentences.append(r.sub("", sentence_split[1].strip()))
                        numbers.append(sentence_split[0])
        numbers = np.array(numbers)

        sentences_set = []
        for s in sentences:
            temp = s.lower().split()
            if len(temp) > 20:
                sentences_set.append(frozenset(temp[-10:] + temp[:10]))
            else:
                sentences_set.append(frozenset(temp))

        russian_stopwords = frozenset({
            'и', 'в', 'во', 'не', 'на', 'с', 'со', 'но',
            'к', 'у', 'же', 'за', 'бы', 'по', 'от', 'о',
            'из', 'ну', 'ли', 'или', 'ни', 'до',
            'нибудь', 'уж', 'для', 'без', 'под',
            'ж', 'при', 'об', 'над', 'про', 'перед'
        })

        personal_pronouns = frozenset({
            'вами', 'она', 'оно', 'ними', 'я', 'вас',
            'неё', 'ими', 'мы', 'они', 'нами', 'меня',
            'он', 'ему', 'им', 'вам', 'нему', 'ней',
            'мне', 'вы', 'его', 'тобою', 'него', 'мною',
            'ты', 'нее', 'нас', 'ей', 'её', 'тебя', 'ею',
            'них', 'нею', 'тобой', 'ним', 'ее', 'мной',
            'их', 'нам', 'тебе'
        })

        possessive_pronouns = frozenset({
            'наших', 'ваших', 'мою', 'моих', 'свои', 'твоих',
            'мое', 'моё', 'нашей', 'ваше', 'моего', 'своих', 'моему',
            'ваши', 'нашу', 'твоими', 'вашими', 'мой', 'твою',
            'нашими', 'твоей', 'твоя', 'твоему', 'вашем', 'наш',
            'своем', 'своём', 'вашему', 'твоего', 'моими', 'своя', 'свой',
            'вашего', 'нашим', 'моим', 'твоим', 'наши', 'моя',
            'вашу', 'моей', 'моем', 'моём', 'нашем', 'наша', 'своему',
            'своим', 'вашим', 'твое', 'твоё', 'нашего', 'свою', 'ваш',
            'твой', 'свое', 'своё', 'твоем', 'твоём', 'мои', 'ваша', 'наше', 'твои',
            'своего', 'нашему', 'своими', 'своей', 'вашей', 'его', 'ее', 'её', 'их'
        })

        demonstrative_pronouns = frozenset({
            'тот', 'таком', 'этою', 'этой', 'эта', 'такова',
            'такая', 'такому', 'свои', 'таких', 'таким',
            'такое', 'таков', 'этот', 'ту', 'такого', 'того',
            'том', 'такою', 'таково', 'той', 'стольких',
            'этому', 'такой', 'теми', 'столькими', 'эти',
            'стольким', 'тех', 'тою', 'таковы', 'тем', 'те',
            'такими', 'та', 'то', 'это', 'этого', 'этом',
            'этих', 'этим', 'такую', 'столько', 'такие',
            'эту', 'тому'
        })

        conjunctions = frozenset({
            'сиречь', 'даже', 'коль', 'пока', 'поскольку',
            'аж', 'словно', 'ли', 'а', 'пускай', 'разве',
            'но', 'зато', 'ровно', 'чтобы', 'как', 'коли',
            'и', 'дабы', 'что', 'лишь', 'ибо', 'пусть',
            'благо', 'или', 'вроде', 'если', 'ежели', 'также',
            'же', 'так', 'покамест', 'чисто', 'якобы', 'хотя',
            'только', 'когда', 'чтоб', 'нежели', 'покуда',
            'буде', 'точно', 'хоть', 'притом', 'едва', 'ан',
            'будто', 'кабы', 'итак', 'абы', 'либо', 'тоже',
            'затем', 'причем', 'да', 'чем', 'раз', 'чуть',
            'однако'
        })

        task_find_borders = [
            re.search("\(1\)", text_lowered).start(),
            re.search("\(" +
                      str(max([int(el["link"][1:-1]) for el in task["question"]["choices"]])) +
                      "\).*[!.?\n]+", text_lowered).end()
        ]
        text_to_find_task = text_lowered[:task_find_borders[0]] + " " + text_lowered[task_find_borders[1]:]
        conditions = [
            regex.search("лексическ\w+\s+повтор", text_to_find_task),
            regex.search("лично\w+\s+местоимени", text_to_find_task),
            regex.search("указательн\w+\s+местоимени", text_to_find_task),
            regex.search("\s+союз", text_to_find_task),
            regex.search("притяжательн\w+\s+местоимени", text_to_find_task),
        ]
        for num_cond, cond in enumerate(conditions):
            if cond:
                conditions[num_cond] = True
            else:
                conditions[num_cond] = False

        conditional_answers = [set() for _ in range(len(conditions))]
        for i in range(1, len(sentences_set)):
            if len(sentences_set[i] & sentences_set[i - 1] - russian_stopwords) != 0:
                conditional_answers[0].add(numbers[i])
            if len(sentences_set[i] & personal_pronouns) != 0:
                conditional_answers[1].add(numbers[i])
            if len(sentences_set[i] & demonstrative_pronouns) != 0:
                conditional_answers[2].add(numbers[i])
            if len(set(sentences[i].lower().split()[:1]) & conjunctions) != 0:
                conditional_answers[3].add(numbers[i])
            if len(sentences_set[i] & possessive_pronouns) != 0:
                conditional_answers[4].add(numbers[i])

        if sum(conditions) > 0:
            answer = set(numbers)
            for num_cond, cond in enumerate(conditions):
                if cond:
                    answer = answer & conditional_answers[num_cond]
            answer = list(map(str, answer))
        else:
            min_choices = task["question"].get("min_choices", 1)
            max_choices = task["question"].get("max_choices", end_sentence_num - start_sentence_num + 1)
            n_choices = random.randint(min_choices, max_choices)
            answer = np.random.choice(list(range(start_sentence_num, end_sentence_num + 1)),
                                      replace=False,
                                      size=n_choices).astype(str).tolist()
    else:
        choices = task["question"]["choices"]
        min_choices = task["question"].get("min_choices", 1)
        max_choices = task["question"].get("max_choices", len(choices))
        n_choices = random.randint(min_choices, max_choices)
        random.shuffle(choices)
        answer = [
            choice["id"]
            for choice in choices[:n_choices]
        ]
    return answer


with open("../models/dictionaries/paronyms_ege.json") as f:
    paronyms_ege = json.load(f)
with open("../models/dictionaries/paronyms_all.json") as f:
    paronyms_all = json.load(f)
if os.path.exists("/misc/models/fasttext/cc.ru.300.bin"):
    model_fasttext = fasttext.load_model("/misc/models/fasttext/cc.ru.300.bin")
else:
    model_fasttext = fasttext.load_model("../models/fasttext/cc.ru.300.bin")


def solver_5(task):

    morph = pymorphy2.MorphAnalyzer()

    text = task["text"]

    sentences = []
    words = []
    normalized_words = []
    contexts = []

    for sent in text.split("\n")[1:]:

        sent = sent.translate(str.maketrans('', '', string.punctuation))

        match = regex.search("[А-ЯЁ]{2,}", sent)
        if match is not None:
            sentences.append(sent)

            words.append(match.group())
            normalized_words.append(morph.parse(match.group())[0].normal_form)

            sent_begin = sentences[-1][:match.start()]
            sent_end = sentences[-1][match.end():]
            contexts.append(sent_begin.strip().lower().split()[-2:] + sent_end.strip().lower().split()[:2])

    word_paronyms = []
    for word in normalized_words:
        par = paronyms_ege.get(word, [])
        if len(par) == 0:
            par = paronyms_all.get(word, [])
        word_paronyms.append(par)

    context_vectors = []
    for c in contexts:
        v = np.zeros(model_fasttext.get_dimension())
        for word in c:
            temp = model_fasttext[word]
            temp /= np.linalg.norm(temp, ord=2)
            v += temp
        v /= np.linalg.norm(v, ord=2)
        context_vectors.append(v)

    max_dist = -1000
    max_dist_indices = (-1, -1)
    for i in range(len(normalized_words)):
        base_vector = model_fasttext[normalized_words[i]]
        base_vector /= np.linalg.norm(base_vector, ord=2)
        dist_base_to_context = np.linalg.norm(context_vectors[i] - base_vector, ord=2)
        for j in range(len(word_paronyms[i])):
            query_vector = model_fasttext[word_paronyms[i][j]]
            query_vector /= np.linalg.norm(query_vector, ord=2)
            dist_query_to_context = np.linalg.norm(context_vectors[i] - query_vector, ord=2)
            dist_diff = dist_base_to_context - dist_query_to_context
            if dist_diff > max_dist:
                max_dist = dist_diff
                max_dist_indices = (i, j)

    initial = morph.parse(words[max_dist_indices[0]])[0]
    ans = morph.parse(word_paronyms[max_dist_indices[0]][max_dist_indices[1]])[0]

    ans_formed = ans.inflect(initial.tag.grammemes)
    if ans_formed is not None:
        return ans_formed.word
    ans_formed = ans.inflect({initial.tag.POS,
                              initial.tag.gender,
                              initial.tag.number,
                              initial.tag.case} - {None})
    if ans_formed is not None:
        return ans_formed.word
    return ans.word


with open("../models/dictionaries/freq_dict_ruscorpora.json") as f:
    freq_dict = json.load(f)


def solver_24(task):
    morph = pymorphy2.MorphAnalyzer()

    text = task["text"]
    boundaries = regex.search("\s\d+[\p{Pd}−]\d+", text)
    if boundaries:
        boundaries = re.split("\D", boundaries.group().strip())
        start_sentence_num = int(boundaries[0])
        end_sentence_num = int(boundaries[1])

        sentences = []
        r = re.compile(r"[^а-яА-ЯёЁ\s]")
        for sentence in re.split("[(.]", text):
            if ")" in sentence:
                sentence_split = sentence.split(")")
                if sentence_split[0].isdigit():
                    if (int(sentence_split[0]) >= start_sentence_num) and \
                            (int(sentence_split[0]) <= end_sentence_num):
                        sentences.append(r.sub("", sentence_split[1].strip()))
        already_met_words = set()
        min_freq = 1000000000
        min_freq_word = ""
        for sent in sentences:
            for word in sent.lower().split(" "):
                if len(word) > 2:
                    word_normal_form = morph.parse(word)[0].normal_form
                    if word_normal_form not in already_met_words:
                        already_met_words.add(word_normal_form)
                        _count = freq_dict.get(word_normal_form, 0)
                        if _count < min_freq:
                            min_freq = _count
                            min_freq_word = word
        return min_freq_word
    else:
        words = [word for word in text.lower().split() if len(word) > 1]
        return random.choice(words)


with open("../models/task_16/task_16_clf.pkl", 'rb') as file:
    clf_task_16 = pickle.load(file)
with open("../models/task_16/task_16_vectorizer_words.pkl", 'rb') as file:
    vectorizer_words_task_16 = pickle.load(file)
with open("../models/task_16/task_16_vectorizer_pos.pkl", 'rb') as file:
    vectorizer_pos_task_16 = pickle.load(file)


def solver_16(task):

    morph = pymorphy2.MorphAnalyzer()

    def _embedder(sentence):

        pos_sentence = " ".join(list(filter(lambda _x: _x,
                                            [morph.parse(word_)[0].tag.POS for word_ in sentence.strip(".").split()])))

        x_pos = vectorizer_pos_task_16.transform([pos_sentence])
        x_word = vectorizer_words_task_16.transform([str(sentence)])

        x = np.concatenate([x_pos.toarray(), x_word.toarray()], axis=1)

        return x

    def _get_2_sentences(probas):
        probas_positions = np.argsort(probas) + 1
        return list(probas_positions[:2])

    def _predict_sentences(list_of_sentences):

        probas = []

        for sent_ in list_of_sentences:
            x = _embedder(sent_)
            proba = clf_task_16.predict_proba(x)
            probas.append(proba)

        sent_probas = np.concatenate(probas)[:, 0]

        return _get_2_sentences(sent_probas)

    sentences = list(map(lambda x: x["text"], task["question"]["choices"]))

    return np.array(_predict_sentences(sentences)).astype(str).tolist()


def solver_1(task):

    lens = [len(choice["text"]) for choice in task["question"]["choices"]]
    argsorted = np.argsort(lens)
    ans = [task["question"]["choices"][argsorted[-1]]["id"], task["question"]["choices"][argsorted[-2]]["id"]]

    return ans


nltk_stopwords = frozenset(nltk.corpus.stopwords.words("russian"))


def solver_6(task):

    morph = pymorphy2.MorphAnalyzer()

    text = task["text"]

    words = text.split("\n")[1].lower().translate(str.maketrans('', '', string.punctuation)).split()

    pos_mapping = {
        "ADJS": "ADJF",
        "INFN": "VERB",
        "PRTS": "PRTF",
        "GRND": "PRTF"
    }

    pos = []
    normalized_words = []
    for w in words:
        w_morph = morph.parse(w)[0]
        normalized_words.append(w_morph.normal_form)
        pos.append(pos_mapping.get(w_morph.tag.POS, w_morph.tag.POS))

    vectors = []
    for w in normalized_words:
        temp = model_fasttext[w]
        temp /= np.linalg.norm(temp, ord=2)
        vectors.append(temp)
    vectors = np.array(vectors)

    dist = np.linalg.norm(np.diff(vectors, axis=0), ord=2, axis=1)

    for i in range(len(words) - 1):
        if {pos[i], pos[i + 1]} == {"ADJF", "NOUN"}:
            if (words[i] in nltk_stopwords) or (normalized_words[i] in nltk_stopwords):
                if i < len(dist):
                    dist[i] = 100000
                if i - 1 >= 0:
                    dist[i - 1] = 100000
        else:
            dist[i] = 100000
    dist[np.isnan(dist)] = 100000

    argmin = np.argmin(dist)

    pos_to_choose_from = pos[argmin:argmin + 2]
    if pos_to_choose_from == ["ADJF", "NOUN"]:
        answer = words[argmin]
    elif pos_to_choose_from == ["NOUN", "ADJF"]:
        answer = words[argmin + 1]
    else:
        answer = words[argmin]

    return answer


def solver_8(task):

    problems_mapping = {
        0: "другое",
        1: "ошибка в построении предложения с деепричастным оборотом",
        2: "ошибка в построении предложения с причастным оборотом",
        3: "ошибка связи между подлежащим и сказуемым",
        4: "неправильное употребление падежной формы существительного с предлогом",
        5: "ошибка в построении предложения с косвенной речью",
        6: "ошибка в построении предложения с несогласованным приложением",
        7: "ошибка в построении предложения с однородными членами",
        8: "ошибка в построении сложного предложения",
        9: "ошибка в построении сложноподчинённого предложения",
        10: "ошибка в употреблении имени числительного",
        11: "ошибка видовременной соотнесённости глагольных форм"
    }

    def question_classifier(question):
        if regex.search("деепричаст", question):
            return 1
        elif regex.search(" причаст", question):
            return 2
        elif regex.search("сказуем", question) and regex.search("подлежащ", question):
            return 3
        elif regex.search("предло", question) and regex.search("падеж", question):
            return 4
        elif regex.search("косвен", question) and regex.search("реч", question):
            return 5
        elif regex.search("несогласован", question) and regex.search("приложени", question):
            return 6
        elif regex.search("однородн", question) and regex.search("член", question):
            return 7
        elif regex.search("сложного", question) or regex.search("сложное", question):
            return 8
        elif regex.search("сложноподчин", question):
            return 9
        elif regex.search("числительн", question):
            return 10
        elif regex.search("глагол", question) and regex.search("врем", question):
            return 11
        else:
            return 0

    morph = pymorphy2.MorphAnalyzer()

    questions = task["question"]["left"]
    choices = task["question"]["choices"]

    # preprocessed_choices = [wordpunct_tokenize(choice["text"].lower()) for choice in choices]
    preprocessed_choices = [regex.findall(r"\w+|[^\w\s]", choice["text"].lower()) for choice in choices]

    pos_choices = []
    tag_choices = []
    for choice in preprocessed_choices:
        pos_choices.append([])
        tag_choices.append([])
        for word in choice:
            morph_word = morph.parse(word)[0]
            tag_choices[-1].append(morph_word.tag)
            if str(morph_word.tag) == "PNCT":
                pos_choices[-1].append("PNCT")
            else:
                pos_choices[-1].append(str(morph_word.tag.POS))
        tag_choices[-1] = np.array(tag_choices[-1])
        pos_choices[-1] = np.array(pos_choices[-1])
    pos_choices_joined = np.array([" ".join(ch) for ch in pos_choices])
    tag_choices = np.array(tag_choices)
    pos_choices = np.array(pos_choices)

    question_classes = []
    for question in questions:
        cls = question_classifier(question["text"].lower().translate(str.maketrans('', '', string.punctuation)))
        question_classes.append(cls)

    possible_answers = []
    for question_num in range(len(questions)):
        possible_answers.append(set())
        if question_classes[question_num] == 1:
            for pos_choice_joined_num in range(len(pos_choices_joined)):
                if "GRND" in pos_choices_joined[pos_choice_joined_num]:
                    possible_answers[-1].add(pos_choice_joined_num)
        elif question_classes[question_num] == 2:
            for pos_choice_joined_num in range(len(pos_choices_joined)):
                if ("PRTF" in pos_choices_joined[pos_choice_joined_num]) or \
                        ("PRTS" in pos_choices_joined[pos_choice_joined_num]):
                    possible_answers[-1].add(pos_choice_joined_num)
        elif question_classes[question_num] == 10:
            for pos_choice_joined_num in range(len(pos_choices_joined)):
                if "NUMR" in pos_choices_joined[pos_choice_joined_num]:
                    possible_answers[-1].add(pos_choice_joined_num)
        elif question_classes[question_num] == 4:
            for pos_choice_joined_num in range(len(pos_choices_joined)):
                if "PREP NOUN" in pos_choices_joined[pos_choice_joined_num]:
                    possible_answers[-1].add(pos_choice_joined_num)
        elif question_classes[question_num] == 11:
            for tag_choice_num in range(len(tag_choices)):
                all_verbs = tag_choices[tag_choice_num][pos_choices[tag_choice_num] == "VERB"]
                if len(all_verbs) >= 2:
                    forms = set()
                    tenses = set()
                    for verb in all_verbs:
                        forms.add(verb.aspect)
                        tenses.add(verb.tense)
                    forms -= {None}
                    tenses -= {None}
                    if (len(forms) > 1) or (len(tenses) > 1):
                        possible_answers[-1].add(tag_choice_num)
        elif question_classes[question_num] == 6:
            for preprocessed_choice_num in range(len(preprocessed_choices)):
                if ("«" in preprocessed_choices[preprocessed_choice_num]) and \
                        ("»" in preprocessed_choices[preprocessed_choice_num]):
                    possible_answers[-1].add(preprocessed_choice_num)
        elif (question_classes[question_num] == 8) or (question_classes[question_num] == 9):
            for preprocessed_choice_num in range(len(preprocessed_choices)):
                if "," in preprocessed_choices[preprocessed_choice_num]:
                    possible_answers[-1].add(preprocessed_choice_num)
        elif question_classes[question_num] == 5:
            for tag_choice_num in range(len(tag_choices)):
                all_nouns = tag_choices[tag_choice_num][np.isin(pos_choices[tag_choice_num], ["NPRO", "NOUN"])]
                if len(all_nouns) >= 2:
                    persons = set()
                    for noun in all_nouns:
                        if noun.person is not None:
                            persons.add(noun.person)
                        elif "Name" in str(noun):
                            persons.add("NAME")
                        elif noun.animacy == "anim":
                            persons.add("anim")
                        else:
                            pass
                    persons -= {None}
                    if len(persons) > 1:
                        possible_answers[-1].add(tag_choice_num)
        else:
            possible_answers[-1].update(range(len(choices)))

    answers = {}
    question_num = 0
    is_solved = [False] * len(questions)
    available_answers = set(range(len(choices)))
    while question_num < len(questions):
        if not is_solved[question_num]:
            if len(possible_answers[question_num]) == 0:
                possible_answers[question_num] = copy.deepcopy(available_answers)
            elif len(possible_answers[question_num]) == 1:
                is_solved[question_num] = True
                correct_choice_num = list(possible_answers[question_num])[0]
                answers[questions[question_num]["id"]] = int(choices[correct_choice_num]["id"])
                for i in range(len(questions)):
                    possible_answers[i] -= {correct_choice_num}
                available_answers -= {correct_choice_num}
                question_num = 0
            else:
                question_num += 1
        else:
            question_num += 1

    question_num = 0
    while question_num < len(questions):
        if not is_solved[question_num]:
            if len(possible_answers[question_num]) == 0:
                possible_answers[question_num] = copy.deepcopy(available_answers)
            is_solved[question_num] = True
            correct_choice_num = random.choice(list(possible_answers[question_num]))
            answers[questions[question_num]["id"]] = int(choices[correct_choice_num]["id"])
            for i in range(len(questions)):
                possible_answers[i] -= {correct_choice_num}
            available_answers -= {correct_choice_num}
        question_num += 1

    return answers
