import os
import re
import sys
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
import stanfordnlp

import pymorphy2

from keras_bert import load_trained_model_from_checkpoint
sys.path.append("/misc/models/bert")
import tokenization

try:
    from solver10 import Solver as Solver10
except:
    from src.solver10 import Solver as Solver10


df_dict_full = pd.read_csv("../models/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
df_dict_full.columns = ["Lemma"]
big_words_set = frozenset(df_dict_full["Lemma"].values)

df_dict = pd.read_table("../models/dictionaries/freqrnc2011.csv")
small_words_dict = df_dict.set_index("Lemma")[["Freq(ipm)"]].to_dict()["Freq(ipm)"]

slovarnie_slova = pd.read_csv("../models/dictionaries/slovarnie_slova.txt", header=None).rename({0: "word"}, axis=1)

morph = pymorphy2.MorphAnalyzer()

bert_folder = '/misc/models/bert'
config_path = bert_folder+'/bert_config.json'
checkpoint_path = bert_folder+'/bert_model.ckpt'
vocab_path = bert_folder+'/vocab.txt'
tokenizer_bert = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=False)
model_bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True)
model_bert._make_predict_function()

synt = stanfordnlp.Pipeline(lang="ru")
synt.processors["tokenize"].config["pretokenized"] = True

solver_10_11_12 = Solver10(vocabulary=big_words_set, morph=morph)


def solver_15(task):
    text = task["text"]
    _splits = regex.split("[\n\xa0]", text)
    questions = _splits[0]
    for split in _splits:
        if regex.search("\(\d+\)", split) is not None:
            options = split
            break
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
            answers.append(str(k))
    if len(answers) == 0:
        answers.append(str(random.choice(list(possible_answers.keys()))))
    return sorted(answers, key=lambda x: int(x))


df_dict_orfoepicheskiy = pd.concat([
    pd.read_csv("../models/dictionaries/orfoepicheckiy_ege2019.txt",
                header=None,
                names=["word"]),
    pd.read_csv("../models/dictionaries/orfoepicheckiy_automatic_povtoru.txt",
                header=None,
                names=["word"]),
    pd.read_csv("../models/dictionaries/orfoepicheskiy_automatic_gde_udarenie_rf.txt",
                header=None,
                names=["word"])
], ignore_index=True)
df_dict_orfoepicheskiy_lowercase = frozenset(df_dict_orfoepicheskiy["word"].str.lower())
df_dict_orfoepicheskiy = frozenset(df_dict_orfoepicheskiy["word"])


def solver_4(task):
    text = task["text"]
    needle = "\n"
    start_index = text.find(needle)
    if start_index == -1:
        # pick a random word from the text
        words = [word for word in text.split() if len(word) > 1]
        answer = random.choice(words).lower()
    else:
        word_count = 0
        words = []
        for word in text[start_index + len(needle):].split("\n"):
            if regex.search("\w+", word) is not None:
                words.append(word)
                word_count += 1
            if word_count == 5:
                break
        words = np.array(words)
        words_new = []
        for word_num in range(len(words)):
            subwords = words[word_num].split()
            subwords = list(filter(lambda x: x.lower() != x, subwords))
            if len(subwords) != 0:
                words_new.append(subwords[0])
        words = np.array(words_new)
        is_met = []
        for word_num in range(len(words)):
            if len(regex.findall("[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", words[word_num])) == 1:
                is_met.append(True)
            elif words[word_num] in df_dict_orfoepicheskiy:
                is_met.append(True)
            else:
                is_met.append(False)
        is_met = np.array(is_met)
        # if every word is met
        if sum(~is_met) == 0:
            # pick a random word from answers
            answer = random.choice(words).split()[0].lower()
        # if one word is not met
        elif sum(~is_met) == 1:
            answer = words[~is_met][0].lower()
        # if more words are not met
        else:
            is_met_in_lowercase = []
            for word_num in range(len(words)):
                word_to_look_for = words[word_num].lower()
                if len(regex.findall("[аеёиоуыэюяАЕЁИОУЫЭЮЯ]", word_to_look_for)) == 1:
                    is_met_in_lowercase.append(True)
                elif word_to_look_for in df_dict_orfoepicheskiy_lowercase:
                    is_met_in_lowercase.append(True)
                else:
                    is_met_in_lowercase.append(False)
            is_met_in_lowercase = np.array(is_met_in_lowercase)
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
    boundaries = regex.search("\s\d+[\p{Pd}−]\d+", text)
    if boundaries:
        boundaries = re.split("\D", boundaries.group().strip())
        start_sentence_num = int(boundaries[0])
        end_sentence_num = int(boundaries[1])

        text = regex.sub("(\d)\)\s*[\.\p{Pd}−]+", "\g<1>)", text)
        text_lowered = text.lower()

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
        normalized_sentences_set = []
        for s in sentences:
            temp = s.lower().split()
            if len(temp) > 20:
                sentences_set.append(frozenset(temp[-10:] + temp[:10]))
                normalized_sentences_set.append(frozenset([morph.parse(word)[0].normal_form
                                                           for word in temp[-10:] + temp[:10]]))
            else:
                sentences_set.append(frozenset(temp))
                normalized_sentences_set.append(frozenset([morph.parse(word)[0].normal_form
                                                           for word in temp]))

        russian_stopwords = frozenset({
            'и', 'в', 'во', 'не', 'на', 'с', 'со', 'но',
            'к', 'у', 'же', 'за', 'бы', 'по', 'от', 'о',
            'из', 'ну', 'ли', 'или', 'ни', 'до',
            'нибудь', 'уж', 'для', 'без', 'под',
            'ж', 'при', 'об', 'над', 'про', 'перед'
        })

        lichniye_mestoimeniya = frozenset({
            'вами', 'она', 'оно', 'ними', 'я', 'вас',
            'неё', 'ими', 'мы', 'они', 'нами', 'меня',
            'он', 'ему', 'им', 'вам', 'нему', 'ней',
            'мне', 'вы', 'его', 'тобою', 'него', 'мною',
            'ты', 'нее', 'нас', 'ей', 'её', 'тебя', 'ею',
            'них', 'нею', 'тобой', 'ним', 'ее', 'мной',
            'их', 'нам', 'тебе'
        })

        prityazatelniye_mestoimeniya = frozenset({
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

        ukazatelniye_mestoimeniya = frozenset({
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

        soyuzy = frozenset({
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

        protivitelniye_soyuzy = frozenset({
            'но', 'зато', 'однако', 'да', 'а'
        })

        ukazatelniye_narechiya = frozenset({
            'здесь', 'тут', 'там', 'туда', 'оттуда', 'так',
            'тогда', 'затем', 'оттого', 'потому', 'поэтому',
            'сюда', 'отсюда', 'сейчас', 'теперь', 'дотуда'
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
            regex.search("противительн", text_to_find_task),
            regex.search("указательн\w+\s+наречи", text_to_find_task),
            regex.search("форм\w*\s+слов", text_to_find_task)
        ]
        for num_cond, cond in enumerate(conditions):
            if cond:
                conditions[num_cond] = True
            else:
                conditions[num_cond] = False

        conditional_answers = [set() for _ in range(len(conditions))]
        for i in range(1, len(sentences_set)):
            if len((sentences_set[i] & sentences_set[i - 1]) - russian_stopwords) != 0:
                conditional_answers[0].add(numbers[i])
            if len(sentences_set[i] & lichniye_mestoimeniya) != 0:
                conditional_answers[1].add(numbers[i])
            if len(sentences_set[i] & ukazatelniye_mestoimeniya) != 0:
                conditional_answers[2].add(numbers[i])
            if len(set(sentences[i].lower().split()[:1]) & soyuzy) != 0:
                conditional_answers[3].add(numbers[i])
            if len(sentences_set[i] & prityazatelniye_mestoimeniya) != 0:
                conditional_answers[4].add(numbers[i])
            if len(set(sentences[i].lower().split()[:1]) & protivitelniye_soyuzy) != 0:
                conditional_answers[5].add(numbers[i])
            if len(sentences_set[i] & ukazatelniye_narechiya) != 0:
                conditional_answers[6].add(numbers[i])
            if len((normalized_sentences_set[i] & normalized_sentences_set[i - 1]) -
                   russian_stopwords - lichniye_mestoimeniya -
                   (sentences_set[i] & sentences_set[i - 1])) != 0:
                conditional_answers[7].add(numbers[i])

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
            str(choice["id"])
            for choice in choices[:n_choices]
        ]
    return sorted(answer, key=lambda x: int(x))


with open("../models/dictionaries/paronyms_ege.json") as f:
    paronyms_ege = json.load(f)
with open("../models/dictionaries/paronyms_all.json") as f:
    paronyms_all = json.load(f)
if os.path.exists("/misc/models/fasttext/cc.ru.300.bin"):
    model_fasttext = fasttext.load_model("/misc/models/fasttext/cc.ru.300.bin")
else:
    model_fasttext = fasttext.load_model("../models/fasttext/cc.ru.300.bin")


def solver_5(task):

    text = task["text"]

    sentences = []
    words = []
    normalized_words = []
    contexts = []

    for sent in regex.split("[\n\xa0\.]", text)[1:]:

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

    return sorted(np.array(_predict_sentences(sentences)).astype(str).tolist(), key=lambda x: int(x))


def solver_1(task):

    lens = [len(choice["text"]) for choice in task["question"]["choices"]]
    argsorted = np.argsort(lens)
    words_split = [regex.findall(r"\w+|[^\w\s]",
                                 choice["text"].lower().translate(str.maketrans('', '', string.punctuation)))
                   for choice in task["question"]["choices"]]

    sent_vectors = []
    for sent in words_split:
        vec = np.zeros(model_fasttext.get_dimension())
        for word in sent:
            temp = model_fasttext[word]
            norm = np.linalg.norm(temp, ord=2)
            if norm != 0:
                temp /= np.linalg.norm(temp, ord=2)
                vec += temp
        vec /= np.linalg.norm(vec, ord=2)
        sent_vectors.append(vec)
    sent_vectors = np.array(sent_vectors)

    dist = np.linalg.norm(sent_vectors - sent_vectors[argsorted[-1]], ord=2, axis=1)

    ans = [str(np.argsort(dist)[0] + 1), str(np.argsort(dist)[1] + 1)]

    return sorted(ans, key=lambda x: int(x))


nltk_stopwords = frozenset(nltk.corpus.stopwords.words("russian"))


def solver_6(task):

    text = task["text"]

    split = regex.split("[\n\xa0]", text)
    split = [s for s in split[1:] if s != ""]
    words = split[0].lower().translate(str.maketrans('', '', string.punctuation)).split()

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

    def check_morph_tag_similarity(a, b):
        if (a.POS == b.POS) and (a.gender == b.gender) and (a.number == b.number) and (a.tense == b.tense):
            return True
        else:
            return False

    prepositions_by_case = {
        "nomn": frozenset(),
        "gent": frozenset(["с", "у", "от", "до", "из", "без", "для", "вокруг", "около", "возле", "кроме"]),
        "datv": frozenset(["к", "по", "благодаря", "вопреки", "согласно"]),
        "accs": frozenset(["под", "за", "про", "через", "в", "на", "во"]),
        "ablt": frozenset(["с", "со", "за", "над", "под", "между", "перед"]),
        "loct": frozenset(["в", "о", "об", "на", "при", "по"]),
        "voct": frozenset([]),
    }
    prepositions_by_case["gen2"] = prepositions_by_case["gent"]
    prepositions_by_case["acc2"] = prepositions_by_case["accs"]
    prepositions_by_case["loc2"] = prepositions_by_case["loct"]
    all_prepositions = frozenset([item for sublist in prepositions_by_case.values() for item in sublist])

    punkts = frozenset([",", ".", "и"])
    prepositions_for_complex_sentences = frozenset(["который", "чтобы", "что", "однако"])

    questions = task["question"]["left"]
    choices = task["question"]["choices"]

    preprocessed_choices = [regex.findall(r"\w+|[^\w\s]", choice["text"].lower()) for choice in choices]

    primary_pos_choices = []
    primary_tag_choices = []
    all_tag_choices = []
    for choice in preprocessed_choices:
        primary_pos_choices.append([])
        primary_tag_choices.append([])
        all_tag_choices.append([])
        for word in choice:
            morph_word = morph.parse(word)
            primary_tag_choices[-1].append(morph_word[0].tag)
            all_tag_choices[-1].append(morph_word)
            if str(morph_word[0].tag) == "PNCT":
                primary_pos_choices[-1].append("PNCT")
            else:
                primary_pos_choices[-1].append(str(morph_word[0].tag.POS))
        primary_tag_choices[-1] = np.array(primary_tag_choices[-1])
        primary_pos_choices[-1] = np.array(primary_pos_choices[-1])
        # all_tag_choices[-1] = np.array(all_tag_choices[-1])
    primary_tag_choices = np.array(primary_tag_choices)
    primary_pos_choices = np.array(primary_pos_choices)
    # all_tag_choices = np.array(all_tag_choices)
    synt_choices = synt(preprocessed_choices)

    question_classes = []
    for question in questions:
        cls = question_classifier(question["text"].lower().translate(str.maketrans('', '', string.punctuation)))
        question_classes.append(cls)

    possible_answers = []
    for question_num in range(len(questions)):
        possible_answers.append(set())
        if question_classes[question_num] == 1:
            for choice_num in range(len(primary_pos_choices)):
                if "GRND" in primary_pos_choices[choice_num]:
                    possible_answers[-1].add(choice_num)
        elif question_classes[question_num] == 2:
            is_finalized = False
            for choice_num in range(len(preprocessed_choices)):
                for word_num in range(len(preprocessed_choices[choice_num])):
                    top1_proba = all_tag_choices[choice_num][word_num][0].score
                    for form in all_tag_choices[choice_num][word_num]:
                        if form.score < top1_proba:
                            break
                        if (form.tag.POS == "PRTF") or (form.tag.POS == "PRTS"):
                            parent_index = synt_choices.sentences[choice_num].words[word_num].governor
                            if parent_index != 0:
                                parent_form = all_tag_choices[choice_num][parent_index - 1][0]
                                grammemes = {parent_form.tag.case,
                                             parent_form.tag.number,
                                             parent_form.tag.gender} - {None}
                                casted_form = form.inflect(grammemes)
                                if casted_form is None:
                                    casted_form = form.inflect(grammemes - {parent_form.tag.gender})
                                if (parent_form.tag.POS in {"NOUN", "NUMR", "NPRO"}) and \
                                        (casted_form.word != form.word):
                                    possible_answers[-1] = {choice_num}
                                    is_finalized = True
                                    break
                                else:
                                    possible_answers[-1].add(choice_num)
                            else:
                                possible_answers[-1].add(choice_num)
                    if is_finalized:
                        break
                if is_finalized:
                    break
        elif question_classes[question_num] == 10:
            for choice_num in range(len(primary_pos_choices)):
                if "NUMR" in primary_pos_choices[choice_num]:
                    possible_answers[-1].add(choice_num)
        elif question_classes[question_num] == 4:
            is_finalized = False
            for choice_num in range(len(primary_tag_choices)):
                for word_num in range(len(primary_tag_choices[choice_num]) - 1):
                    if (primary_pos_choices[choice_num][word_num] == "PREP") and \
                            (primary_pos_choices[choice_num][word_num + 1] == "NOUN"):
                        if (preprocessed_choices[choice_num][word_num] == "по") and \
                                (preprocessed_choices[choice_num][word_num + 1].endswith("ию")):
                            possible_answers[-1] = {choice_num}
                            is_finalized = True
                            break
                        elif preprocessed_choices[choice_num][word_num] not in \
                                prepositions_by_case[primary_tag_choices[choice_num][word_num + 1].case]:
                            possible_answers[-1].add(choice_num)
                        else:
                            pass
                if is_finalized:
                    break
        elif question_classes[question_num] == 11:
            for choice_num in range(len(primary_tag_choices)):
                all_verbs = primary_tag_choices[choice_num][primary_pos_choices[choice_num] == "VERB"]
                if len(all_verbs) >= 2:
                    forms = set()
                    tenses = set()
                    for verb in all_verbs:
                        forms.add(verb.aspect)
                        tenses.add(verb.tense)
                    forms -= {None}
                    tenses -= {None}
                    if (len(forms) > 1) or (len(tenses) > 1):
                        possible_answers[-1].add(choice_num)
        elif question_classes[question_num] == 6:
            for choice_num in range(len(preprocessed_choices)):
                if ("«" in preprocessed_choices[choice_num]) and \
                        ("»" in preprocessed_choices[choice_num]):
                    start_add = False
                    is_good = True
                    for word_num in range(len(preprocessed_choices[choice_num])):
                        if preprocessed_choices[choice_num][word_num] == "«":
                            start_add = True
                        elif preprocessed_choices[choice_num][word_num] == "»":
                            start_add = False
                        else:
                            if start_add:
                                top1_proba = all_tag_choices[choice_num][word_num][0].score
                                is_there_nomn_form = False
                                was_there_noun = False
                                for form in all_tag_choices[choice_num][word_num]:
                                    if form.score < top1_proba:
                                        break
                                    if form.tag.POS in ["NOUN", "NPRO"]:
                                        was_there_noun = True
                                    if form.tag.case == "nomn":
                                        is_there_nomn_form = True
                                if was_there_noun and not is_there_nomn_form:
                                    is_good = False
                                    break
                    if not is_good:
                        possible_answers[-1].add(choice_num)
        elif (question_classes[question_num] == 8) or (question_classes[question_num] == 9):
            for choice_num in range(len(preprocessed_choices)):
                for word_num in range(2, len(all_tag_choices[choice_num])):
                    if all_tag_choices[choice_num][word_num][0].normal_form in prepositions_for_complex_sentences:
                        if (preprocessed_choices[choice_num][word_num - 1] in punkts) or \
                                (preprocessed_choices[choice_num][word_num - 2] in punkts) and \
                                (all_tag_choices[choice_num][word_num - 1][0].tag.POS == "PREP"):
                            possible_answers[-1].add(choice_num)
                            break
        elif question_classes[question_num] == 5:
            for choice_num in range(len(primary_tag_choices)):
                for word_num in range(len(all_tag_choices[choice_num])):
                    if all_tag_choices[choice_num][word_num][0].normal_form in ["сказать",
                                                                                "говорить",
                                                                                "спросить",
                                                                                "рассказать",
                                                                                "рассказывать",
                                                                                "подтвердить",
                                                                                "писать",
                                                                                "утверждать",
                                                                                "произносить"]:
                        possible_answers[-1].add(choice_num)
                        break
        elif question_classes[question_num] == 7:
            for choice_num in range(len(choices)):
                if "не только" in choices[choice_num]["text"].lower():
                    possible_answers[-1].add(choice_num)
                    continue
                for pos_num in range(len(primary_pos_choices[choice_num]) - 2):
                    if (preprocessed_choices[choice_num][pos_num + 1] == "и") and \
                            (primary_pos_choices[choice_num][pos_num] != "PNCT") and \
                            check_morph_tag_similarity(primary_tag_choices[choice_num][pos_num],
                                                       primary_tag_choices[choice_num][pos_num + 2]):
                        possible_answers[-1].add(choice_num)
                        break
        elif question_classes[question_num] == 3:
            for choice_num in range(len(preprocessed_choices)):
                main_predicate_index = np.argmax([word.governor == 0
                                                  for word in synt_choices.sentences[choice_num].words]) + 1
                predicate_morph = all_tag_choices[choice_num][main_predicate_index - 1][0]
                for word_num in range(len(preprocessed_choices[choice_num])):
                    if synt_choices.sentences[choice_num].words[word_num].governor == main_predicate_index:
                        top1_proba = all_tag_choices[choice_num][word_num][0].score
                        for form in all_tag_choices[choice_num][word_num]:
                            if form.score < top1_proba:
                                break
                            if form.tag.POS in {"NOUN", "NUMR", "NPRO", "ADJF", "ADJS"} and (form.tag.case == "nomn"):
                                grammemes = {form.tag.number, form.tag.gender} - {None}
                                casted = predicate_morph.inflect(grammemes)
                                if casted is None:
                                    casted = predicate_morph.inflect(grammemes - {form.tag.gender})
                                if casted is None:
                                    continue
                                if casted.word != predicate_morph.word:
                                    possible_answers[-1].add(choice_num)
                                    break
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
                answers[str(questions[question_num]["id"])] = str(choices[correct_choice_num]["id"])
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
            answers[str(questions[question_num]["id"])] = str(choices[correct_choice_num]["id"])
            for i in range(len(questions)):
                possible_answers[i] -= {correct_choice_num}
            available_answers -= {correct_choice_num}
        question_num += 1

    return answers


def solver_9(task, testing=False):
    def is_unverifiable(w):
        for w2 in slovarnie_slova.word:
            if re.match(re.sub(r"\.\.", ".", w), w2):
                return True
        return False

    def is_stressed(w, pos):
    #     stressed_w = accent.put_stress(w)
    #     if (stressed_w[pos+1] == "'") or ("'" not in stressed_w):
    #         return True
        return False

    def word_exists(w):
        analysis = morph.parse(w)
        if (analysis[0].methods_stack[0][0].__class__.__name__ == "DictionaryAnalyzer") and \
                (analysis[0].methods_stack[0][1] == w):
            return True
        return False

    def possible_variants(w):
        amount = 0
        for candidate in "аоеиы":
            w_n = re.sub(r"\.\.", candidate, w)
            analysis = morph.parse(w_n)
            if (analysis[0].methods_stack[0][0].__class__.__name__ == "DictionaryAnalyzer") and \
                    (analysis[0].methods_stack[0][1] == w_n):
                amount += 1
        if amount == 0:
            amount = 1
        return amount

    def is_alternant(w):
        #зависящие от конечной согласной корня
        patterns_1 = [
            (r"[а-я]*р\.\.(ст|щ)[а-я]*", "а"),
            (r"[а-я]*р\.\.с[а-су-я]*", "о"),
            (r"[а-я]*л\.\.г[а-я]*", "а"),
            (r"[а-я]*л\.\.ж[а-я]*", "о"),
            (r"[а-я]*ск\.\.к[а-я]*", "а"),
            (r"[а-я]*ск\.\.ч[а-я]*", "о"),
        ]
        #зависящие от суффикса "а" после корня
        patterns_2 = [
            (r"[а-я]*(б|д|м|п|т)\.\.ра[а-я]*", "и"),
            (r"[а-я]*бл\.\.ста[а-я]*", "и"),
            (r"[а-я]*ж\.\.га[а-я]*", "и"),
            (r"[а-я]*ст\.\.ла[а-я]*", "и"),
            (r"[а-я]*ч\.\.та[а-я]*", "и"),
            (r"[а-я]*к\.\.са[а-я]*", "а"),
            (r"[а-я]*(б|д|м|п|т)\.\.р[б-я]*", "е"),
            (r"[а-я]*бл\.\.ст[б-я]*", "е"),
            (r"[а-я]*ж\.\.г[б-я]*", "е"),
            (r"[а-я]*ст\.\.л[б-я]*", "е"),
            (r"[а-я]*ч\.\.т[б-я]*", "е"),
            (r"[а-я]*к\.\.с[б-я]*", "о"),
        ]
        #зависящие от ударения (плов-плав хз почему тут, всегда пишется "а" кроме исключений)
        patterns_3a = [
            (r"[а-я]*з\.\.р[а-я]*", "оа"),
            (r"[а-я]*г\.\.р[а-я]*", "ао"),
            (r"[а-я]*тв\.\.р[а-нп-я]+", "ао"),
        ]
        patterns_3b = [
            (r"[а-я]*пл\.\.в[а-я]*", "оа"),
        ]
        #зависящие от лексического значения
        patterns_4 = [
            (r"[а-я]*м\.\.к[а-я]*", "оа"),
            (r"[а-я]*р\.\.вн[а-я]*", "оа"),
        ]

        exceptions = [
            "росток", "ростов", "ростислав", "ростовщик",
            "отрасль", "скачок", "скачу", "сочетать", "сочетание",
            "чета", "зоревать", "зорянка", "пловец", "пловчиха",
            "плывуны", "уровень", "ровесник", "равнина", "равняйсь",
            "равнение ",
        ]
        w = w.lower()
        pos_space = re.search(r"\.", w).span()[0]
        for p in patterns_1:
            if re.match(p[0], w):
                for p_i in p[1]:
                    filled_w = re.sub(r"\.\.", p_i, w)
                    if word_exists(filled_w):
                        stressed = is_stressed(filled_w, pos_space)
                        return True, 1, filled_w, pos_space, stressed
        for p in patterns_2:
            if re.match(p[0], w):
                for p_i in p[1]:
                    filled_w = re.sub(r"\.\.", p_i, w)
                    if word_exists(filled_w):
                        stressed = is_stressed(filled_w, pos_space)
                        return True, 2, filled_w, pos_space, stressed
        for p in patterns_4:
            if re.match(p[0], w):
                for p_i in p[1]:
                    filled_w = re.sub(r"\.\.", p_i, w)
                    if word_exists(filled_w):
                        stressed = is_stressed(filled_w, pos_space)
                        return True, 4, filled_w, pos_space, stressed
        for p in patterns_3a:
            if re.match(p[0], w):
                for q, p_i in enumerate(p[1]):
                    filled_w = re.sub(r"\.\.", p_i, w)
                    if word_exists(filled_w):
                        stress_ind = is_stressed(filled_w, pos_space)
                        if (stress_ind) and (q == 0):
                            return True, 3, filled_w, pos_space, True
                        if (q == 1) and (not stress_ind):
                            return True, 3, filled_w, pos_space, False
        for p in patterns_3b:
            if re.match(p[0], w):
                for p_i in p[1]:
                    filled_w = re.sub(r"\.\.", p_i, w)
                    if word_exists(filled_w):
                        stressed = is_stressed(filled_w, pos_space)
                        return True, 3, filled_w, pos_space, stressed
        return False, None, None, pos_space, None

    words = np.array([re.split(r", ", t["text"]) for t in task["question"]["choices"]])
    #обрезаем скобки
    words = [[re.sub(r"[0-9]+\)", "", re.sub(r"\([а-я ]+\)", "", t2)).strip() for t2 in t1] for t1 in words]
    #в зависимости от числа слов в каждом варианте, мы ожидаем разное число верных ответов
    num_answers = 2
    if len(words[0]) == 1:
        num_answers = 1
    #определяем какой тип нужно искать
    if "чередующ" in task["text"]:
        task_type = 0
    elif "непровер" in task["text"]:
        task_type = 1
    else:
        task_type = 2
    alt_labels = [[is_alternant(t2) for t2 in t1] for t1 in words]
    unver_labels = [[is_unverifiable(t2) for t2 in t1] for t1 in words]
    possible_ways = [[possible_variants(t2) for t2 in t1] for t1 in words]
    scores = np.zeros((len(words), len(words[0]), 3))
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            scores[i, j, 0] = 0
            scores[i, j, 1] = unver_labels[i][j]
            if alt_labels[i][j][0]:
                scores[i, j, 0] = alt_labels[i][j][0]
            scores[i, j, 2] = 1 - scores[i, j, 0] - 10 * scores[i, j, 1] * (possible_ways[i][j]-1)
    if testing: print(scores)
    agg_scores = scores.mean(axis=1)
    if testing: print(agg_scores)
    agg_scores = agg_scores[:, task_type]
    if testing: print(agg_scores)
    max_score = agg_scores.max()
    second_value = agg_scores[agg_scores.argsort()[-2]]
    answer_numbers = np.arange(len(agg_scores))[agg_scores==max_score]
    if (len(answer_numbers) < 2) and (second_value > 0):
        answer_numbers = np.concatenate([answer_numbers,
                                         np.arange(len(agg_scores))[agg_scores==second_value]])
#     answer_numbers = agg_scores.argsort()[2:]
    answer_numbers += 1
    answer_numbers = [str(t) for t in answer_numbers]
    return answer_numbers


def solver_2(task):

    podchinitelniye_soyuzy = ["что", "потомучто"]
    sochinitelniye_soyuzy = ["но", "однако"]
    protivitelniye_soyuzy = ["но", "однако"]
    narechiya = ["потому", "поэтому", "настолько", "сначала", "сейчас", "сегодня"]
    chastizy = ["только", "даже", "и", "именно", "ведь"]
    soyuzniye_slova = ["которых", "который", "которым", "которому", "которой", "которые", "которыми"]
    ukazatelniye_mestoimeniya = [
        'это', 'этот', 'такой', 'эти'
    ]
    otnositelniye_mestoimeniya = soyuzniye_slova
    vvodniye_slova = ["например", "такимобразом"]
    mestoimeniya = ukazatelniye_mestoimeniya

    text = task["text"]
    text = regex.sub("\<[\.…]+\>", "@", text)

    if regex.search("противительн\w+\s*союз", task["text"]) is not None:
        target_set = protivitelniye_soyuzy
    elif regex.search("сочинит\w+\s*союз", task["text"]) is not None:
        target_set = sochinitelniye_soyuzy
    elif regex.search("подчинит\w+\s*союз", task["text"]) is not None:
        target_set = podchinitelniye_soyuzy
    elif regex.search("наречи", task["text"]) is not None:
        target_set = narechiya
    elif regex.search("частиц", task["text"]) is not None:
        target_set = chastizy
    elif regex.search("союзн\w+\s*слов", task["text"]) is None:
        target_set = soyuzniye_slova
    elif regex.search("указательн\w+\s*местоимен", task["text"]) is not None:
        target_set = ukazatelniye_mestoimeniya
    elif regex.search("относительн\w+\s*местоимен", task["text"]) is not None:
        target_set = otnositelniye_mestoimeniya
    elif regex.search("вводн\w+\s*слов", task["text"]) is not None:
        target_set = vvodniye_slova
    elif regex.search("местоимен", task["text"]) is not None:
        target_set = mestoimeniya
    else:
        target_set = ["кто"]

    target_sentence = None
    for sentence in re.split("[(.]", text):
        if ")" in sentence:
            sentence_split = sentence.split(")")
            if sentence_split[0].isdigit():
                if "@" in sentence_split[1]:
                    target_sentence = sentence_split[1].strip()
                    break
    if target_sentence is None:
        return random.choice(target_set)

    words_split = regex.findall(r"\w+|[^\w\s]", target_sentence.lower())

    replace_index = -1
    for i in range(len(words_split)):
        if words_split[i] == "@":
            replace_index = i
            break
    context = words_split[max(replace_index - 5, 0):replace_index] + words_split[replace_index + 1:replace_index + 6]

    context_vector = np.zeros(model_fasttext.get_dimension())
    for word in context:
        temp = model_fasttext[word]
        norm = np.linalg.norm(temp, ord=2)
        if norm != 0:
            temp /= np.linalg.norm(temp, ord=2)
            context_vector += temp
    norm = np.linalg.norm(context_vector, ord=2)
    if norm != 0:
        context_vector /= norm

    target_vectors = []
    for word in target_set:
        temp = model_fasttext[word]
        temp /= np.linalg.norm(temp, ord=2)
        target_vectors.append(temp)
    target_vectors = np.array(target_vectors)

    dist = np.linalg.norm(target_vectors - context_vector, axis=1, ord=2)
    order = np.argsort(dist)

    return target_set[order[0]]


def solver_17(task, threshold=0.5, testing=False):
    max_length = 512

    text = task["text"]
    text = re.sub(r"\(\d\)", "[MASK]", text)
    if testing:
        print(text)
    text = text.replace("[ ]*\[MASK\][ ]*", "[MASK]")
    text = text.split("[MASK]")

    tokens = ["[CLS]"]
    for i in range(len(text)):
        if i == 0:
            tokens = tokens + tokenizer_bert.tokenize(text[i])
        else:
            tokens = tokens + ['[MASK]'] + tokenizer_bert.tokenize(text[i])
    tokens = tokens + ['[SEP]']
    token_input = tokenizer_bert.convert_tokens_to_ids(tokens)
    token_input = np.array(token_input + [0] * (512 - len(token_input)))

    mask_input = np.zeros(max_length)
    mask_input[token_input == 103] = 1

    seg_input = np.zeros(max_length)
    predicts = model_bert.predict([token_input.reshape(1, -1), seg_input.reshape(1, -1), mask_input.reshape(1, -1)])[0]
    comma_likelihoods = predicts[0, :, 117][mask_input.astype(bool)]
    dot_likelihoods = predicts[0, :, 119][mask_input.astype(bool)]
    and_likelihoods = predicts[0, :, 549][mask_input.astype(bool)]
    or_likelihoods = predicts[0, :, 10880][mask_input.astype(bool)]
    complex_likelihoods = [t1 + t2 for t1, t2 in zip(comma_likelihoods, and_likelihoods)]
    if testing:
        print(f"',': {comma_likelihoods}, '.': {dot_likelihoods}, 'и': {and_likelihoods}, 'или': {or_likelihoods}")
    return [str(i + 1) for i, t in enumerate(complex_likelihoods) if t >= threshold]
