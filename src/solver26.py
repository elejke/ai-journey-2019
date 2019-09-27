import re
import pymorphy2
import numpy as np
import pandas as pd

morph = pymorphy2.MorphAnalyzer()

types_dict = dict()

types_dict['вводное слово'] = ['вводное слово', 'вводные конструкции', 'вводные слова',
                               'вводные слова и вставные конструкции',
                               'вводные слова и конструкции', 'вводные слова и предложения']

types_dict['вопросительное предложение'] = (['вопросительное предложение', 'вопросительные предложения'] +
                                            ['риторический вопрос', 'риторический', 'риторические вопросы'])

types_dict['вопросно-ответная форма'] = ['вопросно-ответная форма', 'вопросно-ответная форма изложения']

types_dict['восклицательное предложение'] = (['восклицательное', 'восклицательное предложение',
                                             'восклицательные выражения', 'восклицательные предложения'] +
                                             ['риторическое восклицание','риторические восклицания'])

types_dict['диалектизм'] = ['диалектизм', 'диалектизмы', 'диалектные слова',]

types_dict['индивидуально-авторское слово'] = ['индивидуально-авторские слова', 'индивидуально-авторское слово']

types_dict['книжная лексика'] = ['книжная лексика', 'книжные слова', 'слова книжного стиля']

types_dict['контекстные синонимы'] = ['контекстные синонимы', 'контекстуальные синонимы']

types_dict['лексический повтор'] = ['лексические повторы', 'лексический', 'лексический повтор']

types_dict['литота'] = ['литота', 'литоты']

types_dict['метафора'] = ['метафора', 'метафоры']

types_dict['междометие'] = ['междометие', 'междометия']

types_dict['назывное предложение'] = ['назывное', 'назывное предложение', 'назывные предложения']

types_dict['обращение'] = (['обращение', 'обращения'] +
                           ['риторические обращения', 'риторическое обращение', 'риторическое'])

types_dict['просторечие'] = ['просторечие', 'просторечная лексика','просторечное слово', 'просторечные слова',
                             'разговорная и просторечная лексика', 'разговорная лексика',
                             'разговорно-просторечная лексика', 'разговорное слово',
                             'разговорные синтаксические конструкции', 'разговорные слова',
                             'разговорные слова и просторечия', 'слова разговорного стиля']

types_dict['развёрнутая метафора'] = ['развёрнутая метафора', 'развёрнутое сравнение', 'развёрнутые метафоры']

# types_dict['риторическое обращение'] = ['риторические обращения', 'риторическое обращение', 'риторическое'] # ???

# types_dict['ритореческий вопрос'] = ['риторический вопрос', 'риторический', 'риторические вопросы']

# types_dict['риторическое восклицание'] = ['риторическое восклицание', 'риторические восклицания']

types_dict['ряды однородных членов'] = ['ряд','ряд однородных членов', 'ряд однородных членов предложения',
                                        'ряды однородных членов', 'ряды однородных членов предложения',
                                        'однородные члены', 'однородные члены предложения']

types_dict['сравнительный оборот'] = ['сравнение', 'сравнительные обороты', 'сравнительный оборот']

types_dict['термины'] = ['термин', 'термины']

types_dict['фразеологизм'] = ['фразеологизм', 'фразеологизмы']

types_dict['эмоционально-оценочные слова'] = ['экспрессивно-оценочные слова', 'эмоционально-оценочная лексика',
                                              'эмоционально-оценочные слова']

types_dict['эпитет'] = ['эпитет', 'эпитеты']

types_dict_inverted = {}

for key, value in types_dict.items():
    for string in value:
        types_dict_inverted.setdefault(string, []).append(key)


def voskl_predict_sentences(sentences, additional_info=None):
    if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        return 0.
    if len(sentences):
        sentence_probs = [np.clip(len(re.findall("\!", sentence_)), 0, 1).astype(float) for sentence_ in sentences]
        return round(np.mean(sentence_probs), 2)
    return 0.


def vopros_predict_sentences(sentences, additional_info=None):
    if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        return 0.

    sentence_probs = [np.clip(len(re.findall("\?", sentence_)), 0, 1).astype(float) for sentence_ in sentences]
    return round(np.mean(sentence_probs), 2)


def voprosno_otvetnaya_predict_sentences(sentences, additional_info=None):
    if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        return 0.

    sentence_probs = [np.clip(len(re.findall("\?", sentence_)), 0, 1) for sentence_ in sentences]
    if len(sentences) > 1:
        if int(sentence_probs[0]) == 1 and np.mean(sentence_probs) != 1.0:
            return 1.0
    return 0.


def obrashenie_predict_sentences(sentences, additional_info=None):
    if "Видишь ли" in " ".join(sentences):
        return 1.0
    if ("сетовал" not in "".join(sentences) and
        "говорил" not in "".join(sentences) and
        "сказал" not in "".join(sentences) and
        "отвечал" not in "".join(sentences) and
        "говаривал" not in "".join(sentences)) and len(sentences):
        sentence_probs = [np.clip(len(re.findall("-", sentence)) +
                                  len(re.findall("—", sentence)) +
                                  len(re.findall("–", sentence)), 0, 2) for sentence in sentences]
        return round(np.mean(sentence_probs), 2)
    return 0.0


def citing_predict_sentences(sentences, additional_info=None):
    if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        return 0.0

    if ("сетовал" not in "".join(sentences) and
            "говорил" not in "".join(sentences) and
            "говорит" not in "".join(sentences) and
            "сказал" not in "".join(sentences) and
            "отвечал" not in "".join(sentences) and
            "говаривал" not in "".join(sentences)):
        return 0.0
    else:
        sentence_probs = [np.clip(len(re.findall("-", sentence)) +
                                  len(re.findall("—", sentence)) +
                                  len(re.findall(":", sentence)) +
                                  len(re.findall("–", sentence)), 0., 2.) for sentence in sentences]
        return round(np.mean(sentence_probs), 2)
    return 0.0


def anaphora_predict_sentences(sentences, additional_info=None):
    if len(sentences) > 1:
        if (np.array(list(map(lambda x: len(x.split()), sentences))) < 3).sum() == 0:
            sentences_starts = [sentence.split()[:3] for sentence in sentences]
            return np.mean(np.array(sentences_starts[1:]) == sentences_starts[0])
    return 0.


def epitet_predict_sentences(sentences, additional_info=None):
    placeholder_ = additional_info['placeholder']
    pos_dict_full = dict()
    if "«" in "".join(placeholder_) or "»" in "".join(placeholder_):
        for part_ in placeholder_.split("«"):
            if "»" in part_:
                pos_dict = dict()
                clean_part_ = part_.split("»")[0]
                if clean_part_ not in ["о", "за", "к", "с", "без", "над", "под"]:

                    for word_ in re.findall("[а-яА-ЯёЁ]+", clean_part_):
                        pos_tag_ = morph.parse(word_)[0].tag.POS
                        pos_dict[pos_tag_] = pos_dict.get(pos_tag_, 0.) + 1
                        pos_dict_full[pos_tag_] = pos_dict_full.get(pos_tag_, 0.) + 1

                    if set(pos_dict.keys()) == {'ADJF', 'NOUN'}:
                        return 1.
    #     print(placeholder_, pos_dict_full.get('ADJF', 0))
    if pos_dict_full.get('ADJF', 0) > 2:
        return 1.
    return 0.


def metaphora_predict_sentences(sentences, additional_info=None):

    placeholder_ = additional_info['placeholder']
    if "«" in "".join(placeholder_) or "»" in "".join(placeholder_):
        pos_dict = dict()
        max_words_len = -1.
        for part_ in placeholder_.split("«"):
            if "»" in part_:
                clean_part_ = part_.split("»")[0]
                words_ = re.findall("[а-яА-ЯёЁ]+", clean_part_)
                max_words_len = max(max_words_len, len(words_))
                temp_pos_dict = dict()
                for word_ in words_:
                    pos_tag_ = morph.parse(word_)[0].tag.POS
                    pos_dict[pos_tag_] = pos_dict.get(pos_tag_, 0.) + 1
                    temp_pos_dict[pos_tag_] = pos_dict.get(pos_tag_, 0.) + 1

                if (set(pos_dict.keys()) == {'NOUN'} and "—" not in placeholder_ and
                        "-" not in placeholder_ and "–" not in placeholder_):
                    return 1.
        if set(pos_dict.keys()).issuperset({'VERB', 'NOUN'}):
            return 1.
        elif max_words_len > 4:
            return 0.5
        elif max_words_len > 3 and len(set(pos_dict.keys()).intersection({'VERB', 'INFN'})):
            return 0.5
    return 0.


def razver_metaphora_predict_sentences(sentences, additional_info=None):
    placeholder_ = additional_info['placeholder']
    if "«" in "".join(placeholder_) or "»" in "".join(placeholder_):
        pos_dict = dict()
        max_words_len = -1.
        for part_ in placeholder_.split("«"):
            if "»" in part_:
                clean_part_ = part_.split("»")[0]
                words_ = re.findall("[а-яА-ЯёЁ]+", clean_part_)
                max_words_len = max(max_words_len, len(words_))
                temp_pos_dict = dict()
                for word_ in words_:
                    pos_tag_ = morph.parse(word_)[0].tag.POS
                    pos_dict[pos_tag_] = pos_dict.get(pos_tag_, 0.) + 1
                    temp_pos_dict[pos_tag_] = pos_dict.get(pos_tag_, 0.) + 1

                if (set(pos_dict.keys()) == {'NOUN'} and "—" not in placeholder_ and
                        "-" not in placeholder_ and "–" not in placeholder_):
                    return 1.
        if set(pos_dict.keys()).issuperset({'VERB', 'NOUN'}):
            return 1.
        elif max_words_len > 4:
            return 0.5
        elif max_words_len > 3 and len(set(pos_dict.keys()).intersection({'VERB', 'INFN'})):
            return 0.5
    if len(sentences) == 1:
        return 0.25
    return 0.0


def povtor_predict_sentences(sentences, additional_info=None):
    placeholder_ = additional_info['placeholder']
    if "«" in "".join(placeholder_) and "»" in "".join(placeholder_):
        part_ = placeholder_.split("«")[1].split("»")[0]
        words_ = re.findall(part_, " ".join(sentences))
        return len(words_) / 2.
    return 0.


def sravnenie_predict_sentences(sentences, additional_info=None):
    comparing_words = ["точно", "будто", "словно", "подобно"]
    if len(sentences):
        if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
            word_probs = [len(re.findall(word_, additional_info['placeholder'])) for word_ in comparing_words]
            #             print(word_probs)
            n_words = len(re.findall("\«", additional_info['placeholder']))
            return np.sum(word_probs) / n_words * 2
        elif np.sum([len(re.findall(word_, " ".join(sentences))) for word_ in comparing_words + ["как"]]):
            return 1.0
    elif "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        word_probs = [len(re.findall(word_, additional_info['placeholder'])) for word_ in comparing_words]
        n_words = len(re.findall("\«", additional_info['placeholder']))
        return np.sum(word_probs) / n_words
    return 0.0


def vvodnoe_predict_sentences(sentences, additional_info=None):
    introductory_words = ["конечно,", "кстати,", "правда,", "как можно видеть", "вероятно"]
    if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        introductory_words = ["конечно", "кстати", "Правда,", "Как можно видеть,"]
        word_probs = [len(re.findall(word_, additional_info['placeholder'])) for word_ in introductory_words]
        n_words = len(re.findall("\«", additional_info['placeholder']))
        return np.sum(word_probs) / n_words
    elif np.sum([len(re.findall(word_, " ".join(sentences).lower())) for word_ in introductory_words]) > 0:
        return 1.0
    return 0.0


def nepolnie_predict_sentence(sentences, additional_info=None):
    return np.sum([len(re.findall("\.\.\.", sentence_) +
                       re.findall("\?\.\.", sentence_)) for sentence_ in sentences])

def antonim_predict_sentence(sentences, additional_info=None):
    if "—" in additional_info["placeholder"]:
        return 0.5
    return 0.


def dialog_predict_sentences(sentences, additional_info=None):
    # TODO: FINISH RANGES!!!
    if additional_info["n_sentence_ranges"] > 0:
        return additional_info["n_sentence_ranges"] / 4.
    if "«" in "".join(additional_info['placeholder']) or "»" in "".join(additional_info['placeholder']):
        return 0.0

    if ("сетовал" not in "".join(sentences) and
            "сказал" not in "".join(sentences) and
            "говорил" not in "".join(sentences) and
            "отвечал" not in "".join(sentences) and
            "промолвил" not in "".join(sentences) and
            "говаривал" not in "".join(sentences)):
        return 0.0
    else:
        sentence_probs = [np.clip(len(re.findall("-", sentence)) +
                                  len(re.findall("—", sentence)) +
                                  len(re.findall("–", sentence)) +
                                  len(re.findall("—", sentence)), 0, 2) for sentence in sentences]
        return round(np.mean(sentence_probs), 2)

model_dict = {
    "вопросительное предложение": vopros_predict_sentences,
    "восклицательное предложение": voskl_predict_sentences,
    "обращение": obrashenie_predict_sentences,
    "анафора": anaphora_predict_sentences,
    "цитирование": citing_predict_sentences,
    "сравнительный оборот": sravnenie_predict_sentences,
    "вводное слово": vvodnoe_predict_sentences,
    "диалог": dialog_predict_sentences,
    "вопросно-ответная форма": voprosno_otvetnaya_predict_sentences,
    "эпитет": epitet_predict_sentences,
    "метафора": metaphora_predict_sentences,
    "развёрнутая метафора": razver_metaphora_predict_sentences,
    "неполные предложения": nepolnie_predict_sentence,
    "лексический повтор": povtor_predict_sentences,
    "антонимы": antonim_predict_sentence,
    "контекстные антонимы": antonim_predict_sentence,
    "фразеологизм": metaphora_predict_sentences # TODO: USING DICT
}



def predict_sentences(sentences, model_dict, additional_info=None):
    probas = dict()
    for model_key in model_dict.keys():
        probas[model_key] = model_dict[model_key](sentences, additional_info)

    return probas



def parse_task(task):
    task_text = task['text'].replace("\u2003", " ").replace('(!)', '') # id is hardcoded ??
    question_choices = task['question']['choices']
    question_choices = [_['text'] for _ in question_choices]
    question_choices = [types_dict_inverted.get(word_, [word_])[0] for word_ in question_choices]
    # answer = tests[i]['tasks'][0]['solution']['correct']

    return task_text, question_choices


def parse_placeholders(task_text):
    task_text = task_text.replace("\t", " ")
    a = re.findall("[а-яёЁА-Я\ ]*\(А\)[_ ]*[\ а-яА-Я\,]*\([\ а-яёЁА-Я0-9„”;«»\.\-\—\–\,\"…]+\)", task_text)[0].strip()
    b = re.findall("[а-яёЁА-Я\ ]*\(Б\)[_ ]*[\ а-яА-Я\,]*\([\ а-яёЁА-Я0-9„”;«»\.\-\—\–\,\"…]+\)", task_text)[0].strip()
    c = re.findall("[а-яёЁА-Я\ ]*\(В\)[_ ]*[\ а-яА-Я\,]*\([\ а-яёЁА-Я0-9„”;«»\.\-\—\–\,\"…]+\)", task_text)[0].strip()
    d = re.findall("[а-яёЁА-Я\ ]*\(Г\)[_ ]*[\ а-яА-Я\,]*\([\ а-яёЁА-Я0-9„”;«»\.\-\—\–\,\"…]+\)", task_text)[0].strip()

    return [a, b, c, d]

def parse_sentences(text):
    sentences = []
    for sentence in re.split("[(]", text.replace("\xad", "").replace("(!)", ""). \
            replace(" 13)", " (13)").replace(" 14)", " (14)")):
        if ")" in sentence:
            sentence_split = sentence.split(")")
            if sentence_split[0].isdigit() or sentence_split[0] in ["З", "б", "З0", "ЗО", "ЗЗ", "Зб", "З6", "ll", "12"]:
                sentences.append(sentence_split[1].strip())

    return sentences

def parse_sentences_by_ids(text, ids):
    sentences = parse_sentences(text)
    return list(np.array(sentences)[np.array(ids) - 1])

def predict_parsed_task(placeholders, task_text):
    answers_ = dict()
    for x_id, x in enumerate(placeholders):
        sentence_ids = re.findall("[0-9]+", x)
        sentence_ranges = re.findall("([0-9]+\-[0-9]+)+", x)
        n_kov = re.findall("\"", x)
        if len(n_kov) % 2 == 0:
            j = 1
            for i in range(len(x)):
                if x[i] == "\"":
                    x = x[:i] + "«" * j + "»" * (1 - j) + x[i + 1:]
                    j += 1
                    j %= 2

        additional_info = {"n_sentence_ids": len(sentence_ids),
                           "n_sentence_ranges": len(sentence_ranges),
                           "placeholder": x,
                           "sentence_ids": sentence_ids,
                           "sentence_ranges": sentence_ranges}  # !! TODO:
        sentences_ = []
        if len(sentence_ids):
            sentence_ids = list(np.array(sentence_ids).astype(int))
            sentences_from_ranges_ids = []
            if len(additional_info["sentence_ranges"]):
                sentence_ranges = [re.findall("[0-9]+", _) for _ in additional_info["sentence_ranges"]]
                sentences_from_ranges_ids = list(np.concatenate([list(range(int(range_1), int(range_2)))
                                                                 for (range_1, range_2) in sentence_ranges]))

            sentences_ = parse_sentences_by_ids(task_text, sorted(list(set(sentence_ids + sentences_from_ranges_ids))))

        predicted = predict_sentences(sentences_, model_dict, additional_info)
        #         print(predicted)
        answers_[x_id] = predicted

    return answers_


def create_answer(answers, choices):
    def _filter_answers(_answers, _choices):
        filtered_answer = dict()

        for key_ in _answers.keys():
            temp_answer = dict()
            for value_ in _answers[key_].keys():
                if value_ in _choices:
                    temp_answer[value_] = _answers[key_][value_]

            filtered_answer[key_] = temp_answer
        return filtered_answer

    filtered_answers = _filter_answers(answers, choices)

    df_preds = pd.DataFrame(filtered_answers).T.iloc[:, ::-1]
    df_preds_0 = df_preds * ((df_preds - df_preds.max(axis=0)) >= 0)
    df_preds_1 = (df_preds_0.T * ((df_preds_0.T - df_preds_0.max(axis=1)) >= 0)).T

    res = df_preds_1.apply(lambda x: x.argmax(), axis=1).values
    indicator = (df_preds_1.T == df_preds_1.T.iloc[0]).all(axis=0).values.astype(bool)
    chosen = (1 - indicator) * res
    randomly_chosen = np.random.choice(list(set(choices).difference(chosen)),
                                       size=len(indicator), replace=False).astype(object) * indicator

    result = chosen + randomly_chosen

    return list(result)

def solver_26(task):
    task_text, choices = parse_task(task)
    try:
        placeholders = parse_placeholders(task_text)
        model_answers = predict_parsed_task(placeholders, task_text)

        answer = create_answer(model_answers, choices)
        answer = [str(choices.index(trop_name) + 1) for trop_name in answer]
    except:
        answer = np.random.choice(np.array(list(range(1, len(choices) + 1))).astype(str), replace=False, size=4)

    answer = {"A": int(answer[0]),
              "B": int(answer[1]),
              "C": int(answer[2]),
              "D": int(answer[3])}

    return answer