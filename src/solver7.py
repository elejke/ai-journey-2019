import regex
import pickle
import pymorphy2
import numpy as np
import pandas as pd

from string import punctuation

from textdistance import DamerauLevenshtein

morph = pymorphy2.MorphAnalyzer()

dl = DamerauLevenshtein()


df_hard_nouns = pd.read_csv("../models/dictionaries/hard_nouns.csv")
df_hard_verbs = pd.read_csv("../models/dictionaries/hard_verbs.csv")
df_numeralized = pd.read_csv("../models/dictionaries/numeralized.csv")
df_answers_reshuege_and_pub_train = pd.read_csv("../models/dictionaries/task7_correct_answers_pub_and_reshuege.csv")

# df_dict_full = pd.read_csv("../models/dictionaries/russian_1.5kk_words.txt", encoding="windows-1251", header=None)
# df_dict_full.columns = ["Lemma"]
# big_words_set = set(df_dict_full["Lemma"].str.replace("ё", "е").values)

xgb_clf = pickle.load(open("../models/dictionaries/task_7_xgb.pkl", "rb"))

rnc_path = "../models/dictionaries/1grams-3.txt"

# def solver_7_classifier(question_choices):
#     def _check_known_word(word):
#         return word.lower().replace("  ", " ").replace("ё", "е") in big_words_set
#
#     unknown_choices_ids = []
#
#     X_high = list(map(lambda x: x.lower().strip().replace("  ", " ").replace("ё", "е"),
#                       regex.findall("[А-ЯЁ]+[IА-ЯЁ\ ]+", ",".join(question_choices))))
#
#     for i, x_ in enumerate(X_high):
#         if not _check_known_word(x_):
#             unknown_choices_ids.append(i)
#
#     if not len(unknown_choices_ids):
#         unknown_choices_ids = list(range(5))
#     return np.random.choice(unknown_choices_ids)


def lazy_unigrams(rnc_path):
    unigram_freq_dict = {}
    for line in open(rnc_path, "r", encoding="utf-8").read().split("\n"):
        pair = line.lower().split("\t")
        try:
            freq, unigram = int(pair[0]), " ".join([el for el in pair[1:]
                                                    if el is not "" and el not in punctuation])
            if unigram not in unigram_freq_dict:
                unigram_freq_dict[unigram] = freq
            else:
                unigram_freq_dict[unigram] += freq
        except ValueError:
            pass
    return unigram_freq_dict

rnc_unigrams = lazy_unigrams(rnc_path)

def get_error_word(words):
    frequencies = [rnc_unigrams[w] if w in rnc_unigrams else 10 for w in words]
    error_word = words[np.argmin(frequencies)]
    return error_word


def feature_extractor(x):
    X_high = list(map(lambda x: x.lower().strip().replace("  ", " ").replace("ё", "е"),
                      regex.findall("[А-ЯЁ]+[IА-ЯЁ\ ]+", x)))[0]

    pos_types = [
        'ADJF', 'ADJS', 'ADVB', 'COMP', 'CONJ', 'GRND', 'INFN', 'INTJ',
        'NOUN', 'NPRO', 'NUMR', 'None', 'PRCL', 'PRED', 'PREP', 'PRTF',
        'PRTS', 'VERB'
    ]

    correct_words = [
        'их', 'пятисот', 'поезжайте', 'полощущий', 'благородства',
        'зажжёт', 'директора', 'его', 'погон', 'повара', 'значительный',
        'нужный', 'высоко', 'оденьтесь', 'высохла', 'инженеры',
        'шестистах', 'пятидесяти', 'девятистам', 'промок', 'ярче', 'дёсны',
        'семьюдесятью', 'тысяча', 'ближайшая', 'носков', 'тихо',
        'конструкторы', 'их', 'четырьмястами', 'семьюстами', 'обеими',
        'едкий', 'более громко', 'легчайшее', 'вафель', 'промок',
        'полотенец', 'ложись', 'лягте', 'графику', 'пятистам', 'ботинок',
        'конюхи', 'шестьюстами', 'возрасты', 'супы', 'их', 'две', 'плеч',
        'восьмидесяти', 'заглохший', 'лягте', 'трое', 'им', 'семьюдесятью',
        'пятисот', 'тетерева', 'жёстче', 'директора', 'сплетен',
        'доходчиво', 'поезжай', 'полутора', 'более глубоко', 'свитеры',
        'глубочайшее', 'директора', 'сбережёт', 'поезжай', 'титулы',
        'более просто', 'доктора', 'стрижёт', 'трёхстах', 'обеих',
        'щупалец', 'в двухстах', 'обеих', 'месяцы', 'обеих', 'пятистах',
        'плещет', 'поезжай', 'лёгкая', 'брызг', 'более резкое', 'сотен',
        'надёжный', 'полутора', 'две', 'свитеры', 'более высоко',
        'георгинов', 'обоих', 'надеть', 'абрикосов', 'стрижёт', 'богинь',
        'выговоры', 'наиболее', 'апельсинов', 'испечём', 'шестьюстами',
        'всех быстрее', 'шофёры', 'четверо', 'богаче', 'туфель', 'их',
        'басен', 'поезжай', 'барышень', 'шофёры', 'обеих', 'сардин',
        'полотенец', 'две', 'завидую', 'абрикосов', 'тренеры', 'колокола',
        'аэропорту', 'насмехаться', 'наилучший', 'течёт', 'смелее',
        'обоих', 'обеим', 'тремястами', 'бухгалтеры', 'красивый', 'лягте',
        'сапог', 'падежей', 'мокла', 'клади', 'их', 'солдат', 'туфля',
        'выздоровеешь', 'шестидесяти', 'восьмисот', 'профессора', 'положи',
        'кухонь', 'профессора', 'восьмисот', 'поезжай', 'полотенец', 'их',
        'длинной', 'поскользнулся', 'положи', 'колышет',
        'до тысяча девятьсот пятого', 'сапог', 'пламени', 'более уверенно',
        'ездите', 'уместный', 'двухстах шестидесяти', 'яслей', 'восемьсот',
        'пятидесятая', 'выше', 'строжайше', 'пробуя', 'помидоров',
        'худший', 'двухстах', 'тонкий', 'шкафу', 'критериев',
        'парикмахеры', 'лёгкий', 'семьюстами', 'катера', 'поскользнуться',
        'ляг', 'восьмьюдесятью восьмью', 'семьюстами', 'инженеры',
        'восьмисотому', 'ботинок', 'слаще', 'директора', 'интереснее всех',
        'полотенец', 'бойче', 'пятистах', 'кухонь', 'более крупная',
        'поезжайте', 'добрейший', 'поезжай', 'кухонь', 'клапаны',
        'килограммов', 'серьёзнейшее', 'шестьсот', 'доктора', 'граммов',
        'шестьюдесятью', 'прополощи', 'двое', 'дольше', 'звонче',
        'заезжай', 'их', 'поезжайте', 'шампуня', 'решению', 'яблонь',
        'человек', 'носков', 'деревни', 'поезжай', 'ответственный',
        'громко', 'богатейший', 'полтораста', 'кочерёг', 'пятисот',
        'две тысячи шестом', 'семьюстами', 'яслей', 'широко', 'свежая',
        'свечей', 'девяноста', 'лягте', 'предупрежу',
        'менее привлекательный', 'директора', 'лекторы', 'ездит',
        'помашите', 'облаков', 'чулок', 'трёмстам', 'серёг', 'мандаринов',
        'красивее', 'пять', 'почерк', 'пятьюстами', 'две тысячи пятом',
        'пятистах', 'тридцати пяти', 'одень', 'приезде', 'блюдец',
        'килограммов', 'чище', 'двухсот', 'кухонь', 'аршин', 'яблок',
        'обоих', 'красивейший', 'поезжай', 'комментариев', 'грузин',
        'блюдец', 'помидоров', 'шестьюстами', 'баклажанов', 'лагеря',
        'помидоров', 'макарон', 'георгинов', 'длиннее', 'озяб', 'полутора',
        'времяпрепровождение', 'поезжай', 'триста пятьдесят', 'приговоры',
        'трехстах', 'две тысячи', 'две', 'светлый', 'высохла', 'жаждет',
        'тысяча', 'макарон', 'мяукает', 'тремястами', 'звонче', 'клади',
        'шестьюдесятью', 'ста двадцатью пятью', 'вишен', 'чудеснейший',
        'наисложнейшая', 'выкладывай', 'туфель', 'изрешечённый',
        'консервов', 'яблок', 'обоими', 'помидоров', 'оладий', 'полотенец',
        'поезжайте', 'заморозков', 'две', 'тысяча восьмисотом',
        'приобретший', 'семьюдесятью', 'более весело', 'наиболее сильный',
        'более сильньй', 'смотря', 'чулок', 'передайте', 'помашите',
        'ихние', 'полутораста', 'трое', 'ей', 'крепкий', 'восьмьюдесятью',
        'цапель', 'обгрызенных', 'звонче', 'их', 'шестидесяти', 'яблок',
        'консервов', 'четверо', 'выгодно', 'лажу', 'осетин', 'пятисотым',
        'договоры', 'семисот', 'более надёжный', 'чудеснейшим',
        'выразительно', 'грунте', 'новый', 'макарон', 'граммов',
        'полутораста', 'их', 'наисложнейший', 'семью', 'наилучший',
        'гнездовий', 'умолк', 'менее значимый', 'вишен', 'их',
        'более разговорчивым', 'наилучших', 'поезжайте', 'кладите',
        'пятисот', 'наивысшее', 'их', 'паспорта', 'соболий', 'абрикосов',
        'обеих', 'прессы', 'граммов', 'седьмых', 'паспорта', 'лягте',
        'барышень', 'векторы', 'их', 'тысяча девятьсот двадцать пятом',
        'лжём', 'джемперы', 'полощет', 'долго', 'дешевле', 'солдат', 'ста',
        'девятьюстами', 'снайперы', 'шампунем', 'шестьюдесятью',
        'выздоровею', 'шкафу', 'повеселее', 'более просто', 'тюля',
        'абрикосов', 'джинсов', 'пятистам', 'возврата', 'детьми',
        'нелепейший', 'возрасты', 'джинсов', 'их', 'сладчайший', 'их',
        'обоих', 'килограммов', 'их', 'вкуснее', 'абзацев', 'джинсов',
        'аэропорте', 'строгий', 'наиболее решительно', 'поезжайте',
        'две тысячи', 'их', 'шампунем', 'скорейшего', 'яблонь', 'ботинок',
        'оладий', 'шестьюстами', 'падежей', 'девяноста', 'партизан',
        'лекторы', 'более красивый', 'паспорта', 'граммов', 'обеих',
        'надень', 'лучший', 'башен', 'полотенец', 'трёхстах', 'профессора',
        'спален', 'договоры', 'подожжёшь', 'шестисот', 'инженеры',
        'паспорта', 'двумястами', 'четырьмястами', 'полезно', 'интереснее',
        'выборы', 'двенадцатого', 'договоры', 'семисот', 'семистах',
        'кратчайший', 'его', 'выйди', 'бухгалтеры', 'туфель', 'ответ',
        'сердец', 'лжёте', 'поезжайте', 'поместий', 'ботинок', 'пропуска',
        'наиболее', 'исчез', 'боку', 'наиболее уместный', 'надеть', 'их',
        'договоры', 'абзацев', 'сотрёт', 'выздоровеет', 'крошечный',
        'молодой', 'мандаринов', 'четырьмястами', 'четверых',
        'две тысячи пятому', 'попробует', 'ярко', 'туфля', 'времени',
        'пятьюдесятью', 'их', 'две тысячи семнадцатому', 'джинсов',
        'платьев', 'яблок', 'шестьюстами', 'кастрюль', 'его', 'ветру',
        'торты', 'поезжайте', 'полутора', 'попробуем', 'их', 'открыв',
        'красивейший', 'серёг', 'двухтысячном', 'движет', 'обгрызенная',
        'печёт', 'килограммов', 'напоив', 'тонн', 'две тысячи',
        'наиболее точно', 'сорта', 'попробует', 'с детьми', 'наиболее',
        'поезжай', 'пояса', 'полощет', 'четырьмястами', 'носков',
        'зажжётся', 'поезжайте', 'профессора', 'обоих', 'их', 'ножниц',
        'именин', 'более долго', 'две тысячи шестому', 'вишен', 'паспорта',
        'потолще', 'взгляни', 'сапог', 'посуше', 'восьмисотом',
        'четырёх тысяч', 'поезжайте', 'сильнее', 'выправь', 'искомый',
        'моложе', 'красивее', 'обоих', 'колышется', 'посади', 'исчезнув',
        'барышень', 'менее интересно', 'их', 'недра', 'директора',
        'тарифу', 'более слабый', 'князья', 'звонче', 'зажжётся', 'полиса',
        'пихт', 'лягте', 'полутора', 'погон', 'девятьюстами',
        'полутораста', 'лезь', 'клади', 'четырьмястами', 'попробуем',
        'лягте', 'каникул', 'титулы', 'всего', 'лазайте', 'поскользнулся',
        'тысячу', 'контейнеры', 'умный', 'шорт', 'поезжай', 'поезжайте',
        'их', 'три'
    ]

    pos_dict = dict()

    for pos_ in pos_types:
        pos_dict[pos_] = 0

    features = [len(x), len(x.split()), len(X_high), len(X_high.split()),
                rnc_unigrams[X_high.lower()] if X_high.lower() in rnc_unigrams else 10]

    for word_ in x.lower().split():
        pos_dict[str(morph.parse(word_.lower())[0].tag.POS)] += 1
    pos_features = list(pos_dict.values())

    features.append(X_high.lower() in correct_words)

    features.append(X_high.lower() != solver_7_reconstructor(x))

    return np.array(features + pos_features).astype(int)

def solver_7_xgb_classifier(question_choices):
    feats = []

    for i in range(len(question_choices)):
        feats.append(feature_extractor(question_choices[i]))

    y_preds = xgb_clf.predict_proba(np.array(feats))[:, 0]

    y_final = np.argmin(y_preds)

    return y_final


def _find_nearest_numerical(word, vocab=None, n_neighbours=1):
    sims = []

    for word_ in vocab:
        sim = dl.normalized_similarity(word, word_)
        sims.append(sim)

    if n_neighbours == 1:
        for i in range(min(len(sims), 10)):
            if vocab[np.argsort(sims)[- i - 1]][0] == word[0]:
                return vocab[np.argsort(sims)[- i - 1]], np.array(sims)[np.argsort(sims)[- i - 1]]
        return vocab[np.argsort(sims)[-1]], np.array(sims)[np.argsort(sims)[-1]]
    else:
        #         print(sims)
        return vocab[np.argsort(sims)[-n_neighbours:]], np.array(sims)[np.argsort(sims)[-n_neighbours:]]

def reconstruct_numeralized(word_pair):
    #     print(word_pair)
    word_1, word_2 = word_pair.lower().split()
    #     print(word_1)
    tag_1, tag_2 = morph.parse(word_1)[0].tag, morph.parse(word_2)[0].tag
    pos_1, pos_2 = tag_1.POS, tag_2.POS
    # print(str(tag_2.case))
    nearest_number = _find_nearest_numerical(word_1, list(df_numeralized.numeralized.values))[0]
    # print(nearest_number)
    number_int = df_numeralized[(df_numeralized.numeralized == nearest_number)].number.values[0]
    reconstructed = df_numeralized[(df_numeralized.number == number_int) &
                                   (df_numeralized.gender_name == str(tag_2.gender)) &
                                   (df_numeralized.kase_name == str(tag_2.case))].iloc[0].numeralized
    # print(word_1, df_numeralized[(df_numeralized.number == number_int) &
    #                              (df_numeralized.kase_name == str(tag_2.case))].iloc[0].numeralized)
    #     print(reconstructed + " " + word_2)

    return reconstructed


def solver_ANY(x):
    def _find_nearest(x, vocab=pd.concat([df_hard_nouns, df_hard_verbs,
                                          df_answers_reshuege_and_pub_train]).word.values):
        sims = []
        vocab = list(filter(lambda _: _ != x, vocab))
        for word_ in vocab:
            sim = dl.normalized_similarity(x, word_)
            sims.append(sim)
        if np.sort(sims)[-1] > 0.6:
            return vocab[np.argsort(sims)[-1]]
        else:
            return x

    #     word_1, word_2 = x.lower().split()

    x = regex.findall("[А-ЯЁ]+[А-ЯЁ\ ]+", x)[0].strip()

    return _find_nearest(x.lower())

def solver_VERB(x):
    def _find_nearest(x, vocab=df_hard_verbs.word.values):
        sims = []
        for word_ in vocab:
            sim = dl.normalized_similarity(x, word_)
            sims.append(sim)
        if np.sort(sims)[-1] > 0.6:
            return vocab[np.argsort(sims)[-1]]
        else:
            return x

    #     word_1, word_2 = x.lower().split()

    return _find_nearest(x.lower())


def solver_NOUN_NOUN(x):
    def _find_nearest(x, vocab=df_hard_nouns.word.values):
        sims = []
        for word_ in vocab:
            sim = dl.normalized_similarity(x, word_)
            sims.append(sim)
        if np.sort(sims)[-1] > 0.6:
            return vocab[np.argsort(sims)[-1]]
        else:
            return x

    word_1, word_2 = x.lower().split()

    return _find_nearest(word_2)


def solver_NUM_NOUN(x):
    # numerical case:

    def _find_nearest_numerical(word, vocab=None, n_neighbours=1):
        sims = []

        for word_ in vocab:
            sim = dl.normalized_similarity(word, word_)
            sims.append(sim)

        if n_neighbours == 1:
            for i in range(min(len(sims), 10)):
                if vocab[np.argsort(sims)[- i - 1]][0] == word[0]:
                    return vocab[np.argsort(sims)[- i - 1]], np.array(sims)[np.argsort(sims)[- i - 1]]
            return vocab[np.argsort(sims)[-1]], np.array(sims)[np.argsort(sims)[-1]]
        else:
            #         print(sims)
            return vocab[np.argsort(sims)[-n_neighbours:]], np.array(sims)[np.argsort(sims)[-n_neighbours:]]

    def _reconstruct(x):
        word_1, word_2 = x.lower().split()
        tag_1, tag_2 = morph.parse(word_1)[0].tag, morph.parse(word_2)[0].tag
        pos_1, pos_2 = tag_1.POS, tag_2.POS
        nearest_number = _find_nearest_numerical(word_1, list(df_numeralized.numeralized.values))[0]
        number_int = df_numeralized[(df_numeralized.numeralized == nearest_number)].number.values[0]
        tag_2_gender = tag_2.gender.replace("neut", "masc")
        reconstructed = df_numeralized[(df_numeralized.number == number_int) &
                                       (df_numeralized.gender_name == tag_2_gender) &
                                       (df_numeralized.kase_name == str(tag_2.case))].iloc[0].numeralized

        return reconstructed

    x = x.lower()
    if ("ами" in x or
        "тью" in x or
        "десят" in x or
        "ста" in x or
        "сотый" in x or
        "тыся" in x or
        "пятью" in x) and ("ПОЛТОРАСТАМИ".lower() not in x and
                           "ПОЛУТОРАСТАХ".lower() not in x and
                           "ПОЛТОРАСТАХ".lower() not in x and
                           "ПОЛУТОРАМИ".lower() not in x and
                           "ПОЛУТОРАХ".lower() not in x and
                           "ВОСЬМИДЕСЯТЬЮ".lower() not in x and
                           "ВОСЬМИСТАМИ".lower() not in x):
        return _reconstruct(x)
    elif "ПОЛТОРАСТАМИ".lower() in x:
        return "полутораста"
    elif "ПОЛТОРАСТАХ".lower() in x:
        return "полтораста"
    elif "ПОЛУТОРАСТАХ".lower() in x:
        return "полутораста"
    elif "ПОЛУТОРАМИ".lower() in x:
        return "полутора"
    elif "ПОЛУТОРАХ".lower() in x:
        return "полутора"
    elif "ВОСЬМИДЕСЯТЬЮ".lower() in x:
        return "восьмьюдесятью"
    elif "ВОСЬМИСТАМИ".lower() in x:
        return "восьмьюстами"
    else:
        return x.split()[0]

def solver_ADJF_NOUN_0(x):
    x = x.lower()
    if ("ами" in x or
        "тью" in x or
        "ста" in x or
        "сотый" in x or
        "десят" in x or
        "тыся" in x or
        "полутор" in x or
        "полтора" in x or
        "пятью" in x) and "ПОЛТОРАСТАМИ".lower() not in x:
        words_list = np.array(["сотый", "двухсотый", "трёхсотый", "четырёхсотый", "пятисотый", "семисотый"
                               "восьмисотый" "девятисотый", "тысячный", "двухтысячный", "трёхтысячный"])
        word_1, word_2 = x.split()
        return words_list[np.argmax([dl.normalized_similarity(word_1, word_) for word_ in words_list])]
    elif "ихней" in x.lower():
        return "их"
    elif "ихние" in x.lower():
        return "их"
    elif "евойный" in x.lower():
        return "его"
    elif "ихнего" in x.lower():
        return "их"
    elif "ихний" in x.lower():
        return "их"
    elif "ихних" in x.lower():
        return "их"
    else:
        return x.split()[0].lower()


def solver_ADJF_NOUN_1(x):
    def _inflect(initial_word, context_word):

        initial = morph.parse(initial_word)[0]
        context = morph.parse(context_word)[0]

        ans_formed = initial.inflect({context.tag.number,
                                      context.tag.case} - {None})

        return ans_formed.word

    xs_ = x.split()
    initial = xs_[1].lower()
    context = xs_[0].lower()

    if not morph.word_is_known(xs_[1].lower()):
        if initial[-2:] == "ов":
            initial = initial[:-2]
        elif initial[-2:] == "ки":
            initial = initial[:-2] + "ка"
        else:
            pass

    inflected = _inflect(initial, context)

    return inflected


def solver_NAI_BOLEE(x):
    try:
        xs_ = x.lower().split("более")[1].split()[0].strip()
    except:
        xs_ = x.lower().split()[-1].strip()
    return xs_


def solver_7_reconstructor(x):
    xs_ = x.split()
    parsed_words = []
    for word_ in xs_:
        parsed_words.append(str(morph.parse(word_.lower())[0].tag.POS))

    if "более" in x.lower():
        y_pred = solver_NAI_BOLEE(x)
    elif " ".join(parsed_words) == 'NOUN NOUN' and xs_[0].lower() != xs_[0]:
        y_pred = solver_NUM_NOUN(x)
    #         print(x, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'NOUN NOUN' and xs_[0].lower() == xs_[0]:
        y_pred = solver_NOUN_NOUN(x)
    #         print(x, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'ADJF NOUN' and xs_[0].lower() != xs_[0]:
        y_pred = solver_ADJF_NOUN_0(x)
    #         print(x, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'ADJF NOUN' and xs_[0].lower() == xs_[0]:
        y_pred = solver_ADJF_NOUN_1(x)
    #         print(x, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'PREP NOUN NOUN' and xs_[1].lower() != xs_[1]:
        y_pred = solver_NUM_NOUN(" ".join(xs_[1:]))
    #         print(x, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'ADVB NOUN' and xs_[0].lower() != xs_[0]:
        y_pred = solver_NUM_NOUN(x)
    #         print(x, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'VERB' and xs_[0].lower() != xs_[0]:
        y_pred = solver_VERB(x)
    #         print(xs_, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'VERB NOUN' and xs_[1].lower() == xs_[1]:
        x = xs_[0].lower()
        y_pred = solver_VERB(x)
    #         print(xs_, y_, y_pred, y_ == y_pred)
    elif " ".join(parsed_words) == 'ADVB VERB' and xs_[0].lower() == xs_[0]:
        x = xs_[1].lower()
        y_pred = solver_VERB(x)
    else:
        y_pred = solver_ANY(x)

    return y_pred


def solver_7(task):
    x_ = task['text'].split("\n")[1:][:5]

    incorrect_id = solver_7_xgb_classifier(x_)
    corrected_word = solver_7_reconstructor(x_[incorrect_id])
    return corrected_word