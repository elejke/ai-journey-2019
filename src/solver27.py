import re
import regex
import random

import joblib
import numpy as np
import pandas as pd

import pymorphy2

from summa import summarizer
from fastai.text import *


morph = pymorphy2.MorphAnalyzer()

essay_template = {
    # 1. Формулировка проблемы текста
    # {theme_name} должно стоять в винительном падеже
    # Примеры:
    #   {theme_name} = "тему войны",
    #   {theme_name} = "тему семьи и детства"          (хз ваще насколько это круто,
    #   {theme_name} = "тему деревни и города"          надо подумать, выписал примеры)
    #   {theme_name} = "тему культуры и искусства"
    '1.1': 'Автор в своем тексте раскрывает {theme_name} и рассматривает несколько актуальных проблем современности.',
    # здесь формулировка проблемы может быть сделана только в именительном падеже:
    # Пример:
    #   {problem_formulation} = "проблема эгоизма"
    '1.2b': 'Одной из самых важных среди них является {problem_formulation}.',
    # если оставлять пункт 1.3, то в нем должен быть маппинг на предложение {problem_explanation},
    # в котором идет раскрытие проблемы, это раскрытие является общим для данной проблемы и не зависит от текста:
    # Пример:
    #   {author} = "Л. Н. Толстой"
    # Пример:
    #   {problem_explanation} = "безразличие людей к своим родным и близким, таким же как они сами людям"
    '1.3': '{author} заставляет нас задуматься о такой проблеме как {problem_explanation}.\n',
    # 2. Комментарий проблемы (здесь два примера по проблеме из прочитанного текста, которые помогают понять суть
    # проблемы)
    '2.1': '"{citation1}" - в данном предложении текста находит свое отражение главный тезис рассуждений автора.',
    '2.2': 'Более детально в сути проблемы можно разобраться прочитав предложение "{citation2}".\n',
    # 3. Авторская позиция по обозначенной проблеме.
    # В случае авторской позиции, как мы уже обсуждали с @vovacher, хотелось бы захардкодить все возможные авторские
    # позиции под проблемы из списка. Если же это не будет реализовано, то генерация авторской позиции (???)
    # Пример:
    #   {author_position} = "в обществе распространилась страшная болезнь – «себялюбие»"
    '3.1': 'Развивая свою мысль, автор убеждает нас в том, что {author_position}.',
    # '3.2': 'Также {author} считает, что {water}',
    # вместо {water} считаю, что нужно генерировать другими словами авторскую позицию (нужна подобная опция)
    # Пример:
    #   {author_position_reformulated} = "эгоизм и «себялюбие» захватывают наше общество"
    '3.2': 'Таким образом {author} считает, что {author_position_reformulated}.\n',
    # 4. Собственное мнение по обозначенной проблеме (согласие).
    # Пример:
    #   {own_position} = {author_position_reformulated2} = "эгоизм - это плохо"
    '4.1': "Трудно не согласиться с мнением автора по обозначенной проблеме. Я тоже считаю, что {own_position}.",
    '4.2': '{water}.',
    '4.3': 'Не зря этот вопрос поднимался и во многих других произведениях известных литераторов.\n',
    # 5. Аргумент 1 (из художественной, публицистической или научной литературы).
    # Раздел 5 и 6 в идеальном случае содержит заранее известные аргументы, интегрированные в сочинение (взятые из
    # готовых сочинений).
    # Пример:
    #   {agrument_paragraph1} = "Вспомним, например, рассказ А. П. Чехов «Анна на шее». Его главная героиня,
    #                            Анюта, став по расчету женой состоятельного чиновника, быстро забывает о своем
    #                            отце и братьях, которых прежде так любила. Эгоизм, поселившийся в ее душе после
    #                            замужества, способствует этому."
    '5.1': "{agrument_paragraph1}\n",
    # 6. Аргумент 2 (из жизни). В данном параграфе аналогично пятому используется готовый аргумент, который уже
    # обернут в конструкции цитирования и абсолютно избавлен от зависимости с текстом.
    # Пример:
    #   {agrument_paragraph2} = "Д. Лондон в своем произведении «В далеком краю» повествует читателям о судьбе
    #                            Уэзерби и Катферта. Отправившись на Север за золотом, они вынуждены перезимовать
    #                            вдвоем в хижине, стоящей далеко от обитаемых мест. И здесь с жестокой
    #                            очевидностью выступает их бескрайний эгоизм. Отношения между героями — та же
    #                            конкурентная борьба, только не за прибыль, а за выживание. И в тех условиях, в
    #                            каких они очутились, исход ее не может быть иным, чем в финале новеллы: умирает
    #                            Катферт, придавленный телом Уэзерби, которого он прикончил в звериной драке
    #                            из-за чашки сахара."
    '6.1': "{agrument_paragraph2}\n",
    # 7. Заключение.
    # На деле - заключение {conclusion} должно являться переформулированной позицией автора
    # (у меня была клевая училка по русскому, и она всегда это говорила), так что в этом разделе нам тоже нужна
    # опция с возможностью переформулировать некоторое выражение другими словами:
    # Пример:
    #   {conclusion} = {author_position_reformulated3} = "«себялюбие» - это порок современного общества"
    '7.1': 'Таким образом, вместе с автором текста мы приходим к выводу, что {conclusion}',
    # 'Заканчивая размышление над прочитанным текстом, сделаю вывод: как важно {conclusion}'
}


class EssayWriter(object):
    """
    Note, fastai==1.0.52.
    Простой генератор на основе Ulmfit (https://github.com/mamamot/Russian-ULMFit) и тематического моделирования на
    текстах заданий.
    Дообучается на сочинениях (не учитывает ничего про условие задания).
    Генерация начинается с первой фразы, которая соответствует темам,
    которые были получены в ходе тематического моделирования текста задания.
    В код не включено обучение тематической модели. Интерпретация проведена в ручную.
    Первые фразы сочинений написаны вручную.

    Parameters
    ----------
    seed : path2config, str
        Path to config.
    ulmfit_model_name : str
        Model name for load pretrained ulmfit model and store this.
    ulmfit_dict_name : str
        Dict name for load pretrained ulmfit dict and store this.
    tf_vectorizer_path : str
        Path to vectorizer for topic modeling.
    lda_path : str
        Path to topic model.
    topics_path : str
        Path to topics with first phrases.
    is_load : bool, optional(default=True)
        Load or not pretrained models.
    Examples
    --------
    g = EssayWriter("lm_5_ep_lr2-3_5_stlr", "itos", "tfvect.joblib", "lda.joblib", "topics.csv", is_load=False)
    g = g.fit(df_path="10000.csv", num_epochs=5)
    text = g.generate("Печенье и судьба")
    g.save()
    """

    def __init__(
            self,
            ulmfit_model_name=None,
            ulmfit_dict_name=None,
            tf_vectorizer_path=None,
            lda_path=None,
            topics_path=None,
            is_load=True,
            seed=42
    ):

        self.ulmfit_model_name = ulmfit_model_name
        self.ulmfit_dict_name = ulmfit_dict_name
        self.data = None
        self.learn = None
        self.temperature = None
        self.tf_vectorizer_path = tf_vectorizer_path
        self.lda_path = lda_path
        self.topics_path = topics_path
        self.tf_vectorizer = None
        self.lda = None
        self.topics = None
        self.topic_dic = None
        if is_load:
            self.load()
        self.seed = seed
        self._init_seed()

    def _init_seed(self):
        random.seed(self.seed)

    def get_topic(self, documents):
        tf = self.tf_vectorizer.transform(documents)
        lda_doc_topic = self.lda.transform(tf)
        doc_topics = []
        for n in range(lda_doc_topic.shape[0]):
            topic_most_pr = lda_doc_topic[n].argmax()
            doc_topics.append(topic_most_pr)
        return [self.topic_dic[i] for i in doc_topics]

    def getinfo(self, topic):
        dic = {}
        for i in range(len(self.topics)):
            if self.topics.iloc[i]['Topic'] == topic:
                dic['Первая_фраза'] = self.topics.iloc[i]['First']
                dic['Произведения для аргументов'] = self.topics.iloc[i]['Books']
                dic['Тема'] = self.topics.iloc[i]['Theme']
                dic['Писатели'] = self.topics.iloc[i]['Authors']
        return dic

    def load(self):

        self.tf_vectorizer = joblib.load(self.tf_vectorizer_path)
        self.lda = joblib.load(self.lda_path)
        self.topics = pd.read_csv(self.topics_path, sep="\t")
        self.topic_dic = {int(i): self.topics.iloc[i]['Topic'] for i in range(len(self.topics))}

        self.data = TextList.from_df(
            pd.DataFrame(["tmp", "tmp"]),
            processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")),
                       NumericalizeProcessor(vocab=Vocab.load("models/{}.pkl".format(self.ulmfit_dict_name)))]
        ).random_split_by_pct(.1).label_for_lm().databunch(bs=16)

        conf = awd_lstm_lm_config.copy()
        conf['n_hid'] = 1150
        self.learn = language_model_learner(self.data, AWD_LSTM, pretrained=True, config=conf, drop_mult=0.7,
                                            pretrained_fnames=[self.ulmfit_model_name, self.ulmfit_dict_name], silent=False)

        return self

    def generate(self,
                 task,
                 temperature=0.7):

        self.temperature = temperature
        task, text = split_task_and_text(task)
        author = get_author(task)

        brief_text = clear(summarizer.summarize(text, language="russian", ratio=0.25, split=False))
        citation1, citation2 = np.random.choice(summarizer.summarize(text, language="russian", ratio=0.1, split=True),
                                                size=2, replace=False)

        essay = self._1st_paragraph(brief_text, mention_author(author))
        essay = self._2nd_paragraph(essay, citation1, citation2)
        essay = self._3rd_paragraph(essay, mention_author(author))
        essay = self._4th_paragraph(essay)
        essay = self._5th_paragraph(essay, "проблема плохого примера влияет на результат")
        essay = self._6th_paragraph(essay, "проблема плохого примера влияет на результат")
        essay = self._7th_paragraph(essay)

        #     return essay[len(brief_text):]
        return essay

    def continue_phrase(self, text, n_words=10):
        text = clear(text)
        text = clear(self.learn.predict(text, n_words=n_words, no_unk=True, temperature=self.temperature))
        text = text.replace("xxbos", " ")  # Remove model special symbols
        text = text[:-40] + re.split(r"[.!?]", text[-40:])[0] + '. '  # Cut predicted sentence up to dot
        return clear(text)

    def _1st_paragraph(self, text, author):

        # TODO: theme classifier
        theme_name = "тему семьи и детства"
        # TODO: problem classifier:
        problem_formulation = "проблема эгоизма"
        # TODO: problem -> problem_explanation mapping dict:
        problem_explanation = "безразличие людей к своим родным и близким, таким же людям как и они"

        sentence_1 = essay_template['1.1'].format(theme_name=theme_name)
        sentence_2 = essay_template['1.2'].format(problem_formulation=problem_formulation)
        sentence_3 = essay_template['1.3'].format(author=author, problem_explanation=problem_explanation)

        #     next_sent = essay_template['1.3'].format(problem_formulation='')
        #     essay =  continue_phrase(text + '\n\n' + next_sent, 10)

        #     next_sent = essay_template['1.2'].format(author=author, problem_explanation='')
        #     essay =  continue_phrase(essay + ' ' + next_sent, 30)

        #     return essay + '\n'

        return " ".join([sentence_1, sentence_2, sentence_3])

    def _2nd_paragraph(self, essay, citation1, citation2):

        # TODO: citation detector:
        # citation1, citation2 = self.citation_detector(text, problem_formulation, n_citations=2)
        citation1 = "Все люди должны знать свои права и обязанности и не имеют права нарушать права других людей."
        citation2 = "Эгоистичное отношение к людям давно выведено из норм общественного поведения."

        sentence_1 = essay_template['2.1'].format(citation1=citation1)
        sentence_2 = essay_template['2.2'].format(citation2=citation2)

        #     next_sent = essay_template['2.1'].format(citation=citation)
        #     essay += next_sent

        #     next_sent = essay_template['2.2'].format(water='')
        #     essay =  continue_phrase(essay + ' ' + next_sent, 40)

        #     return essay + '\n'

        return " ".join([essay, sentence_1, sentence_2])

    def _3rd_paragraph(self, essay, author):

        # TODO: author position detector (???)):
        # author_position = self.author_position_detector(text, self.problem_formulation)
        author_position = "в обществе распространилась страшная болезнь – «себялюбие»"
        #     author_position_reformulated = self.reformulate_author_position(author_position)
        author_position_reformulated = "эгоизм и «себялюбие» захватывают наше общество"

        sentence_1 = essay_template['3.1'].format(author_position=author_position)
        sentence_2 = essay_template['3.2'].format(author=author,
                                                  author_position_reformulated=author_position_reformulated)

        #     next_sent = essay_template['3.1'].format(author_position='')
        #     essay =  continue_phrase(essay + next_sent, 25)

        #     next_sent = essay_template['3.2'].format(author=author, water='')
        #     essay =  continue_phrase(essay + ' ' + next_sent, 40)

        #     return essay + '\n'

        return " ".join([essay, sentence_1, sentence_2])

    def _4th_paragraph(self, essay):

        # TODO: OWN POSITION ????:

        # TODO: author position detector (???)):
        # author_position = self.author_position_detector(text, self.problem_formulation)
        # author_position_reformulated = self.reformulate_author_position(author_position)
        # own_position = author_position_reformulated
        own_position = "эгоизм - это плохо"
        # TODO: water generator:
        # water = self.water_generator(text, last_sentnece, ...)
        water = "Я был бы очень рад лично обсудить с автором вопросы, связанные с этой чудовищной проблемой."
        water = " ".join([water,
                          "Несмотря на то, что сами идеи являются основой всего нашего общества, " +
                          "многим они все равно чужды"])
        sentence_1 = essay_template['4.1'].format(own_position=own_position)
        sentence_2 = essay_template['4.2'].format(water=water)

        #     next_sent = essay_template['4.1'].format(own_position='')
        #     essay =  continue_phrase(essay + ' ' + next_sent, 20)

        #     next_sent = essay_template['4.2'].format(water='')
        #     essay =  continue_phrase(essay + ' ' + next_sent, 40)

        #     next_sent = essay_template['4.3'].format(water='')
        #     essay += next_sent

        #     return essay + '\n'
        return " ".join([essay, sentence_1, sentence_2])

    def _5th_paragraph(self, essay, problem_formulation):

        # TODO: argument paragraph generator

        agrument_paragraph1 = ("Д. Лондон в своем произведении «В далеком краю» повествует читателям о судьбе " +
                               "Уэзерби и Катферта. Отправившись на Север за золотом, они вынуждены перезимовать " +
                               "вдвоем в хижине, стоящей далеко от обитаемых мест. И здесь с жестокой " +
                               "очевидностью выступает их бескрайний эгоизм. Отношения между героями — та же " +
                               "конкурентная борьба, только не за прибыль, а за выживание. И в тех условиях, в " +
                               "каких они очутились, исход ее не может быть иным, чем в финале новеллы: умирает " +
                               "Катферт, придавленный телом Уэзерби, которого он прикончил в звериной драке " +
                               "из-за чашки сахара.")
        #     next_sent = essay_template['5.1'].format(argument1_author=argument1_author,
        #                                              argument1_source_name=argument1_source_name)
        #     essay += next_sent

        #     next_sent = essay_template['5.2'].format(water='')
        #     essay =  continue_phrase(essay + next_sent, 40)

        #     return essay + '\n'
        sentence_1 = essay_template['5.1'].format(agrument_paragraph1=agrument_paragraph1)

        return " ".join([essay, sentence_1])

    def _6th_paragraph(self, essay, problem_formulation):

        # TODO: argument paragraph generator

        agrument_paragraph2 = ("Вспомним, например, рассказ А. П. Чехов «Анна на шее». Его главная героиня, " +
                               "Анюта, став по расчету женой состоятельного чиновника, быстро забывает о своем " +
                               "отце и братьях, которых прежде так любила. Эгоизм, поселившийся в ее душе после " +
                               "замужества, способствует этому.")
        #     next_sent = essay_template['6.1'].format(argument2_author=argument2_author,
        #                                              argument2_source_name=argument2_source_name)
        #     essay += next_sent

        #     next_sent = essay_template['6.2'].format(water='')
        #     essay =  continue_phrase(essay + next_sent, 40)

        #     return essay + '\n'

        sentence_1 = essay_template['6.1'].format(agrument_paragraph2=agrument_paragraph2)

        return " ".join([essay, sentence_1])

    def _7th_paragraph(self, essay):

        #     next_sent = essay_template['7.1'].format(conclusion='')
        #     essay =  continue_phrase(essay + next_sent, 40)

        #     return essay

        # example:
        conclusion = "«себялюбие» - это порок современного общества"
        # author_position_reformulated = self.reformulate_author_position(author_position)
        # conclusion = author_position_reformulated

        sentence_1 = essay_template['7.1'].format(conclusion=conclusion)

        return " ".join([essay, sentence_1])

    def __call__(self, task):
        return self.generate(task["text"])


def split_task_and_text(task_text):
    """Split initial task text to the question and actual text fragment. Using texts' sentence counters - (k).
    :return : tuple(Task formulation, Referenced text of the task)
    """

    splitted = re.split(r'\(\d{1,3}\)', task_text)
    formulation = [splitted[0]]
    text = splitted[1:-1]

    last = re.split(r'[!?.]', splitted[-1])
    text.append(last[0])
    formulation.append('.'.join(last[1:]))

    return ''.join(formulation), ''.join(text).strip()


def clear(text):
    text = re.sub("[\t\r]+", "", text)
    text = re.sub("[ ]+[:]", ":",
                  re.sub("[ ]+[.]", ".",
                         re.sub("[«][ ]+", "«",
                                re.sub("[ ]+[»]", "»",
                                       re.sub("[ ]+[,]", ",",
                                              re.sub("[ ]+", " ", text))))))
    text = re.sub("[ ]+[?]", "?", text)
    text = re.sub("[ ]+[!]", "!", text)
    text = re.sub("\n+", "\n", text)
    text = [line.strip() for line in text.split("\n")]
    # text = [line[1:] + line[1].upper() for line in text if len(line)]
    text = "\n".join(text)
    return text


def rus_tok(text):
    reg = '([0-9]|\W|[a-zA-Z])'
    toks = text.split()
    return [morph.parse(t)[0].normal_form for t in toks if not re.match(reg, t)]


def get_author(text):
    name_pattern = "[А-Я][а-яё]+[\p{Pd}−]*\w*"
    short_name_pattern = "[А-Я]\."

    match = regex.search(f"({name_pattern})\s+({name_pattern})\s+({name_pattern})", text)
    if match is not None:
        return list(match.groups())

    match = regex.search(f"({name_pattern})\s+({name_pattern})", text)
    if match is not None:
        return list(match.groups())

    match = regex.search(f"({short_name_pattern})\s*({short_name_pattern})\s*({name_pattern})", text)
    if match is not None:
        name = list(match.groups())
        name[2] = morph.parse(name[2])[0].inflect({"nomn"}).word.capitalize()
        return name

    match = regex.search(f"({short_name_pattern})\s*({name_pattern})", text)
    if match is not None:
        name = list(match.groups())
        name[1] = morph.parse(name[1])[0].inflect({"nomn"}).word.capitalize()
        return name

    return ["автор"]


def mention_author(author, case="nomn"):
    """Упоминает автора в нужном формате и склонении. Юзать правда лучше только в именительном, т.к. некоторые
    фамилии не склоняются. Например, Черных

    nomn	именительный	Кто? Что?	хомяк ест
    gent	родительный	    Кого? Чего?	у нас нет хомяка
    datv	дательный	    Кому? Чему?	сказать хомяку спасибо
    accs	винительный	    Кого? Что?	хомяк читает книгу
    ablt	творительный	Кем? Чем?	зерно съедено хомяком
    loct	предложный	    О ком? О чём? и т.п.	хомяка несут в корзинке
    voct	звательный	    Его формы используются при обращении к человеку.	Саш, пойдем в кино.
    """
    if case not in ["nomn", "gent", "datv", "accs", "ablt", "loct", "voct"]:
        case = "nomn"
    if case != "nomn":
        last_name = morph.parse(author[-1])[0].inflect({case})[0]
    else:
        last_name = author[-1]
    if len(author) > 1:
        last_name = last_name[0].upper() + last_name[1:]
        initials = ". ".join(map(lambda x: x[0].upper(), author[:-1]))
        result = "{}. {}".format(initials, last_name)
    else:
        result = last_name
    return result
