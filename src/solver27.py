import re
import regex
import random

import pickle
import joblib
import numpy as np
import pandas as pd

import pymorphy2

import yake
import fasttext
from sklearn.metrics import pairwise_distances

from summa import summarizer
from fastai.text import *


morph = pymorphy2.MorphAnalyzer()
kw_extractor = yake.KeywordExtractor(lan="ru", n=1, top=30)

essay_template = {
    # 1. Формулировка проблемы текста
    # {theme_name} должно стоять в винительном падеже
    # Примеры:
    #   {theme_name} = "тему войны",
    #   {theme_name} = "тему семьи и детства"          (хз ваще насколько это круто,
    #   {theme_name} = "тему деревни и города"          надо подумать, выписал примеры)
    #   {theme_name} = "тему культуры и искусства"
    '1.1': [
        '{author} в своем тексте раскрывает тему {theme_name} и рассматривает несколько актуальных проблем современности',
        '{author} затрагивает важную общечеловеческую тему {theme_name} и раскрывает ряд актуальных проблем',
    ],
    # здесь формулировка проблемы может быть сделана только в именительном падеже:
    # Пример:
    #   {problem_formulation} = "проблема эгоизма"
    '1.2': [
        'Одной из самых важных среди них является {problem_formulation}',
        'Одной из самых важных среди них является {problem_formulation}'
    ],
    # если оставлять пункт 1.3, то в нем должен быть маппинг на предложение {problem_explanation},
    # в котором идет раскрытие проблемы, это раскрытие является общим для данной проблемы и не зависит от текста:
    # Пример:
    #   {author} = "Л. Н. Толстой"
    # Пример:
    #   {problem_explanation} = "безразличие людей к своим родным и близким, таким же людям как и они"
    '1.3': [
        '{problem_explanation}',
        '{problem_explanation}'
        # '{author} заставляет нас глубоко задуматься о {problem_explanation}',
        # '{problem_explanation} - именно над этим заставляет нас поразмышлять {author}'
    ],
    # 2. Комментарий проблемы (здесь два примера по проблеме из прочитанного текста, которые помогают понять суть
    # проблемы)
    '2.1': [
        '«{citation1}» - в данном предложении текста находит свое отражение главный тезис рассуждений автора',
        'Основной тезис авторской позиции выражен в следующих словах: «{citation1}»',
    ],
    # Трактовка цитаты. Но это только генератором скорее всего.
    '2.2': [
        'Эти слова обращают наше внимание на {water}',
        'Этот отрывок объясняет нам, что {water}',
        # 'В них звучит мысль о том, что {water}.'
    ],
    '2.3': [
        'Более детально в сути проблемы можно разобраться прочитав предложение «{citation2}»',
        'Но на этом рассуждения автора не заканчиваются. Он также пишет: «{citation2}»',
    ],
    '2.4': [
        'Обе приведённых цитаты, дополняя друг друга, позволяют нам убедиться в том, что {water}',
        'Этот пример еще раз показывает нам, что {water}',
    ],
    # 3. Авторская позиция по обозначенной проблеме.
    # В случае авторской позиции, как мы уже обсуждали с @vovacher, хотелось бы захардкодить все возможные авторские
    # позиции под проблемы из списка. Если же это не будет реализовано, то генерация авторской позиции (???)
    # Пример:
    #   {author_position} = "в обществе распространилась страшная болезнь – «себялюбие»"
    '3.1': [
        'Развивая свою мысль, автор убеждает нас в том, что {author_position}',
        'На основании всего вышесказанного можно четко выделить позицию автора: {author_position}'
    ],
    # '3.2': 'Также {author} считает, что {water}',
    # вместо {water} считаю, что нужно генерировать другими словами авторскую позицию (нужна подобная опция)
    # Пример:
    #   {author_position_reformulated} = "эгоизм и «себялюбие» захватывают наше общество"
    '3.2': [
        'Таким образом {author_last_name} считает, что {author_position_reformulated}',
        '{author_last_name} убеждает нас в том, что {author_position_reformulated}'
    ],
    # 4. Собственное мнение по обозначенной проблеме (согласие).
    # Пример:
    #   {own_position} = {author_position_reformulated2} = "эгоизм - это плохо"
    '4.1': [
        "Трудно не согласиться с обозначенными в тексте проблемами. Я тоже считаю, что {own_position}",
        "Я полностью поддерживаю точку зрения автора и также считаю, что {own_position}",
        # "С полной уверенностью могу сказать, что автор прав. {own_position}."
    ],
    '4.2': [
        'Не зря этот вопрос поднимался и во многих других произведениях известных литераторов',
        'Этот вопрос не раз поднимался и в других произведениях известных литераторов',
    ],
    # 5. Аргумент 1 (из художественной, публицистической или научной литературы).
    # Раздел 5 и 6 в идеальном случае содержит заранее известные аргументы, интегрированные в сочинение (взятые из
    # готовых сочинений).
    # Пример:
    #   {argument_paragraph1} = "Вспомним, например, рассказ А. П. Чехов «Анна на шее». Его главная героиня,
    #                            Анюта, став по расчету женой состоятельного чиновника, быстро забывает о своем
    #                            отце и братьях, которых прежде так любила. Эгоизм, поселившийся в ее душе после
    #                            замужества, способствует этому."
    '5.1': "{argument_paragraph1}",
    # 6. Аргумент 2 (из жизни). В данном параграфе аналогично пятому используется готовый аргумент, который уже
    # обернут в конструкции цитирования и абсолютно избавлен от зависимости с текстом.
    # Пример:
    #   {argument_paragraph2} = "Д. Лондон в своем произведении «В далеком краю» повествует читателям о судьбе
    #                            Уэзерби и Катферта. Отправившись на Север за золотом, они вынуждены перезимовать
    #                            вдвоем в хижине, стоящей далеко от обитаемых мест. И здесь с жестокой
    #                            очевидностью выступает их бескрайний эгоизм. Отношения между героями — та же
    #                            конкурентная борьба, только не за прибыль, а за выживание. И в тех условиях, в
    #                            каких они очутились, исход ее не может быть иным, чем в финале новеллы: умирает
    #                            Катферт, придавленный телом Уэзерби, которого он прикончил в звериной драке
    #                            из-за чашки сахара."
    '6.1': "{argument_paragraph2}",
    # 7. Заключение.
    # На деле - заключение {conclusion} должно являться переформулированной позицией автора
    # (у меня была клевая училка по русскому, и она всегда это говорила), так что в этом разделе нам тоже нужна
    # опция с возможностью переформулировать некоторое выражение другими словами:
    # Пример:
    #   {conclusion} = {author_position_reformulated3} = "«себялюбие» - это порок современного общества"
    '7.1': ['Таким образом, вместе с автором текста мы приходим к выводу, что {conclusion}',
            'Подводя итог, можно сказать, что {conclusion}'],
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
    ulmfit_model_name : str
        Model name for load pretrained ulmfit model and store this.
    ulmfit_dict_name : str
        Dict name for load pretrained ulmfit dict and store this.
    lda_tf_vectorizer_path : str
        Path to vectorizer for topic modeling.
    lda_path : str
        Path to topic model.
    lda_topics_path : str
        Path to topics with first phrases.
    is_load : bool, optional(default=True)
        Load or not pretrained models.
    seed : int
        Random seed.
    fasttext_model : str or FastText model loaded
        FastText model.
    custom_topic_keywords_vectors_path : str
        path to the pickle file with dict topic: vectors of keywords
    custom_topics_path : str
        Path to custom topics.
    stopwords_path : str
        path to the pickle file with list of stopwords
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
            lda_tf_vectorizer_path=None,
            lda_path=None,
            lda_topics_path=None,
            is_load=True,
            seed=42,
            fasttext_model=None,
            custom_topic_keywords_vectors_path=None,
            custom_topics_path=None,
            stopwords_path=None
    ):

        self.ulmfit_model_name = ulmfit_model_name
        self.ulmfit_dict_name = ulmfit_dict_name
        self.data = None
        self.learn = None
        self.temperature = None
        self.lda_tf_vectorizer_path = lda_tf_vectorizer_path
        self.lda_path = lda_path
        self.lda_topics_path = lda_topics_path
        self.lda_tf_vectorizer = None
        self.lda = None
        self.lda_topics = None
        self.lda_topic_dic = None
        self.custom_topic_keywords_vectors_path = custom_topic_keywords_vectors_path
        self.custom_topic_keywords_vectors = None
        self.fasttext_model = fasttext_model
        self.stopwords_path = stopwords_path
        self.stopwords = None
        self.custom_topics_path = custom_topics_path
        self.custom_topics = None
        if is_load:
            self.load()
        self.seed = seed
        self._init_seed()

    def _init_seed(self):
        random.seed(self.seed)

    def get_lda_topic(self, documents):
        tf = self.lda_tf_vectorizer.transform(documents)
        lda_doc_topic = self.lda.transform(tf)
        doc_topics = []
        for n in range(lda_doc_topic.shape[0]):
            topic_most_pr = lda_doc_topic[n].argmax()
            doc_topics.append(topic_most_pr)
        return [self.lda_topic_dic[i] for i in doc_topics]

    def get_lda_info(self, topic):
        dic = {}
        for i in range(len(self.lda_topics)):
            if self.lda_topics.iloc[i]['Topic'] == topic:
                dic['Первая_фраза'] = self.lda_topics.iloc[i]['First']
                dic['Произведения для аргументов'] = self.lda_topics.iloc[i]['Books']
                dic['Тема'] = self.lda_topics.iloc[i]['Theme']
                dic['Писатели'] = self.lda_topics.iloc[i]['Authors']
        return dic

    def get_custom_topic(self, text):
        text_keywords = kw_extractor.extract_keywords(" ".join(custom_tok(text, self.stopwords +
                                                                          ["рука", "свой", "самый", "какой-то",
                                                                           "голова", "глаз", "некоторый", "ним"])))
        text_keywords_vectors = []
        text_keywords_dists = []
        for w in text_keywords:
            text_keywords_vectors.append(self.fasttext_model[w[0]])
            text_keywords_dists.append(w[1])
        text_keywords_vectors = np.array(text_keywords_vectors)
        text_keywords_vectors /= np.linalg.norm(text_keywords_vectors, ord=2, axis=1, keepdims=True)
        text_keywords_dists = np.array(text_keywords_dists)

        topics = []
        topics_dist = []
        for topic in self.custom_topic_keywords_vectors:
            distance_matrix = pairwise_distances(text_keywords_vectors, self.custom_topic_keywords_vectors[topic])
            topics.append(topic)
            #         topics_dist.append(np.min(distance_matrix, axis=0).mean())
            topics_dist.append(np.min(distance_matrix * text_keywords_dists.reshape(-1, 1), axis=0).mean())
        topic = topics[np.argmin(topics_dist)]

        return topic

    def load(self):

        self.lda_tf_vectorizer = joblib.load(self.lda_tf_vectorizer_path)
        self.lda = joblib.load(self.lda_path)
        self.lda_topics = pd.read_csv(self.lda_topics_path, sep="\t")
        self.lda_topic_dic = {int(i): self.lda_topics.iloc[i]['Topic'] for i in range(len(self.lda_topics))}

        self.data = TextList.from_df(
            pd.DataFrame(["tmp", "tmp"]),
            processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")),
                       NumericalizeProcessor(vocab=Vocab.load("models/{}.pkl".format(self.ulmfit_dict_name)))]
        ).random_split_by_pct(.1).label_for_lm().databunch(bs=16)

        conf = awd_lstm_lm_config.copy()
        conf['n_hid'] = 1150
        self.learn = language_model_learner(self.data, AWD_LSTM, pretrained=True, config=conf, drop_mult=0.7,
                                            pretrained_fnames=[self.ulmfit_model_name, self.ulmfit_dict_name],
                                            silent=False)

        if isinstance(self.fasttext_model, str):
            self.fasttext_model = fasttext.load_model(self.fasttext_model)
        elif self.fasttext_model is None:
            raise ValueError("Pass loaded 'fasttext' instance")
        else:
            pass
        with open(self.custom_topic_keywords_vectors_path, "rb") as f:
            self.custom_topic_keywords_vectors = pickle.load(f)
        with open(self.stopwords_path, "rb") as f:
            self.stopwords = pickle.load(f)
        self.custom_topics = pd.read_csv(self.custom_topics_path, sep=';', index_col='theme')

        return self

    def generate(self,
                 task,
                 temperature=0.7):

        self.temperature = temperature
        task, text = split_task_and_text(task)

        author = get_author(task)
        theme = self.get_custom_topic(text)

        brief_text = clear(summarizer.summarize(text, language="russian", ratio=0.25, split=False))
        citations = summarizer.summarize(text, language="russian", ratio=0.1, split=True)

        essay = self._1st_paragraph(theme=self.custom_topics.loc[theme]["theme_to_insert_vinitelniy"],
                                    problem_formulation=self.custom_topics.loc[theme]["problem_formulation"],
                                    problem_explanation=self.custom_topics.loc[theme]["problem_explanation"],
                                    author=mention_author(author))
        essay = self._2nd_paragraph(brief_text + '\n\n' + essay, citations[0], citations[1])
        essay = self._3rd_paragraph(essay,
                                    author_last_name=mention_author(author, mode='Aa'),
                                    author_position=self.custom_topics.loc[theme]["author_position"],
                                    author_position_reformulated='water')
        essay = self._4th_paragraph(essay, own_position='water')
        essay = self._5th_paragraph(essay,
                                    argument_paragraph1=self.custom_topics.loc[theme]["argument_paragraph1"])
        essay = self._6th_paragraph(essay,
                                    argument_paragraph2=self.custom_topics.loc[theme]["argument_paragraph2"])
        essay = self._7th_paragraph(essay, conclusion='water')

        return essay[len(brief_text) + 2:]

    def continue_phrase(self, text, n_words=10):
        init_len = len(text)
        text = clear(text)
        text = clear(self.learn.predict(text, n_words=n_words, no_unk=True, temperature=self.temperature))
        text = text.replace("xxbos", " ")  # Remove model special symbols
        text = text[:-40] + re.split(r"[.!?]", text[-40:])[0]  # Cut predicted sentence up to dot
        text = clear(text)
        text = text[:init_len] + re.sub(r"\n+", r" ", text[init_len:])
        return text

    def continue_phrase_with_pattern(self, essay, sent_template, n_words, variable_name, variable_value,
                                     default_value, replacement_dict=None):

        if replacement_dict is None:
            replacement_dict = dict()

        if variable_value == 'water':
            essay = self.continue_phrase(essay + sent_template.format(**replacement_dict, **{variable_name: ''}),
                                         n_words)
        elif (variable_value == '') or variable_value is None:
            essay += sent_template.format(**replacement_dict, **{variable_name: default_value})
        else:
            essay += sent_template.format(**replacement_dict, **{variable_name: variable_value})

        return essay + '.'

    def _1st_paragraph(self,
                       theme=None,
                       problem_formulation=None,
                       problem_explanation=None,
                       author=None):

        if author is None:
            author = "Автор"
        # TODO: theme classifier
        if theme is None:
            theme = "семьи и детства"
        # TODO: problem classifier:
        if problem_formulation is None:
            problem_formulation = "проблема эгоизма"
        # TODO: problem -> problem_explanation mapping dict:
        if problem_explanation is None:
            problem_explanation = "проблема эгоизма"

        var = np.random.choice(range(len(essay_template['1.1'])))

        sentence_1 = essay_template['1.1'][var].format(
            author=author[0].upper() + author[1:], theme_name=theme
        )
        sentence_2 = essay_template['1.2'][var].format(problem_formulation=problem_formulation)
        sentence_3 = essay_template['1.3'][var].format(problem_explanation=problem_explanation).rstrip('.')

        return ". ".join([sentence_1, sentence_2, sentence_3]) + '.\n\n'

    def _2nd_paragraph(self, essay, citation1, citation2):

        # TODO: citation detector:
        # citation1, citation2 = self.citation_detector(text, problem_formulation, n_citations=2)

        var = np.random.choice(range(len(essay_template['2.1'])))

        essay = essay + essay_template['2.1'][var].format(citation1=citation1) + '. '
        essay = self.continue_phrase(essay + essay_template['2.2'][var].format(water=''), 40) + '. '
        essay = essay + ' ' + essay_template['2.3'][var].format(citation2=citation2) + '. '
        essay = self.continue_phrase(essay + essay_template['2.4'][var].format(water=''), 50).rstrip('.')

        return essay + '.\n\n'

    def _3rd_paragraph(self,
                       essay,
                       author_last_name=None,
                       author_position='water',
                       author_position_reformulated='water'):

        # TODO: author position detector (???)):
        if author_last_name is None:
            author_last_name = "автор"

        var = np.random.choice(range(len(essay_template['3.1'])))

        essay = self.continue_phrase_with_pattern(
            essay, essay_template['3.1'][var], 40, 'author_position', author_position,
            "в обществе распространилась страшная болезнь – «себялюбие»"
        )[:-1]

        essay = self.continue_phrase_with_pattern(
            essay + '. ', essay_template['3.2'][var], 60, 'author_position_reformulated',
            author_position_reformulated, "эгоизм и «себялюбие» захватывают наше общество",
            {'author_last_name': author_last_name}
        )

        return essay + '\n\n'

    def _4th_paragraph(self, essay, own_position='water'):

        # TODO: OWN POSITION detector????:
        var = np.random.choice(range(len(essay_template['4.1'])))

        essay = self.continue_phrase_with_pattern(
            essay, essay_template['4.1'][var], 30, 'own_position', own_position, 'эгоизм - это плохо'
        )

        essay += ' ' + essay_template['4.2'][var] + '.'

        return essay + '\n\n'

    def _5th_paragraph(self, essay, argument_paragraph1='water'):

        # TODO: argument paragraph generator

        default_value = (
                "Д. Лондон в своем произведении «В далеком краю» повествует читателям о судьбе " +
                "Уэзерби и Катферта. Отправившись на Север за золотом, они вынуждены перезимовать " +
                "вдвоем в хижине, стоящей далеко от обитаемых мест. И здесь с жестокой " +
                "очевидностью выступает их бескрайний эгоизм. Отношения между героями — та же " +
                "конкурентная борьба, только не за прибыль, а за выживание. И в тех условиях, в " +
                "каких они очутились, исход ее не может быть иным, чем в финале новеллы: умирает " +
                "Катферт, придавленный телом Уэзерби, которого он прикончил в звериной драке " +
                "из-за чашки сахара."
        )
        essay = self.continue_phrase_with_pattern(
            essay, essay_template['5.1'], 90, 'argument_paragraph1', argument_paragraph1, default_value
        )

        return essay[:-1] + '\n\n'

    def _6th_paragraph(self, essay, argument_paragraph2='water'):

        # TODO: argument paragraph generator

        default_value = (
                "Вспомним, например, рассказ А. П. Чехов «Анна на шее». Его главная героиня, " +
                "Анюта, став по расчету женой состоятельного чиновника, быстро забывает о своем " +
                "отце и братьях, которых прежде так любила. Эгоизм, поселившийся в ее душе после " +
                "замужества, способствует этому."
        )
        essay = self.continue_phrase_with_pattern(
            essay, essay_template['6.1'], 90, 'argument_paragraph2', argument_paragraph2, default_value
        )

        return essay[:-1] + '\n\n'

    def _7th_paragraph(self, essay, conclusion='water'):

        var = np.random.choice(range(len(essay_template['7.1'])))

        essay = self.continue_phrase_with_pattern(
            essay, essay_template['7.1'][var], 10, 'conclusion', conclusion,
            "«себялюбие» - это порок современного общества"
        )

        return essay

    def __call__(self, task):
        return self.generate(task["text"])


def split_task_and_text(task_text):
    """Split initial task text to the question and actual text fragment. Using texts' sentence counters - (k).
    :return : tuple(Task formulation, Referenced text of the task)
    """

    splitted = re.split(r'\(\d{1,3}\)', task_text)
    if len(splitted) < 5:
        return "", task_text
    formulation = [splitted[0]]
    text = splitted[1:-1]

    last = re.split(r'[!?.…]', splitted[-1])
    text.append(last[0])
    formulation.append('.'.join(last[1:]))

    return ''.join(formulation), ''.join(text).strip()


def clear(text):
    text = re.sub("[\t\r]+", "", text)
    text = re.sub(r"[ ]+([.,!?»: ])", r"\1", text)
    text = re.sub(r"([«])\s+", r"\1", text)
    text = [line.strip() for line in text.split("\n")]
    # text = [line[1:] + line[1].upper() for line in text if len(line)]
    text = "\n".join(text)
    return text


def custom_tok(text, stop_words=None):
    if stop_words is None:
        stop_words = set()
    reg = "([0-9]|\W|[a-zA-Z])"
    toks = text.split()
    ans = []
    for t in toks:
        if not re.match(reg, t):
            if t not in stop_words:
                form = morph.parse(t)[0]
                if form.tag.POS in ["NOUN", "ADJF"]:
                    if form.normal_form not in stop_words:
                        ans.append(form.normal_form)
    return ans


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


def mention_author(author, mode='A.A. Aa', case="nomn"):
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
        initials = ". ".join(map(lambda x: x[0], author[:-1]))
        result = "{}. {}".format(initials.upper(), last_name)
    else:
        result = last_name

    if mode == 'Aa':
        result = result.split('.')[-1].strip()

    return result
