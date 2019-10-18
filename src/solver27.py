import joblib
import random
from summa import summarizer
import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()
from fastai.text import *
from fastai.callbacks import ReduceLROnPlateauCallback
import numpy as np
import warnings
warnings.filterwarnings('ignore')


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


def rus_tok(text, m=pymorphy2.MorphAnalyzer()):
    reg = '([0-9]|\W|[a-zA-Z])'
    toks = text.split()
    return [m.parse(t)[0].normal_form for t in toks if not re.match(reg, t)]


def get_author(text):
    return re.search('\n\s*\*(\s*\w*){2,3}\s*\(', text).group().strip()[1:-1].split()


def mention_author(author, form='A. A. Aa', case='nomn'):
    """Упоминает автора в нужном формате и склонении. Юзать правда лучше только в именительном, т.к. некоторые
    фамилии не склоняются. Например, Черных

    nomn	именительный	Кто? Что?	хомяк ест
    gent	родительный	Кого? Чего?	у нас нет хомяка
    datv	дательный	Кому? Чему?	сказать хомяку спасибо
    accs	винительный	Кого? Что?	хомяк читает книгу
    ablt	творительный	Кем? Чем?	зерно съедено хомяком
    loct	предложный	О ком? О чём? и т.п.	хомяка несут в корзинке
    voct	звательный	Его формы используются при обращении к человеку.	Саш, пойдем в кино.
    """
    if form == 'A. A. Aa':
        last_name = morph.parse(author[-1])[0].inflect({case})[0].capitalize()
        initials = '. '.join(map(lambda x: x[0].upper(), author[:-1]))
        result = '{}. {}'.format(initials, last_name)
    return result


essay_template = {
    # 1. Формулировка проблемы текста
    '1.1': 'Текст посвящен одной из наиболее актуальных проблем современности - {problem_formulation}',
    '1.2': '{author} заставляет нас задуматься о {problem_explanation}',  # (о чём? над какими вопросами?)
    # 2. Комментарий проблемы (здесь два примера по проблеме из прочитанного текста, которые помогают понять суть
    # проблемы)
    '2.1': 'Главным тезисом в рассуждениях автора можно считать следующий: "{citation}"',
    '2.2': 'Этим он подчеркивает, что {water}',
    # 3. Авторская позиция по обозначенной проблеме.
    '3.1': 'Развивая свою мысль, автор убеждает нас в том, что {author_position}',
    '3.2': 'Также {author} считает, что {water}',
    # 4. Собственное мнение по обозначенной проблеме (согласие).
    '4.1': 'Хочется выразить своё мнение по обозначенной проблеме. И я считаю, что {own_position}',
    '4.2': '{water}',
    '4.3': 'Аргументирую свою точку зрения',
    # 5. Аргумент 1 (из художественной, публицистической или научной литературы).
    '5.1': 'Обратимся к произведению {argument1_author} «{argument1_source_name}».',
    '5.2': '{water}',
    # 6. Аргумент 2 (из жизни).
    '6.1': 'Не так давно по телевидению передавали репортаж о {argument2_description}',
    '6.2': '{water}',
    # 7. Заключение.
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
    model_name : str
        Model name for load pretrained ulmfit model and store this.
    dict_name : str
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
            self, model_name=None, dict_name=None, tf_vectorizer_path=None, lda_path=None,
            topics_path=None, is_load=True, seed=42
    ):

        self.model_name = model_name
        self.dict_name = dict_name
        self.data = None
        self.learn = None
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

    def _init_args(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

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

    def fit(self, texts, num_epochs=5, is_fit_topics=False):

        texts = pd.DataFrame(list(texts))

        self.data = TextList.from_df(
            texts, processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")),
                              NumericalizeProcessor(vocab=Vocab.load("models/{}.pkl".format(self.dict_name)))]
        ).random_split_by_pct(.1).label_for_lm().databunch(bs=16)

        conf = awd_lstm_lm_config.copy()
        conf['n_hid'] = 1150
        self.learn.unfreeze()
        self.learn.lr_find(start_lr=slice(10e-7, 10e-5), end_lr=slice(0.4, 10))
        _ = self.learn.recorder.plot(skip_end=10, suggestion=True)
        best_lm_lr = self.learn.recorder.min_grad_lr
        print(best_lm_lr)

        #         self.learn.fit_one_cycle(
        #             num_epochs, best_lm_lr, callbacks=[ReduceLROnPlateauCallback(self.learn, factor=0.8)])
        self.learn = language_model_learner(self.data, AWD_LSTM, pretrained=True, config=conf, drop_mult=0.7,
                                            pretrained_fnames=[self.model_name, self.dict_name], silent=False)

        self.learn.fit(num_epochs, best_lm_lr, callbacks=[ReduceLROnPlateauCallback(self.learn, factor=0.8)])

        # TODO: fit lda
        if is_fit_topics:
            pass
        return self

    def save(self):
        self.learn.save(self.model_name)
        self.learn.save_encoder(self.model_name + "_enc")

    def load(self):

        self.tf_vectorizer = joblib.load(self.tf_vectorizer_path)
        self.lda = joblib.load(self.lda_path)
        self.topics = pd.read_csv(self.topics_path, sep="\t")
        self.topic_dic = {int(i): self.topics.iloc[i]['Topic'] for i in range(len(self.topics))}

        self.data = TextList.from_df(
            pd.DataFrame(["tmp", "tmp"]),
            processor=[TokenizeProcessor(tokenizer=Tokenizer(lang="xx")),
                       NumericalizeProcessor(vocab=Vocab.load("models/{}.pkl".format(self.dict_name)))]
        ).random_split_by_pct(.1).label_for_lm().databunch(bs=16)

        conf = awd_lstm_lm_config.copy()
        conf['n_hid'] = 1150
        self.learn = language_model_learner(self.data, AWD_LSTM, pretrained=True, config=conf, drop_mult=0.7,
                                            pretrained_fnames=[self.model_name, self.dict_name], silent=False)

        return self

    def generate(self, task, temperature=0.7):

        self.temperature = temperature
        author = get_author(task)
        task, text = split_task_and_text(task)

        brief_text = summarizer.summarize(text, language="russian", ratio=0.25, split=False)
        citation = np.random.choice(summarizer.summarize(text, language="russian", ratio=0.1, split=True))

        essay = self._1st_paragraph(brief_text, mention_author(author))
        essay = self._2nd_paragraph(essay, citation)
        essay = self._3rd_paragraph(essay, mention_author(author))
        essay = self._4th_paragraph(essay)
        essay = self._5th_paragraph(essay, 'Олег Петух', 'Внатуре Петух')
        essay = self._6th_paragraph(essay)
        essay = self._7th_paragraph(essay)

        return essay[len(brief_text):]

    def continue_phrase(self, text, n_words=10):
        text = clear(text)
        text = clear(self.learn.predict(text, n_words=n_words, no_unk=True, temperature=self.temperature))
        text = text.replace("xxbos", " ")  # Remove model special symbols
        text = text[:-40] + re.split(r"[.!?]", text[-40:])[0] + '. '  # Cut predicted sentence up to dot
        return clear(text)

    def _1st_paragraph(self, text, author):

        next_sent = essay_template['1.1'].format(problem_formulation='')
        essay = self.continue_phrase(text + '\n\n' + next_sent, 5)

        next_sent = essay_template['1.2'].format(author=author, problem_explanation='')
        essay = self.continue_phrase(essay + next_sent, 20)

        return essay + '\n'

    def _2nd_paragraph(self, essay, citation):

        next_sent = essay_template['2.1'].format(citation=citation)
        essay += next_sent

        next_sent = essay_template['2.2'].format(water='')
        essay = self.continue_phrase(essay + next_sent, 30)

        return essay + '\n'

    def _3rd_paragraph(self, essay, author):

        next_sent = essay_template['3.1'].format(author_position='')
        essay = self.continue_phrase(essay + next_sent, 15)

        next_sent = essay_template['3.2'].format(author=author, water='')
        essay = self.continue_phrase(essay + next_sent, 30)

        return essay + '\n'

    def _4th_paragraph(self, essay):

        next_sent = essay_template['4.1'].format(own_position='')
        essay = self.continue_phrase(essay + next_sent, 10)

        next_sent = essay_template['4.2'].format(water='')
        essay = self.continue_phrase(essay + next_sent, 30)

        next_sent = essay_template['4.3'].format(water='')
        essay += next_sent

        return essay + '\n'

    def _5th_paragraph(self, essay, argument1_author, argument1_source_name):

        next_sent = essay_template['5.1'].format(argument1_author=argument1_author,
                                                 argument1_source_name=argument1_source_name)
        essay += next_sent

        next_sent = essay_template['5.2'].format(water='')
        essay = self.continue_phrase(essay + next_sent, 30)

        return essay + '\n'

    def _6th_paragraph(self, essay):

        next_sent = essay_template['6.1'].format(argument2_description='')
        essay = self.continue_phrase(essay + next_sent, 15)

        next_sent = essay_template['6.2'].format(water='')
        essay = self.continue_phrase(essay + next_sent, 30)

        return essay + '\n'

    def _7th_paragraph(self, essay):

        next_sent = essay_template['7.1'].format(conclusion='')
        essay = self.continue_phrase(essay + next_sent, 40)

        return essay
