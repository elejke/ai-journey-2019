FROM continuumio/anaconda3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y build-essential unzip cmake && \
    pip install --upgrade pip

RUN pip install tensorflow

RUN apt-get install -y libglib2.0-0 --fix-missing && \
    pip install gensim && \
    pip install xgboost && \
    pip install lightgbm && \
    pip install keras && \
    pip install joblib && \
    pip install tqdm

ENV LD_LIBRARY_PATH=/opt/conda/lib

# custom libs for language
RUN pip install regex && \
    pip install pymystem3 && \
    pip install dawg-python pymorphy2 pymorphy2-dicts-ru

RUN python -c "import pymystem3.mystem ; pymystem3.mystem.autoinstall()" && \
    pip install jellyfish

# fasttext
RUN pip install fasttext

# add custom models to the docker
COPY ./models/fasttext/cc.ru.300.bin /misc/models/fasttext/cc.ru.300.bin

# load NLTK corpus
RUN python -c "import nltk; nltk.download('stopwords')"

# StanfordNLP
RUN pip install stanfordnlp
RUN python -c "import stanfordnlp; stanfordnlp.download('ru', force=True); nlp = stanfordnlp.Pipeline(lang='ru')"

# Additional from baseline
RUN pip install ufal.udpipe \
                pytorch_pretrained_bert \
                python-Levenshtein \
                sklearn_crfsuite \
                fastai \
                fuzzywuzzy \
                summa
RUN pip install catboost
RUN pip install numpy --upgrade
RUN python -c "import nltk; nltk.download('punkt')"