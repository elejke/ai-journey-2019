import re
import copy
import random


def remove_additional(word):
    """ Function take string of words in w/wo brackets and output only "clean" words

    :param word: string of words like: умный (мальчик)
    :return: string with removed words in brackets
    """
    additional_words = re.findall("\([\w\ ]+\)", word)

    for additional_word in additional_words:
        word = word.replace(additional_word, "")

    return word.strip()


def check_pair(word_1, word_2, big_words_set):
    """ Check if pair have the same missed letter or not

    :param word_1: first word in pair
    :param word_2: second word in pair
    :param big_words_set: set of words to compare with
    :return: list of letters which can be inserted to both words
    """
    # TODO: EMBEDDINGS. NEEDED TO USE NOT ONLY WORD BUT WITH CONTEXT
    word_1 = remove_additional(word_1).replace("о́", "о")
    word_2 = remove_additional(word_2).replace("о́", "о")
    russian_letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    letters_1 = []
    letters_2 = []
    for letter_ in russian_letters:
        if word_1.replace("@", letter_) in big_words_set:
            letters_1.append(letter_)
    for letter_ in letters_1:
        if word_2.replace("@", letter_) in big_words_set:
            letters_2.append(letter_)

    return list(set(letters_2).intersection(letters_1))


def repair_words(words, big_words_set, return_repaired=True):
    """ Check if list of words have the same missed letter or not. Repair words if needed

    :param return_repaired: boolean flag, if True return repaired words
    :param words: list of words to test
    :param big_words_set: set of words to compare with
    :return: list of letters which can be inserted to both words
    """

    words = [remove_additional(word_) for word_ in words]

    letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")

    selected_letters = []

    for word_ in words:
        for letter_ in letters:
            if word_.replace("@", letter_) in big_words_set:
                selected_letters.append(letter_)

        letters = selected_letters
        selected_letters = []

    if return_repaired:
        if len(letters):
            letter_ = random.choice(letters)
        else:
            letter_ = random.choice(list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"))

        return "$".join(words).replace("@", letter_).split("$")

    else:
        return letters


def split_task_and_text(task_text):
    """Split initial task text to the question and actual text fragment. Using texts' sentence counters - (k).
    For example, passing the next text:

        Прочитайте текст и выполните следующее задание.(1) В некоторых моделях современных автомобилей раскрытие подушек
        безопасности при непристёгнутых ремнях автоматически блокируется.(2) Дело в том, что срабатывание подушек,
        происходящее с огромной скоростью, сродни сильному удару.

    Split it to:

        Прочитайте текст и выполните следующее задание. Укажите два предложения, в которых верно передана главная
        информация, содержащаяся в тексте.Запишите номера этих предложений.

    and

        (1) В некоторых моделях современных автомобилей раскрытие подушек
        безопасности при непристёгнутых ремнях автоматически блокируется.(2) Дело в том, что срабатывание подушек,
        происходящее с огромной скоростью, сродни сильному удару.

    :param task_text: Input text
    :return : tuple(Task formulation, Referenced text of the task)
    """

    formulation = []
    text = []
    for sentence in task_text.split('.'):
        if re.match(r'^\s*\(\d+\)', sentence):
            text.append(sentence)
        else:
            formulation.append(sentence)

    return '.'.join(formulation), '.'.join(text)


def standardize_task(task):
    if task["id"] in ["10", "11", "12"]:
        task = copy.deepcopy(task)
        if len(task["text"].split("\xa0")) > 1:
            task["text"] = task["text"].replace("о́", "о").split("\xa0")[0]
        if "choices" not in task:
            if "question" in task and "choices" in task["question"]:
                task["choices"] = task["question"]["choices"]
            else:
                parts = task["text"].replace("о́", "о").split("\n")
                task["text"] = parts[0].strip()
                task["choices"] = []
                for i in range(1, len(parts)):
                    task["choices"].append({"id": str(i), "text": parts[i]})

        for i in range(len(task["choices"])):
            choices_text = task["choices"][i]["text"].replace("...", "@").replace("..", "@").replace("о́", "о")
            sep = re.findall("[\;\.\,]", choices_text)
            sep = "," if not len(sep) else sep[0]
            choices_text = re.split("[1-9]\)", choices_text)
            choices_text = choices_text[0] if len(choices_text) == 1 else choices_text[1]
            task["choices"][i]["text"] = choices_text.replace("@", "..").strip()
            parts = [x.strip().replace("@", "..") for x in choices_text.split(sep)]
            task["choices"][i]["parts"] = parts
        if 'choices' in task['question'].keys():
            task['question'].pop('choices')
    return task