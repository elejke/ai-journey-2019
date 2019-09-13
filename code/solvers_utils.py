import re


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
    word_1 = remove_additional(word_1)
    word_2 = remove_additional(word_2)
    russian_letters = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
    letters_1 = []
    letters_2 = []
    for letter_ in russian_letters:
        if word_1.replace("..", letter_) in big_words_set:
            letters_1.append(letter_)
    for letter_ in letters_1:
        if word_2.replace("..", letter_) in big_words_set:
            letters_2.append(letter_)
    return list(set(letters_2).intersection(letters_1))
