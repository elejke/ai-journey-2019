import re

def remove_additional(word):
    additional_words = re.findall("\([\w\ ]+\)", word)

    for additional_word in additional_words:
        word = word.replace(additional_word, "")
        word = word.strip()

    return word


def check_pair(word_1, word_2, big_words_set):
    # TODO: EMBEDDINGS NEEDED TO USE NOT ONLY WORD BUT WITH CONTEXT
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

