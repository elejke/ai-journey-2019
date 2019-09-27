import random
from solvers import solver_11, solver_12, solver_4, solver_10, solver_15, solver_25, solver_5, \
    solver_24, solver_16, solver_1, solver_6, solver_8, solver_9
from solver11 import solver_11
from solver12 import solver_12
from solver26 import solver_26
from flask import Flask, request, jsonify


def take_exam(tasks):
    answers = {}

    for task in tasks:
        question = task['question']

        # choose solver
        if task['id'] in ["1"]:
            answer = solver_1(task)
        elif task['id'] in ["4"]:
            answer = solver_4(task)
        elif task['id'] in ["5"]:
            answer = solver_5(task)
        elif task['id'] in ["6"]:
            answer = solver_6(task)
        elif task['id'] in ["8"]:
            answer = solver_8(task)
        # elif task['id'] in ["9"]:
        #     answer = solver_9(task)
        elif task['id'] in ["10"]:
            answer = solver_10(task)
        elif task['id'] in ["11"]:
            answer = solver_11(task)
        elif task['id'] in ["12"]:
            answer = solver_12(task)
        elif task['id'] in ["15"]:
            answer = solver_15(task)
        elif task['id'] in ["16"]:
            answer = solver_16(task)
        elif task['id'] in ["24"]:
            answer = solver_24(task)
        elif task['id'] in ["25"]:
            try:
                answer = solver_25(task)
            except:
                answer = ["8"]
        elif task['id'] in ["26"]:
            answer = solver_26(task)

        elif question['type'] == 'choice':
            # pick a random answer
            choice = random.choice(question['choices'])
            answer = [str(choice['id'])]

        elif question['type'] == 'multiple_choice':
            # pick a random number of random choices
            min_choices = question.get('min_choices', 1)
            max_choices = question.get('max_choices', len(question['choices']))
            n_choices = random.randint(min_choices, max_choices)
            random.shuffle(question['choices'])
            answer = sorted([
                str(choice['id'])
                for choice in question['choices'][:n_choices]
            ], key=lambda x: int(x))

        elif question['type'] == 'matching':
            # match choices at random
            random.shuffle(question['choices'])
            answer = {
                str(left['id']): str(choice['id'])
                for left, choice in zip(question['left'], question['choices'])
            }

        elif question['type'] == 'text':
            if question.get('restriction') == 'word':
                # pick a random word from the text
                words = [word for word in task['text'].split() if len(word) > 1]
                answer = random.choice(words)

            else:
                # random text generated with https://fish-text.ru
                answer = (
                    'Для современного мира реализация намеченных плановых заданий позволяет '
                    'выполнить важные задания по разработке новых принципов формирования '
                    'материально-технической и кадровой базы. Господа, реализация намеченных '
                    'плановых заданий играет определяющее значение для модели развития. '
                    'Сложно сказать, почему сделанные на базе интернет-аналитики выводы призывают '
                    'нас к новым свершениям, которые, в свою очередь, должны быть в равной степени '
                    'предоставлены сами себе. Ясность нашей позиции очевидна: базовый вектор '
                    'развития однозначно фиксирует необходимость существующих финансовых и '
                    'административных условий.'
                )

        else:
            raise RuntimeError('Unknown question type: {}'.format(question['type']))

        answers[task['id']] = answer

    return answers


app = Flask(__name__)


@app.route('/ready')
def http_ready():
    return 'OK'


@app.route('/take_exam', methods=['POST'])
def http_take_exam():
    request_data = request.get_json()
    tasks = request_data['tasks']
    answers = take_exam(tasks)
    return jsonify({
        'answers': answers
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
