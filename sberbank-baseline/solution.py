import os
import random
import traceback
from collections import defaultdict

from flask import Flask, request, jsonify
import numpy as np

from utils import *
from solvers import *

if "USE_CUSTOM_SOLVERS" in os.environ:
    from src.solvers import solver_1, solver_2, solver_3, solver_4, solver_5, solver_6, solver_7, \
        solver_9, solver_10_11_12, solver_13, solver_14, solver_15, solver_16, solver_17, solver_18, \
        solver_19, solver_20, solver_21, solver_24, solver_25, solver_27

    custom_solvers = {
        1: solver_1,
        2: solver_2,
        3: solver_3,
        4: solver_4,
        5: solver_5,
        6: solver_6,
        7: solver_7,
        9: solver_9,
        10: solver_10_11_12,
        11: solver_10_11_12,
        12: solver_10_11_12,
        13: solver_13,
        14: solver_14,
        15: solver_15,
        16: solver_16,
        17: solver_17,
        18: solver_18,
        19: solver_19,
        20: solver_20,
        21: solver_21,
        24: solver_24,
        25: solver_25,
        27: solver_27
    }
else:
    custom_solvers = {}

if "USE_EMBEDDED_ID" in os.environ:
    use_embedded_id = True
else:
    use_embedded_id = False

solver_param = defaultdict(dict)
solver_param[17]["train_size"] = 0.9
solver_param[18]["train_size"] = 0.85
solver_param[19]["train_size"] = 0.85
solver_param[20]["train_size"] = 0.85


class CuttingEdgeStrongGeneralAI(object):

    def __init__(self, train_path='public_set/train'):
        self.train_path = train_path
        self.classifier = classifier.Solver()
        self.clf_loading()
        solver_classes = [
            solver1,
            solver2,
            solver3,
            solver4,
            solver5,
            solver6,
            solver7,
            solver8,
            solver9,
            solver10,
            solver10,
            solver10,
            solver13,
            solver14,
            solver15,
            solver16,
            solver17,
            solver17,
            solver17,
            solver17,
            solver21,
            solver22,
            solver23,
            solver24,
            solver25,
            solver26,
            solver27
        ]
        self.solvers = self.solver_loading(solver_classes)

    def solver_loading(self, solver_classes):
        solvers = []
        for i, solver_class in enumerate(solver_classes):
            solver_index = i + 1
            if solver_index in custom_solvers:
                solvers.append(None)
                continue
            solver_path = os.path.join("data", "models", "solver{}.pkl".format(solver_index))
            solver = solver_class.Solver(**solver_param[solver_index])
            if os.path.exists(solver_path):
                print("Loading Solver {}".format(solver_index))
                solver.load(solver_path)
            else:
                print("Fitting Solver {}...".format(solver_index))
                try:
                    train_tasks = load_tasks(self.train_path, task_num=solver_index)
                    solver.fit(train_tasks)
                    solver.save(solver_path)
                except Exception as e:
                    print('Exception during fitting: {}'.format(e))
            print("Solver {} is ready!\n".format(solver_index))
            solvers.append(solver)
        return solvers

    def clf_loading(self):
        clf_path = os.path.join("data", "models", "classifier.pkl")
        if os.path.exists(clf_path):
            print("Loading Classifier")
            self.classifier.load(clf_path)
        else:
            try:
                print("Fitting Classifier...")
                self.classifier.fit_from_dir(self.train_path)
                self.classifier.save(clf_path)
            except Exception as e:
                print('Exception during fitting: {}'.format(e))
        print("Classifier is ready!\n")
        return self

    def not_so_strong_task_solver(self, task):
        question = task['question']
        if question['type'] == 'choice':
            # pick a random answer
            choice = random.choice(question['choices'])
            answer = choice['id']
        elif question['type'] == 'multiple_choice':
            # pick a random number of random choices
            min_choices = question.get('min_choices', 1)
            max_choices = question.get('max_choices', len(question['choices']))
            n_choices = random.randint(min_choices, max_choices)
            random.shuffle(question['choices'])
            answer = [
                choice['id']
                for choice in question['choices'][:n_choices]
            ]
        elif question['type'] == 'matching':
            # match choices at random
            random.shuffle(question['choices'])
            answer = {
                left['id']: choice['id']
                for left, choice in zip(question['left'], question['choices'])
            }
        elif question['type'] == 'text':
            if question.get('restriction') == 'word':
                # pick a random word from the text
                words = [word for word in task['text'].split() if len(word) > 1]
                answer = random.choice(words)

            else:
                # random text generated with https://fish-text.ru
                answer = 'Автор затрагивает важную общечеловеческую тему культуры и искусства и раскрывает ряд актуальных проблем. Одной из самых важных среди них является проблема влияния искусства на человека. Автор заставляет нас задуматься о роли творчества на формирование личности.\n\n«Есть книги, не сокращающие жизнь на часы, что потрачены на их чтение, а удлиняющие её». В данном предложении находит свое отражение главный тезис автора. Эти слова обращают наше внимание на то, что художественное произведение может стать для человека чем-то большим, чем просто развлечением: требовательным учителем, домом, школой жизни или даже лучшим другом. Более детально в сути проблемы можно разобраться, прочитав предложение «И если в юности не попалась книга, перепахавшая душу, читатель для литературы потерян». Обе приведённые цитаты, дополняя друг друга, позволяют нам убедиться в том, что правильное восприятие искусства является важной частью развития общества.\n\nНа основании всего вышесказанного можно четко выделить позицию автора: искусство может не только изменить мнение о жизни, но и сформировать взгляды, расширить мировоззрение. Автор убеждает нас в том, что творчество должно воспитывать людей, учить их добру и милосердию.\n\nТрудно не согласиться с обозначенными в тексте проблемами. Я тоже считаю, что настоящее искусство должно быть не бесцельным рисованием перед публикой, а тем, что пробуждает в людях прекрасные мысли и светлые чувства. Данный вопрос часто привлекает к себе внимание и других писателей и публицистов.\n\nПрекрасный пример влияния книги на мировоззрение человека приводит А. Грин в рассказе "Зеленая лампа", повествующем о восхитительной судьбе бродяги Ива, ставшего знаменитым врачом и состоятельным человеком. И все это благодаря книгам, которые Ив читал, читал и читал.\n\nВ романе А. И. Куприна “Гранатовый браслет” влияние искусства проявилось в том, что теплое чувство Желткова к княгине Шеиной возникло именно при прослушивании бетховенской композиции “Аппосионата”. Вера Николаевна испытывала в душе благодарность за такие чувства к ней и было понимание, что настоящая любовь коснулась ее жизни, именно при прослушивании той же сонаты.\n\nТаким образом, вместе с автором текста мы приходим к выводу, что в жизни человека очень высока роль искусства. Поэтому любое творчество мы должны воспринимать глубже и серьезнее.'

        else:
            raise RuntimeError('Unknown question type: {}'.format(question['type']))

        return answer

    def take_exam(self, exam):
        answers = {}
        # pprint.pprint(exam)
        if "tasks" in exam:
            variant = exam["tasks"]
            if isinstance(variant, dict):
                if "tasks" in variant.keys():
                    variant = variant["tasks"]
        else:
            variant = exam
        if use_embedded_id:
            print("Use embedded ids...")
        else:
            print("Use predicted ids...")
        task_number = self.classifier.predict(variant, use_embedded_id)
        print("Classifier results: ", task_number)
        for i, task in enumerate(variant):
            task_id = task['id']
            task_index, task_type = i + 1, task["question"]["type"]
            try:
                if task_number[i] in custom_solvers:
                    prediction = custom_solvers[task_number[i]](task)
                    print("From custom solver")
                else:
                    prediction = self.solvers[task_number[i] - 1].predict_from_model(task)
                    print("From baseline solver")
            except Exception as e:
                print(traceback.format_exc())
                prediction = self.not_so_strong_task_solver(task)
                print("From random solver")
            print("Prediction: ", prediction)
            print()
            if isinstance(prediction, np.ndarray):
                prediction = list(prediction)
            answers[task_id] = prediction
        return answers


app = Flask(__name__)

ai = CuttingEdgeStrongGeneralAI()


@app.route('/ready')
def http_ready():
    return 'OK'


@app.route('/take_exam', methods=['POST'])
def http_take_exam():
    request_data = request.get_json()
    answers = ai.take_exam(request_data)
    return jsonify({
        'answers': answers
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
