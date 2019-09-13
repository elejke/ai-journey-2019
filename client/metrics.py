import numpy as np
import pandas as pd

from utils import read_str


def eval_choice(row):
    if not pd.isnull(row["gt_unique"]):
        correct_answers = read_str(row["gt_unique"])
    elif not pd.isnull(row["gt_variants"]):
        correct_answers = read_str(row["gt_variants"])
    else:
        return np.NaN

    if str(row["prediction"]) in correct_answers:
        return row["score"]
    else:
        return 0.


def eval_multiple_choice(row):
    if not pd.isnull(row["gt_unique"]):
        correct_answers = set(read_str(row["gt_unique"]))
    elif not pd.isnull(row["gt_variants"]):
        correct_answers = set(read_str(row["gt_variants"])[0])
    else:
        return np.NaN

    my_answers = read_str(row["prediction"])
    if not isinstance(my_answers, list):
        return np.NaN
    my_answers = set(my_answers)

    return len(correct_answers & my_answers) / len(correct_answers | my_answers) * row["score"]


def eval_row(row):
    if row["type"] == "choice":
        return eval_choice(row)
    elif row["type"] == "multiple_choice":
        return eval_multiple_choice(row)
    else:
        return np.NaN


def get_all_metrics(inp):

    if isinstance(inp, str):
        data = pd.read_csv(inp)
    elif isinstance(inp, pd.DataFrame):
        data = inp
    else:
        TypeError("'inp' should be either a pandas.DataFrame or a path to the csv!")

    data["metric"] = data.apply(eval_row, axis=1)

    if isinstance(inp, str):
        data.to_csv(inp, index=False)
        return inp
    elif isinstance(inp, pd.DataFrame):
        return data
