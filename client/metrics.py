import os

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
        return float(row["score"])
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


def eval_matching(row):
    if not pd.isnull(row["gt_unique"]):
        correct_answers = read_str(row["gt_unique"])
    elif not pd.isnull(row["gt_variants"]):
        correct_answers = read_str(row["gt_variants"])[0]
    else:
        return np.NaN

    my_answers = read_str(row["prediction"])
    if not isinstance(my_answers, dict):
        return np.NaN

    correct_matches = 0
    total_matches = len(correct_answers)
    for k in correct_answers:
        if str(correct_answers[k]) == str(my_answers.get(k, "###")):
            correct_matches += 1

    return correct_matches / total_matches * row["score"]


def eval_text_word(row):
    if not pd.isnull(row["gt_unique"]):
        correct_answers = [row["gt_unique"]]
    elif not pd.isnull(row["gt_variants"]):
        correct_answers = read_str(row["gt_variants"])
    else:
        return np.NaN

    if str(row["prediction"]) in correct_answers:
        return float(row["score"])
    else:
        return 0.


def eval_row(row):
    if row["type"] == "choice":
        return eval_choice(row)
    elif row["type"] == "multiple_choice":
        return eval_multiple_choice(row)
    elif row["type"] == "matching":
        return eval_matching(row)
    elif row["type"] == "text":
        if int(row["id"]) != 27:
            return eval_text_word(row)
        else:
            return np.NaN
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


def get_aggregative_metrics(folder):

    conversion_from_primary_to_secondary = {
        0: 0,
        1: 3,
        2: 5,
        3: 8,
        4: 10,
        5: 12,
        6: 15,
        7: 17,
        8: 20,
        9: 22,
        10: 24,
        11: 26,
        12: 28,
        13: 30,
        14: 32,
        15: 34,
        16: 36,
        17: 38,
        18: 39,
        19: 40,
        20: 41,
        21: 43,
        22: 44,
        23: 45,
        24: 46,
        25: 48,
        26: 49,
        27: 50,
        28: 51,
        29: 53,
        30: 54,
        31: 55,
        32: 56,
        33: 57,
        34: 59,
        35: 60,
        36: 61,
        37: 62,
        38: 64,
        39: 65,
        40: 66,
        41: 67,
        42: 69,
        43: 70,
        44: 71,
        45: 72,
        46: 73,
        47: 76,
        48: 78,
        49: 80,
        50: 82,
        51: 85,
        52: 87,
        53: 89,
        54: 91,
        55: 94,
        56: 96,
        57: 98,
        58: 100
    }

    def interp_score(x):
        unzipped_primary, unzipped_secondary = zip(*conversion_from_primary_to_secondary.items())
        return np.interp(x, unzipped_primary, unzipped_secondary)

    if isinstance(folder, str):
        data = pd.read_csv(os.path.join(folder, "parsed_answers.csv"))
    else:
        TypeError("'inp' should be either a path to the folder!")

    metrics_by_id = data.groupby("id")[["metric", "score"]].mean()
    metrics_by_id.loc["TOTAL"] = [
        metrics_by_id["metric"].sum(),
        metrics_by_id["score"].sum()
    ]
    metrics_by_id.loc["TOTAL_CONVERTED"] = [
        interp_score(metrics_by_id.fillna(0.).loc["TOTAL", "metric"]),
        100
    ]

    metrics_by_exam = data.groupby("path")[["metric"]].sum()
    metrics_by_exam["converted_metric"] = metrics_by_exam["metric"].apply(interp_score)
    metrics_by_exam.loc["TOTAL"] = [
        metrics_by_exam["metric"].mean(),
        metrics_by_exam["converted_metric"].mean()
    ]

    metrics_by_exam.to_csv(os.path.join(folder, "metrics_by_exam.csv"))
    metrics_by_id.to_csv(os.path.join(folder, "metrics_by_id.csv"))

    return folder
