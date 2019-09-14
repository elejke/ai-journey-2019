import json
import pandas as pd


def answers_enricher(path):
    parsed_answers = pd.read_csv(path)

    parsed_answers["type"] = None
    parsed_answers["source"] = None
    parsed_answers["score"] = None
    parsed_answers["gt_unique"] = None
    parsed_answers["gt_variants"] = None

    # try to append correct answers to the parsed answers
    for file in parsed_answers["path"].drop_duplicates().values:
        with open(file, "r") as f:
            file_data = json.load(f)
        for question in file_data["tasks"]:
            col_id = (parsed_answers["path"] == file) & (parsed_answers["id"] == int(question["id"]))
            parsed_answers.loc[col_id, "type"] = question["question"]["type"]
            parsed_answers.loc[col_id, "source"] = question["meta"]["source"]
            parsed_answers.loc[col_id, "score"] = float(question["score"])
            if "correct" in question["solution"]:
                parsed_answers.loc[col_id, "gt_unique"] = str(question["solution"]["correct"])
            elif "correct_variants" in question["solution"]:
                parsed_answers.loc[col_id, "gt_variants"] = str(question["solution"]["correct_variants"])
            else:
                pass
    parsed_answers = parsed_answers.sort_values(by=["path", "id"]).reset_index(drop=True)
    parsed_answers.to_csv(path, index=False)

    return parsed_answers


def read_str(x):
    if isinstance(x, str):
        return eval(x)
    else:
        return x
