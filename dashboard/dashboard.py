import os
import glob
import dash
import pathlib
import collections
import pandas as pd
import dash_html_components as html

SCORES_DIR = "../scores_dir/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df_scores = pd.read_csv(os.path.join(SCORES_DIR, "scores.csv"))[["id", "score"]]

df_list = collections.deque(maxlen=5)
full_model_list = []

def generate_table(max_rows=29):

    df_path_list = glob.glob(os.path.join(SCORES_DIR, "*.csv"))
    df_path_list.sort(key=os.path.getmtime)

    for df_path in df_path_list:
        if "scores.csv" not in df_path:
            model_id = pathlib.Path(df_path).name.strip(".csv")
            if model_id not in full_model_list:
                df_temp = pd.read_csv(df_path)[["metric"]]
                df_temp.columns = ["model_" + model_id]
                full_model_list.append(model_id)
                df_list.append(df_temp)

    df = pd.concat([df_scores[["id", "score"]]] + list(df_list), axis=1).round(2)

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), max_rows))]
    )

def serve_layout():
    return html.Div(children=[
        html.H4(children='Metric by task id'),
        generate_table()
    ])

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = serve_layout

if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')