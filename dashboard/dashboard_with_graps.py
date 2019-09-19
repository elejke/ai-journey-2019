import os
import glob
import regex
import pathlib
import pandas as pd

from multiprocessing import Manager

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

SCORES_DIR = "../scores_dir/"

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

df_scores = pd.read_csv(os.path.join(SCORES_DIR, "scores.csv"))[["id", "score"]]

df_list = []
full_model_list = []

manager = Manager()
lock = manager.Lock()


def update_table(lock):

    lock.acquire()

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

    lock.release()


def generate_table(max_rows=29):

    update_table(lock)

    df = pd.concat([df_scores[["id", "score"]]] + list(df_list)[-5:], axis=1).round(2)

    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [html.Tr([
            html.Td(dcc.Link(df.iloc[i][col], href=f"/graph/{df.iloc[i]['id']}")) if col == "id" else html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), max_rows))]
    )


def serve_dashboard_layout():
    return html.Div([
        html.H4(children='Metric by task id'),
        generate_table()
    ])


def serve_graph_layout(default="1"):
    update_table(lock)
    xaxisrange = list(range(1, len(df_list) + 1))
    df = pd.concat([df_scores[["id"]]] + df_list, axis=1)
    df.set_index("id", drop=True, inplace=True)
    return html.Div([
        html.Div([
            dcc.Graph(
                id="quality-graph",
                figure={
                    "data": [
                        {
                            "x": xaxisrange,
                            "y": df.loc[default],
                            "name": "Current score"
                        },
                        {
                            "x": xaxisrange,
                            "y": [df_scores[df_scores["id"] == default].iloc[0]["score"]] * len(xaxisrange),
                            "mode": "lines",
                            "name": "Max score"
                        }
                    ],
                    'layout': dict(
                        autosize=True,
                        hovermode="closest",
                        title="Metrics history for task 1",
                        xaxis={"title": "Model",
                               "ticktext": list(df.columns),
                               "tickvals": list(range(1, len(df.loc["1"]) + 1)),
                               "tickangle": -90,
                               "automargin": True},
                        yaxis={"title": "Metric"}
                    )
                },
                style={"height": "80vh"}
            )
        ]
        ),
        html.Div([
            dcc.Dropdown(
                id="dist-drop",
                options=[{"label": ind, "value": ind} for ind in df.index],
                value=default)
        ]
        ),
        html.Br(),
        dcc.Link("Go to dashboard", href="/dashboard"),
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(
    Output(component_id='quality-graph', component_property='figure'),
    [Input(component_id='dist-drop', component_property='value')]
)
def update_map(value):
    xaxisrange = list(range(1, len(df_list) + 1))
    df = pd.concat([df_scores[["id"]]] + df_list, axis=1)
    df.set_index("id", drop=True, inplace=True)
    return {
                "data": [
                    {
                        "x": xaxisrange,
                        "y": df.loc[value],
                        "name": "Current score"
                    },
                    {
                        "x": xaxisrange,
                        "y": [df_scores[df_scores["id"] == value].iloc[0]["score"]] * len(xaxisrange),
                        "mode": "lines",
                        "name": "Max score"
                    }
                ],
                'layout': dict(
                    autosize=True,
                    hovermode="closest",
                    title=f"Metrics history for task {value}",
                    xaxis={"title": "Model",
                           "ticktext": list(df.columns),
                           "tickvals": list(range(1, len(df.loc[value]) + 1)),
                           "tickangle": -90,
                           "automargin": True},
                    yaxis={"title": "Metric"}
                )
            }


# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/graph':
        return serve_graph_layout()
    elif regex.search("\/graph\/\d+", str(pathname)):
        return serve_graph_layout(str(pathname).split("/")[-1])
    elif pathname == '/dashboard':
        return serve_dashboard_layout()
    else:
        return serve_dashboard_layout()


if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='0.0.0.0')
