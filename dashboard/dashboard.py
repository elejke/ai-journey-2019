import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

#df = pd.read_csv(
#    'https://gist.githubusercontent.com/chriddyp/'
#    'c78bf172206ce24f77d6363a2d754b59/raw/'
#    'c353e8ef842413cae56ae3920b8fd78468aa4cb2/'
#    'usa-agricultural-exports-2011.csv')

df_1 = pd.read_csv("./scores_dir/metrics_by_id_1.csv")[["metric"]]
df_2 = pd.read_csv("./scores_dir/metrics_by_id_2.csv")
df_1.columns = ["metric_model_1"]
df_2.columns = ["id", "metric_model_2", "score"]

df = pd.concat([df_2[["id", "score", "metric_model_2"]], df_1], axis=1).round(2)


def generate_table(dataframe, max_rows=29):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    html.H4(children='Metric by task id'),
    generate_table(df)
])

if __name__ == '__main__':
    app.run_server(debug=False)
