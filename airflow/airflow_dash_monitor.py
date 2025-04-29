import dash
from dash import dcc, html
import pandas as pd
import psycopg2
import plotly.express as px

# Connect to Airflow's Postgres metadata DB
conn = psycopg2.connect(
    dbname="airflow", user="airflow", password="airflow", host="postgres", port="5432"
)


def get_latest_task_statuses():
    query = """
    SELECT dag_id, task_id, state, execution_date
    FROM task_instance
    ORDER BY execution_date DESC
    LIMIT 50;
    """
    return pd.read_sql(query, conn)

# Initialize Dash
app = dash.Dash(__name__)
app.title = "Airflow Pipeline Monitor"

app.layout = html.Div([
    html.H1("📊 Airflow DAG Monitoring Dashboard"),
    dcc.Interval(id="interval", interval=60000, n_intervals=0),
    html.Div(id="table-container")
])

@app.callback(
    dash.dependencies.Output("table-container", "children"),
    [dash.dependencies.Input("interval", "n_intervals")]
)
def update_table(n):
    df = get_latest_task_statuses()
    fig = px.timeline(
        df, x_start="execution_date", x_end="execution_date", y="task_id", color="state",
        title="Latest Task States", labels={"execution_date": "Timestamp"}
    )
    fig.update_yaxes(autorange="reversed")
    return dcc.Graph(figure=fig)

if __name__ == "__main__":
    app.run(debug=True, port=8050)
