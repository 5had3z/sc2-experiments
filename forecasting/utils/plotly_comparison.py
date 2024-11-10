from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from konductor.metadata.database.sqlite import DEFAULT_FILENAME, SQLiteDB

from .xgboost_492 import d as xgboost_492
from .xgboost_tourn import d as xgboost_tourn


class TimePoint:
    __slots__ = "value"

    def __init__(self, value: float):
        self.value = value

    def as_db_key(self):
        return "t_" + str(self.value).replace(".", "_")

    def as_pq_key(self):
        return "binary_acc_" + str(self.value)

    def as_float(self):
        return float(self.value)


time_points = [TimePoint(t) for t in np.arange(2, 20, 0.5)]


def get_labels():
    return {str(t.as_float()): "FLOAT" for t in time_points}


layout = html.Div(
    children=[
        dbc.Row(
            [
                dbc.Col(html.H3("Eval Accuracy Over Time")),
                dbc.Col(
                    dbc.Button(
                        "Download CSV", id="btn-render", style={"float": "right"}
                    )
                ),
                dcc.Download(id="download-csv-render"),
            ]
        ),
        dbc.Row(dcc.Graph(id="ts2-line-graph", selectedData={})),
        dbc.Row(
            [
                dbc.Col(html.H3("Eval Accuracy At A Time Step")),
                dbc.Col(
                    dcc.Dropdown(id="ts2-metric", options=list(get_labels().keys()))
                ),
            ]
        ),
        dbc.Row(dcc.Graph(id="ts2-graph", selectedData={})),
    ]
)


def get_performance_data(root: Path, metric: str):
    db_handle = SQLiteDB(root / DEFAULT_FILENAME)
    time_step = TimePoint(float(metric)).as_db_key()
    output = (
        db_handle.cursor()
        .execute(f"SELECT hash, {time_step} FROM binary_accuracy")
        .fetchall()
    )

    return output


def get_all_performance_data(root: Path, keys: list[str]):
    db_handle = SQLiteDB(root / DEFAULT_FILENAME)
    output = (
        db_handle.cursor()
        .execute(f"SELECT hash, {','.join(keys)} FROM binary_accuracy")
        .fetchall()
    )

    return output


def hash_to_brief(root: Path):
    db_handle = SQLiteDB(root / DEFAULT_FILENAME)
    output = db_handle.cursor().execute("SELECT hash, brief FROM metadata").fetchall()
    return {x[0]: x[1] for x in output}


@callback(
    Output("ts2-graph", "figure"),
    Input("ts2-metric", "value"),
    Input("root-dir", "data"),
    prevent_initial_call=True,
)
def update_col_graph(metric: str, root: str):
    if not all((metric, root)):
        raise PreventUpdate

    df = get_performance_data(Path(root), metric)

    df.append(("xgboost_tournament", xgboost_tourn.get(float(metric), 0)))
    df.append(("xgboost_492", xgboost_492.get(float(metric), 0)))

    fig = go.Figure()
    column_names, values = zip(*df)

    hb_map = hash_to_brief(Path(root))

    column_names = [hb_map[cn] if cn in hb_map else cn for cn in column_names]

    # Create a column chart using Plotly
    fig = go.Figure(data=[go.Bar(x=column_names, y=values)])

    # Update layout for better readability (optional)
    fig.update_layout(
        title_text=f"Results over time={metric}",
        xaxis_title="Brief",
        yaxis_title="Accuracy (%)",
    )

    return fig


@callback(
    Output("ts2-line-graph", "figure"),
    Input("root-dir", "data"),
    prevent_initial_call=False,
)
def update_line_graph(root: str):
    if not root:
        raise PreventUpdate()

    keys = [tp.as_db_key() for tp in time_points]
    df = get_all_performance_data(Path(root), keys)

    legend_names = [d[0] for d in df]
    data = [d[1:] for d in df]

    xgboost_data = [xgboost_tourn.get(tp.as_float(), 0) for tp in time_points]
    data.append(xgboost_data)
    legend_names.append("xgboost_tourn")

    xgboost_492_data = [xgboost_492.get(tp.as_float(), 0) for tp in time_points]
    data.append(xgboost_492_data)
    legend_names.append("xgboost_492")

    hb_map = hash_to_brief(Path(root))

    column_names = [hb_map[cn] if cn in hb_map else cn for cn in legend_names]

    # Create a line chart using Plotly
    fig = go.Figure()

    x = [tp.as_float() for tp in time_points]

    for legend, values in zip(column_names, data):
        fig.add_trace(go.Scatter(x=x, y=values, mode="lines", name=legend))

    # Update layout for better readability (optional)
    fig.update_layout(
        title_text="Results Over Game Time",
        xaxis_title="Time (minutes)",
        yaxis_title="Accuracy (%)",
    )

    return fig


@callback(
    Output("download-csv-render", "data"),
    Input("btn-render", "n_clicks"),
    State("ts2-line-graph", "figure"),
    prevent_initial_call=True,
)
def render_csv(n_clicks, figure):
    """"""
    if n_clicks is None:
        raise PreventUpdate

    if figure is None:
        return ""

    series: list[pd.Series] = []
    for trace in figure["data"]:
        if not trace["visible"] is True:
            continue
        series.append(pd.Series(trace["y"], index=trace["x"], name=trace["name"]))

    df = pd.concat(series, axis=1)

    # Convert dataframe to CSV format
    csv_data = df.to_csv(index=True)

    return {
        "content": csv_data,
        "filename": "sc2-outcome-results.csv",
        "mime_type": "text/csv",
    }
