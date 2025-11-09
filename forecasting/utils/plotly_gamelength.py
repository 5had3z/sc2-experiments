import csv
import io
import sqlite3
from pathlib import Path

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.dependencies import State
from dash.exceptions import PreventUpdate
from konductor.metadata.database.interface import DEFAULT_SQLITE_FILENAME

layout = html.Div(
    children=[
        dbc.Row(
            [
                html.H3("Eval Accuracy Over % Game Length"),
                dcc.Dropdown(id="ts2-eval-folder", options=["tournament", "492"]),
                html.Button("Save", id="save-button"),
                dcc.Download(id="download-csv"),
                dcc.Graph(id="ts2-length-win", selectedData={}),
            ]
        ),
    ]
)


def hash_to_brief(root: Path):
    db_handle = sqlite3.connect(root / DEFAULT_SQLITE_FILENAME)
    output = db_handle.cursor().execute("SELECT hash, brief FROM metadata").fetchall()
    return {x[0]: x[1] for x in output}


@callback(
    Output("ts2-length-win", "figure"),
    Input("ts2-eval-folder", "value"),
    Input("root-dir", "data"),
    prevent_initial_call=False,
)
def update_game_length(eval_folder: str, root_: str):
    if not root_:
        raise PreventUpdate()

    root = Path(root_)

    hb_map = hash_to_brief(root)
    fig = go.Figure()

    for folder in filter(lambda x: x.is_dir(), root.iterdir()):
        csv_path = folder / f"percent_{eval_folder}" / "game_length_results_50"
        if not csv_path.exists():
            continue

        with open(csv_path, encoding="utf-8") as f:
            lines = f.readlines()

        counts = lines[0].split(",")
        totals = lines[1].split(",")

        percentage = [float(c) / float(t) for c, t in zip(counts[1:], totals[1:])]
        times = list(range(2, 100, 2))

        # Add a scatter plot
        fig.add_trace(
            go.Scatter(
                x=times, y=percentage, mode="lines+markers", name=hb_map[folder.name]
            )
        )

    # Update layout
    fig.update_layout(
        title="Outcome forecast acc over Game Duration",
        xaxis_title="% Game Duration",
        yaxis_title="Accuracy (%)",
        yaxis_range=[0, 1],
    )

    # Show the plot
    return fig


@callback(
    Output("download-csv", "data"),
    Input("save-button", "n_clicks"),
    State("ts2-length-win", "figure"),
    prevent_initial_call=True,
)
def save_graph_data_to_csv(n_clicks, figure):
    if n_clicks:
        buffer = io.StringIO()

        writer = csv.writer(buffer)

        if len(figure["data"]) == 0:
            raise PreventUpdate

        print(figure)

        max_index = max(
            [len(figure["data"][idx]["x"]) for idx in range(len(figure["data"]))]
        )
        for idx in range(-1, max_index):
            row = []
            for data in figure["data"]:
                if idx == -1:
                    row += [f"{data['name']}_x", f"{data['name']}_y"]
                    continue
                try:
                    row += [data["x"][idx], data["y"][idx]]
                except IndexError:
                    row += [0, 0]
            writer.writerow(row)

    return dict(content=buffer.getvalue(), filename="graph_data.csv")
