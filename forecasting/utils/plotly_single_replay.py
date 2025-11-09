from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dash.exceptions import PreventUpdate
from konductor.metadata.database import Metadata

layout = html.Div(
    children=[
        html.H2("Replay Outcome Prediction"),
        dbc.Row(
            [
                dbc.Col(html.H4("Experiment:"), width="auto"),
                dbc.Col(dcc.Dropdown(id="sr-run-brief"), width=True),
                dbc.Col(html.H4("Replay:"), width="auto"),
                dbc.Col(dcc.Dropdown(id="sr-replay-hash"), width=True),
            ]
        ),
        dbc.Row([dcc.Graph(id="sr-replay-preds", selectedData={})]),
    ]
)

EXPERIMENTS: list[Metadata] = []


def get_experiment_by_brief(brief: str):
    return next(filter(lambda x: x.brief == brief, EXPERIMENTS))


def get_experiment_data(root: str, brief: str) -> pd.DataFrame:
    exp_meta = get_experiment_by_brief(brief)
    data = pd.read_csv(Path(root) / exp_meta.hash / "outcome_prediction.csv")
    return data


@callback(
    Output("sr-run-brief", "options"),
    Input("root-dir", "data"),
    prevent_initial_call=False,
)
def update_runs(root: str):
    if not root:
        raise PreventUpdate()

    for item in Path(root).iterdir():
        if (item / "outcome_prediction.csv").exists():
            exp_meta = Metadata.from_yaml(item / "metadata.yaml")
            EXPERIMENTS.append(exp_meta)

    return [exp.brief for exp in EXPERIMENTS]


@callback(
    Output("sr-replay-hash", "options"),
    Input("root-dir", "data"),
    Input("sr-run-brief", "value"),
)
def update_replays(root: str, run_brief: str):
    """Update the list of replays to choose from"""
    if not root or not run_brief:
        raise PreventUpdate()

    data = get_experiment_data(root, run_brief)
    return list(data["replay"].unique())


@callback(
    Output("sr-replay-preds", "figure"),
    Input("root-dir", "data"),
    Input("sr-run-brief", "value"),
    Input("sr-replay-hash", "value"),
)
def update_game_length(root: str, run_brief: str, replay_hash: str):
    if not all((root, run_brief, replay_hash)):
        raise PreventUpdate

    data = get_experiment_data(root, run_brief)
    col_to_rm = ["replay"]
    if "Unnamed: 0" in data.columns:
        col_to_rm.append("Unnamed: 0")
    data = data[data["replay"] == replay_hash].drop(columns=col_to_rm)

    aux_cols = {"playerId", "outcome"}
    times = pd.Series(
        [float(time) for time in filter(lambda x: x not in aux_cols, data.columns)],
        name="time [min]",
    )

    fig = go.Figure()
    winner = 9999
    for playerId in [1, 2]:
        # Add a scatter plot
        playerData = data[data["playerId"] == playerId]
        outcome = playerData.drop(columns=["playerId", "outcome"]).iloc[0]
        if playerData["outcome"].all():
            winner = playerId
        fig.add_trace(
            go.Scatter(
                x=times, y=outcome, mode="lines+markers", name=f"Player {playerId}"
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Outcome Forecast, Player {winner} Win",
        xaxis_title="Game Time [min]",
        yaxis_title="Forecasted Win (%)",
        yaxis_range=[0, 1],
    )

    # Show the plot
    return fig
