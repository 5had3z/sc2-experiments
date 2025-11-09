#!/usr/bin/env python3
import typer
from src.baseline import minimap, outcome, outcome_numpy

app = typer.Typer()
app.add_typer(outcome.app, name="outcome")
app.add_typer(outcome_numpy.app, name="outcome-numpy")
app.add_typer(minimap.app, name="minimap")

if __name__ == "__main__":
    app()
