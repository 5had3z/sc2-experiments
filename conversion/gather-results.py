#!/usr/bin/env python3

"""Gather and format results from perflog files"""

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(path: Path = Path.cwd()):
    """Gather performance logging files and print formatted results"""
    files = sorted(filter(lambda x: x.suffix == ".csv", path.iterdir()))

    results = pd.DataFrame(
        columns=["program", "n_threads", "playerId", "mean", "std"],
        index=pd.RangeIndex(0, 2 * len(files)),
    )

    for fidx, file in enumerate(files):
        data = pd.read_csv(file)
        program, threads = file.stem.split("_")
        for pidx, player in enumerate(["p1", "p2"]):
            filt = data[data["playerId"] == player]["time"]
            row = results.iloc[2 * fidx + pidx]
            row["program"] = program
            row["n_threads"] = int(threads)
            row["playerId"] = player
            row["mean"] = filt.mean()
            row["std"] = filt.std()

    print(results)


if __name__ == "__main__":
    app()
