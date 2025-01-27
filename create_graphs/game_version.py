import sqlite3
from contextlib import closing
from pathlib import Path
from packaging.version import Version

import typer
import pandas as pd

app = typer.Typer()


def _remove_build_version(version: str | None):
    if version is None:
        return version
    i = version.rindex(".")
    return version[:i]


def test_build_removal():
    version = "1.2.3.456"
    no_build = _remove_build_version(version)
    assert no_build == "1.2.3"


@app.command()
def main(
    path: Path,
    output: Path = Path("version_stats.csv"),
    remove_build: bool = True,
):
    """Create"""
    assert output.suffix == ".csv", "Output must be .csv"
    with closing(sqlite3.connect(path)) as con:
        metadata = pd.read_sql("SELECT gameVersion FROM game_data", con)
    version_data = metadata["gameVersion"]

    if remove_build:
        version_data = version_data.transform(_remove_build_version)
    counts = version_data.value_counts()
    counts = counts.reindex(index=pd.Index(sorted(counts.index, key=Version)))
    counts.to_csv(output, index_label="version")


if __name__ == "__main__":
    test_build_removal()
    app()
