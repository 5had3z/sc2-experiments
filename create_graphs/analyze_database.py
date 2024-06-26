import sqlite3
from functools import reduce
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import typer
from sc2_serializer import Race, Result
from violin_plot import plot_single, plot_vs

app = typer.Typer()
TABLE_NAME = "game_data"

custom_colors = {
    Race.Terran: (85 / 255, 115 / 255, 205 / 255),  # Replace with your RGB values
    Race.Zerg: (137 / 255, 26 / 255, 185 / 255),  # Replace with your RGB values
    Race.Protoss: (223 / 255, 193 / 255, 32 / 255),  # Replace with your RGB values
}
race_palette = {int(race): tuple(color) for race, color in custom_colors.items()}

custom_colors2 = {
    (Race.Terran, Race.Terran): (
        85 / 255,
        115 / 255,
        205 / 255,
    ),  # Replace with your RGB values
    (Race.Terran, Race.Zerg): (
        116 / 255,
        128 / 255,
        140 / 255,
    ),  # Replace with your RGB values
    (Race.Terran, Race.Protoss): (
        16 / 255,
        111 / 255,
        13 / 255,
    ),  # Replace with your RGB values
    (Race.Zerg, Race.Terran): (
        116 / 255,
        128 / 255,
        140 / 255,
    ),  # Replace with your RGB values
    (Race.Zerg, Race.Zerg): (
        137 / 255,
        26 / 255,
        185 / 255,
    ),  # Replace with your RGB values
    (Race.Zerg, Race.Protoss): (
        205 / 255,
        125 / 255,
        152 / 255,
    ),  # Replace with your RGB values
    (Race.Protoss, Race.Terran): (
        16 / 255,
        111 / 255,
        13 / 255,
    ),  # Replace with your RGB values
    (Race.Protoss, Race.Zerg): (
        205 / 255,
        125 / 255,
        152 / 255,
    ),  # Replace with your RGB values
    (Race.Protoss, Race.Protoss): (
        223 / 255,
        193 / 255,
        32 / 255,
    ),  # Replace with your RGB values
}
vs_palette = {race: tuple(color) for race, color in custom_colors2.items()}


def compose(*funcs):
    return lambda x: reduce(lambda acc, f: f(acc), reversed(funcs), x)


def get_races(cursor: sqlite3.Cursor):
    query = f"""
        SELECT playerrace, COUNT(*) as count
        FROM {TABLE_NAME}
        WHERE playerrace IN (0, 1, 2)
        GROUP BY playerrace
        """
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchall()

    for row in results:
        playerrace, count = row
        typer.echo(f"Player Race {Race(int(playerrace))}: {count} occurrences")


def get_rows(cursor: sqlite3.Cursor):
    query = f"""
        SELECT COUNT(*) as count
        FROM {TABLE_NAME}
        """
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchone()

    typer.echo(f"Number of rows: {results[0]:,}")
    return results[0]


def count_invalid(cursor: sqlite3.Cursor, valid_rows: int | None = None):
    min_game_time_minutes = 5
    game_steps = int(min_game_time_minutes * 60 * 22.4)
    invalid_criteria = [
        "playerMMR <= 0",
        "read_success = 0",
        "playerAPM <= 0",
        "final_score_float <= 0",
        f"game_length < {game_steps}",
    ]
    invalid_criteria.append(" OR ".join(invalid_criteria))
    for ic in invalid_criteria:
        query = f"""
            SELECT COUNT(*) as count
            FROM {TABLE_NAME}
            WHERE {ic}
            """
        cursor.execute(query)

        # Fetch the results
        results = cursor.fetchone()

        if valid_rows is not None:
            typer.echo(
                f"Invalid rows ({ic}): {results[0]:,} ({results[0] * 100 / valid_rows:.2f}%)"
            )
        else:
            typer.echo(f"Invalid rows ({ic}): {results[0]:,}")


def race_match_length(cursor: sqlite3.Cursor, prefix: str):
    def _race_match_length(race1: int, race2: int):
        query = f"""
            SELECT game_length
            FROM {TABLE_NAME}
            WHERE playerRace IN ({race1}, {race2})
            GROUP BY replayHash
            HAVING COUNT(DISTINCT playerRace) = {len(set((race1,race2)))}
                AND COUNT(*) = 2;
            """
        cursor.execute(query)
        results = [x[0] / 22.4 / 60 for x in cursor.fetchall()]
        return results

    numbers = [x for x in Race.__members__.values() if x != Race.Random]
    result = [(num1, num2) for num1 in numbers for num2 in numbers]

    def canonical_order(pair):
        return tuple(sorted((int(pair[0]), int(pair[1]))))

    result = sorted(result, key=canonical_order)

    data = {}
    for c in result:
        output = _race_match_length(int(c[0]), int(c[1]))
        data[c] = output

    _vs_palette = {idx: vs_palette[key] for idx, key in enumerate(data.keys())}

    plot_vs(
        data,
        _vs_palette,
        f"{prefix}_racematch_length",
        y_max=80,
        ylabel="Game Length (minutes)",
    )


def racevsrace_length(cursor: sqlite3.Cursor, prefix: str):
    def _racevsrace_length(winning_race: int, losing_race: int):
        query = f"""
        SELECT game_length
        FROM {TABLE_NAME}
        WHERE (playerRace = {winning_race} AND playerResult = {int(Result.Win)})
               OR (playerRace = {losing_race} AND playerResult = {int(Result.Loss)})
        GROUP BY replayHash
        HAVING COUNT(DISTINCT playerRace) = {len(set((winning_race,losing_race)))}
        AND COUNT(*) = 2;
        """
        cursor.execute(query)
        results = [x[0] / 22.4 / 60 for x in cursor.fetchall()]
        return results

    numbers = [x for x in Race.__members__.values() if x != Race.Random]
    result = [(num1, num2) for num1 in numbers for num2 in numbers]

    def canonical_order(pair):
        return tuple(sorted((int(pair[0]), int(pair[1]))))

    result = sorted(result, key=canonical_order)

    data = {}
    for c in result:
        output = _racevsrace_length(int(c[0]), int(c[1]))
        data[c] = output

    _vs_palette = {idx: vs_palette[key] for idx, key in enumerate(data.keys())}

    plot_vs(
        data,
        _vs_palette,
        f"{prefix}_racevsrace_length",
        y_max=80,
        ylabel="Game Length (minutes)",
    )


def racewin(cursor: sqlite3.Cursor, prefix: str):
    def _racewin(winning_race: int):
        query = f"""
        SELECT game_length
        FROM {TABLE_NAME}
        WHERE (playerRace = {winning_race} AND playerResult = {int(Result.Win)})
        GROUP BY replayHash
        """
        cursor.execute(query)
        results = [x[0] / 22.4 / 60 for x in cursor.fetchall()]
        return results

    data = {}

    for c in Race.__members__.values():
        if c == Race.Random:
            continue
        output = _racewin(int(c))
        data[str(c)] = output

    plot_single(
        data,
        race_palette,
        f"{prefix}_win_lengths",
        y_max=100,
        ylabel="Game Length (minutes)",
    )


def apm_per_race(cursor: sqlite3.Cursor, prefix: str):
    def _apm_per_race(winning_race: int):
        query = f"""
        SELECT playerAPM
        FROM {TABLE_NAME}
        WHERE playerRace = {winning_race} and playerAPM > 0
        """
        cursor.execute(query)
        results = [x[0] for x in cursor.fetchall()]
        return results

    data = {}

    for c in Race.__members__.values():
        if c == Race.Random:
            continue
        output = _apm_per_race(int(c))
        data[str(c)] = output

    plot_single(
        data,
        race_palette,
        f"{prefix}_apm_per_race",
        y_max=1200,
        ylabel="Actions Per Miniute",
    )


def count_discrete_values(
    cursor: sqlite3.Cursor,
    columns: Dict[str, Tuple[List[Any], Callable[[str], Any] | None]],
):
    def convert(k, v):
        return f"{k} in ({','.join([str(i) for i in v[0]])})"

    query = f"""
            SELECT {", ".join(columns.keys())}, COUNT(*) as count
            FROM {TABLE_NAME}
            WHERE {" AND ".join(convert(k,v) for k,v in columns.items())}
            GROUP BY {", ".join(columns.keys())}
        """
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchall()

    for row in results:
        # Extract column values and count from the row
        values = row[:-1]
        count = row[-1]

        # Format and print the output dynamically
        formatted_values = ", ".join(
            f"{k}: {v[1](str(val)) if v[1] is not None else val}"
            for (k, v), val in zip(columns.items(), values)
        )

        typer.echo(f"{formatted_values}, Count: {count} occurrences")


def mad(data):
    median = np.median(data)
    diff = np.abs(data - median)
    mad = np.median(diff)
    return mad


def calculate_bounds(data, z_thresh=3.5):
    MAD = mad(data)
    median = np.median(data)
    const = z_thresh * MAD / 0.6745
    return (median - const, median + const)


def outlier_aware_hist(data, lower=None, upper=None):
    if not lower or lower < data.min():
        lower = data.min()
        lower_outliers = False
    else:
        lower_outliers = True

    if not upper or upper > data.max():
        upper = data.max()
        upper_outliers = False
    else:
        upper_outliers = True

    n, bins, patches = plt.hist(data, range=(lower, upper), density=True, bins=100)

    csv_data = np.column_stack((bins[:-1], bins[1:], n))
    np.savetxt(
        "histogram_data.csv",
        csv_data,
        delimiter=",",
        header="Lower Bin Edge,Upper Bin Edge,Frequency",
        comments="",
    )

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(
            patches[0].get_height() + n_lower_outliers / data.shape[0]
        )
        patches[0].set_facecolor("c")
        patches[0].set_label(
            "Lower outliers: ({:.2f}, {:.2f})".format(data.min(), lower)
        )

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        patches[-1].set_height(
            patches[-1].get_height() + n_upper_outliers / data.shape[0]
        )
        patches[-1].set_facecolor("m")
        patches[-1].set_label(
            "Upper outliers, freq: ({:.2f}, {:.2f}, {:.2f})".format(
                upper, data.max(), n_upper_outliers / data.shape[0]
            )
        )

    if lower_outliers or upper_outliers:
        plt.legend()
    return upper


def plot_game_lengths(cursor):
    cursor.execute(f"SELECT game_length FROM {TABLE_NAME}")
    game_lengths = [x[0] / 22.4 for x in cursor.fetchall()]

    data = np.array(game_lengths)
    upper = outlier_aware_hist(data, *calculate_bounds(data))

    # Set x ticks every 60 units, up to 600
    plt.xticks(range(0, int(upper), 60), rotation="vertical")

    plt.title("Normalized Distribution of Game Lengths")
    plt.xlabel("Game Length (s)")
    plt.ylabel("Probability Density")
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.grid(True)
    plt.savefig("game_length.png")


def get_column_diff(cursor: sqlite3.Cursor, column: str):
    """
    Calculate the difference between the two values of a column inn a game in a SQLite table.

    Parameters:
    - cursor (sqlite3.Cursor): SQLite cursor for executing queries.
    - column (str): Name of the column for which the difference is to be calculated.

    The function performs the following steps:
    1. Retrieves data from the specified column and playerResult from the given table where:
       - replayHash is common in both rows.
       - playerMMR > 0 and playerAPM > 0 for both rows.
       - Groups the data by replayHash and selects rows where playerResult is 0 or 1.
       - Sorted ascending to ensure, 'Win' is the first value (0)
    2. Constructs a subquery to calculate APM_Difference by subtracting the two values
    3. Normalizes the APM_Difference values.
    4. Writes the normalized values to a file in a specific format.
    """
    query = f"""
    WITH Subquery AS (
        SELECT replayHash,
            GROUP_CONCAT({column}) AS APM,
            GROUP_CONCAT(playerResult) AS Result,
            SUBSTR(GROUP_CONCAT({column}), 0, INSTR(GROUP_CONCAT({column}), ',')) AS open,
            SUBSTR(GROUP_CONCAT({column}), INSTR(GROUP_CONCAT({column}), ',')+1) AS close
        FROM (
            SELECT replayHash, {column}, playerResult
            FROM {TABLE_NAME}
            WHERE replayHash IN (
                SELECT replayHash
                FROM {TABLE_NAME}
                WHERE playerMMR > 0 AND playerAPM > 0
                GROUP BY replayHash
                HAVING COUNT(*) = 2
            )
            ORDER BY playerResult ASC
        )
        GROUP BY replayHash
        HAVING playerResult IN (0, 1)
    )
    SELECT (open - close) AS APM_Difference
    FROM Subquery;
    """

    cursor.execute(query)
    output = [x[0] for x in cursor.fetchall()]
    normal_value = max([abs(x) for x in output]) * 5
    output = [x / normal_value for x in output]

    with open(f"{column}_{normal_value}.dat", "w") as out:
        out.write("A\n")
        out.write("\n".join(map(str, output)))


def generate_scatter(
    cursor: sqlite3.Cursor, column_x: str, column_y: str, x_filter=None, y_filter=None
):
    # Construct the SQL query with optional filters
    query = f"SELECT {column_y}, {column_x} FROM {TABLE_NAME}"
    if x_filter is not None and y_filter is not None:
        query += f" WHERE {column_x} {x_filter} AND {column_y} {y_filter}"
    elif x_filter is not None:
        query += f" WHERE {column_x} {x_filter}"
    elif y_filter is not None:
        query += f" WHERE {column_y} {y_filter}"

    cursor.execute(query)
    data = cursor.fetchall()
    y_vals, x_vals = zip(*data)

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_vals, x_vals, alpha=0.5, color="green", edgecolors="black")

    # Set labels and title
    plt.title(f"Scatter Plot of {column_y} vs {column_x}")
    plt.xlabel(column_y)
    plt.ylabel(column_x)

    plt.savefig(f"{column_y.replace(' ', '_')}_vs_{column_x.replace(' ', '_')}.png")


@app.command()
def main(
    database: Annotated[Path, typer.Option()],
    output_prefix: Annotated[str, typer.Option()] = "tournament",
):
    database_str = str(database)

    # Use the SQLite connection
    try:
        conn = sqlite3.connect(database_str)
        cursor = conn.cursor()

        # get_column_diff(cursor, "playerAPM")
        # get_column_diff(cursor, "playerMMR")
        racewin(cursor, output_prefix)
        apm_per_race(cursor, output_prefix)
        racevsrace_length(cursor, output_prefix)
        race_match_length(cursor, output_prefix)
        # rows = get_rows(cursor)
        # count_invalid(cursor, rows)
        # plot_game_lengths(cursor)

        # count_discrete_values(cursor, {"playerRace": ([0, 1, 2], compose(Race, int))})
        # count_discrete_values(
        #     cursor,
        #     {"playerRace": ([0, 1, 2], compose(Race, int)), "playerId": ([1, 2], None)},
        # )
        # count_discrete_values(
        #     cursor,
        #     {
        #         "playerRace": ([0, 1, 2], compose(Race, int)),
        #         "playerResult": ([0, 1], compose(Result, int)),
        #     },
        # )
        # count_discrete_values(
        #     cursor,
        #     {
        #         "playerId": ([1, 2], None),
        #         "playerResult": ([0, 1], compose(Result, int)),
        #     },
        # )
        # generate_scatter(cursor, "playerMMR", "game_length", x_filter="> 0")

        conn.close()
    except sqlite3.Error as e:
        typer.echo(f"SQLite error: {e}")


if __name__ == "__main__":
    app()
