from datetime import datetime
from sqlite3 import Cursor
from typing import Iterable, Literal, Mapping

DBTYPESTR = Literal["INT", "FLOAT", "TEXT", "TIMESTAMP"]
DBTYPE = int | float | str | datetime


def create_table(cur: Cursor, name: str, cols: Iterable[str] | Mapping[str, DBTYPESTR]):
    """Create table with columns in sqlite if it doesn't already exist"""
    if not isinstance(cols, Mapping):
        cols = {c: "FLOAT" for c in cols}

    col_str = "hash TEXT PRIMARY KEY"
    for col, dtype in cols.items():
        col_str += f", {col} {dtype}"

    cur.execute(f"CREATE TABLE IF NOT EXISTS {name} ({col_str})")


def write_entry(
    cur: Cursor, table_name: str, run_hash: str, data: Mapping[str, DBTYPE]
):
    """Write entry into sqlite database"""
    cur.execute(f"INSERT OR IGNORE INTO {table_name} (hash) VALUES (?)", [run_hash])

    # Create update query
    set_str = "SET "
    set_val = []
    for name, value in data.items():
        set_str += f"{name} = ?, "
        set_val.append(value)
    set_val.append(run_hash)

    # set_str[:-2] Remove extra ', '
    cur.execute(f"UPDATE {table_name} {set_str[:-2]} WHERE hash = ?;", set_val)
    cur.close()
