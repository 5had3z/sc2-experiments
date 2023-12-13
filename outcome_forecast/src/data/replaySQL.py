from dataclasses import dataclass
from pathlib import Path
from typing import List

from konductor.data import DATASET_REGISTRY, Split

import sqlite3
from .baseDataset import SC2ReplayBase, TimeRange, SC2ReplayConfigBase


class SC2SQLReplay(SC2ReplayBase):
    def __init__(
        self,
        basepath: Path,
        split: Split,
        train_ratio: float,
        features: set[str] | None,
        timepoints: TimeRange,
        min_game_time: float,
        sql_query: str,
        database: Path,
    ) -> None:
        self.sql_query = sql_query
        self.database = database

        super().__init__(
            basepath, split, train_ratio, features, timepoints, min_game_time
        )

        # Extract and print column names
        self.cursor.execute("PRAGMA table_info('game_data');")
        # Fetch all rows containing column information
        columns_info = self.cursor.fetchall()
        self.column_names = [column[1] for column in columns_info]
        self.file_name_idx = self.column_names.index("partition")
        self.idx_idx = self.column_names.index("idx")

    def load_files(self, basepath):
        self.conn = sqlite3.connect(self.database)
        self.cursor = self.conn.cursor()
        self.cursor.execute(self.sql_query.replace(" * ", " COUNT(*) "))

        self.n_replays = self.cursor.fetchone()[0]
        self.train_test_split()

    def __getitem__(self, index: int):
        squery = self.sql_query[:-1] + f" LIMIT 1 OFFSET {index};"
        self.cursor.execute(squery)
        result = self.cursor.fetchone()

        return self.getitem(
            self.basepath / result[self.file_name_idx], result[self.idx_idx]
        )


@dataclass
@DATASET_REGISTRY.register_module("sc2-sql-replay")
class SC2ReplayConfig(SC2ReplayConfigBase):
    database: Path = Path("./")
    sql_filters: List[str] | None = None
    sql_query: str = ""

    def get_class(self):
        return SC2SQLReplay

    def _known_unused(self):
        return {"train_loader", "val_loader", "basepath", "sql_filters"}

    def __post_init__(self):
        super().__post_init__()

        sql_filter_string = (
            ""
            if self.sql_filters is None or len(self.sql_filters) == 0
            else (" " + " AND ".join(self.sql_filters))
        )
        self.sql_query = "SELECT * FROM game_data" + sql_filter_string + ";"
        assert sqlite3.complete_statement(self.sql_query), "Incomplete SQL Statement"
        with sqlite3.connect(self.database) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(self.sql_query)
            except sqlite3.OperationalError as e:
                raise AssertionError("Invalid SQL Syntax", e)
