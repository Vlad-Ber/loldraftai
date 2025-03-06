from enum import Enum
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import pandas as pd

POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


class ColumnType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    LIST = "list"


@dataclass
class ColumnDefinition:
    name: str
    column_type: ColumnType
    getter: Optional[Callable[[pd.DataFrame], pd.Series]] = None


def get_numerical_elo(df: pd.DataFrame) -> pd.Series:
    """Convert tier/division to numerical value."""

    def map_elo(row):
        tier = row["averageTier"]
        division = row["averageDivision"]

        if (tier, division) in [
            ("DIAMOND", "II"),
            ("DIAMOND", "I"),
            ("MASTER", "I"),
            ("GRANDMASTER", "I"),
            ("CHALLENGER", "I"),
        ]:
            return 0
        elif (tier, division) == ("DIAMOND", "III"):
            return 1
        elif (tier, division) == ("DIAMOND", "IV"):
            return 2
        elif (tier, division) == ("EMERALD", "I"):
            return 3
        else:
            raise ValueError(f"Unknown elo: {tier} {division}")

    return df.apply(map_elo, axis=1)


def get_numerical_patch(df: pd.DataFrame) -> pd.Series:
    """Convert patch version to numerical value."""
    return df["gameVersionMajorPatch"] * 50 + df["gameVersionMinorPatch"]


def get_champion_ids(df: pd.DataFrame) -> pd.Series:
    champion_ids = []
    # Blue team, then red team from top to bottom
    for team in [100, 200]:
        for role in POSITIONS:
            champion_ids.append(df[f"team_{team}_{role}_championId"])
    # Concatenate horizontally and apply list conversion to each row
    return pd.concat(champion_ids, axis=1).apply(lambda x: x.tolist(), axis=1)


# Define all columns
COLUMNS: Dict[str, ColumnDefinition] = {
    # Computed columns
    "numerical_elo": ColumnDefinition(
        name="numerical_elo", column_type=ColumnType.NUMERICAL, getter=get_numerical_elo
    ),
    # special case for patch number, applied in prepare-data.py
    "numerical_patch": ColumnDefinition(
        name="numerical_patch",
        column_type=ColumnType.NUMERICAL,
        # getter=get_numerical_patch,
    ),
    # TODO: Could change to categorical for simplification
    "champion_ids": ColumnDefinition(
        name="champion_ids", column_type=ColumnType.LIST, getter=get_champion_ids
    ),
}

# Helper lists for different column types
CATEGORICAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.CATEGORICAL
]
NUMERICAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.NUMERICAL
]
LIST_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.LIST
]
