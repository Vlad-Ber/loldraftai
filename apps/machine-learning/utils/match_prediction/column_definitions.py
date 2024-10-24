# utils/match_prediction/column_definitions.py
from enum import Enum
from typing import Any, Callable, Dict

from utils.match_prediction.database import Match


class ColumnType(Enum):
    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    LIST = "list"


# Type alias for the extraction function
ExtractorFunc = Callable[[Match], Any]


class ColumnDefinition:
    def __init__(self, column_type: ColumnType, extractor: ExtractorFunc):
        self.column_type = column_type
        self.extractor = extractor


def extract_numerical_elo(match: Match):
    tier = match.averageTier.value
    division = match.averageDivision.value

    if (tier, division) in [
        ("DIAMOND", "II"),
        ("DIAMOND", "I"),
        ("MASTER", "I"),
        ("GRANDMASTER", "I"),
        ("CHALLENGER", "I"),
    ]:
        return 0
    elif (tier, division) in [("DIAMOND", "III")]:
        return 1
    elif (tier, division) in [("DIAMOND", "IV")]:
        return 2
    elif (tier, division) in [("EMERALD", "I")]:
        return 3
    else:
        raise ValueError(f"Unknown elo: {tier} {division}")


def extract_numerical_patch(match: Match):
    # max minor patch last 3 years is 24
    return match.gameVersionMajorPatch * 50 + match.gameVersionMinorPatch


POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]


def extract_champion_ids(match: Match):
    champion_ids = []
    for team_id in sorted(match.teams.keys()):
        team = match.teams[team_id]
        for position in POSITIONS:
            champion_ids.append(team["participants"][position]["championId"])
    return champion_ids


COLUMNS: Dict[str, ColumnDefinition] = {
    "champion_ids": ColumnDefinition(ColumnType.LIST, extract_champion_ids),
    "champion_role_percentages": ColumnDefinition(
        ColumnType.LIST, lambda match: None
    ),  # This will be filled by the MatchDataset
    "numerical_elo": ColumnDefinition(ColumnType.NUMERICAL, extract_numerical_elo),
    "numerical_patch": ColumnDefinition(ColumnType.NUMERICAL, extract_numerical_patch),
}

CATEGORICAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.CATEGORICAL
]
NUMERICAL_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.NUMERICAL
]
LIST_COLUMNS = [
    col for col, def_ in COLUMNS.items() if def_.column_type == ColumnType.LIST
]


def extract_raw_features(match: Match):
    return {col: def_.extractor(match) for col, def_ in COLUMNS.items()}
