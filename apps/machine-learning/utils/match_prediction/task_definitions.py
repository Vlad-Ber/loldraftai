# utils/match_prediction/task_definitions.py

import pandas as pd
from enum import Enum
from typing import Callable

TEAMS = ["100", "200"]
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
DAMAGE_STATS = [
    "magicDamageDoneToChampions",
    "physicalDamageDoneToChampions",
    "trueDamageDoneToChampions",
]
TIMESTAMPS = ["900000", "1200000", "1500000", "1800000"]
TEAM_STATS = [
    # "totalKills",
    # "totalDeaths",
    # "totalAssists",
    # "totalGold",
    "towerKills",
    "inhibitorKills",
    "baronKills",
    "dragonKills",
    "riftHeraldKills",
]

INDIVIDUAL_STATS = [
    "kills",
    "level",
    "deaths",
    "assists",
    "totalGold",
    "creepScore",
]


class TaskType(Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


class TaskDefinition:
    def __init__(
        self,
        name: str,
        task_type: TaskType,
        weight: float,
        getter: Callable[[pd.DataFrame], pd.Series] = None,
    ):
        self.name = name
        self.getter = getter
        self.task_type = task_type
        self.weight = weight


def get_win_prediction(df: pd.DataFrame) -> pd.Series:
    return df["team_100_win"]


# Define tasks
TASKS = {
    "win_prediction": TaskDefinition(
        name="win_prediction",
        getter=get_win_prediction,
        task_type=TaskType.BINARY_CLASSIFICATION,
        weight=0.8,
    ),
    "gameDuration": TaskDefinition(
        name="gameDuration",
        task_type=TaskType.REGRESSION,
        weight=0.01,
    ),
}
for stat in INDIVIDUAL_STATS:
    for position in POSITIONS:
        for team_id in TEAMS:
            for timestamp in TIMESTAMPS:
                task_name = f"team_{team_id}_{position}_{stat}_at_{timestamp}"
                TASKS[task_name] = TaskDefinition(
                    name=task_name,
                    task_type=TaskType.REGRESSION,
                    weight=0.05
                    / (
                        len(INDIVIDUAL_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# For damage stats
for damage_type in DAMAGE_STATS:
    for timestamp in TIMESTAMPS:
        for position in POSITIONS:
            for team_id in TEAMS:
                task_name = f"team_{team_id}_{position}_{damage_type}_at_{timestamp}"
                TASKS[task_name] = TaskDefinition(
                    name=task_name,
                    task_type=TaskType.REGRESSION,
                    weight=0.05
                    / (
                        len(DAMAGE_STATS)
                        * len(POSITIONS)
                        * len(TEAMS)
                        * len(TIMESTAMPS)
                    ),
                )

# For team stats
for timestamp in TIMESTAMPS:
    for stat in TEAM_STATS:
        for team_id in TEAMS:
            task_name = f"team_{team_id}_{stat}_at_{timestamp}"
            TASKS[task_name] = TaskDefinition(
                name=task_name,
                task_type=TaskType.REGRESSION,
                weight=0.1 / (len(TIMESTAMPS) * len(TEAM_STATS) * len(TEAMS)),
            )
