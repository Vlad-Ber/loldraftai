#!/usr/bin/env python3

import argparse
import json
import pandas as pd
from typing import Dict, Any, List

# Constants
POSITIONS = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
TEAMS = ["100", "200"]
TIMESTAMPS = ["900000", "1200000", "1500000", "1800000"]
INDIVIDUAL_STATS = ["kills", "level", "deaths", "assists", "totalGold", "creepScore"]
DAMAGE_STATS = [
    "magicDamageDoneToChampions",
    "physicalDamageDoneToChampions",
    "trueDamageDoneToChampions",
]
TEAM_STATS = [
    "towerKills",
    "inhibitorKills",
    "baronKills",
    "dragonKills",
    "riftHeraldKills",
]


def extract_teams_data(match_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and flatten teams data from a match"""
    result = {}

    teams = match_data.get("teams", {})

    # Extract champion IDs and role data
    for team_id in TEAMS:
        if team_id not in teams:
            return None

        team = teams[team_id]
        for position in POSITIONS:
            if position not in team.get("participants", {}):
                return None

            participant = team["participants"][position]
            prefix = f"team_{team_id}_{position}"

            # Champion ID
            result[f"{prefix}_championId"] = participant["championId"]

            # Timeline data
            timeline = participant.get("timeline", {})
            for timestamp in TIMESTAMPS:
                if timestamp not in timeline:
                    continue

                # Individual stats
                for stat in INDIVIDUAL_STATS:
                    result[f"{prefix}_{stat}_at_{timestamp}"] = timeline[timestamp].get(
                        stat
                    )

                # Damage stats
                damage_stats = timeline[timestamp].get("damageStats", {})
                for damage_type in DAMAGE_STATS:
                    result[f"{prefix}_{damage_type}_at_{timestamp}"] = damage_stats.get(
                        damage_type
                    )

    # Extract team stats
    for team_id in TEAMS:
        team = teams[team_id]
        prefix = f"team_{team_id}"

        # Win status
        result[f"{prefix}_win"] = team.get("win", False)

        # Team stats at different timestamps
        team_stats = team.get("teamStats", {})
        for timestamp in TIMESTAMPS:
            if timestamp not in team_stats:
                continue

            stats = team_stats[timestamp]
            for stat in TEAM_STATS:
                result[f"{prefix}_{stat}_at_{timestamp}"] = stats.get(stat)

    return result


def process_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single match into a flat dictionary"""
    # Start with the base fields we want to keep
    result = {
        "id": match.get("id"),
        "matchId": match.get("matchId"),
        "queueId": match.get("queueId"),
        "region": match.get("region"),
        "averageTier": match.get("averageTier"),
        "averageDivision": match.get("averageDivision"),
        "gameVersionMajorPatch": match.get("gameVersionMajorPatch"),
        "gameVersionMinorPatch": match.get("gameVersionMinorPatch"),
        "gameDuration": match.get("gameDuration"),
        "gameStartTimestamp": match.get("gameStartTimestamp"),
    }

    # Extract and flatten teams data
    teams_data = extract_teams_data(match)
    if teams_data is None:
        return None

    result.update(teams_data)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-file", required=True, help="Input JSON file containing matches"
    )
    parser.add_argument("--output-file", required=True, help="Output parquet file path")
    args = parser.parse_args()

    # Read input JSON
    with open(args.batch_file, "r") as f:
        matches = json.load(f)

    # Process all matches
    processed_matches = []
    for match in matches:
        processed = process_match(match)
        if processed is not None:
            processed_matches.append(processed)

    if not processed_matches:
        print("No valid matches to process")
        return

    # Create DataFrame
    df = pd.DataFrame(processed_matches)

    # Convert timestamp to datetime if it exists
    if "gameStartTimestamp" in df.columns:
        df["gameStartTimestamp"] = pd.to_datetime(df["gameStartTimestamp"])

    # Save to parquet
    df.to_parquet(args.output_file, index=False)
    print(f"Successfully created parquet file with {len(df)} matches")


if __name__ == "__main__":
    main()
