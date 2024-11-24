import type { Team } from "@/app/types";

// Backend URL - can be moved to a config file for cloud deployment
const backendUrl =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000/";

interface DataFormat {
  topChampionName: string;
  jungleChampionName: string;
  middleChampionName: string;
  bottomChampionName: string;
  utilityChampionName: string;
}

export const formatTeamData = (team: Team) => {
  // Transform team data into the required format for the API
  const formattedData: DataFormat = {} as DataFormat;
  if (team[0]) {
    formattedData.topChampionName = team[0].name;
  } else {
    formattedData.topChampionName = "";
  }
  if (team[1]) {
    formattedData.jungleChampionName = team[1].name;
  } else {
    formattedData.jungleChampionName = "";
  }
  if (team[2]) {
    formattedData.middleChampionName = team[2].name;
  } else {
    formattedData.middleChampionName = "";
  }
  if (team[3]) {
    formattedData.bottomChampionName = team[3].name;
  } else {
    formattedData.bottomChampionName = "";
  }
  if (team[4]) {
    formattedData.utilityChampionName = team[4].name;
  } else {
    formattedData.utilityChampionName = "";
  }
  return formattedData;
};

interface Prediction {
  prediction: number;
}

export const predictGame = async (
  team1: Team,
  team2: Team,
  elo: string
): Promise<Prediction> => {
  const requestBody = {
    team_100: formatTeamData(team1),
    team_200: formatTeamData(team2),
    elo: elo.toLowerCase(),
  };

  const endpoint = "predict";
  const url = new URL(endpoint, backendUrl).toString();
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return (await response.json()) as Prediction;
  } catch (error) {
    console.error("There was a problem with the fetch operation:", error);
    throw error; // Rethrow to handle it in the calling function
  }
};

interface PredictionWithShap {
  prediction: number;
  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  shap: { [key: string]: number };
}

export const predictGameWithShap = async (
  team1: Team,
  team2: Team,
  elo: string
): Promise<PredictionWithShap> => {
  const requestBody = {
    team_100: formatTeamData(team1),
    team_200: formatTeamData(team2),
    elo: elo.toLowerCase(),
  };

  const endpoint = "predict_shap";
  const url = new URL(endpoint, backendUrl).toString();
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return (await response.json()) as PredictionWithShap;
  } catch (error) {
    console.error("There was a problem with the fetch operation:", error);
    throw error; // Rethrow to handle it in the calling function
  }
};

// eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
export const featureMappings: { [key: string]: string } = {
  "200_TOP_winrate": "Red Top Win Rate",
  "200_JUNGLE_winrate": "Red Jungle Win Rate",
  "200_MIDDLE_winrate": "Red Mid Win Rate",
  "200_BOTTOM_winrate": "Red Bot Win Rate",
  "200_UTILITY_winrate": "Red Support Win Rate",
  "200_AVG_WINRATE": "Red Average Win Rate",
  "200_NON_META": "Red Non-meta Picks",
  "200_MASKED_AND_NON_META": "Red Masked and Non-meta Picks",
  "100_TOP_winrate": "Blue Top Win Rate",
  "100_JUNGLE_winrate": "Blue Jungle Win Rate",
  "100_MIDDLE_winrate": "Blue Mid Win Rate",
  "100_BOTTOM_winrate": "Blue Bot Win Rate",
  "100_UTILITY_winrate": "Blue Support Win Rate",
  "100_AVG_WINRATE": "Blue Average Win Rate",
  "100_NON_META": "Blue Non-meta Picks",
  "100_MASKED_AND_NON_META": "Blue Masked and Non-meta Picks",
  AVG_WINRATE_DIFF: "Overall Stronger Champions",
  TOP_WINRATE_DIFF: "Stronger Top Champion",
  JUNGLE_WINRATE_DIFF: "Stronger Jungle Champion",
  MIDDLE_WINRATE_DIFF: "Stronger Mid Champion",
  BOTTOM_WINRATE_DIFF: "Stronger Bot Champion",
  UTILITY_WINRATE_DIFF: "Stronger Support Champion",
  "200_TOP_against_100_TOP_winrate": "Top Matchup",
  "200_JUNGLE_with_200_MIDDLE_winrate": "Red Jungle and Mid Synergy",
  "200_JUNGLE_against_100_JUNGLE_winrate": "Jungle Matchup",
  "200_MIDDLE_against_100_MIDDLE_winrate": "Mid Matchup",
  "200_BOTTOM_with_200_UTILITY_winrate": "Red Bot and Support Synergy",
  "200_BOTTOM_against_100_BOTTOM_winrate": "Bot Matchup",
  "200_BOTTOM_against_100_UTILITY_winrate": "Blue Support vs. Red Bot Matchup",
  "200_UTILITY_against_100_BOTTOM_winrate": "Blue Bot vs. Red Support Matchup",
  "200_UTILITY_against_100_UTILITY_winrate": "Support Matchup",
  "100_JUNGLE_with_100_MIDDLE_winrate": "Blue Jungle and Mid Synergy",
  "100_BOTTOM_with_100_UTILITY_winrate": "Blue Bot and Support Synergy",
  sum_100_synergies: "Overall Blue Synergy",
  //'n_100_synergies': 'Number of Synergies (Blue)',
  sum_200_synergies: "Overall Red Synergy",
  //'n_200_synergies': 'Number of Synergies (Red)',
  sum_matchups: "Overall Matchups",
  "200_TOTAL_PHYS DMG_DPM": "Red Total Physical Damage",
  "200_TOTAL_AP DMG_DPM": "Red Total AP Damage",
  "200_TOTAL_TRUE DMG_DPM": "Red Total True Damage",
  "200_TOTAL_DPM": "Red Total Damage",
  "100_TOTAL_PHYS DMG_DPM": "Blue Total Physical Damage",
  "100_TOTAL_AP DMG_DPM": "Blue Total AP Damage",
  "100_TOTAL_TRUE DMG_DPM": "Blue Total True Damage",
  "100_TOTAL_DPM": "Blue Total Damage",
};
