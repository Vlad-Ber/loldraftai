import type { Team, Elo, ChampionIndex } from "@/app/types";
import { eloToNumerical } from "@/app/types";

// Backend URL - can be moved to a config file for cloud deployment
const backendUrl =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://127.0.0.1:8000/";

export const formatTeamData = (team: Team) => {
  // Transform team data into the required format for the API
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as ChampionIndex]?.id ?? "UNKNOWN");
  }
  return championsIds;
};

interface Prediction {
  win_probability: number;
}

export const predictGame = async (
  team1: Team,
  team2: Team,
  elo: Elo
): Promise<Prediction> => {
  const requestBody = {
    champion_ids: [...formatTeamData(team1), ...formatTeamData(team2)],
    numerical_elo: eloToNumerical(elo),
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

    const data = (await response.json()) as Prediction;

    // Convert to percentage
    return { win_probability: data.win_probability * 100 };
  } catch (error) {
    console.error("There was a problem with the fetch operation:", error);
    throw error; // Rethrow to handle it in the calling function
  }
};
