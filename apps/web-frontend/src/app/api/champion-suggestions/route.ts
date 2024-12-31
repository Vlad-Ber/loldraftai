import { NextResponse } from "next/server";
import { Champion, Team } from "@draftking/ui/lib/types";

interface ChampionSuggestionRequest {
  team1: Team;
  team2: Team;
  championIds: number[];
  selectedTeamIndex: 1 | 2;
  selectedChampionIndex: number;
  elo: number;
  patch: string;
}

const backendUrl = process.env.INFERENCE_BACKEND_URL ?? "http://127.0.0.1:8000";

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

export async function POST(request: Request) {
  try {
    const body: ChampionSuggestionRequest = await request.json();
    const {
      team1,
      team2,
      championIds,
      selectedTeamIndex,
      selectedChampionIndex,
      elo,
      patch,
    } = body;

    // Make parallel requests to the inference API
    const predictions = await Promise.all(
      championIds.map(async (champId) => {
        const newTeam = selectedTeamIndex === 1 ? { ...team1 } : { ...team2 };
        newTeam[selectedChampionIndex as keyof Team] = {
          id: champId,
        } as Champion;

        const response = await fetch(`${backendUrl}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            champion_ids: [
              ...formatTeamData(selectedTeamIndex === 1 ? newTeam : team1),
              ...formatTeamData(selectedTeamIndex === 2 ? newTeam : team2),
            ],
            numerical_elo: elo,
            patch,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        return {
          championId: champId,
          winrate:
            selectedTeamIndex === 1
              ? result.win_probability * 100
              : (1 - result.win_probability) * 100,
        };
      })
    );

    return NextResponse.json(predictions);
  } catch (error) {
    console.error("Champion suggestions error:", error);
    return NextResponse.json(
      { error: "Failed to get champion suggestions" },
      { status: 500 }
    );
  }
}

// Helper function
const formatTeamData = (team: Team): (number | "UNKNOWN")[] => {
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as keyof Team]?.id ?? "UNKNOWN");
  }
  return championsIds;
};
