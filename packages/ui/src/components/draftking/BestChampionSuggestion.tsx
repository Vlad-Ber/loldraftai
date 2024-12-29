import { useState, useEffect, useMemo } from "react";
import { Loader2 } from "lucide-react";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
  Elo,
  ImageComponent,
} from "@draftking/ui/lib/types";
import { champions } from "@draftking/ui/lib/champions";
import { championIndexToFavoritesPosition } from "@draftking/ui/lib/types";
import { eloToNumerical } from "@draftking/ui/lib/draftLogic";
import { WinrateBar } from "./WinrateBar";

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: Elo;
  patch: string;
  baseApiUrl: string;
  ImageComponent: ImageComponent;
}

// Helper functions from DraftAnalysis
const formatTeamData = (team: Team): (number | "UNKNOWN")[] => {
  const championsIds: (number | "UNKNOWN")[] = [];
  for (let i = 0; i < 5; i++) {
    championsIds.push(team[i as keyof Team]?.id ?? "UNKNOWN");
  }
  return championsIds;
};

export const BestChampionSuggestion = ({
  team1,
  team2,
  selectedSpot,
  favorites,
  remainingChampions,
  elo,
  patch,
  baseApiUrl,
  ImageComponent,
}: BestChampionSuggestionProps) => {
  const [championWinrates, setChampionWinrates] = useState<
    Array<{ champion: Champion; winrate: number }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const championsIdsToConsider = useMemo(() => {
    const favoritesForSpot =
      favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)];
    return favoritesForSpot;
  }, [selectedSpot, favorites]);

  const championData = useMemo(() => {
    return championWinrates.map(({ champion, winrate }) => ({
      champion,
      winrate,
      isAvailable: remainingChampions.some((c) => c.id === champion.id),
    }));
  }, [championWinrates, remainingChampions]);

  useEffect(() => {
    const fetchWinrates = async () => {
      setLoading(true);
      try {
        const predictionPromises = championsIdsToConsider.map(
          async (champId) => {
            const newTeam =
              selectedSpot.teamIndex === 1 ? { ...team1 } : { ...team2 };
            const champion = champions.find((c) => c.id === champId);
            if (!champion) return null;

            newTeam[selectedSpot.championIndex] = champion;

            const requestBody = {
              champion_ids: [
                ...formatTeamData(
                  selectedSpot.teamIndex === 1 ? newTeam : team1
                ),
                ...formatTeamData(
                  selectedSpot.teamIndex === 2 ? newTeam : team2
                ),
              ],
              numerical_elo: eloToNumerical(elo),
              patch,
            };

            const response = await fetch(`${baseApiUrl}/api/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(requestBody),
            });

            if (!response.ok) {
              throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            const winrate = result.win_probability * 100;

            return {
              champion,
              winrate: selectedSpot.teamIndex === 1 ? winrate : 100 - winrate,
            };
          }
        );

        const results = await Promise.all(predictionPromises);
        const winrates = results
          .filter(
            (result): result is NonNullable<typeof result> => result !== null
          )
          .sort((a, b) => b.winrate - a.winrate);

        setChampionWinrates(winrates);
      } catch (error) {
        console.error("Error in finding best champion:", error);
        setError("Failed to load champion suggestions. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    void fetchWinrates();
  }, [
    selectedSpot,
    team1,
    team2,
    championsIdsToConsider,
    elo,
    patch,
    baseApiUrl,
  ]);

  useEffect(() => {
    if (loading) {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [loading]);

  return (
    <div className="mt-5 rounded border p-4">
      <div>
        <h6 className="mb-2 text-lg font-semibold">
          Champion Suggestions (Best First)
        </h6>
        <div>
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <p>Loading best champion suggestion...</p>
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            championData.map((championWinrate, index) => (
              <div
                key={index}
                className={`mb-2 flex flex-col items-center ${
                  !championWinrate.isAvailable ? "opacity-50" : ""
                }`}
              >
                <div className="flex items-center gap-2 mb-1 relative">
                  <ImageComponent
                    src={`/icons/champions/${championWinrate.champion.icon}`}
                    alt={championWinrate.champion.name}
                    width={32}
                    height={32}
                    className={`inline-block ${
                      !championWinrate.isAvailable ? "grayscale" : ""
                    }`}
                  />
                  <h6
                    className={`text-lg font-semibold ${
                      !championWinrate.isAvailable ? "line-through" : ""
                    }`}
                  >
                    {`${
                      championWinrate.champion.name
                    }: ${championWinrate.winrate.toFixed(1)}%`}
                  </h6>
                </div>
                <WinrateBar
                  team1Winrate={
                    selectedSpot.teamIndex === 1
                      ? championWinrate.winrate
                      : 100 - championWinrate.winrate
                  }
                />
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
