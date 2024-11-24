import React, { useState, useEffect, useMemo } from "react";
import { predictGame } from "./api";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
  Elo,
} from "@/app/types";
import { championIndexToFavoritesPosition } from "@/app/types";
import { getChampionById } from "@/app/champions";
import { WinrateBar } from "./WinrateBar";

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: Elo;
}

interface ChampionWinrate {
  champion: Champion;
  winrate: number;
}

export const BestChampionSuggestion = ({
  team1,
  team2,
  selectedSpot,
  favorites,
  remainingChampions,
  elo,
}: BestChampionSuggestionProps) => {
  const [championData, setChampionData] = useState<ChampionWinrate[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null); // Add an error state

  const championsIdsToConsider = useMemo(() => {
    if (selectedSpot) {
      const favoritesForSpot =
        favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)];
      const remainingChampionsIds = remainingChampions.map(
        (champion) => champion.id
      );
      const filteredFavorites = favoritesForSpot.filter((favorite) =>
        remainingChampionsIds.includes(favorite)
      );
      return filteredFavorites;
    }
    return [];
  }, [selectedSpot, favorites, remainingChampions]);

  useEffect(() => {
    const findBestChampion = async () => {
      setLoading(true);

      try {
        // Create array of promises for each champion prediction
        const predictionPromises = championsIdsToConsider.map(
          async (champId) => {
            const newTeam =
              selectedSpot.teamIndex === 1 ? { ...team1 } : { ...team2 };
            const champion = getChampionById(champId);

            if (!champion) {
              console.error(
                "Champion in favorites not found in champions list:",
                champId
              );
              return null;
            }

            newTeam[selectedSpot.championIndex] = champion;

            const result = await predictGame(
              selectedSpot.teamIndex === 1 ? newTeam : team1,
              selectedSpot.teamIndex === 2 ? newTeam : team2,
              elo
            );

            const winrate =
              selectedSpot.teamIndex === 1
                ? result.win_probability
                : 100 - result.win_probability;

            return { champion, winrate };
          }
        );

        // Wait for all predictions to complete
        const results = await Promise.all(predictionPromises);

        // Filter out null results and sort by winrate
        const championsWinrates = results
          .filter((result): result is ChampionWinrate => result !== null)
          .sort((a, b) => b.winrate - a.winrate);

        setChampionData(championsWinrates);
      } catch (error) {
        console.error("Error in finding best champion:", error);
        setError("Failed to load champion suggestions. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    if (selectedSpot) {
      void findBestChampion();
    }
  }, [selectedSpot, team1, team2, championsIdsToConsider, elo]);

  useEffect(() => {
    // Scroll to bottom when analysis is shown
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
            <>
              <p>Loading best champion suggestion...</p>
              <p>
                The first request could take up to 15s because we scale capacity
                to 0 when traffic is low.
              </p>
            </>
          ) : error ? ( // Conditionally render error message
            <p className="text-red-500">{error}</p>
          ) : (
            championData.map((championWinrate, index) => (
              <div key={index} className="mb-2 flex flex-col items-center">
                <h6 className="text-lg font-semibold">{`${
                  championWinrate.champion.name
                }: ${championWinrate.winrate.toFixed(1)}%`}</h6>
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
