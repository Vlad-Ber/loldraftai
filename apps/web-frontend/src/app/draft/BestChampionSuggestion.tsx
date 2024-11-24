import React, { useState, useEffect, useMemo } from "react";
import { predictGame } from "./api";
import type {
  Champion,
  Team,
  SelectedSpot,
  FavoriteChampions,
} from "@/app/types";
import { championIndexToFavoritesPosition } from "@/app/types";
import { createChampion } from "@/app/champions";
import { WinrateBar } from "./WinrateBar";

interface BestChampionSuggestionProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  elo: string;
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

  const championsToConsider = useMemo(() => {
    if (selectedSpot) {
      const favoritesForSpot =
        favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)];
      const remainingChampionsNames = remainingChampions.map(
        (champion) => champion.searchName
      );
      const filteredFavorites = favoritesForSpot.filter((favorite) =>
        remainingChampionsNames.includes(favorite)
      );
      return filteredFavorites;
    }
    return [];
  }, [selectedSpot, favorites, remainingChampions]);

  useEffect(() => {
    const findBestChampion = async () => {
      setLoading(true);
      const championsWinrates = [];

      for (const champ of championsToConsider) {
        const newTeam =
          selectedSpot.teamIndex === 1 ? { ...team1 } : { ...team2 };
        const champion = createChampion(champ);
        newTeam[selectedSpot.championIndex] = champion;

        try {
          const result = await predictGame(
            selectedSpot.teamIndex === 1 ? newTeam : team1,
            selectedSpot.teamIndex === 2 ? newTeam : team2,
            elo
          );
          const winrate =
            selectedSpot.teamIndex === 1
              ? result.prediction
              : 100 - result.prediction;
          championsWinrates.push({ champion, winrate });
        } catch (error) {
          console.error("Error in finding best champion:", error);
          setError("Failed to load champion suggestions. Please try again."); // Set error message on failure
          setLoading(false);
          return; // Exit the function early on error
        }
      }

      // Sort champions by winrate in descending order
      championsWinrates.sort((a, b) => b.winrate - a.winrate);
      setChampionData(championsWinrates);
      setLoading(false);
    };

    if (selectedSpot) {
      void findBestChampion();
    }
  }, [selectedSpot, team1, team2, championsToConsider, elo]);

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
