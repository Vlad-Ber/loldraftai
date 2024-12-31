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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";
import { HelpCircle } from "lucide-react";

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
        const response = await fetch(`${baseApiUrl}/api/champion-suggestions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            team1,
            team2,
            championIds: championsIdsToConsider,
            selectedTeamIndex: selectedSpot.teamIndex,
            selectedChampionIndex: selectedSpot.championIndex,
            elo: eloToNumerical(elo),
            patch,
          }),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const predictions: { championId: number; winrate: number }[] =
          await response.json();
        const winrates = predictions
          .map((pred: { championId: number; winrate: number }) => ({
            champion: champions.find((c) => c.id === pred.championId)!,
            winrate: pred.winrate,
          }))
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

  return (
    <div className="mt-5 rounded-lg border border-gray-200 bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 p-6 shadow-sm">
      <div>
        <div className="flex items-center gap-2 mb-4">
          <h6 className="text-xl font-semibold brand-text">
            LoLDraftAI Champion Suggestions
          </h6>
          <TooltipProvider delayDuration={0}>
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-6 w-6 text-gray-500" />
              </TooltipTrigger>
              <TooltipContent className="max-w-[350px] whitespace-normal">
                <p className="mb-2">
                  Champions suggestions are ranked by the{" "}
                  <span className="brand-text">LoLDraftAI</span> winrate
                  prediction with that champion in the team.
                </p>
                <p className="mb-2">
                  <strong>Note:</strong>
                </p>
                <ul className="list-disc pl-4 space-y-1">
                  <li>
                    When the draft is not complete, the model predictions are
                    for the average draft,{" "}
                    <strong>
                      not necessarily for drafts where a champion ends up hard
                      countered
                    </strong>
                    . If not many enemy champions are visible, prioritize strong
                    blind picks.
                  </li>
                  <li>
                    The model doesn't take into account player skill. It
                    predicts the average performance on each champion.
                  </li>
                </ul>
                <p className="mt-2">
                  For these reasons you as the player should still make the
                  final call on what the best champion is.
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>

        <div className="space-y-1">
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              <p>Loading best champion suggestion...</p>
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : (
            championData.map((championWinrate, index) => (
              <div
                key={index}
                className={`relative rounded-md p-3 ${
                  !championWinrate.isAvailable ? "opacity-50" : ""
                }`}
              >
                <div className="flex flex-col items-center gap-2">
                  <div className="flex items-center gap-3 mb-1">
                    <div className="relative">
                      <div className="absolute -left-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-slate-700 text-xs font-bold text-white">
                        {index + 1}
                      </div>
                      <ImageComponent
                        src={`/icons/champions/${championWinrate.champion.icon}`}
                        alt={championWinrate.champion.name}
                        width={40}
                        height={40}
                        className={`rounded-full ${
                          !championWinrate.isAvailable ? "grayscale" : ""
                        }`}
                      />
                    </div>
                    <h6
                      className={`text-lg font-semibold ${
                        !championWinrate.isAvailable ? "line-through" : ""
                      }`}
                    >
                      {championWinrate.champion.name}
                      <span
                        className={`ml-2 ${
                          selectedSpot.teamIndex === 1
                            ? "text-blue-500"
                            : "text-red-500"
                        }`}
                      >
                        {championWinrate.winrate.toFixed(1)}%
                      </span>
                    </h6>
                  </div>
                  <div className="w-full">
                    <WinrateBar
                      team1Winrate={
                        selectedSpot.teamIndex === 1
                          ? championWinrate.winrate
                          : 100 - championWinrate.winrate
                      }
                    />
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};
