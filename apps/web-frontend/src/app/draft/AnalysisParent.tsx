import React, { useEffect, useMemo, useState } from "react";
import type {
  Champion,
  Team,
  FavoriteChampions,
  SelectedSpot,
  Elo,
} from "@/app/types";
import { championIndexToFavoritesPosition, elos } from "@/app/types";
import { DraftAnalysis } from "./DraftAnalysis";
import { BestChampionSuggestion } from "./BestChampionSuggestion";

interface AnalysisParentProps {
  team1: Team;
  team2: Team;
  selectedSpot: SelectedSpot | null;
  favorites: FavoriteChampions;
  remainingChampions: Champion[];
  analysisTrigger: number;
}

interface EloSelectProps {
  elo: Elo;
  setElo: (elo: Elo) => void;
}

const EloSelect = ({ elo, setElo }: EloSelectProps) => (
  <>
    <label htmlFor="elo-select" className="sr-only">
      Select Elo
    </label>
    <select
      id="elo-select"
      className="rounded p-2 text-black"
      value={elo}
      onChange={(e) => setElo(e.target.value as Elo)}
      aria-label="Select Elo"
    >
      {elos.map((eloOption) => (
        <option key={eloOption} value={eloOption}>
          {eloOption.toUpperCase()}
        </option>
      ))}
    </select>
  </>
);

interface AnalyzeDraftButtonProps {
  toggleAnalyzeDraft: () => void;
  showAnalysis: boolean;
}
const AnalyzeDraftButton = ({
  toggleAnalyzeDraft,
  showAnalysis,
}: AnalyzeDraftButtonProps) => (
  <button
    className={`hover:bg-blue-700} rounded bg-blue-500 p-2 font-bold text-white`}
    onClick={toggleAnalyzeDraft}
  >
    {showAnalysis ? "Hide Analysis" : "Analyze Draft"}
  </button>
);

interface TooltipProps {
  children: React.ReactNode;
  text: string;
}

const Tooltip = ({ children, text }: TooltipProps) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div className="relative flex items-center">
      <div
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        {children}
      </div>
      {isHovered && (
        <div className="absolute bottom-full z-10 mb-2 rounded-md bg-black px-2 py-1 text-xs text-white">
          {text}
        </div>
      )}
    </div>
  );
};

interface ChampionSuggestionButtonProps {
  enableChampionSuggestion: boolean;
  toggleChampionSuggestion: () => void;
  showChampionSuggestion: boolean;
  selectedSpot: SelectedSpot | null;
}

const ChampionSuggestionButton = ({
  enableChampionSuggestion,
  toggleChampionSuggestion,
  showChampionSuggestion,
  selectedSpot,
}: ChampionSuggestionButtonProps) =>
  enableChampionSuggestion ? (
    <button
      className="rounded bg-blue-500 p-2 font-bold text-white hover:bg-blue-700"
      onClick={toggleChampionSuggestion}
    >
      {showChampionSuggestion ? "Hide Suggestions" : "Suggest Champion"}
    </button>
  ) : (
    <Tooltip
      text={
        !selectedSpot
          ? "Click on a team position to select a position."
          : "Right click a champion in the list to add to favorites."
      }
    >
      <button
        className="cursor-not-allowed rounded bg-blue-500 p-2 font-bold text-white opacity-50"
        onClick={toggleChampionSuggestion}
        disabled={true}
      >
        {showChampionSuggestion ? "Hide Suggestions" : "Suggest Champion"}
      </button>
    </Tooltip>
  );

const AnalysisParent = ({
  team1,
  team2,
  selectedSpot,
  favorites,
  remainingChampions,
  analysisTrigger,
}: AnalysisParentProps) => {
  const [showChampionSuggestion, setShowChampionSuggestion] = useState(false);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [elo, setElo] = useState<Elo>("emerald");

  useEffect(() => {
    // reset when team changes
    setShowChampionSuggestion(false);
  }, [selectedSpot, team1, team2, elo]);

  useEffect(() => {
    setShowAnalysis(false);
  }, [team1, team2, elo]);

  useEffect(() => {
    if (analysisTrigger > 0) {
      setShowAnalysis(true);
    }
  }, [analysisTrigger]);

  useEffect(() => {
    //scroll to bottom when analysis is shown
    if (showAnalysis) {
      window.scrollTo({
        top: document.body.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [showAnalysis]);

  const toggleChampionSuggestion = () => {
    setShowChampionSuggestion(!showChampionSuggestion);
  };

  const enableChampionSuggestion = useMemo(() => {
    return (
      (selectedSpot &&
        favorites[championIndexToFavoritesPosition(selectedSpot.championIndex)]
          .length > 0) ??
      false
    );
  }, [selectedSpot, favorites]);

  return (
    <div className="draft-analysis p-5">
      <div className="-mx-2 flex flex-wrap items-stretch">
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <EloSelect elo={elo} setElo={setElo} />
          </div>
        </div>
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <AnalyzeDraftButton
              toggleAnalyzeDraft={() => {
                setShowAnalysis(!showAnalysis);
              }}
              showAnalysis={showAnalysis}
            />
          </div>
        </div>
        <div className="flex w-full p-1 sm:w-auto">
          <div className="flex-1">
            <ChampionSuggestionButton
              enableChampionSuggestion={enableChampionSuggestion}
              toggleChampionSuggestion={toggleChampionSuggestion}
              showChampionSuggestion={showChampionSuggestion}
              selectedSpot={selectedSpot}
            />
          </div>
        </div>
      </div>

      {showAnalysis && <DraftAnalysis team1={team1} team2={team2} elo={elo} />}

      {showChampionSuggestion && selectedSpot && (
        <BestChampionSuggestion
          team1={team1}
          team2={team2}
          selectedSpot={selectedSpot}
          favorites={favorites}
          remainingChampions={remainingChampions}
          elo={elo}
        />
      )}
    </div>
  );
};

export default AnalysisParent;
