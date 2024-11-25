import React, { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
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
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

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
  <Select value={elo} onValueChange={setElo}>
    <SelectTrigger>
      <SelectValue placeholder="Select Elo" />
    </SelectTrigger>
    <SelectContent>
      <SelectGroup>
        {elos.map((eloOption) => (
          <SelectItem key={eloOption} value={eloOption}>
            {eloOption.toUpperCase()}
          </SelectItem>
        ))}
      </SelectGroup>
    </SelectContent>
  </Select>
);

interface AnalyzeDraftButtonProps {
  toggleAnalyzeDraft: () => void;
  showAnalysis: boolean;
}
const AnalyzeDraftButton = ({
  toggleAnalyzeDraft,
  showAnalysis,
}: AnalyzeDraftButtonProps) => (
  <Button variant="outline" onClick={toggleAnalyzeDraft}>
    {showAnalysis ? "Hide Analysis" : "Analyze Draft"}
  </Button>
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
    <Button variant="outline" onClick={toggleChampionSuggestion}>
      {showChampionSuggestion ? "Hide Suggestions" : "Suggest Champion"}
    </Button>
  ) : (
    <Tooltip
      text={
        !selectedSpot
          ? "Click on a team position to select a position."
          : "Right click a champion in the list to add to favorites."
      }
    >
      <Button
        variant="outline"
        onClick={toggleChampionSuggestion}
        disabled={true}
      >
        {showChampionSuggestion ? "Hide Suggestions" : "Suggest Champion"}
      </Button>
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
