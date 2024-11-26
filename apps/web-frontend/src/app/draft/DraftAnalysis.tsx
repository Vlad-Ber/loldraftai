import React, { useEffect, useState } from "react";
import type { Elo, Team } from "@/app/types";
import { predictGame } from "./api";
import { DraftAnalysisShowcase } from "./DraftAnalysisShowcase";
import { Loader2 } from "lucide-react";

interface DraftAnalysisProps {
  team1: Team;
  team2: Team;
  elo: Elo;
}

export const DraftAnalysis = ({ team1, team2, elo }: DraftAnalysisProps) => {
  const [teamWinrate, setTeamWinrate] = useState<number | null>(null);

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchWinrate = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await predictGame(team1, team2, elo);
        setTeamWinrate(result.win_probability);
      } catch (err) {
        console.error("Error fetching winrate:", err);
        setError("Failed to load winrate. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    void fetchWinrate();
  }, [team1, team2, elo]);

  return (
    <div className="mt-5 rounded border border-gray-200 p-4">
      <div className="flex items-center justify-between">
        <h6 className="text-lg font-semibold">Draft Analysis</h6>
      </div>
      <div className="mb-2.5 mt-2 grid grid-cols-1 gap-2">
        <div>
          {loading ? (
            <div className="flex items-center gap-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              <p>Loading...</p>
            </div>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : teamWinrate ? (
            <DraftAnalysisShowcase winrate={teamWinrate} />
          ) : (
            <p>No data available</p>
          )}
        </div>
      </div>
    </div>
  );
};
