import React, { useEffect, useState } from "react";
import type { Team } from "@/app/types";
import { predictGameWithShap } from "./api";
import { DraftAnalysisShowcase } from "./DraftAnalysisShowcase";

const generateShareableLink = (team1: Team, team2: Team) => {
  //TODO: maybe add elo to the link
  let teamOneIds = Array.from({ length: 5 }, (_, i) => team1[i]?.id ?? "").join(
    ","
  );
  let teamTwoIds = Array.from({ length: 5 }, (_, i) => team2[i]?.id ?? "").join(
    ","
  );
  //remove trailing commas
  teamOneIds = teamOneIds.replace(/,+$/, "");
  teamTwoIds = teamTwoIds.replace(/,+$/, "");

  const baseUrl = window.location.origin + window.location.pathname;
  const shareableLink = `${baseUrl}?team1=${teamOneIds}&team2=${teamTwoIds}`;
  void navigator.clipboard.writeText(shareableLink);
};

interface DraftAnalysisProps {
  team1: Team;
  team2: Team;
  elo: string;
}

export const DraftAnalysis = ({ team1, team2, elo }: DraftAnalysisProps) => {
  const [teamWinrate, setTeamWinrate] = useState<number | null>(null);
  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  const [shapMapping, setShapMapping] = useState<{
    [key: string]: number;
  } | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchWinrate = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await predictGameWithShap(team1, team2, elo);
        setTeamWinrate(result.prediction);
        setShapMapping(result.shap);
      } catch (err) {
        console.error("Error fetching winrate:", err);
        setError("Failed to load winrate. Please try again.");
      } finally {
        setLoading(false);
      }
    };

    void fetchWinrate();
  }, [team1, team2, elo]);

  const handleShareDraft = () => {
    generateShareableLink(team1, team2);
  };
  const ShareButton = () => {
    const [buttonText, setButtonText] = useState("Share Analysis");

    const handleClick = () => {
      handleShareDraft();
      //TODO: something cleaner than changing the button text
      setButtonText("Link Copied!");
      setTimeout(() => {
        setButtonText("Share Analysis");
      }, 2000);
    };

    return (
      <button
        className="rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-700"
        onClick={handleClick}
      >
        {buttonText}
      </button>
    );
  };

  return (
    <div className="mt-5 rounded border border-gray-200 p-4">
      <div className="flex items-center justify-between">
        <h6 className="text-lg font-semibold">Draft Analysis</h6>
        <ShareButton />
      </div>
      <div className="mb-2.5 mt-2 grid grid-cols-1 gap-2">
        <div>
          {loading ? (
            <>
              <p>Loading...</p>
              <p>
                The first request could take up to 15s because we scale capacity
                to 0 when traffic is low.
              </p>
            </>
          ) : error ? (
            <p className="text-red-500">{error}</p>
          ) : teamWinrate && shapMapping ? (
            <DraftAnalysisShowcase
              winrate={teamWinrate}
              shapMapping={shapMapping}
              team1={team1}
              team2={team2}
            />
          ) : (
            <p>No data available</p>
          )}
        </div>
      </div>
    </div>
  );
};
