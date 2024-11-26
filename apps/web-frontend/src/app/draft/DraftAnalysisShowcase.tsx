import React from "react";
import { WinrateBar } from "./WinrateBar";

interface DraftAnalysisShowcaseProps {
  winrate: number;
}

export const DraftAnalysisShowcase = ({
  winrate,
}: DraftAnalysisShowcaseProps) => {
  const team1Winrate = winrate;
  const team2Winrate = 100 - winrate;

  return (
    <>
      <div className="flex w-full flex-col items-center">
        <h2 className="mt-4 text-center text-lg font-bold">
          Predicted Winrate
        </h2>
        <div className="mb-1 flex w-full justify-between">
          <p>{`${Math.round(team1Winrate)}% Blue Side`}</p>
          <p>{`Red Side ${Math.round(team2Winrate)}%`}</p>
        </div>
        <WinrateBar team1Winrate={team1Winrate} />
      </div>
    </>
  );
};
