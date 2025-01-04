import React from "react";
import type { Team } from "@draftking/ui/lib/types";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from "../ui/tooltip";
import { WinrateBar } from "./WinrateBar";
import type { ImageComponent } from "@draftking/ui/lib/types";

interface DraftAnalysisShowcaseProps {
  prediction: {
    win_probability: number;
    gold_diff_15min: number[];
    champion_impact: number[];
  };
  team1: Team;
  team2: Team;
  ImageComponent: ImageComponent; // Add ImageComponent prop
}

const HeaderTooltip = ({
  children,
  content,
}: {
  children: React.ReactNode;
  content: string;
}) => (
  <th className="p-2 text-right">
    <Tooltip>
      <TooltipTrigger className="cursor-help">{children}</TooltipTrigger>
      <TooltipContent className="max-w-[200px] whitespace-normal text-center">
        {content}
      </TooltipContent>
    </Tooltip>
  </th>
);

export const DraftAnalysisShowcase = ({
  prediction,
  team1,
  team2,
  ImageComponent,
}: DraftAnalysisShowcaseProps) => {
  const { win_probability, gold_diff_15min, champion_impact } = prediction;

  return (
    <TooltipProvider delayDuration={0}>
      <div className="flex w-full flex-col items-center space-y-4">
        <div className="w-full">
          <h2 className="text-center text-lg font-bold">Predicted Winrate</h2>
          <div className="mb-1 flex w-full justify-between">
            <p>{`${win_probability.toFixed(1)}% Blue Side`}</p>
            <p>{`Red Side ${(100 - win_probability).toFixed(1)}%`}</p>
          </div>
          <WinrateBar team1Winrate={win_probability} />
        </div>

        <div className="w-full overflow-x-auto rounded-lg bg-white dark:bg-gray-900 shadow-sm">
          <table className="w-full min-w-[600px] table-auto">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 text-sm text-white">
                <th className="p-3 text-left font-medium">BLUE SIDE</th>
                <HeaderTooltip content="How much this champion contributes to their team's win probability. Tip: play around champions with high impact, they are your key to victory.">
                  IMPACT
                </HeaderTooltip>
                <HeaderTooltip content="Predicted gold lead for this champion at 15 minutes">
                  G@15 LEAD
                </HeaderTooltip>
                <HeaderTooltip content="Predicted gold lead for this champion at 15 minutes">
                  G@15 LEAD
                </HeaderTooltip>
                <HeaderTooltip content="How much this champion contributes to their team's win probability. Tip: play around champions with high impact, they are your key to victory.">
                  IMPACT
                </HeaderTooltip>
                <th className="p-3 text-right font-medium">RED SIDE</th>
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 5 }).map((_, i) => (
                <tr
                  key={i}
                  className="border-b border-gray-100 dark:border-gray-800"
                >
                  <td className="p-3">
                    {team1[i as 0 | 1 | 2 | 3 | 4] && (
                      <ImageComponent
                        src={`/icons/champions/${
                          team1[i as 0 | 1 | 2 | 3 | 4]?.icon
                        }`}
                        alt={team1[i as 0 | 1 | 2 | 3 | 4]?.name ?? ""}
                        width={40}
                        height={40}
                        className="inline-block "
                      />
                    )}
                  </td>
                  <td
                    className={`p-3 text-right ${
                      (champion_impact[i] as number) > 0
                        ? "text-green-600"
                        : (champion_impact[i] as number) < 0
                        ? "text-red-600"
                        : ""
                    }`}
                  >
                    {champion_impact[i] !== 0 &&
                      `(${((champion_impact[i] as number) * 100).toFixed(1)}%)`}
                  </td>
                  <td className="p-3 text-center">
                    {(gold_diff_15min[i] ?? 0) > 0
                      ? Math.abs(gold_diff_15min[i] ?? 0).toFixed(0)
                      : ""}
                  </td>
                  <td className="p-3 text-center">
                    {(gold_diff_15min[i] ?? 0) < 0
                      ? Math.abs(gold_diff_15min[i] ?? 0).toFixed(0)
                      : ""}
                  </td>
                  <td
                    className={`p-3 text-right ${
                      (champion_impact[i + 5] as number) > 0
                        ? "text-green-600"
                        : (champion_impact[i + 5] as number) < 0
                        ? "text-red-600"
                        : ""
                    }`}
                  >
                    {(champion_impact[i + 5] as number) !== 0 &&
                      `(${((champion_impact[i + 5] as number) * 100).toFixed(
                        1
                      )}%)`}
                  </td>
                  <td className="p-3 text-right">
                    {team2[i as 0 | 1 | 2 | 3 | 4] && (
                      <ImageComponent
                        src={`/icons/champions/${
                          team2[i as 0 | 1 | 2 | 3 | 4]?.icon
                        }`}
                        alt={team2[i as 0 | 1 | 2 | 3 | 4]?.name ?? ""}
                        width={40}
                        height={40}
                        className="inline-block"
                      />
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </TooltipProvider>
  );
};
