import { type Team, type ChampionIndex } from "../../lib/types";
import { getChampionPlayRates, getChampionById } from "../../lib/champions";
import { ExclamationTriangleIcon } from "@heroicons/react/24/solid";

interface LowPickrateWarningProps {
  teamOne: Team;
  teamTwo: Team;
  currentPatch: string;
}

const PICKRATE_THRESHOLD = 0.3;

const roleIndexToKey = {
  0: "TOP",
  1: "JUNGLE",
  2: "MIDDLE",
  3: "BOTTOM",
  4: "UTILITY",
} as const;

const roleToDisplayName = {
  TOP: "Toplane",
  JUNGLE: "Jungle",
  MIDDLE: "Midlane",
  BOTTOM: "Botlane",
  UTILITY: "Support",
} as const;

export function LowPickrateWarning({
  teamOne,
  teamTwo,
  currentPatch,
}: LowPickrateWarningProps) {
  // Find low pickrate champion-role combinations
  const findLowPickrateChampions = (team: Team) => {
    const lowPickrateChampions: string[] = [];

    Object.entries(team).forEach(([index, champion]) => {
      if (!champion) return;

      const roleKey = roleIndexToKey[index as unknown as ChampionIndex];
      const playRates = getChampionPlayRates(champion.id, currentPatch);

      if (playRates && playRates[roleKey] < PICKRATE_THRESHOLD) {
        const championData = getChampionById(champion.id);
        if (championData) {
          lowPickrateChampions.push(
            `${championData.name} / ${roleToDisplayName[roleKey]}`
          );
        }
      }
    });

    return lowPickrateChampions;
  };

  const teamOneLowPickrate = findLowPickrateChampions(teamOne);
  const teamTwoLowPickrate = findLowPickrateChampions(teamTwo);
  const allLowPickrateChampions = [
    ...teamOneLowPickrate,
    ...teamTwoLowPickrate,
  ];

  if (allLowPickrateChampions.length === 0) return null;

  return (
    <div className="text-amber-600 text-sm font-medium mb-4">
      <div className="flex items-center gap-2 mb-1">
        <ExclamationTriangleIcon className="h-5 w-5" />
        Warning: Uncommon champion-role combination(s) detected. Predictions may
        have lower confidence for these picks:
      </div>
      <ul className="list-disc pl-10">
        {allLowPickrateChampions.map((champRole, index) => (
          <li key={index}>{champRole}</li>
        ))}
      </ul>
    </div>
  );
}
