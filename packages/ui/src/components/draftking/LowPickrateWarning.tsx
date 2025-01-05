import { type Team, type ChampionIndex } from "../../lib/types";
import { getChampionPlayRates } from "../../lib/champions";
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

export function LowPickrateWarning({
  teamOne,
  teamTwo,
  currentPatch,
}: LowPickrateWarningProps) {
  // Check both teams for low pickrate champions
  const hasLowPickrate = (team: Team) => {
    return Object.entries(team).some(([index, champion]) => {
      if (!champion) return false;

      const roleKey = roleIndexToKey[index as unknown as ChampionIndex];
      const playRates = getChampionPlayRates(champion.id, currentPatch);

      return playRates ? playRates[roleKey] < PICKRATE_THRESHOLD : false;
    });
  };

  const showWarning = hasLowPickrate(teamOne) || hasLowPickrate(teamTwo);

  if (!showWarning) return null;

  return (
    <div className="text-amber-600 text-sm font-medium mb-4 flex items-center gap-2">
      <ExclamationTriangleIcon className="h-5 w-5" />
      Warning: Uncommon champion-role combination(s) detected. Predictions may
      have lower confidence for these picks.
    </div>
  );
}
