import React from "react";
import { WinrateBar } from "./WinrateBar";
import { featureMappings } from "./api";
import type { Team } from "@/app/types";
import Image from "next/image";

const filterFeaturesByTeamPresence = (
  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  shapMapping: { [key: string]: number },
  team1: Team,
  team2: Team
) => {
  const isPositionFilledInTeam = (team: Team, positionIndex: number) =>
    team[positionIndex] !== undefined;

  const positionRegex = /(100|200)_(TOP|JUNGLE|MIDDLE|BOTTOM|UTILITY)/g;
  const winrateDiffRegex = /(TOP|JUNGLE|MIDDLE|BOTTOM|UTILITY)_WINRATE_DIFF/g;

  const positionIndexMap = {
    TOP: 0,
    JUNGLE: 1,
    MIDDLE: 2,
    BOTTOM: 3,
    UTILITY: 4,
  };

  return Object.entries(shapMapping)
    .filter(([key]) => {
      const positionMatches = key.match(positionRegex);
      const winrateDiffMatches = key.match(winrateDiffRegex);

      if (!positionMatches && !winrateDiffMatches) return true;

      if (positionMatches) {
        return positionMatches.every((match) => {
          const [teamPrefix, position] = match.split("_");
          const team = teamPrefix === "100" ? team1 : team2;

          const positionIndex =
            positionIndexMap[position as keyof typeof positionIndexMap];

          return isPositionFilledInTeam(team, positionIndex);
        });
      }

      if (winrateDiffMatches) {
        return winrateDiffMatches.every((match) => {
          const position = match.split("_")[0];
          const positionIndex =
            positionIndexMap[position as keyof typeof positionIndexMap];

          return (
            isPositionFilledInTeam(team1, positionIndex) &&
            isPositionFilledInTeam(team2, positionIndex)
          );
        });
      }
    })
    .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});
};

interface TeamAdvantagesShowcaseProps {
  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  shapMapping: { [key: string]: number };
  team1: Team;
  team2: Team;
}

const TeamAdvantagesShowcase = ({
  shapMapping,
  team1,
  team2,
}: TeamAdvantagesShowcaseProps) => {
  // Filter shapMapping for absent roles
  const filteredShapMappingByTeamPresence = filterFeaturesByTeamPresence(
    shapMapping,
    team1,
    team2
  );

  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  const prepareAdvantages = (filteredShapMapping: {
    [key: string]: number;
  }) => {
    return Object.entries(filteredShapMapping)
      .map(([key, value]) => {
        const iconsComponents: JSX.Element[] = [];

        // Mapping from position to index in the team array
        const positionToIndex: {
          [key in "TOP" | "JUNGLE" | "MIDDLE" | "BOTTOM" | "UTILITY"]: number;
        } = {
          TOP: 0,
          JUNGLE: 1,
          MIDDLE: 2,
          BOTTOM: 3,
          UTILITY: 4,
        };
        // Extract all {100|200}_{POSITION} substrings and map them to icons
        const positionMatches = key.match(
          /(100|200)_(TOP|JUNGLE|MIDDLE|BOTTOM|UTILITY)/g
        );
        if (positionMatches) {
          // Sort positions with 100 before those with 200
          const sortedPositionMatches = positionMatches.sort((a) =>
            a.includes("100") ? -1 : 1
          );
          sortedPositionMatches.forEach((match) => {
            const [teamPrefix, position] = match.split("_");
            const relevantTeam = teamPrefix === "100" ? team1 : team2;
            const champion =
              relevantTeam[
                positionToIndex[
                  position as "TOP" | "JUNGLE" | "MIDDLE" | "BOTTOM" | "UTILITY"
                ]
              ];
            if (champion) {
              iconsComponents.push(
                <Image
                  key={champion.icon}
                  src={"/icons/champions/" + champion.icon}
                  alt={"Icon of " + champion.name}
                  width={50}
                  height={50}
                  style={{
                    border:
                      teamPrefix === "100" ? "2px solid blue" : "2px solid red",
                  }}
                />
              );
            }
          });
        }

        return { feature: key, value, icons: iconsComponents };
      })
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value)); // Sort by absolute value
  };

  const advantages = prepareAdvantages(filteredShapMappingByTeamPresence);

  // Split into blue and red team advantages based on value
  const blueTeamAdvantages = advantages.filter(({ value }) => value > 0);
  const redTeamAdvantages = advantages.filter(({ value }) => value < 0);

  interface TeamAdvantagesProps {
    advantages: { feature: string; value: number; icons: JSX.Element[] }[];
    isBlueTeam: boolean;
  }

  const TeamAdvantages: React.FC<TeamAdvantagesProps> = ({
    advantages,
    isBlueTeam,
  }) => (
    <div className="mt-3 flex-1 p-1">
      <h3
        className={`text-center text-lg font-bold text-white ${
          isBlueTeam ? "bg-blue-500" : "bg-red-500"
        } rounded-t-lg py-2`}
      >
        {isBlueTeam ? "Blue Side Advantages" : "Red Side Advantages"}
      </h3>
      <table className="min-w-full divide-y divide-gray-200">
        <tbody className="divide-y divide-gray-200 bg-white">
          {advantages.map(({ feature, value, icons }) => {
            if (!featureMappings[feature]) return null; // Do not show the row if the feature is not in featureMappings
            return (
              <tr key={feature}>
                <td className="px-6 py-4 text-sm font-medium text-gray-900">
                  <div className="flex flex-wrap items-center">
                    <span className="mr-2">{featureMappings[feature]}</span>
                    <div className="flex -space-x-2 overflow-hidden">
                      {icons.map((icon, index) => (
                        <div key={index} className="p-1.5">
                          {icon}
                        </div>
                      ))}
                    </div>
                  </div>
                </td>
                <td className="whitespace-normal px-6 py-4 text-sm text-gray-500 sm:whitespace-nowrap">
                  +{(Math.abs(value) * 100).toFixed(1)}%
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  return (
    <>
      <p className="text-secondary p-1 text-center">
        Note: Displayed advantages are key factors; thus, percentages may not
        sum to the total predicted winrate.
      </p>
      <div className="flex w-full flex-wrap">
        <TeamAdvantages advantages={blueTeamAdvantages} isBlueTeam={true} />
        <TeamAdvantages advantages={redTeamAdvantages} isBlueTeam={false} />
      </div>
    </>
  );
};

interface DraftAnalysisShowcaseProps {
  winrate: number;
  // eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
  shapMapping: { [key: string]: number };
  team1: Team;
  team2: Team;
}

export const DraftAnalysisShowcase = ({
  winrate,
  shapMapping,
  team1,
  team2,
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
          <p className="text-secondary">
            {`${Math.round(team1Winrate)}% Blue Side`}
          </p>
          <p className="text-secondary">
            {`Red Side ${Math.round(team2Winrate)}%`}
          </p>
        </div>
        <WinrateBar team1Winrate={team1Winrate} />
        <h2 className="mt-4 text-center text-lg font-bold">Team Advantages</h2>
        <TeamAdvantagesShowcase
          shapMapping={shapMapping}
          team1={team1}
          team2={team2}
        />
      </div>
    </>
  );
};
