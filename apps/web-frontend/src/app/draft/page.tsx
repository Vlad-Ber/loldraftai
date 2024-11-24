/* eslint-disable @typescript-eslint/non-nullable-type-assertion-style */
"use client";

import React, { useEffect, useState } from "react";

import ChampionGrid from "./ChampionGrid";
import TeamPanel from "./TeamPanel";
import HelpModal from "./HelpModal";
import AnalysisParent from "./AnalysisParent";
import type {
  Champion,
  Team,
  TeamIndex,
  ChampionIndex,
  SelectedSpot,
  FavoriteChampions,
} from "@/app/types";
import { champions, championToRolesMap, roleToIndexMap } from "@/app/champions";
import Cookies from "js-cookie";

const emptyTeam: Team = {
  0: undefined,
  1: undefined,
  2: undefined,
  3: undefined,
  4: undefined,
};

export default function Draft() {
  const [remainingChampions, setRemainingChampions] =
    useState<Champion[]>(champions);
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [analysisTrigger] = useState(0);
  const teams = [teamOne, teamTwo];
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
  const [favorites, setFavorites] = useState<FavoriteChampions>({
    top: [],
    jungle: [],
    mid: [],
    bot: [],
    support: [],
  });

  const [showHelpModal, setShowHelpModal] = useState(false);

  const openHelpModal = () => setShowHelpModal(true);
  const closeHelpModal = () => setShowHelpModal(false);

  useEffect(() => {
    // Existing code to initialize favorites
    const savedFavorites = Cookies.get("favorites");
    if (savedFavorites) {
      setFavorites(JSON.parse(savedFavorites) as FavoriteChampions);
    }
  }, [remainingChampions]);

  const resetDraft = () => {
    setRemainingChampions(champions);
    setTeamOne(emptyTeam);
    setTeamTwo(emptyTeam);
    setSelectedSpot(null);
  };

  const handleSpotSelection = (index: ChampionIndex, team: TeamIndex) => {
    // If no spot is currently selected, select the new spot
    if (!selectedSpot) {
      setSelectedSpot({ teamIndex: team, championIndex: index });
      return;
    }

    // If the same spot is clicked again, deselect it
    if (
      selectedSpot.teamIndex === team &&
      selectedSpot.championIndex === index
    ) {
      setSelectedSpot(null);
      return;
    }

    const teamOneCopy = { ...teamOne };
    const teamTwoCopy = { ...teamTwo };
    const championFromSelectedSpot =
      selectedSpot.teamIndex === 1
        ? teamOneCopy[selectedSpot.championIndex]
        : teamTwoCopy[selectedSpot.championIndex];
    const targetChampion = team === 1 ? teamOneCopy[index] : teamTwoCopy[index];

    //if both spots are empty, select the new spot
    if (
      championFromSelectedSpot === undefined &&
      targetChampion === undefined
    ) {
      setSelectedSpot({ teamIndex: team, championIndex: index });
      return;
    }

    // Swap the champions between the two spots
    if (selectedSpot.teamIndex === 1) {
      if (team === 1) {
        // Swap within the same team
        teamOneCopy[selectedSpot.championIndex] = targetChampion;
        teamOneCopy[index] = championFromSelectedSpot;
        setTeamOne(teamOneCopy);
      } else {
        // Swap between different teams
        teamOneCopy[selectedSpot.championIndex] = targetChampion;
        teamTwoCopy[index] = championFromSelectedSpot;
        setTeamOne(teamOneCopy);
        setTeamTwo(teamTwoCopy);
      }
    } else {
      if (team === 2) {
        // Swap within the same team
        teamTwoCopy[selectedSpot.championIndex] = targetChampion;
        teamTwoCopy[index] = championFromSelectedSpot;
        setTeamTwo(teamTwoCopy);
      } else {
        // Swap between different teams
        teamTwoCopy[selectedSpot.championIndex] = targetChampion;
        teamOneCopy[index] = championFromSelectedSpot;
        setTeamOne(teamOneCopy);
        setTeamTwo(teamTwoCopy);
      }
    }

    // Reset the selected spot
    setSelectedSpot(null);
  };

  const addChampion = (champion: Champion) => {
    if (selectedSpot !== null) {
      const team = teams[selectedSpot.teamIndex - 1];
      if (!team) return; // or handle the error as appropriate

      let updatedRemainingChampions = handleDeleteChampion(
        selectedSpot.championIndex,
        team
      );

      if (selectedSpot.teamIndex === 1) {
        const newTeam = { ...teamOne };
        newTeam[selectedSpot.championIndex] = champion;
        setTeamOne(newTeam);
      } else {
        const newTeam = { ...teamTwo };
        newTeam[selectedSpot.championIndex] = champion;
        setTeamTwo(newTeam);
      }
      updatedRemainingChampions = updatedRemainingChampions.filter(
        (c) => c.id !== champion.id
      );
      setRemainingChampions(updatedRemainingChampions);

      setSelectedSpot(null);
      return;
    }
    const pickOrder = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1];

    const teamOneLength = Object.values(teamOne).filter(
      (c) => c !== undefined
    ).length;
    const teamTwoLength = Object.values(teamTwo).filter(
      (c) => c !== undefined
    ).length;
    const championsPicked = teamOneLength + teamTwoLength;
    if (championsPicked >= 10) {
      return;
    }

    const champions = remainingChampions.filter((c) => c.id !== champion.id);
    setRemainingChampions(champions);

    //add according to pick order, unless the team is full
    let teamToAddToIndex: number;
    if (pickOrder[championsPicked] === 0) {
      //team one's turn
      if (teamOneLength >= 5) {
        teamToAddToIndex = 1;
      } else {
        teamToAddToIndex = 0;
      }
    } else {
      //team two's turn
      if (teamTwoLength >= 5) {
        teamToAddToIndex = 0;
      } else {
        teamToAddToIndex = 1;
      }
    }
    let potentialRoles = championToRolesMap[champion.searchName];
    if (!potentialRoles) {
      potentialRoles = [];
    }
    const potentialRolesIndexes = potentialRoles.map(
      (role) => roleToIndexMap[role]
    );
    // add rest of indexes at the end of potentialRolesIndexes
    for (let i = 0; i < 5; i++) {
      if (!potentialRolesIndexes.includes(i)) {
        potentialRolesIndexes.push(i);
      }
    }
    // add to first available spot
    for (let i = 0; i < 5; i++) {
      const roleIndex = potentialRolesIndexes[i];
      if (teamToAddToIndex === 0 && !teamOne[roleIndex as keyof Team]) {
        setTeamOne({
          ...teams[teamToAddToIndex],
          [roleIndex as keyof Team]: champion,
        });
        return;
      } else if (teamToAddToIndex === 1 && !teamTwo[roleIndex as keyof Team]) {
        setTeamTwo({
          ...teams[teamToAddToIndex],
          [roleIndex as keyof Team]: champion,
        });
        return;
      }
    }
  };

  const handleDeleteChampion = (index: ChampionIndex, team: Team) => {
    const champion = team[index];
    if (champion === undefined) {
      return remainingChampions;
    }
    // Check if the champion is already in the remaining champions list
    const isChampionAlreadyRemaining = remainingChampions.some(
      (remainingChampion) => remainingChampion.id === champion.id
    );
    let champions;
    if (!isChampionAlreadyRemaining) {
      champions = [...remainingChampions, champion].sort((a, b) =>
        a.name.localeCompare(b.name)
      );
      setRemainingChampions(champions);
    } else {
      champions = [...remainingChampions];
    }

    if (team === teamOne) {
      const newTeam = { ...teamOne };
      delete newTeam[index];
      setTeamOne(newTeam);
    } else {
      const newTeam = { ...teamTwo };
      delete newTeam[index];
      setTeamTwo(newTeam);
    }
    return champions;
  };

  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-gradient-to-b from-gray-700 to-gray-900 text-white">
      <div className="mx-auto lg:w-2/3">
        <button
          className="m-2 rounded bg-blue-500 px-4 py-2 font-bold text-white hover:bg-blue-700"
          onClick={resetDraft}
        >
          Reset Draft
        </button>
        <button
          className="m-2 rounded bg-green-600 px-4 py-2 font-bold text-white hover:bg-green-700"
          onClick={openHelpModal}
        >
          Help
        </button>
        {showHelpModal && <HelpModal closeHandler={closeHelpModal} />}

        <div className="flex flex-wrap items-stretch justify-evenly">
          <div className="flex w-full justify-between">
            {/* Team Panel 1 */}
            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamOne}
                is_first_team={true}
                onDeleteChampion={(index: ChampionIndex) =>
                  handleDeleteChampion(index, teamOne)
                }
                selectedSpot={selectedSpot}
                onSpotSelected={handleSpotSelection}
              />
            </div>

            {/* Champion Grid */}
            <div className="grow p-1">
              <ChampionGrid
                champions={remainingChampions}
                addChampion={addChampion}
                favorites={favorites}
                setFavorites={setFavorites}
              />
            </div>

            {/* Team Panel 2 */}
            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamTwo}
                is_first_team={false}
                onDeleteChampion={(index: ChampionIndex) =>
                  handleDeleteChampion(index, teamTwo)
                }
                selectedSpot={selectedSpot}
                onSpotSelected={handleSpotSelection}
              />
            </div>
          </div>

          {/* Draft Analysis */}
          <div className="flex w-full justify-center">
            <AnalysisParent
              team1={teamOne}
              team2={teamTwo}
              selectedSpot={selectedSpot}
              favorites={favorites}
              remainingChampions={remainingChampions}
              analysisTrigger={analysisTrigger}
            />
          </div>
        </div>
      </div>
    </main>
  );
}
