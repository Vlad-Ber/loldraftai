import { useState, useEffect, useMemo } from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { TeamPanel } from "@draftking/ui/components/draftking/TeamPanel";
import { ChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import { AnalysisParent } from "./components/AnalysisParent";
import { PlainImage } from "./components/PlainImage";
import { HelpModal } from "@draftking/ui/components/draftking/HelpModal";
import { champions, roleToIndexMap } from "@draftking/ui/lib/champions";
import { useDraftStore } from "./stores/draftStore";
import type {
  Team,
  SelectedSpot,
  ChampionIndex,
  TeamIndex,
  Champion,
  FavoriteChampions,
} from "@draftking/ui/lib/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@draftking/ui/components/ui/select";
import {
  emptyTeam,
  DRAFT_ORDERS,
  handleSpotSelection,
  addChampion as addChampionLogic,
  handleDeleteChampion as handleDeleteChampionLogic,
  type DraftOrderKey,
} from "@draftking/ui/lib/draftLogic";
import { getChampionRoles } from "@draftking/ui/lib/champions";
import { StatusMessage } from "@draftking/ui/components/draftking/StatusMessage";
import { useToast } from "@draftking/ui/hooks/use-toast";
import { usePersistedState } from "@draftking/ui/hooks/usePersistedState";

function App() {
  // Draft state
  const [teamOne, setTeamOne] = useState<Team>(emptyTeam);
  const [teamTwo, setTeamTwo] = useState<Team>(emptyTeam);
  const [selectedSpot, setSelectedSpot] = useState<SelectedSpot | null>(null);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [favorites, setFavorites] = usePersistedState<FavoriteChampions>(
    "favorites",
    {
      top: [],
      jungle: [],
      mid: [],
      bot: [],
      support: [],
    }
  );
  const [selectedDraftOrder, setSelectedDraftOrder] =
    useState<DraftOrderKey>("Draft Order");
  const [remainingChampions, setRemainingChampions] =
    useState<Champion[]>(champions);
  const [isLiveTracking, setIsLiveTracking] = useState(false);
  const [bannedChampions, setBannedChampions] = useState<Champion[]>([]);

  // Store
  const { currentPatch, patches, setCurrentPatch, setPatchList } =
    useDraftStore();

  const { toast } = useToast();

  // Handlers
  const handleDeleteChampion = (index: ChampionIndex, team: Team) => {
    return handleDeleteChampionLogic(
      index,
      team,
      teamOne,
      teamTwo,
      remainingChampions,
      setTeamOne,
      setTeamTwo,
      setRemainingChampions
    );
  };

  const handleSpotSelected = (index: ChampionIndex, team: TeamIndex) => {
    handleSpotSelection(
      index,
      team,
      selectedSpot,
      teamOne,
      teamTwo,
      setTeamOne,
      setTeamTwo,
      setSelectedSpot
    );
  };

  const handleAddChampion = (champion: Champion) => {
    addChampionLogic(
      champion,
      selectedSpot,
      teamOne,
      teamTwo,
      remainingChampions,
      currentPatch,
      selectedDraftOrder,
      setTeamOne,
      setTeamTwo,
      setRemainingChampions,
      setSelectedSpot,
      handleDeleteChampion
    );
  };

  const resetDraft = () => {
    setTeamOne(emptyTeam);
    setTeamTwo(emptyTeam);
    setSelectedSpot(null);
    setBannedChampions([]);
    setRemainingChampions(champions);
  };

  useEffect(() => {
    // Listen for update notifications
    window.electronAPI.onUpdateNotification((info) => {
      toast({
        title: info.title,
        description: info.body,
      });
    });
  }, [toast]);

  useEffect(() => {
    if (!isLiveTracking) return;

    let intervalId: NodeJS.Timeout;

    async function updateFromLiveGame() {
      try {
        const champSelect = await window.electronAPI.getChampSelect();
        if (!champSelect) {
          setIsLiveTracking(false);
          toast({
            title: "Live Tracking Stopped",
            description: "No live draft lobby found",
          });
          return;
        }

        // Process bans
        const bannedActions = champSelect.actions
          .flat()
          .filter(
            (action: any) => action.type === "ban" && action.completed === true
          );

        const newBannedChampions = bannedActions
          .map((action: any) =>
            champions.find((c) => c.id === action.championId)
          )
          .filter(
            (champion: any): champion is Champion => champion !== undefined
          );

        // Update bans if changed
        setBannedChampions((current) => {
          const currentIds = new Set(current.map((c) => c.id));
          const newIds = new Set(newBannedChampions.map((c: any) => c.id));

          if (
            currentIds.size !== newIds.size ||
            newBannedChampions.some((c: any) => !currentIds.has(c.id))
          ) {
            return newBannedChampions;
          }
          return current;
        });

        // Process picks (existing code)
        const completedActions = champSelect.actions
          .flat()
          .filter(
            (action: any) => action.type === "pick" && action.completed === true
          );

        // Process all players
        const allPlayers = [...champSelect.myTeam, ...champSelect.theirTeam];

        for (const player of allPlayers) {
          // Find if this player has a completed pick action
          const completedAction = completedActions.find(
            (action: any) => action.actorCellId === player.cellId
          );

          // Skip if no completed action found or no champion selected
          if (!completedAction || completedAction.championId === 0) continue;

          const champion = champions.find(
            (c) => c.id === completedAction.championId
          );
          if (!champion) continue;

          const targetTeam = player.team === 1 ? teamOne : teamTwo;
          const setTeamFn = player.team === 1 ? setTeamOne : setTeamTwo;

          // Skip if champion is already in team
          const isAlreadyInTeam = Object.values(targetTeam).some(
            (c) => c && c.id === champion.id
          );
          if (isAlreadyInTeam) continue;

          // Get potential roles based on play rates
          const potentialRoles = getChampionRoles(champion.id, currentPatch);
          const potentialRolesIndexes = potentialRoles.map(
            (role) => roleToIndexMap[role]
          );

          // Add any remaining roles at the end
          for (let i = 0; i < 5; i++) {
            if (!potentialRolesIndexes.includes(i)) {
              potentialRolesIndexes.push(i);
            }
          }

          // Try to place champion in their most played role first
          for (const roleIndex of potentialRolesIndexes) {
            if (!targetTeam[roleIndex as keyof Team]) {
              setTeamFn({
                ...targetTeam,
                [roleIndex]: champion,
              });
              break;
            }
          }
        }

        // Check if draft is complete
        if (champSelect.timer.phase === "GAME_STARTING") {
          setIsLiveTracking(false);
        }
      } catch (error) {
        console.error("Error updating from live game:", error);
        setIsLiveTracking(false);
      }
    }

    void updateFromLiveGame();
    intervalId = setInterval(updateFromLiveGame, 1000);

    return () => {
      clearInterval(intervalId);
      setBannedChampions([]);
    };
  }, [isLiveTracking]);

  const toggleLiveTracking = () => {
    if (!isLiveTracking) {
      resetDraft();
    }
    setIsLiveTracking(!isLiveTracking);
  };

  const remainingNonBannedChampions = useMemo(() => {
    const bannedIds = new Set(bannedChampions.map((c) => c.id));
    return champions.filter((c) => !bannedIds.has(c.id));
  }, [bannedChampions]);

  return (
    <div className="container mx-auto mt-12">
      <div className="mx-auto">
        <div className="flex flex-wrap items-stretch justify-start mb-4">
          <div className="flex w-full p-1 sm:w-auto">
            <div className="flex-1">
              <Button variant="outline" onClick={resetDraft}>
                Reset Draft
              </Button>
            </div>
          </div>
          <div className="flex w-full p-1 sm:w-auto">
            <div className="flex-1">
              <Select
                value={selectedDraftOrder}
                onValueChange={(value: DraftOrderKey) =>
                  setSelectedDraftOrder(value)
                }
              >
                <SelectTrigger className="w-[180px]">
                  <SelectValue placeholder="Select draft order" />
                </SelectTrigger>
                <SelectContent>
                  {Object.keys(DRAFT_ORDERS).map((order) => (
                    <SelectItem key={order} value={order}>
                      {order}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex w-full p-1 sm:w-auto">
            <div className="flex-1">
              <Button variant="outline" onClick={() => setShowHelpModal(true)}>
                Help
              </Button>
            </div>
          </div>
          <div className="flex w-full p-1 sm:w-auto">
            <div className="flex-1">
              <Button
                variant="outline"
                onClick={toggleLiveTracking}
                className="inline-flex items-center gap-2"
              >
                <span className="relative flex h-2 w-2">
                  {isLiveTracking && (
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
                  )}
                  <span
                    className={`relative inline-flex h-2 w-2 rounded-full ${
                      isLiveTracking ? "bg-red-500" : "bg-gray-200"
                    }`}
                  />
                </span>
                {isLiveTracking ? "Stop Live Tracking" : "Start Live Tracking"}
              </Button>
            </div>
          </div>
        </div>

        <HelpModal
          isOpen={showHelpModal}
          closeHandler={() => setShowHelpModal(false)}
        />

        <div className="text-center text-lg font-semibold mb-4">
          <StatusMessage
            selectedSpot={selectedSpot}
            teamOne={teamOne}
            teamTwo={teamTwo}
            selectedDraftOrder={selectedDraftOrder}
          />
        </div>

        <div className="flex flex-wrap items-stretch justify-evenly">
          <div className="flex w-full justify-between">
            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamOne}
                is_first_team={true}
                selectedSpot={selectedSpot}
                onDeleteChampion={(index) =>
                  handleDeleteChampion(index, teamOne)
                }
                onSpotSelected={handleSpotSelected}
                ImageComponent={PlainImage}
              />
            </div>

            <div className="grow p-1">
              <ChampionGrid
                champions={remainingNonBannedChampions}
                addChampion={handleAddChampion}
                favorites={favorites}
                setFavorites={setFavorites}
                ImageComponent={PlainImage}
              />
            </div>

            <div className="flex w-auto max-w-xs p-1">
              <TeamPanel
                team={teamTwo}
                is_first_team={false}
                selectedSpot={selectedSpot}
                onDeleteChampion={(index) =>
                  handleDeleteChampion(index, teamTwo)
                }
                onSpotSelected={handleSpotSelected}
                ImageComponent={PlainImage}
              />
            </div>
          </div>
        </div>

        <div className="mt-4">
          <AnalysisParent
            team1={teamOne}
            team2={teamTwo}
            selectedSpot={selectedSpot}
            favorites={favorites}
            remainingChampions={remainingNonBannedChampions}
            analysisTrigger={0}
            currentPatch={currentPatch}
            patches={patches}
            setCurrentPatch={setCurrentPatch}
            setPatchList={setPatchList}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
