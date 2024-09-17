import {
  RiotAPIClient,
  MatchDto,
  TimelineDto,
  ParticipantFrameDto,
  ObjectivesDto,
  ChampionKillEventSchema,
  BuildingKillEventSchema,
  EliteMonsterKillEventDto,
  EliteMonsterKillEventSchema,
  EventsTimeLineDto,
  ChampionKillEventDto,
  BuildingKillEventDto,
  FramesTimeLineDto,
  ParticipantIdSchema,
  ParticipantId,
  TeamId,
} from "@draftking/riot-api";

interface ProcessedParticipantData {
  championId: number;
  level: number;
  kills: number;
  deaths: number;
  assists: number;
  creepScore: number;
  totalGold: number;
  damageStats: {
    magicDamageDoneToChampions: number;
    physicalDamageDoneToChampions: number;
    trueDamageDoneToChampions: number;
  };
}

interface ProcessedTeamStats {
  totalKills: number;
  totalDeaths: number;
  totalAssists: number;
  totalGold: number;
  towerKills: number;
  inhibitorKills: number;
  baronKills: number;
  dragonKills: number;
  riftHeraldKills: number;
}

interface ProcessedTimelineData {
  participants: Record<ParticipantId, ProcessedParticipantData>;
  teamStats: Record<TeamId, ProcessedTeamStats>;
}

interface ProcessedMatchData {
  gameId: number;
  gameDuration: number;
  gameVersion: string;
  queueId: number;
  teams: Record<
    TeamId,
    {
      win: boolean;
      objectives: ObjectivesDto;
    }
  >;
  timeline: Record<number, ProcessedTimelineData>;
}

type ParticipantStats = Record<
  ParticipantId,
  {
    championId: number;
    teamId: TeamId;
    kills: number;
    deaths: number;
    assists: number;
  }
>;

function initializeProcessedData(matchData: MatchDto): ProcessedMatchData {
  const processedData = {
    gameId: matchData.info.gameId,
    gameDuration: matchData.info.gameDuration,
    gameVersion: matchData.info.gameVersion,
    queueId: matchData.info.queueId,
    teams: {
      100: {
        win: matchData.info.teams[0].win,
        objectives: matchData.info.teams[0].objectives,
      },
      200: {
        win: matchData.info.teams[1].win,
        objectives: matchData.info.teams[1].objectives,
      },
    },
    timeline: {},
  };

  matchData.info.teams.forEach((team) => {
    processedData.teams[team.teamId] = {
      win: team.win,
      objectives: team.objectives,
    };
  });

  return processedData;
}

function initializeParticipantStats(matchData: MatchDto): ParticipantStats {
  const participantStats = {} as ParticipantStats;
  matchData.info.participants.forEach((participant) => {
    participantStats[participant.participantId] = {
      championId: participant.championId,
      teamId: participant.teamId,
      kills: 0,
      deaths: 0,
      assists: 0,
    };
  });
  return participantStats;
}

function initializeTeamStats(): Record<TeamId, ProcessedTeamStats> {
  return {
    100: {
      totalKills: 0,
      totalDeaths: 0,
      totalAssists: 0,
      totalGold: 0,
      towerKills: 0,
      inhibitorKills: 0,
      baronKills: 0,
      dragonKills: 0,
      riftHeraldKills: 0,
    },
    200: {
      totalKills: 0,
      totalDeaths: 0,
      totalAssists: 0,
      totalGold: 0,
      towerKills: 0,
      inhibitorKills: 0,
      baronKills: 0,
      dragonKills: 0,
      riftHeraldKills: 0,
    },
  };
}

function updateChampionKillStats(
  event: ChampionKillEventDto,
  participantStats: ParticipantStats,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  if (participantStats[event.killerId]) {
    participantStats[event.killerId].kills++;
    teamStats[participantStats[event.killerId].teamId].totalKills++;
  }
  if (participantStats[event.victimId]) {
    participantStats[event.victimId].deaths++;
    teamStats[participantStats[event.victimId].teamId].totalDeaths++;
  }
  event.assistingParticipantIds?.forEach((assistId) => {
    if (participantStats[assistId]) {
      participantStats[assistId].assists++;
      teamStats[participantStats[assistId].teamId].totalAssists++;
    }
  });
}

function updateBuildingKillStats(
  event: BuildingKillEventDto,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  if (event.buildingType === "TOWER_BUILDING") {
    teamStats[event.teamId].towerKills++;
  } else if (event.buildingType === "INHIBITOR_BUILDING") {
    teamStats[event.teamId].inhibitorKills++;
  }
}

function updateEliteMonsterKillStats(
  event: EliteMonsterKillEventDto,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  switch (event.monsterType) {
    case "BARON_NASHOR":
      teamStats[event.killerTeamId].baronKills++;
      break;
    case "DRAGON":
      teamStats[event.killerTeamId].dragonKills++;
      break;
    case "RIFTHERALD":
      teamStats[event.killerTeamId].riftHeraldKills++;
      break;
  }
}

function processEvents(
  events: EventsTimeLineDto[],
  participantStats: ParticipantStats,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  events.forEach((event) => {
    switch (event.type) {
      case "CHAMPION_KILL":
        updateChampionKillStats(
          ChampionKillEventSchema.parse(event),
          participantStats,
          teamStats
        );
        break;
      case "BUILDING_KILL":
        updateBuildingKillStats(
          BuildingKillEventSchema.parse(event),
          teamStats
        );
        break;
      case "ELITE_MONSTER_KILL":
        updateEliteMonsterKillStats(
          EliteMonsterKillEventSchema.parse(event),
          teamStats
        );
        break;
    }
  });
}

function processTimelineFrame(
  frame: FramesTimeLineDto,
  timestamp: number,
  processedData: ProcessedMatchData,
  participantStats: ParticipantStats,
  teamStats: Record<TeamId, ProcessedTeamStats>
) {
  processedData.timeline[timestamp] = {
    participants: {} as Record<ParticipantId, ProcessedParticipantData>,
    teamStats: JSON.parse(JSON.stringify(teamStats)), // Deep copy current team stats
  };

  Object.entries(frame.participantFrames).forEach(
    ([participantId, participantFrame]: [string, ParticipantFrameDto]) => {
      const pId = ParticipantIdSchema.parse(parseInt(participantId));
      // We know timestamp exists because we set it just before
      processedData.timeline[timestamp]!.participants[pId] = {
        championId: participantStats[pId].championId,
        level: participantFrame.level,
        kills: participantStats[pId].kills,
        deaths: participantStats[pId].deaths,
        assists: participantStats[pId].assists,
        creepScore:
          participantFrame.minionsKilled + participantFrame.jungleMinionsKilled,
        totalGold: participantFrame.totalGold,
        damageStats: {
          magicDamageDoneToChampions:
            participantFrame.damageStats.magicDamageDoneToChampions,
          physicalDamageDoneToChampions:
            participantFrame.damageStats.physicalDamageDoneToChampions,
          trueDamageDoneToChampions:
            participantFrame.damageStats.trueDamageDoneToChampions,
        },
      };

      // Update team gold
      const teamId = participantStats[pId].teamId;
      // We know timestamp exists because we set it just before
      processedData.timeline[timestamp]!.teamStats[teamId].totalGold +=
        participantFrame.totalGold;
    }
  );
}

async function processMatchData(
  client: RiotAPIClient,
  matchId: string
): Promise<ProcessedMatchData> {
  const matchData: MatchDto = await client.getMatchById(matchId);
  const timelineData: TimelineDto = await client.getMatchTimelineById(matchId);

  const processedData = initializeProcessedData(matchData);
  const participantStats = initializeParticipantStats(matchData);
  const teamStats = initializeTeamStats();

  const relevantTimestamps = [
    15 * 60 * 1000,
    20 * 60 * 1000,
    25 * 60 * 1000,
    30 * 60 * 1000,
  ];

  timelineData.info.frames.forEach((frame) => {
    if (relevantTimestamps.includes(frame.timestamp)) {
      processTimelineFrame(
        frame,
        frame.timestamp,
        processedData,
        participantStats,
        teamStats
      );
    }

    processEvents(frame.events, participantStats, teamStats);
  });

  return processedData;
}

export { processMatchData, type ProcessedMatchData };
