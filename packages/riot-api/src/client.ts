import { z } from "zod";
import axios, { AxiosInstance } from "axios";
import {
  type QueueType,
  LeagueEntryDTOSchema,
  type LeagueEntryDTO,
  type TierDivisionPair,
  type Region,
} from "./schemas";
import { LeagueListDTOSchema, type LeagueListDTO } from "./internalSchemas";

export class RiotAPIClient {
  private axiosInstance: AxiosInstance;

  constructor(apiKey: string, region: Region = "EUW1") {
    this.axiosInstance = axios.create({
      baseURL: `https://${region}.api.riotgames.com`,
      headers: {
        "X-Riot-Token": apiKey,
      },
    });
  }

  async getLeagueEntries(
    queue: QueueType,
    tierDivision: TierDivisionPair,
    page: number = 1
  ): Promise<LeagueEntryDTO[]> {
    const [tier, division] = tierDivision;

    // The riot api is inconsistent, it has different endpoints for master+ leagues
    // We internally map the answers to have a more consistent api
    if (["CHALLENGER", "GRANDMASTER", "MASTER"].includes(tier)) {
      if (page !== 1) return []; // Return empty array for pages > 1 for these tiers
      const response = await this.axiosInstance.get(
        `/lol/league/v4/${tier.toLowerCase()}leagues/by-queue/${queue}`
      );
      const leagueList = LeagueListDTOSchema.parse(response.data);
      return this.mapLeagueItemsToLeagueEntries(leagueList);
    } else {
      const response = await this.axiosInstance.get(
        `/lol/league/v4/entries/${queue}/${tier}/${division}`,
        {
          params: { page },
        }
      );
      return z.array(LeagueEntryDTOSchema).parse(response.data);
    }
  }

  private mapLeagueItemsToLeagueEntries(
    leagueList: LeagueListDTO
  ): LeagueEntryDTO[] {
    return leagueList.entries.map((entry) => ({
      leagueId: leagueList.leagueId,
      summonerId: entry.summonerId,
      queueType: leagueList.queue,
      tier: leagueList.tier,
      rank: entry.rank,
      leaguePoints: entry.leaguePoints,
      wins: entry.wins,
      losses: entry.losses,
      hotStreak: entry.hotStreak,
      veteran: entry.veteran,
      freshBlood: entry.freshBlood,
      inactive: entry.inactive,
      miniSeries: entry.miniSeries,
    }));
  }
}
