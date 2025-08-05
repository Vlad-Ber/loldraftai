import { z } from "zod";
import fs from "fs";
import axios, { AxiosInstance } from "axios";
import {
  type QueueType,
  LeagueEntryDTOSchema,
  type LeagueEntryDTO,
  type TierDivisionPair,
  type Region,
  SummonerDTOSchema,
  type SummonerDTO,
  type MatchType,
  MatchDtoSchema,
  type MatchDto,
  TimelineDtoSchema,
  type TimelineDto,
} from "./schemas";
import { LeagueListDTOSchema, type LeagueListDTO } from "./internalSchemas";

const DEBUG_SAVE_REQUESTS = false;

const REGION_TO_PLATFORM_ROUTING: Record<Region, string> = {
  BR1: "AMERICAS",
  EUN1: "EUROPE",
  EUW1: "EUROPE",
  JP1: "ASIA",
  KR: "ASIA",
  LA1: "AMERICAS",
  LA2: "AMERICAS",
  ME1: "EUROPE",
  NA1: "AMERICAS",
  OC1: "SEA",
  PH2: "SEA",
  RU: "EUROPE",
  SG2: "SEA",
  TH2: "SEA",
  TR1: "EUROPE",
  TW2: "SEA",
  VN2: "SEA",
};

export class RiotAPIClient {
  private axiosInstance: AxiosInstance;
  private platformAxiosInstance: AxiosInstance;
  private platformRoutingValue: string;
  private skipValidation: boolean;
  private maxRetries: number;
  private baseDelay: number;

  constructor(
    apiKey: string,
    region: Region,
    maxRetries: number = 3,
    baseDelay: number = 10000
  ) {
    this.axiosInstance = axios.create({
      baseURL: `https://${region}.api.riotgames.com`,
      headers: {
        "X-Riot-Token": apiKey,
      },
    });
    this.platformRoutingValue = REGION_TO_PLATFORM_ROUTING[region];
    this.skipValidation = process.env.SKIP_VALIDATION === "true";
    this.maxRetries = maxRetries;
    this.baseDelay = baseDelay;
  }

  private validate<T>(schema: z.ZodSchema<T>, data: unknown): T {
    if (this.skipValidation) {
      return data as T;
    }
    return schema.parse(data);
  }

  private async withRetry<T>(
    operation: () => Promise<T>,
    retryCount: number = 0
  ): Promise<T> {
    try {
      return await operation();
    } catch (error) {
      if (!axios.isAxiosError(error)) throw error;

      // Handle rate limit (429)
      if (error.response?.status === 429) {
        const retryAfter = parseInt(
          error.response.headers["retry-after"] || "1",
          10
        );
        console.warn(
          `Rate limit exceeded. Waiting ${retryAfter} seconds before retrying...`
        );
        await new Promise((resolve) => setTimeout(resolve, retryAfter * 1000));
        return this.withRetry(operation, retryCount); // Don't increment retry count for rate limits
      }

      // Handle other retryable errors
      if (
        retryCount < this.maxRetries &&
        (error.code === "ECONNRESET" ||
          error.code === "ETIMEDOUT" ||
          (error.response?.status && error.response.status >= 500)) // Also retry on server errors
      ) {
        const delay = this.baseDelay * Math.pow(2, retryCount);
        console.warn(
          `Request failed with ${
            error.response?.status || error.code
          }, retrying in ${delay}ms... (Attempt ${retryCount + 1}/${
            this.maxRetries
          })`
        );
        await new Promise((resolve) => setTimeout(resolve, delay));
        return this.withRetry(operation, retryCount + 1);
      }
      throw error;
    }
  }

  async getLeagueEntries(
    queue: QueueType,
    tierDivision: TierDivisionPair,
    page: number = 1
  ): Promise<LeagueEntryDTO[]> {
    return this.withRetry(async () => {
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
        return this.validate(z.array(LeagueEntryDTOSchema), response.data);
      }
    });
  }

  async getSummonerById(encryptedSummonerId: string): Promise<SummonerDTO> {
    return this.withRetry(async () => {
      const response = await this.axiosInstance.get(
        `/lol/summoner/v4/summoners/${encryptedSummonerId}`
      );
      return this.validate(SummonerDTOSchema, response.data);
    });
  }

  async getMatchIdsByPuuid(
    puuid: string,
    options: {
      startTime?: number;
      endTime?: number;
      queue?: number;
      type?: MatchType;
      start?: number;
      count?: number;
    } = {}
  ): Promise<string[]> {
    return this.withRetry(async () => {
      const response = await this.axiosInstance.get(
        `https://${this.platformRoutingValue}.api.riotgames.com/lol/match/v5/matches/by-puuid/${puuid}/ids`,
        { params: options }
      );
      return this.validate(z.array(z.string()), response.data);
    });
  }

  async getMatchById(matchId: string): Promise<MatchDto> {
    return this.withRetry(async () => {
      const response = await this.axiosInstance.get(
        `https://${this.platformRoutingValue}.api.riotgames.com/lol/match/v5/matches/${matchId}`
      );
      // save to /tmp/match.json
      if (DEBUG_SAVE_REQUESTS) {
        fs.writeFileSync(
          `/tmp/match.json`,
          JSON.stringify(response.data, null, 2)
        );
      }
      return this.validate(MatchDtoSchema, response.data);
    });
  }

  async getMatchTimelineById(matchId: string): Promise<TimelineDto> {
    return this.withRetry(async () => {
      const response = await this.axiosInstance.get(
        `https://${this.platformRoutingValue}.api.riotgames.com/lol/match/v5/matches/${matchId}/timeline`
      );
      // save to /tmp/timeline.json
      if (DEBUG_SAVE_REQUESTS) {
        fs.writeFileSync(
          `/tmp/timeline.json`,
          JSON.stringify(response.data, null, 2)
        );
      }
      return this.validate(TimelineDtoSchema, response.data);
    });
  }

  private mapLeagueItemsToLeagueEntries(
    leagueList: LeagueListDTO
  ): LeagueEntryDTO[] {
    return leagueList.entries.map((entry) => ({
      leagueId: leagueList.leagueId,
      puuid: entry.puuid,
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
