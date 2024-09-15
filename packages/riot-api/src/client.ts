import { z } from "zod";
import axios, { AxiosInstance } from "axios";
import {
  type QueueType,
  LeagueListDTOSchema,
  type LeagueListDTO,
  LeagueEntryDTOSchema,
  type LeagueEntryDTO,
  type Tier,
  type Division,
} from "./schemas";

export class RiotAPIClient {
  private axiosInstance: AxiosInstance;

  constructor(apiKey: string, region: string = "euw1") {
    this.axiosInstance = axios.create({
      baseURL: `https://${region}.api.riotgames.com`,
      headers: {
        "X-Riot-Token": apiKey,
      },
    });
  }

  private async getLeagueByQueue(
    tier: string,
    queue: QueueType
  ): Promise<LeagueListDTO> {
    const response = await this.axiosInstance.get(
      `/lol/league/v4/${tier}leagues/by-queue/${queue}`
    );
    return LeagueListDTOSchema.parse(response.data);
  }

  async getChallengerLeague(queue: QueueType): Promise<LeagueListDTO> {
    return this.getLeagueByQueue("challenger", queue);
  }

  async getGrandmasterLeague(queue: QueueType): Promise<LeagueListDTO> {
    return this.getLeagueByQueue("grandmaster", queue);
  }

  async getMasterLeague(queue: QueueType): Promise<LeagueListDTO> {
    return this.getLeagueByQueue("master", queue);
  }

  async getLeagueEntries(
    queue: QueueType,
    tier: Tier,
    division: Division,
    page: number = 1
  ): Promise<LeagueEntryDTO[]> {
    const response = await this.axiosInstance.get(
      `/lol/league/v4/entries/${queue}/${tier}/${division}`,
      {
        params: { page },
      }
    );
    return z.array(LeagueEntryDTOSchema).parse(response.data);
  }
}
