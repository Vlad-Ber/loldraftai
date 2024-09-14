import { z } from "zod";
import axios, { AxiosInstance } from "axios";
import { QueueTypeSchema, LeagueListDTOSchema } from "./schemas";

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

  async getChallengerLeague(queue: z.infer<typeof QueueTypeSchema>) {
    const response = await this.axiosInstance.get(
      `/lol/league/v4/challengerleagues/by-queue/${queue}`
    );
    return LeagueListDTOSchema.parse(response.data);
  }
}
