import { RiotAPIClient } from "@draftking/riot-api";
import { config } from "dotenv";

config();

const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const client = new RiotAPIClient(apiKey);

await client.getLeagueEntries("RANKED_SOLO_5x5", ["DIAMOND", "I"], 1);

const response = await client.getLeagueEntries(
  "RANKED_SOLO_5x5",
  ["CHALLENGER", "I"],
  1
);

if (!response[0]) {
  throw new Error("No summoner found");
}

const summoner = await client.getSummonerById(response[0].summonerId);

const matchIds = await client.getMatchIdsByPuuid(summoner.puuid, {
  type: "ranked",
});

if (!matchIds[0]) {
  throw new Error("No match ids found");
}

const match = await client.getMatchById(matchIds[0]);

console.log(match);

const timeline = await client.getMatchTimelineById(matchIds[0]);

console.log(timeline);
