import { RiotAPIClient } from "@draftking/riot-api";
import { config } from "dotenv";

config();

const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const client = new RiotAPIClient(apiKey);

// client.getChallengerLeague("RANKED_SOLO_5x5");

// client.getGrandmasterLeague("RANKED_SOLO_5x5");

// client.getMasterLeague("RANKED_SOLO_5x5");

await client.getLeagueEntries("RANKED_SOLO_5x5", ["DIAMOND", "I"], 1);

const response = await client.getLeagueEntries("RANKED_SOLO_5x5", ["CHALLENGER", "I"], 1);

if (!response[0]) {
  throw new Error("No summoner found");
}
const summonedId = response[0].summonerId;

const summoner = await client.getSummonerById(summonedId);

console.log(summoner);
