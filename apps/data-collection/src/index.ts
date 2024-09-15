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

client.getLeagueEntries("RANKED_SOLO_5x5", "DIAMOND", "I");
