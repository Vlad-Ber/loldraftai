import yargs from "yargs";
import axios from "axios";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RegionSchema } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient, Summoner } from "@draftking/riot-database";
import { config } from "dotenv";
import { telemetry } from "../utils/telemetry";

config();

const argv = await yargs(hideBin(process.argv))
  .option("region", {
    type: "string",
    demandOption: true,
    describe: "The region to fetch PUUIDs for",
  })
  .parse();

const region = RegionSchema.parse(argv.region);

const apiKey = process.env.X_RIOT_API_KEY;

if (!apiKey) {
  throw new Error("X_RIOT_API_KEY is not set");
}

const riotApiClient = new RiotAPIClient(apiKey);
const prisma = new PrismaClient();

// We do the same limiter as processMatches.ts as there is not point going faster
// Except we get 100 matches per player, so we can go even slower
const limiter = new Bottleneck({
  minTime: 500, // 20ms between requests (50 requests per second)
  // Limit: 2000 requests every 10 seconds
  reservoir: 250,
  reservoirRefreshAmount: 250,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 50,
});

async function collectMatchIds() {
  try {
    console.log(`[${new Date().toISOString()}] Starting collectMatchIds for region: ${region}`);
    
    while (true) {
      console.log(`[${new Date().toISOString()}] Fetching next batch of summoners...`);
      const summoners = (await prisma.$queryRaw`
        SELECT *
        FROM "Summoner"
        WHERE puuid IS NOT NULL
        AND region = ${region}::text::"Region"
        AND (
          "matchesFetchedAt" IS NULL -- Older than 3 days, because we get 100 games
          OR "matchesFetchedAt" < ${new Date(
            Date.now() - 3 * 24 * 60 * 60 * 1000
          )}
        )
        AND "rankUpdateTime" > ${new Date(
          Date.now() - 7 * 24 * 60 * 60 * 1000
        )}
        AND "matchFetchErrored" = false
        LIMIT 100
      `) as Summoner[];

      console.log(`[${new Date().toISOString()}] Found ${summoners.length} summoners to process`);

      if (summoners.length === 0) {
        console.log(`[${new Date().toISOString()}] No summoners found, waiting 60 seconds...`);
        await sleep(60 * 1000);
        continue;
      }

      for (const summoner of summoners) {
        console.log(`[${new Date().toISOString()}] Processing summoner: ${summoner.summonerId}`);
        
        await limiter.schedule(async () => {
          try {
            console.log(`[${new Date().toISOString()}] Fetching matches for summoner: ${summoner.summonerId}`);
            const matchIds = await riotApiClient.getMatchIdsByPuuid(
              summoner.puuid!,
              {
                type: "ranked",
                queue: 420,
                count: 100,
              }
            );
            console.log(`[${new Date().toISOString()}] Found ${matchIds.length} matches for summoner: ${summoner.summonerId}`);

            const matchCreates = matchIds.map((matchId) => ({
              matchId,
              region,
              processed: false,
              averageTier: summoner.tier,
              averageDivision: summoner.rank,
            }));

            const { count } = await prisma.match.createMany({
              data: matchCreates,
              skipDuplicates: true,
            });
            console.log(`[${new Date().toISOString()}] Created ${count} new matches for summoner: ${summoner.summonerId}`);

            telemetry.trackEvent("MatchesCollected", {
              count: matchIds.length,
              region,
            });
            telemetry.trackEvent("NewMatchesCreated", {
              count,
              region,
            });

            await prisma.summoner.update({
              where: {
                id: summoner.id,
              },
              data: {
                matchesFetchedAt: new Date(),
              },
            });
            console.log(`[${new Date().toISOString()}] Updated matchesFetchedAt for summoner: ${summoner.summonerId}`);

          } catch (error: any) {
            if (axios.isAxiosError(error) && error.response?.status === 400) {
              console.log(`[${new Date().toISOString()}] 400 error for summoner ${summoner.summonerId}, marking as errored`);
              await prisma.summoner.update({
                where: { id: summoner.id },
                data: {
                  matchFetchErrored: true,
                  matchesFetchedAt: new Date(),
                },
              });
            } else {
              console.error(
                `[${new Date().toISOString()}] Error fetching match IDs for summoner ${summoner.summonerId}:`,
                error
              );
            }
          }
        });
      }
    }
  } catch (error) {
    console.error(`[${new Date().toISOString()}] Fatal error in collectMatchIds:`, error);
  } finally {
    await prisma.$disconnect();
  }
}

collectMatchIds();
