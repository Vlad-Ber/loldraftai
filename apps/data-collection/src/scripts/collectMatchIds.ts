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
import {
  DatabaseBackoff,
  LoggerFunction,
} from "../utils/databaseErrorHandling";

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

const riotApiClient = new RiotAPIClient(apiKey, region);
const prisma = new PrismaClient();

// We do the same limiter as processMatches.ts as there is not point going faster
// Except we get 100 matches per player, so we can go even slower
// here it is around 2 times slower than processMatches.ts
const limiter = new Bottleneck({
  minTime: 400,
  reservoir: 25,
  reservoirRefreshAmount: 25,
  reservoirRefreshInterval: 10 * 1000,
  maxConcurrent: 5,
});

const dbBackoff = new DatabaseBackoff();

// Add a simple logger function
const log: LoggerFunction = (level, message) => {
  console.log(`[${new Date().toISOString()}] [${level}] ${message}`);
};

async function collectMatchIds() {
  try {
    while (true) {
      // Fetch summoners with retry
      const summoners = await dbBackoff.withRetry(async () => {
        return prisma.$queryRaw`
          SELECT * FROM "Summoner"
          WHERE puuid IS NOT NULL
          AND region = ${region}::text::"Region"
          AND (
            "matchesFetchedAt" IS NULL
            OR "matchesFetchedAt" < ${new Date(
              Date.now() - 1 * 24 * 60 * 60 * 1000
            )}
          )
          AND "rankUpdateTime" > ${new Date(
            Date.now() - 7 * 24 * 60 * 60 * 1000
          )}
          AND "matchFetchErrored" = false
          LIMIT 25
        ` as Promise<Summoner[]>;
      });

      if (summoners.length === 0) {
        await sleep(60 * 1000);
        continue;
      }

      for (const summoner of summoners) {
        await limiter.schedule(async () => {
          try {
            const matchIds = await riotApiClient.getMatchIdsByPuuid(
              summoner.puuid!,
              {
                // queue: 700, // Summoner's rift clash, 420, // Ranked Solo/Duo queue
                queue: 420, // TODO: automatically collect both queues, instead of manually changing this
                // Less games for lower elos(to avoid having to many low elo games)
                count: ["PLATINUM", "GOLD", "SILVER"].includes(summoner.tier)
                  ? 50
                  : 100, // max count
              }
            );

            // Prepare batch create data
            const matchCreates = matchIds.map((matchId) => ({
              matchId,
              region,
              processed: false,
              averageTier: summoner.tier,
              averageDivision: summoner.rank,
            }));

            // Wrap database operations in retry logic
            await dbBackoff.withRetry(async () => {
              const { count } = await prisma.match.createMany({
                data: matchCreates,
                skipDuplicates: true,
              });

              telemetry.trackEvent("MatchesCollected", {
                count: matchIds.length,
                region,
              });
              telemetry.trackEvent("NewMatchesCreated", {
                count,
                region,
              });

              await prisma.summoner.update({
                where: { id: summoner.id },
                data: { matchesFetchedAt: new Date() },
              });
            });
          } catch (error: any) {
            if (axios.isAxiosError(error) && error.response?.status === 400) {
              await dbBackoff.withRetry(async () => {
                await prisma.summoner.update({
                  where: { id: summoner.id },
                  data: {
                    matchFetchErrored: true,
                    matchesFetchedAt: new Date(),
                  },
                });
              });
            } else {
              // Log other errors but don't mark as errored (could be temporary API issues)
              console.error(
                `Error fetching match IDs for summoner ${summoner.summonerId}:`,
                error
              );
            }
          }
        });
      }
    }
  } catch (error) {
    console.error("Error in collectMatchIds:", error);
  } finally {
    await prisma.$disconnect();
  }
}

collectMatchIds();
