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

const PATCH_14_START = new Date("2024-01-01").getTime() / 1000; // Convert to seconds for Riot API
const MAX_MATCHES_HIGH_ELO = 800;
const MAX_MATCHES_LOW_ELO = 10;

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
              Date.now() - 2 * 24 * 60 * 60 * 1000
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
        try {
          const isHighElo = ["CHALLENGER", "GRANDMASTER", "MASTER"].includes(
            summoner.tier
          );
          const batchSize = isHighElo ? 100 : MAX_MATCHES_LOW_ELO; // Optimize batch size based on elo
          let allMatchIds: string[] = [];
          let start = 0;

          // For high elo, we'll fetch multiple pages
          const maxPages = isHighElo
            ? Math.ceil(MAX_MATCHES_HIGH_ELO / 100)
            : 1;

          for (let page = 0; page < maxPages; page++) {
            // Move limiter to wrap just the API call
            const matchIds = await limiter.schedule(() =>
              riotApiClient.getMatchIdsByPuuid(summoner.puuid!, {
                queue: 420, // Ranked Solo/Duo queue
                startTime: PATCH_14_START,
                start: start,
                count: batchSize,
              })
            );

            // If we got less than requested matches, we've reached the end
            if (matchIds.length < batchSize) {
              allMatchIds.push(...matchIds);
              break;
            }

            allMatchIds.push(...matchIds);

            // Only continue pagination for high elo
            if (isHighElo) {
              start += batchSize;
            }
          }

          // Prepare batch create data
          const matchCreates = allMatchIds.map((matchId) => ({
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
              count: allMatchIds.length,
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
            console.error(
              `Error fetching match IDs for summoner ${summoner.summonerId}:`,
              error
            );
          }
        }
      }
    }
  } catch (error) {
    console.error("Error in collectMatchIds:", error);
  } finally {
    await prisma.$disconnect();
  }
}

collectMatchIds();
