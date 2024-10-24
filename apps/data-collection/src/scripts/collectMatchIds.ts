import yargs from "yargs";
import axios from "axios";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RegionSchema } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient } from "@draftking/riot-database";
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
  highWater: 1000,
  strategy: Bottleneck.strategy.BLOCK,
});

async function collectMatchIds() {
  try {
    while (true) {
      // Fetch summoners with a PUUID and outdated matchesFetchedAt
      const summoners = await prisma.summoner.findMany({
        where: {
          puuid: {
            not: null,
          },
          region: region,
          OR: [
            { matchesFetchedAt: null },
            {
              matchesFetchedAt: {
                lt: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000), // Older than 3 days, because we get 100 games
              },
            },
          ],
          // Updated less than 1 week ago (up to date)
          rankUpdateTime: {
            gt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
          },
          // Not errored
          matchFetchErrored: false,
        },
        take: 100, // Not too many to avoid spikes in db usage
      });

      if (summoners.length === 0) {
        console.log(
          "No summoners to fetch match IDs for. Sleeping for 1 minute."
        );
        await sleep(60 * 1000);
        continue;
      }

      for (const summoner of summoners) {
        await limiter.schedule(async () => {
          try {
            const matchIds = await riotApiClient.getMatchIdsByPuuid(
              summoner.puuid!,
              {
                type: "ranked",
                queue: 420, // Ranked Solo/Duo queue
                count: 100, // max count
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

            // Perform batch create, skipping duplicates
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

            // Update the summoner's matchesFetchedAt timestamp
            await prisma.summoner.update({
              where: {
                id: summoner.id,
              },
              data: {
                matchesFetchedAt: new Date(),
              },
            });
          } catch (error: any) {
            if (axios.isAxiosError(error) && error.response?.status === 400) {
              // Mark summoner as having match fetch errors
              await prisma.summoner.update({
                where: { id: summoner.id },
                data: {
                  matchFetchErrored: true,
                  matchesFetchedAt: new Date(),
                },
              });
              // TODO: could track error as event
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
