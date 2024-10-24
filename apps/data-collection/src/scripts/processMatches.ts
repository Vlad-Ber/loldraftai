import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RegionSchema } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient } from "@draftking/riot-database";
import { config } from "dotenv";
import { processMatchData } from "../utils/matchProcessing";
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

// Rate limiter settings based on the API rate limits
const limiter = new Bottleneck({
  minTime: 100, // 20ms between requests (50 requests per second)
  // Limit: 2000 requests every 10 seconds
  reservoir: 500,
  reservoirRefreshAmount: 500,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 50,
  highWater: 1000, // Add this line
  strategy: Bottleneck.strategy.BLOCK, // Add this line
});

async function processMatches() {
  try {
    while (true) {
      // Fetch unprocessed matches
      const matches = await prisma.match.findMany({
        where: {
          processed: false,
          processingErrored: false,
          region: region,
        },
        take: 100, // Not too many to avoid spikes in db usage
      });

      if (matches.length === 0) {
        console.log("No unprocessed matches found. Sleeping for 1 minute.");
        await sleep(60 * 1000);
        continue;
      }

      console.log(`Processing ${matches.length} matches.`);

      // Replace Promise.all with a for...of loop
      for (const match of matches) {
        await limiter.schedule(async () => {
          try {
            const processedData = await processMatchData(
              riotApiClient,
              match.matchId
            );

            // Update the match record with processed data
            await prisma.match.update({
              where: {
                id: match.id,
              },
              data: {
                processed: true,
                gameDuration: processedData.gameDuration,
                gameStartTimestamp: new Date(processedData.gameStartTimestamp),
                queueId: processedData.queueId,
                gameVersionMajorPatch: processedData.gameVersionMajorPatch,
                gameVersionMinorPatch: processedData.gameVersionMinorPatch,
                teams: processedData.teams,
              },
            });

            telemetry.trackEvent("MatchesProcessed", {
              count: 1,
              region,
            });
          } catch (error) {
            // Mark the match as processed but with an error
            await prisma.match.update({
              where: {
                id: match.id,
              },
              data: {
                processed: true,
                processingErrored: true,
              },
            });
          }
        });
      }
    }
  } catch (error) {
    console.error("Error in processMatches:", error);
  } finally {
    await prisma.$disconnect();
  }
}

processMatches();
