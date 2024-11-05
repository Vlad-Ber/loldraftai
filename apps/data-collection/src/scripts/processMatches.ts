import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RegionSchema } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { Match, PrismaClient } from "@draftking/riot-database";
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
  minTime: 200,
  // Limit: 2000 requests every 10 seconds
  reservoir: 50,
  reservoirRefreshAmount: 50,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // we limit the number of concurrent, to avoid having too many that trigger when a rate limit await is activated
  // this happens when server says to wait before retrying
  maxConcurrent: 10,
});

// Add debug logger
const debug = process.env.DEBUG === "true";
const log = (message: string) => {
  if (debug) {
    console.log(`[${new Date().toISOString()}] ${message}`);
  }
};

async function processMatches() {
  try {
    let nextMatchesPromise: Promise<Match[]> | null = null;

    while (true) {
      log("Starting new batch fetch");

      // Use the pre-fetched matches if available, or fetch the first batch
      const matches = await (nextMatchesPromise ??
        (prisma.$queryRaw`
        SELECT *
        FROM "Match"
        WHERE processed = false 
        AND "processingErrored" = false
        AND region = ${region}::text::"Region"
        LIMIT 500
      ` as Promise<Match[]>));

      // Start fetching next batch immediately, excluding currently processing IDs
      const currentlyProcessingIds = new Set(matches.map((m) => m.id));
      nextMatchesPromise = prisma.$queryRaw`
        SELECT *
        FROM "Match"
        WHERE processed = false 
        AND "processingErrored" = false
        AND region = ${region}::text::"Region"
        AND id NOT IN (${Array.from(currentlyProcessingIds)})
        LIMIT 500
      ` as Promise<Match[]>;

      if (matches.length === 0) {
        log("No matches found, sleeping");
        await sleep(60 * 1000);
        continue;
      }

      // Process matches in parallel without rate limiting the entire block
      await Promise.all(
        matches.map(async (match) => {
          log(`Starting to process match ${match.matchId}`);

          try {
            // Time the API call
            const apiStart = Date.now();
            const processedData = await processMatchData(
              riotApiClient,
              match.matchId,
              limiter
            );
            log(
              `API call took ${Date.now() - apiStart}ms for match ${
                match.matchId
              }`
            );

            // Database updates and telemetry continue without rate limiting
            const updateStart = Date.now();
            await prisma.match.update({
              where: { id: match.id },
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
            log(
              `Database update took ${Date.now() - updateStart}ms for match ${
                match.matchId
              }`
            );

            const telemetryStart = Date.now();
            telemetry.trackEvent("MatchesProcessed", {
              count: 1,
              region,
            });
            log(
              `Telemetry took ${Date.now() - telemetryStart}ms for match ${
                match.matchId
              }`
            );
          } catch (error) {
            log(`Error processing match ${match.matchId}: ${error}`);

            const errorUpdateStart = Date.now();
            await prisma.match.update({
              where: { id: match.id },
              data: {
                processed: true,
                processingErrored: true,
              },
            });
            log(
              `Error update took ${Date.now() - errorUpdateStart}ms for match ${
                match.matchId
              }`
            );
          }
        })
      );
    }
  } catch (error) {
    log(`Fatal error in processMatches: ${error}`);
  } finally {
    await prisma.$disconnect();
  }
}

processMatches();
