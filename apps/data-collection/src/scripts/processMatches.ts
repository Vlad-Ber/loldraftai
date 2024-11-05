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
  minTime: 100, // 20ms between requests (50 requests per second)
  // Limit: 2000 requests every 10 seconds
  reservoir: 500,
  reservoirRefreshAmount: 500,
  reservoirRefreshInterval: 10 * 1000, // 10 seconds

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 50,
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
    while (true) {
      log("Starting new batch fetch");
      const fetchStart = Date.now();

      // Fetch unprocessed matches
      const matches = (await prisma.$queryRaw`
        SELECT *
        FROM "Match"
        WHERE processed = false 
        AND "processingErrored" = false
        AND region = ${region}::text::"Region"
        LIMIT 100
        `) as Match[];

      log(
        `Database fetch took ${Date.now() - fetchStart}ms for ${
          matches.length
        } matches`
      );

      if (matches.length === 0) {
        log("No matches found, sleeping");
        await sleep(60 * 1000);
        continue;
      }

      // Process matches in parallel within rate limits
      await Promise.all(
        matches.map((match) =>
          limiter.schedule(async () => {
            log(`Starting to process match ${match.matchId}`);

            try {
              // Time the API call
              const apiStart = Date.now();
              const processedData = await processMatchData(
                riotApiClient,
                match.matchId
              );
              log(
                `API call took ${Date.now() - apiStart}ms for match ${
                  match.matchId
                }`
              );

              // Time the database update
              const updateStart = Date.now();
              await prisma.match.update({
                where: { id: match.id },
                data: {
                  processed: true,
                  gameDuration: processedData.gameDuration,
                  gameStartTimestamp: new Date(
                    processedData.gameStartTimestamp
                  ),
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
                `Error update took ${
                  Date.now() - errorUpdateStart
                }ms for match ${match.matchId}`
              );
            }
          })
        )
      );
    }
  } catch (error) {
    log(`Fatal error in processMatches: ${error}`);
  } finally {
    await prisma.$disconnect();
  }
}

processMatches();
