import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RiotAPIClient, RegionSchema } from "@draftking/riot-api";
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

// Rate limiter settings based on the API rate limits
const limiter = new Bottleneck({
  // Short-term limit: 30 requests every 10 seconds
  minTime: 200, // 200ms between requests (5 requests per second)

  // Limit: 1600 requests every 1 minute
  reservoir: 500,
  reservoirRefreshAmount: 500,
  reservoirRefreshInterval: 60 * 1000, // 1 minute

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 30,
  highWater: 1000,
  strategy: Bottleneck.strategy.BLOCK,
});

async function fetchPuuids() {
  try {
    while (true) {
      // Fetch summoners without a PUUID
      const summoners = (await prisma.$queryRaw`
        SELECT *
        FROM "Summoner"
        WHERE puuid IS NULL
        AND region = ${region}::text::"Region"
        LIMIT 100
      `) as Summoner[];

      if (summoners.length === 0) {
        await sleep(60 * 1000);
        continue;
      }

      for (const summoner of summoners) {
        await limiter.schedule(async () => {
          try {
            const summonerDTO = await riotApiClient.getSummonerById(
              summoner.summonerId
            );

            // Update the summoner record with the fetched PUUID
            await prisma.summoner.update({
              where: {
                id: summoner.id,
              },
              data: {
                puuid: summonerDTO.puuid,
              },
            });

            telemetry.trackEvent("PUUIDsFetched", {
              count: 1,
              region,
            });
          } catch (error) {
            console.error(
              `Error updating summoner ${summoner.summonerId}:`,
              error
            );
          }
        });
      }
    }
  } catch (error) {
    console.error("Error in convertSummonerIdsToPuuids:", error);
  } finally {
    await prisma.$disconnect();
  }
}

fetchPuuids();
