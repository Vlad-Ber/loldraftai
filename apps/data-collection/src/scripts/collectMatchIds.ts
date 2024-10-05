import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import Bottleneck from "bottleneck";
import { sleep } from "../utils";
import { RegionSchema } from "@draftking/riot-api";
import { RiotAPIClient } from "@draftking/riot-api";
import { PrismaClient } from "@draftking/riot-database";
import { config } from "dotenv";

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
  // Limit: 2000 requests every 10 seconds, but we do 500 to be safe
  reservoir: 500,
  reservoirRefreshAmount: 500,
  reservoirRefreshInterval: 11 * 1000, // 10 seconds

  // Adjust maxConcurrent based on your needs and system capabilities
  maxConcurrent: 50,
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
        },
        take: 1000, // Adjust the batch size as needed
      });

      if (summoners.length === 0) {
        console.log(
          "No summoners to fetch match IDs for. Sleeping for 1 minute."
        );
        await sleep(60 * 1000);
        continue;
      }
      console.log(`Processing ${summoners.length} summoners.`);

      await Promise.all(
        summoners.map((summoner) =>
          limiter.schedule(async () => {
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
              await prisma.match.createMany({
                data: matchCreates,
                skipDuplicates: true,
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
            } catch (error) {
              console.error(
                `Error fetching match IDs for summoner ${summoner.summonerId}:`,
                error
              );
            }
          })
        )
      );
    }
  } catch (error) {
    console.error("Error in collectMatchIds:", error);
  } finally {
    await prisma.$disconnect();
  }
}

collectMatchIds();
