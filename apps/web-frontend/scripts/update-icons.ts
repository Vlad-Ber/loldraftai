import { champions } from "@draftking/ui/lib/champions";
import fsPromises from "fs/promises";
import path from "path";
import https from "https";

const ICON_PATHS = ["../../../apps/desktop/public/icons/champions"];

async function downloadAndCompareImage(
  url: string,
  filepath: string
): Promise<void> {
  return new Promise((resolve, reject) => {
    https
      .get(url, async (response) => {
        if (response.statusCode !== 200) {
          reject(new Error(`Failed to download: ${response.statusCode}`));
          return;
        }

        // Collect the image data in memory first
        const chunks: Buffer[] = [];
        response.on("data", (chunk) => chunks.push(Buffer.from(chunk)));

        response.on("end", async () => {
          const newImageData = Buffer.concat(chunks);

          try {
            // Check if file exists and compare content
            let shouldWrite = true;
            try {
              const existingImageData = await fsPromises.readFile(filepath);
              if (existingImageData.equals(newImageData)) {
                console.log(
                  `⏭️ Skipping ${path.basename(filepath)} (unchanged)`
                );
                shouldWrite = false;
              }
            } catch (err) {
              // File doesn't exist, we should write it
            }

            if (shouldWrite) {
              await fsPromises.writeFile(filepath, newImageData);
              console.log(`✓ Updated ${path.basename(filepath)}`);
            }

            resolve();
          } catch (error) {
            reject(error);
          }
        });
      })
      .on("error", reject);
  });
}

async function ensureDirectoryExists(dirPath: string): Promise<void> {
  try {
    await fsPromises.access(dirPath);
  } catch {
    await fsPromises.mkdir(dirPath, { recursive: true });
  }
}

// First, fetch the champion data from Data Dragon to get the mapping
async function getDataDragonMapping(
  version: string
): Promise<Record<number, string>> {
  return new Promise((resolve, reject) => {
    https
      .get(
        `https://ddragon.leagueoflegends.com/cdn/${version}/data/en_US/champion.json`,
        (response) => {
          let data = "";
          response.on("data", (chunk) => (data += chunk));
          response.on("end", () => {
            try {
              const championData = JSON.parse(data).data;
              const mapping: Record<number, string> = {};

              interface ChampionData {
                key: string;
                id: string;
              }

              Object.values<ChampionData>(championData).forEach((champion) => {
                mapping[parseInt(champion.key)] = champion.id;
              });

              resolve(mapping);
            } catch (error) {
              reject(error);
            }
          });
        }
      )
      .on("error", reject);
  });
}

async function getLatestVersion(): Promise<string> {
  return new Promise((resolve, reject) => {
    https
      .get(
        "https://ddragon.leagueoflegends.com/api/versions.json",
        (response) => {
          let data = "";
          response.on("data", (chunk) => (data += chunk));
          response.on("end", () => {
            try {
              const versions = JSON.parse(data);
              resolve(versions[0]); // First version is always the latest
            } catch (error) {
              reject(error);
            }
          });
        }
      )
      .on("error", reject);
  });
}

async function main(): Promise<void> {
  const version = await getLatestVersion();
  console.log(`Using Data Dragon version: ${version}`);

  const ICON_BASE_URL = `https://ddragon.leagueoflegends.com/cdn/${version}/img/champion/`;
  const dataDragonMapping = await getDataDragonMapping(version);

  for (const iconPath of ICON_PATHS) {
    const fullPath = path.resolve(__dirname, iconPath);
    await ensureDirectoryExists(fullPath);

    console.log(`\nProcessing icons in ${fullPath}...`);

    for (const champion of champions) {
      const dataDragonName = dataDragonMapping[champion.id];
      if (!dataDragonName) {
        console.error(
          `❌ No Data Dragon mapping found for ${champion.name} (ID: ${champion.id})`
        );
        continue;
      }

      const iconUrl = `${ICON_BASE_URL}${dataDragonName}.png`;
      const filePath = path.join(fullPath, champion.icon);

      try {
        await downloadAndCompareImage(iconUrl, filePath);
      } catch (error) {
        console.error(`✗ Failed to process ${champion.icon}:`, error);
      }
    }
  }
}

main().catch(console.error);
