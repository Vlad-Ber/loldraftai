// src/scripts/extractToAzure.ts
import fs from "fs";
import { BlobServiceClient, ContainerClient } from "@azure/storage-blob";
import { PrismaClient } from "@prisma/client";
import { spawn } from "child_process";
import path from "path";
import { config } from "dotenv";

config();

const BATCH_SIZE = 512 * 4;
const AZURE_CONNECTION_STRING = process.env.AZURE_CONNECTION_STRING!;
if (!AZURE_CONNECTION_STRING) {
  throw new Error("AZURE_CONNECTION_STRING is not set");
}
const CONTAINER_NAME = "league-matches";
const RAW_DATA_PREFIX = "raw";
const PROCESSED_DATA_PREFIX = "processed";

const currentFileDir = path.dirname(new URL(import.meta.url).pathname);

interface ExtractorConfig {
  batchSize: number;
  tempDir: string;
}

class MatchExtractor {
  private prisma: PrismaClient;
  private blobServiceClient: BlobServiceClient;
  private containerClient: ContainerClient;

  constructor(private config: ExtractorConfig) {
    this.prisma = new PrismaClient();
    this.blobServiceClient = BlobServiceClient.fromConnectionString(
      AZURE_CONNECTION_STRING
    );
    this.containerClient =
      this.blobServiceClient.getContainerClient(CONTAINER_NAME);
    this.config = config;
  }

  async initialize() {
    // Ensure container exists
    await this.containerClient.createIfNotExists();
    // Ensure temp dir exists
    fs.mkdirSync(this.config.tempDir, { recursive: true });
  }

  async extractBatch() {
    // Get batch of unprocessed matches
    const matches = await this.prisma.match.findMany({
      where: {
        exported: false,
        processed: true,
        processingErrored: false,
      },
      take: this.config.batchSize,
    });

    if (matches.length < this.config.batchSize) {
      return 0; // We wait until we can have at least a full batch
    }

    const batchId = new Date().toISOString().replace(/[:.]/g, "-");
    const rawFilePath = `${this.config.tempDir}/batch_${batchId}.json`;
    const processedFilePath = `${this.config.tempDir}/processed_${batchId}.parquet`;

    // Save raw JSON
    const rawBlob = this.containerClient.getBlockBlobClient(
      `${RAW_DATA_PREFIX}/${batchId}.json`
    );
    await rawBlob.upload(
      JSON.stringify(matches),
      JSON.stringify(matches).length
    );

    // Save raw JSON to temp dir
    fs.writeFileSync(rawFilePath, JSON.stringify(matches));

    // Create temporary parquet file using Python
    const pythonProcess = spawn("python", [
      path.join(currentFileDir, "createParquet.py"),
      "--batch-file",
      rawFilePath,
      "--output-file",
      processedFilePath,
    ]);

    await new Promise((resolve, reject) => {
      pythonProcess.on("exit", (code) => {
        if (code === 0) resolve(null);
        else reject(new Error(`Python process exited with code ${code}`));
      });
    });

    // Upload processed parquet to Azure
    const processedBlob = this.containerClient.getBlockBlobClient(
      `${PROCESSED_DATA_PREFIX}/${batchId}.parquet`
    );
    await processedBlob.uploadFile(processedFilePath);

    // Mark matches as exported
    await this.prisma.match.updateMany({
      where: {
        id: {
          in: matches.map((m) => m.id),
        },
      },
      data: {
        exported: true,
      },
    });

    // Clean up temp files
    fs.rmSync(this.config.tempDir, { recursive: true, force: true });

    return matches.length;
  }

  async run() {
    await this.initialize();

    while (true) {
      const processedCount = await this.extractBatch();
      if (processedCount === 0) {
        console.log("No more matches to process");
        // Sleep for 1 hour
        await new Promise((resolve) => setTimeout(resolve, 1000 * 60 * 60));
      }
      console.log(`Processed ${processedCount} matches`);
    }
  }
}

// Run the extractor
const extractor = new MatchExtractor({
  batchSize: BATCH_SIZE,
  tempDir: "/tmp/leaguedraft-extractor",
});

extractor.run().catch(console.error);
