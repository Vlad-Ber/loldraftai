// src/scripts/extractToAzure.ts
import fs from "fs";
import { BlobServiceClient, ContainerClient } from "@azure/storage-blob";
import { Match, PrismaClient } from "@prisma/client";
import { spawn } from "child_process";
import path from "path";
import { config } from "dotenv";
import { telemetry } from "../utils/telemetry";

config();

const AZURE_CONNECTION_STRING = process.env.AZURE_CONNECTION_STRING!;

if (!AZURE_CONNECTION_STRING) {
  throw new Error("AZURE_CONNECTION_STRING is not set");
}

const BATCH_SIZE = 512 * 4;

const CONTAINER_NAME = "league-matches";
const RAW_DATA_PREFIX = "raw";
const PROCESSED_DATA_PREFIX = "processed";
const currentFileDir = path.dirname(new URL(import.meta.url).pathname);

interface ExtractorConfig {
  batchSize: number;
  tempDir: string;
}

interface FileSaver {
  saveFile(fileName: string, data: string | Buffer): Promise<void>;
}

class AzureFileSaver implements FileSaver {
  constructor(private containerClient: ContainerClient) {}

  async saveFile(fileName: string, data: string | Buffer): Promise<void> {
    const blob = this.containerClient.getBlockBlobClient(fileName);
    if (typeof data === "string") {
      await blob.upload(data, data.length);
    } else {
      await blob.uploadData(data);
    }
  }
}

class LocalFileSaver implements FileSaver {
  private tempDir: string;

  constructor() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    this.tempDir = `/tmp/${timestamp}`;
    fs.mkdirSync(this.tempDir, { recursive: true });
  }

  async saveFile(fileName: string, data: string | Buffer): Promise<void> {
    const filePath = path.join(this.tempDir, fileName);
    const dir = path.dirname(filePath);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(filePath, data);
  }
}

class MatchExtractor {
  private prisma: PrismaClient;
  private remoteFileSaver: FileSaver;

  constructor(private config: ExtractorConfig) {
    this.prisma = new PrismaClient();
    const blobServiceClient = BlobServiceClient.fromConnectionString(
      AZURE_CONNECTION_STRING
    );
    const containerClient =
      blobServiceClient.getContainerClient(CONTAINER_NAME);

    this.remoteFileSaver =
      process.env.NODE_ENV === "development"
        ? new LocalFileSaver()
        : new AzureFileSaver(containerClient);
  }

  async initialize() {
    fs.mkdirSync(this.config.tempDir, { recursive: true });
  }

  async extractBatch() {
    try {
      // Get batch of unprocessed matches
      const matches = (await this.prisma.$queryRaw`
        SELECT *
        FROM "Match"
        WHERE exported = false
        AND processed = true
        AND "processingErrored" = false
        LIMIT ${this.config.batchSize}
      `) as Match[];

      // Process full batches only
      if (matches.length < this.config.batchSize) {
        return 0;
      }

      const now = new Date();
      const year = now.getFullYear();
      const month = (now.getMonth() + 1).toString().padStart(2, "0");
      const batchId = now.toISOString().replace(/[:.]/g, "-");

      // Use hierarchical paths
      const remoteRawFileName = `${RAW_DATA_PREFIX}/${year}/${month}/${batchId}.json`;
      const processedFileName = `${PROCESSED_DATA_PREFIX}/${year}/${month}/${batchId}.parquet`;

      // Save raw JSON to remote storage
      await this.remoteFileSaver.saveFile(
        remoteRawFileName,
        JSON.stringify(matches)
      );

      // Save raw JSON to local storage(to be processed by python)
      const rawLocalFilePath = path.join(
        this.config.tempDir,
        `raw_${batchId}.json`
      );
      const processedLocalFilePath = path.join(
        this.config.tempDir,
        `processed_${batchId}.parquet`
      );
      fs.mkdirSync(path.dirname(rawLocalFilePath), { recursive: true }); // in case the /tmp directory was deleted automatically
      fs.writeFileSync(rawLocalFilePath, JSON.stringify(matches));

      // Create parquet file
      const pythonProcess = spawn("python", [
        path.join(currentFileDir, "createParquet.py"),
        "--batch-file",
        rawLocalFilePath,
        "--output-file",
        processedLocalFilePath,
      ]);

      let pythonOutput = "";
      let pythonError = "";

      pythonProcess.stdout.on("data", (data) => {
        pythonOutput += data.toString();
      });

      pythonProcess.stderr.on("data", (data) => {
        pythonError += data.toString();
      });

      await new Promise((resolve, reject) => {
        pythonProcess.on("exit", (code) => {
          if (code === 0) {
            resolve(null);
          } else {
            reject(
              new Error(
                `Python process exited with code ${code}.\nError: ${pythonError}\nOutput: ${pythonOutput}`
              )
            );
          }
        });
      });

      // Save processed parquet
      const processedData = fs.readFileSync(processedLocalFilePath);
      await this.remoteFileSaver.saveFile(processedFileName, processedData);

      // Mark as exported
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

      // Clean up
      fs.rmSync(this.config.tempDir, { recursive: true, force: true });

      // Track the number of matches extracted
      telemetry.trackEvent("MatchesExtracted", {
        count: matches.length,
      });

      if (matches.length > 0) {
        // Add a small delay between batches
        await new Promise((resolve) => setTimeout(resolve, 5000));
      }

      return matches.length;
    } catch (error) {
      console.error("Batch extraction failed:", error);
      throw error;
    }
  }

  async run() {
    try {
      await this.initialize();

      while (true) {
        const processedCount = await this.extractBatch();

        if (processedCount === 0) {
          // Sleep for 1 hour
          await new Promise((resolve) => setTimeout(resolve, 1000 * 60 * 60));
        }
      }
    } catch (error) {
      await telemetry.flush();
      console.error("Extractor crashed:", error);
      throw error;
    }
  }
}

// Run the extractor
const extractor = new MatchExtractor({
  batchSize: BATCH_SIZE,
  tempDir: "/tmp/leaguedraft-extractor",
});

extractor.run().catch(console.error);
