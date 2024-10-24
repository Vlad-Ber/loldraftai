-- DropIndex
DROP INDEX "Match_exported_idx";

-- DropIndex
DROP INDEX "Match_gameStartTimestamp_idx";

-- DropIndex
DROP INDEX "Match_processed_idx";

-- DropIndex
DROP INDEX "Match_processed_processingErrored_idx";

-- DropIndex
DROP INDEX "Match_region_idx";

-- DropIndex
DROP INDEX "Summoner_matchesFetchedAt_idx";

-- DropIndex
DROP INDEX "Summoner_puuid_idx";

-- DropIndex
DROP INDEX "Summoner_region_idx";

-- CreateIndex
CREATE INDEX "Match_processed_region_processingErrored_idx" ON "Match"("processed", "region", "processingErrored");

-- CreateIndex
CREATE INDEX "Match_processed_exported_processingErrored_idx" ON "Match"("processed", "exported", "processingErrored");

-- CreateIndex
CREATE INDEX "Summoner_matchesFetchedAt_rankUpdateTime_region_idx" ON "Summoner"("matchesFetchedAt", "rankUpdateTime", "region");

-- CreateIndex
CREATE INDEX "Summoner_puuid_region_idx" ON "Summoner"("puuid", "region");
