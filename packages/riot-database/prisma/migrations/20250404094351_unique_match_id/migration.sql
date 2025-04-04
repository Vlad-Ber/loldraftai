/*
  Warnings:

  - A unique constraint covering the columns `[matchId]` on the table `Match` will be added. If there are existing duplicate values, this will fail.

*/
-- DropIndex
DROP INDEX "Match_matchId_region_key";

-- CreateIndex
CREATE UNIQUE INDEX "Match_matchId_key" ON "Match"("matchId");
