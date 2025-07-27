/*
  Warnings:

  - You are about to drop the column `summonerId` on the `Summoner` table. All the data in the column will be lost.
  - A unique constraint covering the columns `[puuid]` on the table `Summoner` will be added. If there are existing duplicate values, this will fail.
  - Made the column `puuid` on table `Summoner` required. This step will fail if there are existing NULL values in that column.

*/
-- DropIndex
DROP INDEX "Summoner_puuid_region_idx";

-- DropIndex
DROP INDEX "Summoner_summonerId_region_key";

-- AlterTable
ALTER TABLE "Summoner" DROP COLUMN "summonerId",
ALTER COLUMN "puuid" SET NOT NULL;

-- CreateIndex
CREATE UNIQUE INDEX "Summoner_puuid_key" ON "Summoner"("puuid");
