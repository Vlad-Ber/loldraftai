-- AlterTable
ALTER TABLE "Match" ADD COLUMN     "exported" BOOLEAN NOT NULL DEFAULT false;

-- CreateIndex
CREATE INDEX "Match_exported_idx" ON "Match"("exported");
