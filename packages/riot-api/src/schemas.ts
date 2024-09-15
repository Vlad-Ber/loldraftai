import { z } from "zod";

export const QueueTypeSchema = z.enum([
  "RANKED_SOLO_5x5",
  "RANKED_FLEX_SR",
  "RANKED_FLEX_TT",
]);
export type QueueType = z.infer<typeof QueueTypeSchema>;

export const DivisionSchema = z.enum(["I", "II", "III", "IV"]);
export type Division = z.infer<typeof DivisionSchema>;

export const TierSchema = z.enum([
  "DIAMOND",
  "EMERALD",
  "PLATINUM",
  "GOLD",
  "SILVER",
  "BRONZE",
  "IRON",
]);
export type Tier = z.infer<typeof TierSchema>;

export const MiniSeriesDTOSchema = z.object({
  losses: z.number(),
  progress: z.string(),
  target: z.number(),
  wins: z.number(),
});
export type MiniSeriesDTO = z.infer<typeof MiniSeriesDTOSchema>;

// For master to challenger
export const LeagueItemDTOSchema = z.object({
  freshBlood: z.boolean(),
  wins: z.number(),
  miniSeries: MiniSeriesDTOSchema.optional(),
  inactive: z.boolean(),
  veteran: z.boolean(),
  hotStreak: z.boolean(),
  rank: z.string(),
  leaguePoints: z.number(),
  losses: z.number(),
  summonerId: z.string(),
});
export type LeagueItemDTO = z.infer<typeof LeagueItemDTOSchema>;

// For Iron to Diamond
export const LeagueEntryDTOSchema = z.object({
  leagueId: z.string(),
  summonerId: z.string(),
  queueType: z.string(),
  tier: TierSchema,
  rank: z.string(),
  leaguePoints: z.number(),
  wins: z.number(),
  losses: z.number(),
  hotStreak: z.boolean(),
  veteran: z.boolean(),
  freshBlood: z.boolean(),
  inactive: z.boolean(),
  miniSeries: MiniSeriesDTOSchema.optional(),
});
export type LeagueEntryDTO = z.infer<typeof LeagueEntryDTOSchema>;

export const LeagueListDTOSchema = z.object({
  leagueId: z.string(),
  entries: z.array(LeagueItemDTOSchema),
  tier: z.string(),
  name: z.string(),
  queue: z.string(),
});
export type LeagueListDTO = z.infer<typeof LeagueListDTOSchema>;
