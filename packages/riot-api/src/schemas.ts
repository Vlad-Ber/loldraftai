import { z } from "zod";

export const QueueTypeSchema = z.enum([
  "RANKED_SOLO_5x5",
  "RANKED_FLEX_SR",
  "RANKED_FLEX_TT",
]);

export const MiniSeriesDTOSchema = z.object({
  losses: z.number(),
  progress: z.string(),
  target: z.number(),
  wins: z.number(),
});

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

export const LeagueListDTOSchema = z.object({
  leagueId: z.string(),
  entries: z.array(LeagueItemDTOSchema),
  tier: z.string(),
  name: z.string(),
  queue: z.string(),
});
