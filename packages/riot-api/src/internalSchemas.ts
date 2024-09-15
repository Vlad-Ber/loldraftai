import { z } from "zod";
import {
  MiniSeriesDTOSchema,
  QueueTypeSchema,
  TierSchema,
  DivisionSchema,
} from "./schemas";

export const LeagueItemDTOSchema = z.object({
  freshBlood: z.boolean(),
  wins: z.number(),
  miniSeries: MiniSeriesDTOSchema.optional(),
  inactive: z.boolean(),
  veteran: z.boolean(),
  hotStreak: z.boolean(),
  rank: DivisionSchema,
  leaguePoints: z.number(),
  losses: z.number(),
  summonerId: z.string(),
});
export type LeagueItemDTO = z.infer<typeof LeagueItemDTOSchema>;

export const LeagueListDTOSchema = z.object({
  leagueId: z.string(),
  entries: z.array(LeagueItemDTOSchema),
  tier: TierSchema,
  name: z.string(),
  queue: QueueTypeSchema,
});
export type LeagueListDTO = z.infer<typeof LeagueListDTOSchema>;
