import { z } from "zod";

export const RegionSchema = z.enum([
  "BR1",
  "EUN1",
  "EUW1",
  "JP1",
  "KR",
  "LA1",
  "LA2",
  "ME1",
  "NA1",
  "OC1",
  "PH2",
  "RU",
  "SG2",
  "TH2",
  "TR1",
  "TW2",
  "VN2",
]);
export type Region = z.infer<typeof RegionSchema>;

export const QueueTypeSchema = z.enum([
  "RANKED_SOLO_5x5",
  "RANKED_FLEX_SR",
  "RANKED_FLEX_TT",
]);
export type QueueType = z.infer<typeof QueueTypeSchema>;

export const TierSchema = z.enum([
  "CHALLENGER",
  "GRANDMASTER",
  "MASTER",
  "DIAMOND",
  "EMERALD",
  "PLATINUM",
  "GOLD",
  "SILVER",
  "BRONZE",
  "IRON",
]);
export type Tier = z.infer<typeof TierSchema>;

export const DivisionSchema = z.enum(["I", "II", "III", "IV"]);
export type Division = z.infer<typeof DivisionSchema>;

export const TierDivisionPairSchema = z.union([
  z.tuple([z.enum(["CHALLENGER", "GRANDMASTER", "MASTER"]), z.literal("I")]),
  z.tuple([
    z.enum([
      "DIAMOND",
      "EMERALD",
      "PLATINUM",
      "GOLD",
      "SILVER",
      "BRONZE",
      "IRON",
    ]),
    DivisionSchema,
  ]),
]);
export type TierDivisionPair = z.infer<typeof TierDivisionPairSchema>;

export const MiniSeriesDTOSchema = z.object({
  losses: z.number(),
  progress: z.string(),
  target: z.number(),
  wins: z.number(),
});

export const LeagueEntryDTOSchema = z.object({
  leagueId: z.string(),
  summonerId: z.string(),
  queueType: QueueTypeSchema,
  tier: TierSchema,
  rank: DivisionSchema,
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

export const SummonerDTOSchema = z.object({
  accountId: z.string().max(56),
  profileIconId: z.number(),
  revisionDate: z.number(),
  id: z.string().max(63),
  puuid: z.string().length(78),
  summonerLevel: z.number(),
});
export type SummonerDTO = z.infer<typeof SummonerDTOSchema>;
