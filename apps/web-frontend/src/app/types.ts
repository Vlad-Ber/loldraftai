export interface Champion {
  id: string;
  name: string;
  searchName: string;
  icon: string;
}

// eslint-disable-next-line @typescript-eslint/consistent-indexed-object-style
export interface Team {
  //TODO: championIndex or maybe indexed object style eslint fix
  [key: number]: Champion | undefined;
}

export type TeamIndex = 1 | 2;
export type ChampionIndex = 0 | 1 | 2 | 3 | 4;

export const championIndexToFavoritesPosition = (index: ChampionIndex) => {
  switch (index) {
    case 0:
      return "top";
    case 1:
      return "jungle";
    case 2:
      return "mid";
    case 3:
      return "bot";
    case 4:
      return "support";
  }
};

export type FavoriteChampions = {
  top: string[];
  jungle: string[];
  mid: string[];
  bot: string[];
  support: string[];
};

export interface SelectedSpot {
  teamIndex: TeamIndex;
  championIndex: ChampionIndex;
}
