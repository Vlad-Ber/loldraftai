"use client";
import React from "react";
import Image from "next/image";
import { ChampionGrid as SharedChampionGrid } from "@draftking/ui/components/draftking/ChampionGrid";
import type {
  Champion,
  FavoriteChampions,
  ImageComponent,
} from "@draftking/ui/components/draftking/ChampionGrid";

interface ChampionGridProps {
  champions: Champion[];
  addChampion: (champion: Champion) => void;
  favorites: FavoriteChampions;
  setFavorites: (favorites: FavoriteChampions) => void;
}

const ChampionGrid: React.FC<ChampionGridProps> = (props) => {
  return (
    <SharedChampionGrid
      {...props}
      ImageComponent={Image as ImageComponent}
      setFavorites={props.setFavorites}
    />
  );
};

export default ChampionGrid;
