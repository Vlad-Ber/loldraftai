"use client";
import React, { useState, useEffect, useCallback } from "react";
import Image from "next/image";
import { StarIcon } from "@heroicons/react/24/solid";
import Cookies from "js-cookie";
import type { Champion, FavoriteChampions } from "@/app/types";
import { Input } from "@/components/ui/input";
import {
  ContextMenu,
  ContextMenuTrigger,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuPortal,
} from "@/components/ui/context-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SearchBarProps {
  searchTerm: string;
  setSearchTerm: (searchTerm: string) => void;
  handleKeyDown: (e: React.KeyboardEvent) => void;
  filteredChampions: Champion[];
}

const SearchBar = ({
  searchTerm,
  setSearchTerm,
  handleKeyDown,
  filteredChampions,
}: SearchBarProps) => {
  return (
    // Relative z-10 to make sure it stays above icons when scrolling(search bar grows when selected, which could cause overlap)
    <div className="w-full relative z-10">
      <TooltipProvider>
        <Tooltip open={filteredChampions.length === 1}>
          <TooltipTrigger asChild>
            <Input
              className="rounded-t"
              type="text"
              placeholder="Search..."
              onChange={(e) => setSearchTerm(e.target.value)}
              value={searchTerm}
              onKeyDown={(e) => handleKeyDown(e)}
            />
          </TooltipTrigger>
          <TooltipContent side="top" align="start">
            Press enter to select {filteredChampions[0]?.name}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};

interface ChampionCardProps {
  champion: Champion;
  favorites: FavoriteChampions;
}

const ChampionCard = ({ champion, favorites }: ChampionCardProps) => {
  const isFavorite = Object.values(favorites).some((favoriteList) =>
    favoriteList.includes(champion.id)
  );

  return (
    <div
      key={champion.id}
      className="relative m-0 cursor-pointer rounded-none p-0 shadow-none"
    >
      {isFavorite && (
        <div className="absolute right-0 top-0">
          <StarIcon
            className="h-6 w-6 text-yellow-500"
            stroke="black"
            strokeWidth={2}
          />
        </div>
      )}
      <Image
        src={`/icons/champions/${champion.icon}`}
        alt={champion.name}
        width={80}
        height={80}
      />
    </div>
  );
};

interface ChampionGridProps {
  champions: Champion[];
  addChampion: (champion: Champion) => void;
  favorites: FavoriteChampions;
  setFavorites: (favorites: FavoriteChampions) => void;
}

const ChampionGrid = ({
  champions,
  addChampion,
  favorites,
  setFavorites,
}: ChampionGridProps) => {
  const [, setSelectedChampion] = useState<null | Champion>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredChampions, setFilteredChampions] = useState(champions);

  const debouncedFilter = useCallback(
    (term: string) => {
      const results = champions.filter((champion) =>
        champion.searchName.includes(term.toLowerCase())
      );
      setFilteredChampions(results);
    },
    [champions]
  );

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      debouncedFilter(searchTerm);
    }, 300); // 300ms delay

    return () => clearTimeout(timeoutId);
  }, [searchTerm, debouncedFilter]);

  let touchTimeout: number | null = null;

  const handleTouchStart = (event: React.TouchEvent, champion: Champion) => {
    // Prevent firing the context menu for non-touch devices
    //event.preventDefault();

    console.log("touchstart with id: ", champion.id);

    // Clear any existing timeout to prevent multiple triggers
    if (touchTimeout) clearTimeout(touchTimeout);

    touchTimeout = window.setTimeout(() => {
      handleContextMenu(
        event as unknown as React.MouseEvent<HTMLElement>,
        champion
      );
    }, 300);
  };

  const handleTouchEnd = () => {
    // Clear the timeout when the user lifts their finger off the screen
    if (touchTimeout) clearTimeout(touchTimeout);
  };

  const handleTouchMove = () => {
    // Clear the timeout if the user moves their finger, indicating they are not performing a long press
    if (touchTimeout) clearTimeout(touchTimeout);
  };

  const preventDefaultActions = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleContextMenu = (
    event: React.MouseEvent<HTMLElement>,
    champion: Champion
  ) => {
    //TODO: seperate into 1 that takes even and other that doesn't because touch can't be prevented
    event.preventDefault();
    setSelectedChampion(champion);
  };

  const handleAddToFavorites = (
    champion: Champion,
    position: keyof FavoriteChampions
  ) => {
    const updatedFavorites = {
      ...favorites,
      [position]: [...favorites[position], champion.id],
    };

    setFavorites(updatedFavorites);
    Cookies.set("favorites", JSON.stringify(updatedFavorites), {
      expires: 365,
    });
  };

  const handleRemoveFromFavorites = (
    champion: Champion,
    position: keyof FavoriteChampions
  ) => {
    const updatedFavorites = {
      ...favorites,
      [position]: favorites[position].filter((id) => id !== champion.id),
    };

    setFavorites(updatedFavorites);
    Cookies.set("favorites", JSON.stringify(updatedFavorites), {
      expires: 365,
    });
  };

  const handleChampionSelection = (champion: Champion) => {
    addChampion(champion);
    setSearchTerm("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (
      e.key === "Enter" &&
      filteredChampions.length === 1 &&
      filteredChampions[0]
    ) {
      handleChampionSelection(filteredChampions[0]);
    }
  };

  return (
    <div className="h-[560px] rounded border border-gray-200 bg-zinc-800">
      <SearchBar
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        handleKeyDown={handleKeyDown}
        filteredChampions={filteredChampions}
      />
      <div className="h-[505px] overflow-y-auto p-1">
        <div className="grid grid-cols-[repeat(auto-fill,80px)] justify-center gap-2">
          {champions.map((champion) => (
            // Using ContextMenu from shadcn/ui which provides individual exports rather than dot notation
            <ContextMenu key={champion.id}>
              {/* 
                Using absolute positioning for hidden champions instead of display: none
                This keeps the components mounted (preserving loaded images) while removing them from layout flow
                pointer-events-none prevents any interaction with hidden elements
              */}
              <ContextMenuTrigger
                className={`${
                  filteredChampions.map((c) => c.id).includes(champion.id)
                    ? "relative"
                    : "absolute invisible pointer-events-none"
                }`}
              >
                <div
                  onClick={() => handleChampionSelection(champion)}
                  onTouchStart={(e) => handleTouchStart(e, champion)}
                  onTouchEnd={handleTouchEnd}
                  onTouchMove={handleTouchMove}
                  onDragStart={preventDefaultActions}
                  onDrop={preventDefaultActions}
                >
                  <ChampionCard champion={champion} favorites={favorites} />
                </div>
              </ContextMenuTrigger>
              {/* 
                ContextMenuPortal ensures menu content is rendered in a portal
                This prevents the menu from affecting grid layout or being clipped
                by parent containers
              */}
              <ContextMenuPortal>
                <ContextMenuContent>
                  {["top", "jungle", "mid", "bot", "support"].map(
                    (position) => {
                      const isFavorite = favorites[
                        position as keyof FavoriteChampions
                      ].includes(champion.id);
                      return (
                        <ContextMenuItem
                          key={position}
                          onClick={() =>
                            isFavorite
                              ? handleRemoveFromFavorites(
                                  champion,
                                  position as keyof FavoriteChampions
                                )
                              : handleAddToFavorites(
                                  champion,
                                  position as keyof FavoriteChampions
                                )
                          }
                        >
                          <StarIcon
                            className={`mr-2 h-5 w-5 ${
                              isFavorite ? "text-yellow-500" : "text-white"
                            }`}
                            stroke="black"
                            strokeWidth={2}
                          />
                          {isFavorite
                            ? `Remove from ${
                                position.charAt(0).toUpperCase() +
                                position.slice(1)
                              } Favorites`
                            : `Add to ${
                                position.charAt(0).toUpperCase() +
                                position.slice(1)
                              } Favorites`}
                        </ContextMenuItem>
                      );
                    }
                  )}
                </ContextMenuContent>
              </ContextMenuPortal>
            </ContextMenu>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ChampionGrid;
