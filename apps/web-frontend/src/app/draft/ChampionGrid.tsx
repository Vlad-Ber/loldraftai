"use client";
import React, { useState, useEffect, useRef } from "react";
import Image from "next/image";
import { StarIcon } from "@heroicons/react/24/solid";
import Cookies from "js-cookie";
import type { Champion, FavoriteChampions } from "@/app/types";

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
    <div className="w-full text-black">
      <input
        className="w-full rounded-t border p-2"
        type="text"
        placeholder="Search..."
        onChange={(e) => setSearchTerm(e.target.value)}
        value={searchTerm}
        onKeyDown={(e) => handleKeyDown(e)}
      />
      {filteredChampions.length === 1 && (
        <div className="text-left text-white">
          (press enter to select {filteredChampions[0]?.name})
        </div>
      )}
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

interface ChampionMenuProps {
  anchorEl: HTMLElement | null;
  handleClose: () => void;
  selectedChampion: Champion | null;
  favorites: FavoriteChampions;
  handleAddToFavorites: (position: keyof FavoriteChampions) => void;
  handleRemoveFromFavorites: (position: keyof FavoriteChampions) => void;
}

const ChampionMenu = ({
  anchorEl,
  handleClose,
  selectedChampion,
  favorites,
  handleAddToFavorites,
  handleRemoveFromFavorites,
}: ChampionMenuProps) => {
  const menuRef = useRef<HTMLDivElement>(null);
  const rect = anchorEl?.getBoundingClientRect();

  useEffect(() => {
    const checkIfClickedOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        handleClose();
      }
    };

    document.addEventListener("mousedown", checkIfClickedOutside);

    return () => {
      document.removeEventListener("mousedown", checkIfClickedOutside);
    };
  }, [handleClose]);

  return (
    <div
      ref={menuRef}
      id="champion-menu"
      className={`absolute rounded-md bg-white text-black shadow-md ${
        anchorEl ? "block" : "hidden"
      }`}
      style={{
        top: `${window.scrollY + (rect?.bottom ?? 0)}px`,
        left: `${window.scrollX + (rect?.left ?? 0)}px`,
        zIndex: 1000,
      }}
    >
      <ul className="m-0 list-none p-0">
        {["top", "jungle", "mid", "bot", "support"].map((position) =>
          selectedChampion &&
          favorites[position as keyof FavoriteChampions].includes(
            selectedChampion.id
          ) ? (
            <li
              key={position}
              className="flex cursor-pointer items-center p-2 hover:bg-gray-100"
              onClick={() =>
                handleRemoveFromFavorites(position as keyof FavoriteChampions)
              }
            >
              <StarIcon
                className="h-5 w-5 text-white"
                stroke="black"
                strokeWidth={2}
              />
              <span className="ml-2">
                Remove from{" "}
                {position.charAt(0).toUpperCase() + position.slice(1)} Favorites
              </span>
            </li>
          ) : (
            <li
              key={position}
              className="flex cursor-pointer items-center p-2 hover:bg-gray-100"
              onClick={() =>
                handleAddToFavorites(position as keyof FavoriteChampions)
              }
            >
              <StarIcon
                className="h-5 w-5 text-yellow-500"
                stroke="black"
                strokeWidth={2}
              />
              <span className="ml-2">
                Add to {position.charAt(0).toUpperCase() + position.slice(1)}{" "}
                Favorites
              </span>
            </li>
          )
        )}
      </ul>
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
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [selectedChampion, setSelectedChampion] = useState<null | Champion>(
    null
  );
  const [open, setOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [filteredChampions, setFilteredChampions] = useState(champions);

  // Update the filtered champions whenever the search term or the list of champions changes
  // TODO: can probably optmize all the id lookups in filteredChampions(and when hiding cards)
  useEffect(() => {
    const results = champions.filter((champion) =>
      champion.searchName.includes(searchTerm.toLowerCase())
    );
    setFilteredChampions(results);
  }, [searchTerm, champions]);

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
    setAnchorEl(event.currentTarget);
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
    setAnchorEl(null);
  };

  const handleAddToFavorites = (position: keyof FavoriteChampions) => {
    if (selectedChampion) {
      const updatedFavorites = {
        ...favorites,
        [position]:
          selectedChampion.id in favorites[position]
            ? favorites[position]
            : [...favorites[position], selectedChampion.id],
      };

      setFavorites(updatedFavorites);
      Cookies.set("favorites", JSON.stringify(updatedFavorites), {
        expires: 365,
      }); // Set cookie with 1 year expiry
    }

    handleClose();
  };

  const handleRemoveFromFavorites = (position: keyof FavoriteChampions) => {
    if (selectedChampion) {
      const updatedFavorites = {
        ...favorites,
        [position]: favorites[position].filter(
          (champion) => champion !== selectedChampion.id
        ),
      };

      setFavorites(updatedFavorites);
      Cookies.set("favorites", JSON.stringify(updatedFavorites), {
        expires: 365,
      }); // Set cookie with 1 year expiry
    }

    handleClose();
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
        <div className="grid grid-cols-1 justify-items-center gap-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
          {champions.map((champion) => (
            <div
              key={champion.id}
              onContextMenu={(e) => handleContextMenu(e, champion)}
              onClick={() => handleChampionSelection(champion)}
              onTouchStart={(e) => handleTouchStart(e, champion)}
              onTouchEnd={handleTouchEnd}
              onTouchMove={handleTouchMove}
              onDragStart={preventDefaultActions}
              onDrop={preventDefaultActions}
              /* Using hidden to keep image in memory when doing search */
              className={`cursor-pointer ${
                filteredChampions.map((c) => c.id).includes(champion.id)
                  ? "block"
                  : "hidden"
              }`}
            >
              <ChampionCard champion={champion} favorites={favorites} />
            </div>
          ))}
        </div>
      </div>
      {open && (
        <ChampionMenu
          anchorEl={anchorEl}
          handleClose={handleClose}
          selectedChampion={selectedChampion}
          favorites={favorites}
          handleAddToFavorites={handleAddToFavorites}
          handleRemoveFromFavorites={handleRemoveFromFavorites}
        />
      )}
    </div>
  );
};

export default ChampionGrid;
